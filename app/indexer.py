"""
Indexer module for Second Brain daemon.
Handles file indexing, updates, and deletion.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

from app.config import config
from app.db import Database, db
from app.parser import MarkdownParser, parser
from app.embeddings import embedding_service

logger = logging.getLogger(__name__)


class Indexer:
    """Indexes markdown files into the database."""
    
    def __init__(
        self, 
        database: Optional[Database] = None,
        md_parser: Optional[MarkdownParser] = None,
        vault_path: Optional[Path] = None
    ):
        self.db = database or db
        self.parser = md_parser or parser
        self.vault_path = vault_path or config.VAULT_PATH
    
    def get_relative_path(self, file_path: Path) -> str:
        """Get path relative to vault for storage."""
        try:
            return str(file_path.relative_to(self.vault_path))
        except ValueError:
            return str(file_path)
    
    def should_index(self, file_path: Path) -> bool:
        """Check if file should be indexed."""
        return (
            file_path.is_file() and
            file_path.suffix.lower() in config.MARKDOWN_EXTENSIONS and
            not file_path.name.startswith('.')
        )
    
    def needs_reindex(self, file_path: Path) -> bool:
        """Check if file needs re-indexing based on mtime."""
        if not self.should_index(file_path):
            return False
        
        rel_path = self.get_relative_path(file_path)
        stored_mtime = self.db.get_file_mtime(rel_path)
        
        if stored_mtime is None:
            return True
        
        current_mtime = file_path.stat().st_mtime
        return current_mtime > stored_mtime
    
    def index_file(self, file_path: Path) -> bool:
        """Index or re-index a single file."""
        if not self.should_index(file_path):
            logger.debug(f"Skipping non-indexable file: {file_path}")
            return False
        
        rel_path = self.get_relative_path(file_path)
        logger.info(f"Indexing file: {rel_path}")
        
        try:
            # Parse the markdown file
            parsed = self.parser.parse_file(file_path)
            mtime = file_path.stat().st_mtime
            
            # Convert sections to dicts for the atomic method
            sections = [
                {
                    "heading": section.heading,
                    "level": section.level,
                    "content": section.content
                }
                for section in parsed.sections
            ]
            
            # Index everything in a single atomic transaction
            file_id = self.db.index_file_atomic(
                path=rel_path,
                mtime=mtime,
                title=parsed.title,
                content=parsed.content,
                sections=sections,
                tags=list(parsed.tags),
                links=list(parsed.links)
            )
            
            # Create chunks for embedding
            db_sections = self.db.get_sections_by_file(file_id)
            if db_sections:
                chunk_ids = embedding_service.create_chunks_for_file(file_id, db_sections)
                logger.debug(f"Created {len(chunk_ids)} chunks for {rel_path}")

                # Auto-generate embeddings in background when event loop is running
                if chunk_ids:
                    self._schedule_embed(file_id)

            logger.debug(
                f"Indexed {rel_path}: {len(parsed.sections)} sections, "
                f"{len(parsed.tags)} tags, {len(parsed.links)} links"
            )
            return True
            
        except Exception as e:
            logger.error(f"Error indexing {rel_path}: {e}")
            return False

    # ------------------------------------------------------------------
    # Auto-embed helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _schedule_embed(file_id: int) -> None:
        """
        Schedule embedding generation for a file's chunks.

        Tries to fire-and-forget on the running event loop.  During
        startup / tests where no loop is running, the task is silently
        skipped — the user can still call POST /embeddings/generate.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop — skip (batch mode / tests)
            return

        async def _embed() -> None:
            try:
                ok, fail = await embedding_service.embed_file_chunks(file_id)
                if ok:
                    logger.debug("Auto-embedded %d chunks for file %d", ok, file_id)
                if fail:
                    logger.warning("Auto-embed: %d failures for file %d", fail, file_id)
            except Exception as exc:
                logger.warning("Auto-embed error for file %d: %s", file_id, exc)

        loop.create_task(_embed())
    
    def delete_file(self, file_path: Path) -> bool:
        """Remove a file from the index."""
        rel_path = self.get_relative_path(file_path)
        logger.info(f"Deleting from index: {rel_path}")
        
        deleted = self.db.delete_file(rel_path)
        if deleted:
            logger.debug(f"Removed {rel_path} from index")
        else:
            logger.debug(f"File {rel_path} was not in index")
        
        return deleted
    
    def rename_file(self, old_path: Path, new_path: Path) -> bool:
        """Handle file rename: delete old and index new."""
        logger.info(f"Renaming: {old_path} -> {new_path}")
        
        # Delete old entry
        self.delete_file(old_path)
        
        # Index new location
        if self.should_index(new_path):
            return self.index_file(new_path)
        
        return True
    
    def full_scan(self) -> tuple[int, int]:
        """
        Perform a full vault scan.
        Returns (indexed_count, error_count).
        """
        logger.info(f"Starting full vault scan: {self.vault_path}")
        
        indexed = 0
        errors = 0
        
        # Get all currently indexed paths
        indexed_paths = self.db.get_all_indexed_paths()
        seen_paths = set()
        
        # Walk through vault
        for file_path in self.vault_path.rglob("*"):
            if not self.should_index(file_path):
                continue
            
            rel_path = self.get_relative_path(file_path)
            seen_paths.add(rel_path)
            
            if self.needs_reindex(file_path):
                if self.index_file(file_path):
                    indexed += 1
                else:
                    errors += 1
        
        # Remove files that no longer exist
        removed_paths = indexed_paths - seen_paths
        for path in removed_paths:
            logger.info(f"Removing deleted file from index: {path}")
            self.db.delete_file(path)
        
        # Update last indexed time
        self.db.set_last_indexed(datetime.utcnow())
        
        logger.info(
            f"Full scan complete: {indexed} indexed, {errors} errors, "
            f"{len(removed_paths)} removed"
        )
        
        return indexed, errors
    
    def incremental_scan(self) -> tuple[int, int]:
        """
        Scan for changes since last index.
        Returns (indexed_count, error_count).
        """
        logger.info("Starting incremental scan")
        
        indexed = 0
        errors = 0
        
        for file_path in self.vault_path.rglob("*"):
            if self.needs_reindex(file_path):
                if self.index_file(file_path):
                    indexed += 1
                else:
                    errors += 1
        
        if indexed > 0:
            self.db.set_last_indexed(datetime.utcnow())
        
        logger.info(f"Incremental scan complete: {indexed} indexed, {errors} errors")
        
        return indexed, errors


# Singleton indexer instance
indexer = Indexer()
