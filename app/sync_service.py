"""
Sync Service for Second Brain.
Synchronizes data from local SQLite to remote PostgreSQL for Next.js consumption.
"""

import asyncio
import logging
import struct
import time
from datetime import datetime
from typing import Optional

from app.config import Config
from app.db import db
from app.db_postgres import get_postgres_db, PostgresDatabase

logger = logging.getLogger(__name__)


class SyncService:
    """
    Synchronizes Second Brain SQLite data to PostgreSQL.
    
    Supports:
    - Full sync: Transfers all data from SQLite to PostgreSQL
    - Incremental sync: Only syncs files modified since last sync
    - On-demand file sync: Sync specific files when changed
    """
    
    def __init__(self, postgres_db: Optional[PostgresDatabase] = None):
        self.postgres_db = postgres_db or get_postgres_db()
    
    async def full_sync(self) -> dict:
        """
        Perform full synchronization from SQLite to PostgreSQL.
        Clears and rebuilds all PostgreSQL data.
        
        Returns:
            dict: Sync statistics
        """
        start_time = time.time()
        stats = {
            'files_added': 0,
            'files_updated': 0,
            'files_deleted': 0,
            'sections': 0,
            'tags': 0,
            'links': 0,
            'chunks': 0,
            'embeddings': 0,
            'errors': []
        }
        
        try:
            logger.info("Starting full sync to PostgreSQL...")
            
            # Get all files from SQLite
            sqlite_files = db.get_all_files()
            pg_files = await self.postgres_db.get_all_files()
            pg_paths = {f['path'] for f in pg_files}
            sqlite_paths = {f['path'] for f in sqlite_files}
            
            # Delete files that no longer exist in SQLite
            for pg_file in pg_files:
                if pg_file['path'] not in sqlite_paths:
                    await self.postgres_db.delete_file(pg_file['path'])
                    stats['files_deleted'] += 1
                    logger.debug(f"Deleted: {pg_file['path']}")
            
            # Sync each file
            for file_row in sqlite_files:
                try:
                    await self._sync_file(file_row, stats, is_new=file_row['path'] not in pg_paths)
                except Exception as e:
                    stats['errors'].append({
                        'path': file_row['path'],
                        'error': str(e)
                    })
                    logger.error(f"Error syncing {file_row['path']}: {e}")
            
            duration_ms = int((time.time() - start_time) * 1000)
            
            # Log sync
            await self.postgres_db.log_sync(
                files_added=stats['files_added'],
                files_updated=stats['files_updated'],
                files_deleted=stats['files_deleted'],
                status='completed' if not stats['errors'] else 'completed_with_errors',
                error=str(stats['errors']) if stats['errors'] else None,
                duration_ms=duration_ms
            )
            
            logger.info(
                f"Full sync completed in {duration_ms}ms: "
                f"{stats['files_added']} added, {stats['files_updated']} updated, "
                f"{stats['files_deleted']} deleted"
            )
            
            return stats
            
        except Exception as e:
            logger.error(f"Full sync failed: {e}")
            await self.postgres_db.log_sync(
                files_added=0,
                files_updated=0,
                files_deleted=0,
                status='failed',
                error=str(e),
                duration_ms=int((time.time() - start_time) * 1000)
            )
            raise
    
    async def incremental_sync(self) -> dict:
        """
        Sync only files modified since last sync.
        
        Returns:
            dict: Sync statistics
        """
        start_time = time.time()
        stats = {
            'files_added': 0,
            'files_updated': 0,
            'files_deleted': 0,
            'sections': 0,
            'tags': 0,
            'links': 0,
            'chunks': 0,
            'embeddings': 0,
            'errors': []
        }
        
        try:
            # Get last sync time
            last_sync = await self.postgres_db.get_last_sync()
            last_sync_time = last_sync['synced_at'] if last_sync else None
            
            logger.info(
                f"Starting incremental sync "
                f"(since {last_sync_time or 'beginning'})"
            )
            
            # Get all files from both databases
            sqlite_files = db.get_all_files()
            pg_files = await self.postgres_db.get_all_files()
            pg_by_path = {f['path']: f for f in pg_files}
            sqlite_paths = {f['path'] for f in sqlite_files}
            
            # Delete files that no longer exist
            for pg_file in pg_files:
                if pg_file['path'] not in sqlite_paths:
                    await self.postgres_db.delete_file(pg_file['path'])
                    stats['files_deleted'] += 1
            
            # Sync new or modified files
            for file_row in sqlite_files:
                try:
                    pg_file = pg_by_path.get(file_row['path'])
                    
                    # Check if file needs sync
                    is_new = pg_file is None
                    is_modified = (
                        pg_file and 
                        file_row['mtime'] > pg_file['mtime']
                    )
                    
                    if is_new or is_modified:
                        await self._sync_file(file_row, stats, is_new=is_new)
                        
                except Exception as e:
                    stats['errors'].append({
                        'path': file_row['path'],
                        'error': str(e)
                    })
                    logger.error(f"Error syncing {file_row['path']}: {e}")
            
            duration_ms = int((time.time() - start_time) * 1000)
            
            # Log sync
            await self.postgres_db.log_sync(
                files_added=stats['files_added'],
                files_updated=stats['files_updated'],
                files_deleted=stats['files_deleted'],
                status='completed' if not stats['errors'] else 'completed_with_errors',
                error=str(stats['errors']) if stats['errors'] else None,
                duration_ms=duration_ms
            )
            
            logger.info(
                f"Incremental sync completed in {duration_ms}ms: "
                f"{stats['files_added']} added, {stats['files_updated']} updated, "
                f"{stats['files_deleted']} deleted"
            )
            
            return stats
            
        except Exception as e:
            logger.error(f"Incremental sync failed: {e}")
            raise
    
    async def sync_file(self, path: str) -> bool:
        """
        Sync a single file to PostgreSQL.
        
        Args:
            path: Path to the file (relative to vault)
            
        Returns:
            bool: True if sync succeeded
        """
        try:
            file_id = db.get_file_id_by_path(path)
            if file_id is None:
                # File deleted, remove from PostgreSQL
                await self.postgres_db.delete_file(path)
                logger.info(f"Removed from PostgreSQL: {path}")
                return True
            
            # Get file data
            file_data = db.get_file_by_path(path)
            if not file_data:
                return False
            
            stats = {
                'files_added': 0,
                'files_updated': 0,
                'sections': 0,
                'tags': 0,
                'links': 0,
                'chunks': 0,
                'embeddings': 0
            }
            
            pg_file = await self.postgres_db.get_file_by_path(path)
            await self._sync_file(file_data, stats, is_new=pg_file is None)
            
            logger.info(f"Synced to PostgreSQL: {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error syncing file {path}: {e}")
            return False
    
    async def _sync_file(self, file_row: dict, stats: dict, is_new: bool) -> None:
        """
        Sync a file and all its related data.
        
        Args:
            file_row: File data from SQLite
            stats: Stats dict to update
            is_new: Whether file is new in PostgreSQL
        """
        path = file_row['path']
        
        # Upsert file
        pg_file_id = await self.postgres_db.upsert_file(
            path=path,
            mtime=file_row['mtime'],
            title=file_row['title'],
            content=file_row['content']
        )
        
        if is_new:
            stats['files_added'] += 1
        else:
            stats['files_updated'] += 1
            # Clear existing relations for update
            await self.postgres_db.clear_file_sections(pg_file_id)
            await self.postgres_db.clear_file_tags(pg_file_id)
            await self.postgres_db.clear_file_links(pg_file_id)
            await self.postgres_db.clear_file_chunks(pg_file_id)
        
        # Sync sections
        sqlite_file_id = db.get_file_id_by_path(path)
        sections = db.get_sections_by_file(sqlite_file_id)
        section_id_map = {}  # SQLite section ID -> PostgreSQL section ID
        
        for section in sections:
            pg_section_id = await self.postgres_db.add_section(
                file_id=pg_file_id,
                heading=section['heading'],
                level=section['level'],
                content=section['content']
            )
            section_id_map[section['id']] = pg_section_id
            stats['sections'] += 1
        
        # Sync tags
        tags = db.get_file_tags(path)
        for tag_name in tags:
            tag_id = await self.postgres_db.get_or_create_tag(tag_name)
            await self.postgres_db.add_file_tag(pg_file_id, tag_id)
            stats['tags'] += 1
        
        # Sync links
        links = db.get_file_links(sqlite_file_id)
        for link in links:
            await self.postgres_db.add_link(pg_file_id, link['to_path'])
            stats['links'] += 1
        
        # Sync chunks and embeddings
        chunks = db.get_chunks_by_file(sqlite_file_id)
        for chunk in chunks:
            pg_section_id = (
                section_id_map.get(chunk['section_id']) 
                if chunk['section_id'] else None
            )
            
            pg_chunk_id = await self.postgres_db.add_chunk(
                file_id=pg_file_id,
                section_id=pg_section_id,
                chunk_index=chunk['chunk_index'],
                content=chunk['content'],
                token_count=chunk['token_count']
            )
            stats['chunks'] += 1
            
            # Get and sync embedding
            embedding_row = db.get_chunk_embedding(chunk['id'])
            if embedding_row:
                # Deserialize embedding from SQLite
                embedding_bytes = embedding_row['embedding']
                dimensions = embedding_row['dimensions']
                embedding = list(struct.unpack(f'{dimensions}f', embedding_bytes))
                
                await self.postgres_db.add_embedding(
                    chunk_id=pg_chunk_id,
                    embedding=embedding,
                    model=embedding_row['model'],
                    dimensions=dimensions
                )
                stats['embeddings'] += 1


# Singleton instance
_sync_service: Optional[SyncService] = None


def get_sync_service() -> SyncService:
    """Get sync service instance."""
    global _sync_service
    if _sync_service is None:
        _sync_service = SyncService()
    return _sync_service


async def run_sync(mode: str = 'incremental') -> dict:
    """
    Run synchronization.
    
    Args:
        mode: 'full' or 'incremental'
        
    Returns:
        dict: Sync statistics
    """
    service = get_sync_service()
    
    if mode == 'full':
        return await service.full_sync()
    else:
        return await service.incremental_sync()
