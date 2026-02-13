"""
File service: file retrieval, tags, and backlinks.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.config import config
from app.db import db
from app.api.models.files import (
    FileResponse,
    BacklinkItem,
    BacklinksResponse,
    TagItem,
    TagsResponse,
)

logger = logging.getLogger(__name__)


def _parse_frontmatter(content: str) -> dict:
    """Extract YAML frontmatter from markdown content."""
    if not content.startswith("---"):
        return {}
    try:
        import yaml
        parts = content.split("---", 2)
        if len(parts) >= 3:
            return yaml.safe_load(parts[1]) or {}
    except Exception:
        pass
    return {}


def _resolve_created_at(frontmatter: dict, fallback: Optional[str]) -> Optional[str]:
    """Resolve created_at from frontmatter or fallback value."""
    created_at = frontmatter.get("created") or frontmatter.get("date") or fallback
    if isinstance(created_at, datetime):
        return created_at.isoformat()
    if created_at and not isinstance(created_at, str):
        return str(created_at)
    return created_at


class FileService:
    """Handles file retrieval, tags, and backlinks."""

    def get_file(self, path: str) -> FileResponse:
        """
        Retrieve a file by its vault-relative path.

        Falls back to reading from the filesystem when the file isn't indexed.

        Args:
            path: Vault-relative path (e.g. "subfolder/note.md").

        Returns:
            FileResponse with content, tags, and metadata.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            IOError: If the file can't be read.
        """
        file_record = db.get_file_by_path(path)

        if not file_record:
            return self._read_from_filesystem(path)

        return self._build_from_db(file_record, path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read_from_filesystem(self, path: str) -> FileResponse:
        """Read directly from the filesystem when not indexed."""
        from app.parser import MarkdownParser

        file_path = config.VAULT_PATH / path
        if not file_path.exists() or not file_path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        parser = MarkdownParser(config.VAULT_PATH)
        parsed = parser.parse_file(file_path)
        content = file_path.read_text(encoding="utf-8")
        mtime = file_path.stat().st_mtime
        modified_at = datetime.fromtimestamp(mtime).isoformat()

        frontmatter = _parse_frontmatter(content)
        created_at = _resolve_created_at(frontmatter, modified_at)

        logger.info("File not in database, reading from filesystem: %s", path)

        return FileResponse(
            path=path,
            title=parsed.title,
            content=content,
            tags=parsed.tags,
            created_at=created_at,
            modified_at=modified_at,
            frontmatter=frontmatter,
        )

    def _build_from_db(self, file_record: dict, path: str) -> FileResponse:
        """Build response from database record."""
        tags = []
        with db.cursor() as cur:
            cur.execute(
                """
                SELECT t.name FROM tags t
                JOIN file_tags ft ON t.id = ft.tag_id
                JOIN files f ON ft.file_id = f.id
                WHERE f.path = ?
                """,
                (path,),
            )
            tags = [row["name"] for row in cur.fetchall()]

        mtime = file_record.get("mtime")
        modified_at = datetime.fromtimestamp(mtime).isoformat() if mtime else None

        content = file_record["content"]
        frontmatter = _parse_frontmatter(content)
        created_at = _resolve_created_at(frontmatter, modified_at)

        return FileResponse(
            path=file_record["path"],
            title=file_record["title"],
            content=content,
            tags=tags,
            created_at=created_at,
            modified_at=modified_at,
            frontmatter=frontmatter,
        )

    # ------------------------------------------------------------------
    # Tags & Backlinks
    # ------------------------------------------------------------------

    def list_tags(self) -> TagsResponse:
        """Return all tags with usage counts."""
        with db.cursor() as cur:
            cur.execute("""
                SELECT t.name, COUNT(ft.file_id) as file_count
                FROM tags t
                LEFT JOIN file_tags ft ON t.id = ft.tag_id
                GROUP BY t.id
                ORDER BY file_count DESC, t.name
            """)
            tags = [TagItem(name=row["name"], file_count=row["file_count"]) for row in cur.fetchall()]
        return TagsResponse(tags=tags, count=len(tags))

    def get_backlinks(self, path: str) -> BacklinksResponse:
        """Find all files that link to *path*."""
        target_name = Path(path).stem

        with db.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT f.path, f.title
                FROM links l
                JOIN files f ON l.from_file_id = f.id
                WHERE l.to_path = ? OR l.to_path = ? OR l.to_path LIKE ?
                """,
                (path, target_name, f"%{target_name}%"),
            )
            backlinks = [BacklinkItem(path=row["path"], title=row["title"]) for row in cur.fetchall()]

        return BacklinksResponse(target=path, backlinks=backlinks, count=len(backlinks))


# Singleton
file_service = FileService()
