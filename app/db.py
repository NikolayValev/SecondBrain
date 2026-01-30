"""
Database module for Second Brain daemon.
Manages SQLite database with FTS5 for full-text search.
"""

import sqlite3
import logging
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime
from typing import Optional, Generator

from app.config import config

logger = logging.getLogger(__name__)


class Database:
    """SQLite database manager with FTS5 support."""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or config.DATABASE_PATH
        self._connection: Optional[sqlite3.Connection] = None
        self._last_indexed: Optional[datetime] = None
    
    def connect(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._connection is None:
            self._connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False
            )
            self._connection.row_factory = sqlite3.Row
            # Enable foreign keys
            self._connection.execute("PRAGMA foreign_keys = ON")
        return self._connection
    
    def close(self) -> None:
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
    
    @contextmanager
    def cursor(self) -> Generator[sqlite3.Cursor, None, None]:
        """Context manager for database cursor with auto-commit."""
        conn = self.connect()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    
    def initialize(self) -> None:
        """Create database schema if not exists."""
        logger.info(f"Initializing database at {self.db_path}")
        
        with self.cursor() as cur:
            # Files table - main file metadata
            cur.execute("""
                CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT UNIQUE NOT NULL,
                    mtime REAL NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL
                )
            """)
            
            # Sections table - headings and their content
            cur.execute("""
                CREATE TABLE IF NOT EXISTS sections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id INTEGER NOT NULL,
                    heading TEXT NOT NULL,
                    level INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
                )
            """)
            
            # Tags table - unique tag names
            cur.execute("""
                CREATE TABLE IF NOT EXISTS tags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL
                )
            """)
            
            # File-Tags junction table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS file_tags (
                    file_id INTEGER NOT NULL,
                    tag_id INTEGER NOT NULL,
                    PRIMARY KEY (file_id, tag_id),
                    FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE,
                    FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE
                )
            """)
            
            # Links table - outbound links from files
            cur.execute("""
                CREATE TABLE IF NOT EXISTS links (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    from_file_id INTEGER NOT NULL,
                    to_path TEXT NOT NULL,
                    FOREIGN KEY (from_file_id) REFERENCES files(id) ON DELETE CASCADE
                )
            """)
            
            # Metadata table for tracking
            cur.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            
            # Create indexes for performance
            cur.execute("CREATE INDEX IF NOT EXISTS idx_sections_file_id ON sections(file_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_file_tags_file_id ON file_tags(file_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_file_tags_tag_id ON file_tags(tag_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_links_from_file_id ON links(from_file_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_links_to_path ON links(to_path)")
            
            # FTS5 virtual table for full-text search
            # Standalone FTS table (not content-synced) for flexibility
            cur.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS fts_content USING fts5(
                    title,
                    heading,
                    content,
                    file_path,
                    section_id UNINDEXED
                )
            """)
            
            # Triggers to keep FTS index in sync
            cur.execute("""
                CREATE TRIGGER IF NOT EXISTS sections_ai AFTER INSERT ON sections BEGIN
                    INSERT INTO fts_content(title, heading, content, file_path, section_id)
                    SELECT f.title, NEW.heading, NEW.content, f.path, NEW.id
                    FROM files f WHERE f.id = NEW.file_id;
                END
            """)
            
            cur.execute("""
                CREATE TRIGGER IF NOT EXISTS sections_ad AFTER DELETE ON sections BEGIN
                    DELETE FROM fts_content WHERE section_id = OLD.id;
                END
            """)
            
            cur.execute("""
                CREATE TRIGGER IF NOT EXISTS sections_au AFTER UPDATE ON sections BEGIN
                    DELETE FROM fts_content WHERE section_id = OLD.id;
                    INSERT INTO fts_content(title, heading, content, file_path, section_id)
                    SELECT f.title, NEW.heading, NEW.content, f.path, NEW.id
                    FROM files f WHERE f.id = NEW.file_id;
                END
            """)
        
        logger.info("Database initialized successfully")
    
    def get_file_by_path(self, path: str) -> Optional[dict]:
        """Get file record by path."""
        with self.cursor() as cur:
            cur.execute("SELECT * FROM files WHERE path = ?", (path,))
            row = cur.fetchone()
            return dict(row) if row else None
    
    def get_file_mtime(self, path: str) -> Optional[float]:
        """Get file modification time from database."""
        with self.cursor() as cur:
            cur.execute("SELECT mtime FROM files WHERE path = ?", (path,))
            row = cur.fetchone()
            return row["mtime"] if row else None
    
    def upsert_file(self, path: str, mtime: float, title: str, content: str) -> int:
        """Insert or update file record. Returns file ID."""
        with self.cursor() as cur:
            cur.execute("""
                INSERT INTO files (path, mtime, title, content)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    mtime = excluded.mtime,
                    title = excluded.title,
                    content = excluded.content
            """, (path, mtime, title, content))
            
            cur.execute("SELECT id FROM files WHERE path = ?", (path,))
            return cur.fetchone()["id"]
    
    def delete_file(self, path: str) -> bool:
        """Delete file and all related records. Returns True if deleted."""
        with self.cursor() as cur:
            cur.execute("DELETE FROM files WHERE path = ?", (path,))
            return cur.rowcount > 0
    
    def clear_file_relations(self, file_id: int) -> None:
        """Clear all sections, tags, and links for a file."""
        with self.cursor() as cur:
            cur.execute("DELETE FROM sections WHERE file_id = ?", (file_id,))
            cur.execute("DELETE FROM file_tags WHERE file_id = ?", (file_id,))
            cur.execute("DELETE FROM links WHERE from_file_id = ?", (file_id,))
    
    def add_section(self, file_id: int, heading: str, level: int, content: str) -> int:
        """Add a section to a file."""
        with self.cursor() as cur:
            cur.execute("""
                INSERT INTO sections (file_id, heading, level, content)
                VALUES (?, ?, ?, ?)
            """, (file_id, heading, level, content))
            return cur.lastrowid
    
    def get_or_create_tag(self, name: str) -> int:
        """Get existing tag ID or create new tag."""
        with self.cursor() as cur:
            cur.execute("SELECT id FROM tags WHERE name = ?", (name,))
            row = cur.fetchone()
            if row:
                return row["id"]
            
            cur.execute("INSERT INTO tags (name) VALUES (?)", (name,))
            return cur.lastrowid
    
    def add_file_tag(self, file_id: int, tag_id: int) -> None:
        """Associate a tag with a file."""
        with self.cursor() as cur:
            cur.execute("""
                INSERT OR IGNORE INTO file_tags (file_id, tag_id)
                VALUES (?, ?)
            """, (file_id, tag_id))
    
    def add_link(self, from_file_id: int, to_path: str) -> None:
        """Add an outbound link from a file."""
        with self.cursor() as cur:
            cur.execute("""
                INSERT INTO links (from_file_id, to_path)
                VALUES (?, ?)
            """, (from_file_id, to_path))
    
    def search(self, query: str, limit: int = 20) -> list[dict]:
        """Full-text search across sections."""
        with self.cursor() as cur:
            cur.execute("""
                SELECT 
                    fts.file_path,
                    fts.title,
                    fts.heading,
                    snippet(fts_content, 2, '<mark>', '</mark>', '...', 32) as snippet,
                    bm25(fts_content) as rank
                FROM fts_content fts
                WHERE fts_content MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (query, limit))
            
            results = []
            for row in cur.fetchall():
                results.append({
                    "file_path": row["file_path"],
                    "title": row["title"],
                    "heading": row["heading"],
                    "snippet": row["snippet"],
                    "rank": row["rank"]
                })
            return results
    
    def get_stats(self) -> dict:
        """Get database statistics."""
        with self.cursor() as cur:
            cur.execute("SELECT COUNT(*) as count FROM files")
            file_count = cur.fetchone()["count"]
            
            cur.execute("SELECT COUNT(*) as count FROM sections")
            section_count = cur.fetchone()["count"]
            
            cur.execute("SELECT COUNT(*) as count FROM tags")
            tag_count = cur.fetchone()["count"]
            
            cur.execute("SELECT COUNT(*) as count FROM links")
            link_count = cur.fetchone()["count"]
            
            cur.execute("SELECT value FROM metadata WHERE key = 'last_indexed'")
            row = cur.fetchone()
            last_indexed = row["value"] if row else None
            
            return {
                "file_count": file_count,
                "section_count": section_count,
                "tag_count": tag_count,
                "link_count": link_count,
                "last_indexed": last_indexed
            }
    
    def set_last_indexed(self, timestamp: Optional[datetime] = None) -> None:
        """Update last indexed timestamp."""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        with self.cursor() as cur:
            cur.execute("""
                INSERT INTO metadata (key, value)
                VALUES ('last_indexed', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """, (timestamp.isoformat(),))
    
    def get_all_indexed_paths(self) -> set[str]:
        """Get all currently indexed file paths."""
        with self.cursor() as cur:
            cur.execute("SELECT path FROM files")
            return {row["path"] for row in cur.fetchall()}
    
    def get_file_content(self, path: str) -> Optional[str]:
        """Get full file content by path."""
        with self.cursor() as cur:
            cur.execute("SELECT content FROM files WHERE path = ?", (path,))
            row = cur.fetchone()
            return row["content"] if row else None


# Singleton database instance
db = Database()
