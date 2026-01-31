"""
PostgreSQL Database Adapter for Second Brain.
Provides async PostgreSQL operations compatible with Prisma schema.
Enables sync between local SQLite and remote PostgreSQL for Next.js consumption.
"""

import asyncio
import json
import logging
import struct
from datetime import datetime
from typing import Optional, Any
from contextlib import asynccontextmanager

import asyncpg

from app.config import Config

logger = logging.getLogger(__name__)


class PostgresDatabase:
    """
    Async PostgreSQL database manager.
    Mirrors the SQLite schema for Prisma/Next.js integration.
    """
    
    def __init__(self, connection_url: Optional[str] = None):
        self.connection_url = connection_url or Config.POSTGRES_URL
        self._pool: Optional[asyncpg.Pool] = None
    
    async def connect(self) -> asyncpg.Pool:
        """Get or create connection pool."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                self.connection_url,
                min_size=2,
                max_size=10,
                command_timeout=60
            )
            logger.info("PostgreSQL connection pool created")
        return self._pool
    
    async def close(self) -> None:
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("PostgreSQL connection pool closed")
    
    @asynccontextmanager
    async def acquire(self):
        """Context manager for acquiring a connection."""
        pool = await self.connect()
        async with pool.acquire() as conn:
            yield conn
    
    @asynccontextmanager
    async def transaction(self):
        """Context manager for a database transaction."""
        async with self.acquire() as conn:
            async with conn.transaction():
                yield conn
    
    # =========================================================================
    # File Operations
    # =========================================================================
    
    async def upsert_file(
        self,
        path: str,
        mtime: float,
        title: str,
        content: str
    ) -> int:
        """Insert or update a file record."""
        async with self.acquire() as conn:
            row = await conn.fetchrow("""
                INSERT INTO files (path, mtime, title, content, "updatedAt")
                VALUES ($1, $2, $3, $4, NOW())
                ON CONFLICT (path) DO UPDATE SET
                    mtime = EXCLUDED.mtime,
                    title = EXCLUDED.title,
                    content = EXCLUDED.content,
                    "updatedAt" = NOW()
                RETURNING id
            """, path, mtime, title, content)
            return row['id']
    
    async def get_file_by_path(self, path: str) -> Optional[dict]:
        """Get file by path."""
        async with self.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM files WHERE path = $1",
                path
            )
            return dict(row) if row else None
    
    async def get_file_by_id(self, file_id: int) -> Optional[dict]:
        """Get file by ID."""
        async with self.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM files WHERE id = $1",
                file_id
            )
            return dict(row) if row else None
    
    async def delete_file(self, path: str) -> bool:
        """Delete a file and cascade relations."""
        async with self.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM files WHERE path = $1",
                path
            )
            return result == "DELETE 1"
    
    async def get_all_files(self) -> list[dict]:
        """Get all files."""
        async with self.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM files ORDER BY path")
            return [dict(row) for row in rows]
    
    # =========================================================================
    # Section Operations
    # =========================================================================
    
    async def add_section(
        self,
        file_id: int,
        heading: str,
        level: int,
        content: str
    ) -> int:
        """Add a section to a file."""
        async with self.acquire() as conn:
            row = await conn.fetchrow("""
                INSERT INTO sections (file_id, heading, level, content)
                VALUES ($1, $2, $3, $4)
                RETURNING id
            """, file_id, heading, level, content)
            return row['id']
    
    async def get_sections_by_file(self, file_id: int) -> list[dict]:
        """Get all sections for a file."""
        async with self.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM sections WHERE file_id = $1 ORDER BY id",
                file_id
            )
            return [dict(row) for row in rows]
    
    async def clear_file_sections(self, file_id: int) -> None:
        """Delete all sections for a file."""
        async with self.acquire() as conn:
            await conn.execute(
                "DELETE FROM sections WHERE file_id = $1",
                file_id
            )
    
    # =========================================================================
    # Tag Operations
    # =========================================================================
    
    async def get_or_create_tag(self, name: str) -> int:
        """Get existing tag or create new one."""
        async with self.acquire() as conn:
            # Try to get existing
            row = await conn.fetchrow(
                "SELECT id FROM tags WHERE name = $1",
                name
            )
            if row:
                return row['id']
            
            # Create new
            row = await conn.fetchrow(
                "INSERT INTO tags (name) VALUES ($1) RETURNING id",
                name
            )
            return row['id']
    
    async def add_file_tag(self, file_id: int, tag_id: int) -> None:
        """Associate a tag with a file."""
        async with self.acquire() as conn:
            await conn.execute("""
                INSERT INTO file_tags (file_id, tag_id)
                VALUES ($1, $2)
                ON CONFLICT DO NOTHING
            """, file_id, tag_id)
    
    async def get_file_tags(self, file_id: int) -> list[str]:
        """Get all tags for a file."""
        async with self.acquire() as conn:
            rows = await conn.fetch("""
                SELECT t.name FROM tags t
                JOIN file_tags ft ON ft.tag_id = t.id
                WHERE ft.file_id = $1
                ORDER BY t.name
            """, file_id)
            return [row['name'] for row in rows]
    
    async def get_all_tags(self) -> list[str]:
        """Get all tag names."""
        async with self.acquire() as conn:
            rows = await conn.fetch("SELECT name FROM tags ORDER BY name")
            return [row['name'] for row in rows]
    
    async def clear_file_tags(self, file_id: int) -> None:
        """Remove all tags from a file."""
        async with self.acquire() as conn:
            await conn.execute(
                "DELETE FROM file_tags WHERE file_id = $1",
                file_id
            )
    
    # =========================================================================
    # Link Operations
    # =========================================================================
    
    async def add_link(self, from_file_id: int, to_path: str) -> int:
        """Add an outbound link."""
        async with self.acquire() as conn:
            row = await conn.fetchrow("""
                INSERT INTO links (from_file_id, to_path)
                VALUES ($1, $2)
                RETURNING id
            """, from_file_id, to_path)
            return row['id']
    
    async def get_file_links(self, file_id: int) -> list[str]:
        """Get outbound links from a file."""
        async with self.acquire() as conn:
            rows = await conn.fetch(
                "SELECT to_path FROM links WHERE from_file_id = $1",
                file_id
            )
            return [row['to_path'] for row in rows]
    
    async def get_backlinks(self, path: str) -> list[dict]:
        """Get files that link to this path."""
        async with self.acquire() as conn:
            rows = await conn.fetch("""
                SELECT f.id, f.path, f.title
                FROM files f
                JOIN links l ON l.from_file_id = f.id
                WHERE l.to_path = $1 OR l.to_path LIKE $2
            """, path, f"%{path}")
            return [dict(row) for row in rows]
    
    async def clear_file_links(self, file_id: int) -> None:
        """Remove all links from a file."""
        async with self.acquire() as conn:
            await conn.execute(
                "DELETE FROM links WHERE from_file_id = $1",
                file_id
            )
    
    # =========================================================================
    # Chunk Operations
    # =========================================================================
    
    async def add_chunk(
        self,
        file_id: int,
        section_id: Optional[int],
        chunk_index: int,
        content: str,
        token_count: int
    ) -> int:
        """Add a text chunk."""
        async with self.acquire() as conn:
            row = await conn.fetchrow("""
                INSERT INTO chunks (file_id, section_id, chunk_index, content, token_count)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING id
            """, file_id, section_id, chunk_index, content, token_count)
            return row['id']
    
    async def get_chunks_by_file(self, file_id: int) -> list[dict]:
        """Get all chunks for a file."""
        async with self.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM chunks WHERE file_id = $1 ORDER BY chunk_index",
                file_id
            )
            return [dict(row) for row in rows]
    
    async def clear_file_chunks(self, file_id: int) -> None:
        """Delete all chunks for a file (cascades to embeddings)."""
        async with self.acquire() as conn:
            await conn.execute(
                "DELETE FROM chunks WHERE file_id = $1",
                file_id
            )
    
    # =========================================================================
    # Embedding Operations
    # =========================================================================
    
    async def add_embedding(
        self,
        chunk_id: int,
        embedding: list[float],
        model: str,
        dimensions: int
    ) -> int:
        """Add an embedding for a chunk."""
        # Serialize embedding to bytes
        embedding_bytes = struct.pack(f'{len(embedding)}f', *embedding)
        
        async with self.acquire() as conn:
            row = await conn.fetchrow("""
                INSERT INTO embeddings (chunk_id, embedding, model, dimensions, created_at)
                VALUES ($1, $2, $3, $4, NOW())
                ON CONFLICT (chunk_id) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    model = EXCLUDED.model,
                    dimensions = EXCLUDED.dimensions,
                    created_at = NOW()
                RETURNING id
            """, chunk_id, embedding_bytes, model, dimensions)
            return row['id']
    
    async def get_embedding(self, chunk_id: int) -> Optional[list[float]]:
        """Get embedding vector for a chunk."""
        async with self.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT embedding, dimensions FROM embeddings WHERE chunk_id = $1",
                chunk_id
            )
            if not row:
                return None
            
            # Deserialize bytes to floats
            embedding_bytes = row['embedding']
            dimensions = row['dimensions']
            return list(struct.unpack(f'{dimensions}f', embedding_bytes))
    
    # =========================================================================
    # Search Operations
    # =========================================================================
    
    async def search_files(
        self,
        query: str,
        limit: int = 20
    ) -> list[dict]:
        """Full-text search across files and sections."""
        async with self.acquire() as conn:
            # PostgreSQL full-text search
            rows = await conn.fetch("""
                SELECT 
                    f.path as file_path,
                    f.title,
                    s.heading,
                    LEFT(s.content, 200) as snippet,
                    ts_rank(
                        to_tsvector('english', f.title || ' ' || s.heading || ' ' || s.content),
                        plainto_tsquery('english', $1)
                    ) as rank
                FROM sections s
                JOIN files f ON f.id = s.file_id
                WHERE to_tsvector('english', f.title || ' ' || s.heading || ' ' || s.content) 
                      @@ plainto_tsquery('english', $1)
                ORDER BY rank DESC
                LIMIT $2
            """, query, limit)
            return [dict(row) for row in rows]
    
    async def search_by_tag(self, tag: str, limit: int = 50) -> list[dict]:
        """Find files with a specific tag."""
        async with self.acquire() as conn:
            rows = await conn.fetch("""
                SELECT f.id, f.path, f.title
                FROM files f
                JOIN file_tags ft ON ft.file_id = f.id
                JOIN tags t ON t.id = ft.tag_id
                WHERE t.name = $1
                ORDER BY f.title
                LIMIT $2
            """, tag, limit)
            return [dict(row) for row in rows]
    
    # =========================================================================
    # Sync Log Operations
    # =========================================================================
    
    async def log_sync(
        self,
        files_added: int,
        files_updated: int,
        files_deleted: int,
        status: str = "completed",
        error: Optional[str] = None,
        duration_ms: Optional[int] = None
    ) -> int:
        """Log a sync operation."""
        async with self.acquire() as conn:
            row = await conn.fetchrow("""
                INSERT INTO sync_logs 
                (files_added, files_updated, files_deleted, status, error, duration_ms)
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING id
            """, files_added, files_updated, files_deleted, status, error, duration_ms)
            return row['id']
    
    async def get_last_sync(self) -> Optional[dict]:
        """Get most recent sync log."""
        async with self.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM sync_logs 
                ORDER BY synced_at DESC 
                LIMIT 1
            """)
            return dict(row) if row else None
    
    # =========================================================================
    # Stats Operations
    # =========================================================================
    
    async def get_stats(self) -> dict:
        """Get database statistics."""
        async with self.acquire() as conn:
            stats = {}
            
            stats['file_count'] = await conn.fetchval(
                "SELECT COUNT(*) FROM files"
            )
            stats['section_count'] = await conn.fetchval(
                "SELECT COUNT(*) FROM sections"
            )
            stats['tag_count'] = await conn.fetchval(
                "SELECT COUNT(*) FROM tags"
            )
            stats['link_count'] = await conn.fetchval(
                "SELECT COUNT(*) FROM links"
            )
            stats['chunk_count'] = await conn.fetchval(
                "SELECT COUNT(*) FROM chunks"
            )
            stats['embedding_count'] = await conn.fetchval(
                "SELECT COUNT(*) FROM embeddings"
            )
            
            last_sync = await self.get_last_sync()
            stats['last_sync'] = last_sync['synced_at'].isoformat() if last_sync else None
            
            return stats
    
    # =========================================================================
    # Conversation Operations (for Next.js app)
    # =========================================================================
    
    async def create_conversation(
        self,
        session_id: Optional[str] = None,
        title: Optional[str] = None
    ) -> int:
        """Create a new conversation."""
        async with self.acquire() as conn:
            row = await conn.fetchrow("""
                INSERT INTO conversations (session_id, title)
                VALUES ($1, $2)
                RETURNING id
            """, session_id, title)
            return row['id']
    
    async def add_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        sources: Optional[list[dict]] = None
    ) -> int:
        """Add a message to a conversation."""
        sources_json = json.dumps(sources) if sources else None
        
        async with self.acquire() as conn:
            row = await conn.fetchrow("""
                INSERT INTO messages (conversation_id, role, content, sources)
                VALUES ($1, $2, $3, $4::jsonb)
                RETURNING id
            """, conversation_id, role, content, sources_json)
            
            # Update conversation timestamp
            await conn.execute("""
                UPDATE conversations SET "updated_at" = NOW()
                WHERE id = $1
            """, conversation_id)
            
            return row['id']
    
    async def get_conversation(self, conversation_id: int) -> Optional[dict]:
        """Get a conversation with all messages."""
        async with self.acquire() as conn:
            conv = await conn.fetchrow(
                "SELECT * FROM conversations WHERE id = $1",
                conversation_id
            )
            if not conv:
                return None
            
            messages = await conn.fetch("""
                SELECT * FROM messages 
                WHERE conversation_id = $1 
                ORDER BY created_at
            """, conversation_id)
            
            return {
                **dict(conv),
                'messages': [dict(m) for m in messages]
            }
    
    async def get_recent_conversations(
        self,
        session_id: Optional[str] = None,
        limit: int = 20
    ) -> list[dict]:
        """Get recent conversations."""
        async with self.acquire() as conn:
            if session_id:
                rows = await conn.fetch("""
                    SELECT * FROM conversations 
                    WHERE session_id = $1
                    ORDER BY updated_at DESC
                    LIMIT $2
                """, session_id, limit)
            else:
                rows = await conn.fetch("""
                    SELECT * FROM conversations 
                    ORDER BY updated_at DESC
                    LIMIT $1
                """, limit)
            
            return [dict(row) for row in rows]


# Singleton instance (lazily initialized)
_postgres_db: Optional[PostgresDatabase] = None


def get_postgres_db() -> PostgresDatabase:
    """Get PostgreSQL database instance."""
    global _postgres_db
    if _postgres_db is None:
        _postgres_db = PostgresDatabase()
    return _postgres_db


async def close_postgres_db() -> None:
    """Close PostgreSQL database connection."""
    global _postgres_db
    if _postgres_db:
        await _postgres_db.close()
        _postgres_db = None
