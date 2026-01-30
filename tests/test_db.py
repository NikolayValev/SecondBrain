"""
Unit tests for the database module.
"""

import pytest
from pathlib import Path
from datetime import datetime

from app.db import Database


class TestDatabaseInitialization:
    """Tests for database initialization."""
    
    def test_creates_database_file(self, temp_dir: Path):
        """Database file should be created on initialization."""
        db_path = temp_dir / "new.db"
        db = Database(db_path)
        db.initialize()
        
        assert db_path.exists()
        db.close()
    
    def test_creates_required_tables(self, temp_db: Database):
        """All required tables should be created."""
        with temp_db.cursor() as cur:
            cur.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' 
                ORDER BY name
            """)
            tables = {row["name"] for row in cur.fetchall()}
        
        assert "files" in tables
        assert "sections" in tables
        assert "tags" in tables
        assert "file_tags" in tables
        assert "links" in tables
        assert "metadata" in tables
        assert "fts_content" in tables
    
    def test_creates_indexes(self, temp_db: Database):
        """Performance indexes should be created."""
        with temp_db.cursor() as cur:
            cur.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='index' AND name LIKE 'idx_%'
            """)
            indexes = {row["name"] for row in cur.fetchall()}
        
        assert "idx_sections_file_id" in indexes
        assert "idx_file_tags_file_id" in indexes
        assert "idx_links_from_file_id" in indexes
    
    def test_idempotent_initialization(self, temp_db: Database):
        """Multiple initializations should not cause errors."""
        # Initialize again
        temp_db.initialize()
        temp_db.initialize()
        
        # Should still work
        stats = temp_db.get_stats()
        assert stats["file_count"] == 0


class TestDatabaseFileOperations:
    """Tests for file CRUD operations."""
    
    def test_upsert_new_file(self, temp_db: Database):
        """Should insert new file record."""
        file_id = temp_db.upsert_file(
            path="test/note.md",
            mtime=1234567890.0,
            title="Test Note",
            content="# Test Note\n\nContent here."
        )
        
        assert file_id > 0
        
        file = temp_db.get_file_by_path("test/note.md")
        assert file is not None
        assert file["title"] == "Test Note"
        assert file["mtime"] == 1234567890.0
    
    def test_upsert_updates_existing(self, temp_db: Database):
        """Should update existing file on conflict."""
        # Insert
        file_id1 = temp_db.upsert_file(
            path="test/note.md",
            mtime=1000.0,
            title="Original Title",
            content="Original content"
        )
        
        # Update
        file_id2 = temp_db.upsert_file(
            path="test/note.md",
            mtime=2000.0,
            title="Updated Title",
            content="Updated content"
        )
        
        assert file_id1 == file_id2
        
        file = temp_db.get_file_by_path("test/note.md")
        assert file["title"] == "Updated Title"
        assert file["mtime"] == 2000.0
    
    def test_get_file_by_path_not_found(self, temp_db: Database):
        """Should return None for non-existent file."""
        file = temp_db.get_file_by_path("nonexistent.md")
        assert file is None
    
    def test_get_file_mtime(self, temp_db: Database):
        """Should return file mtime."""
        temp_db.upsert_file("test.md", 1234.5, "Test", "Content")
        
        mtime = temp_db.get_file_mtime("test.md")
        assert mtime == 1234.5
    
    def test_get_file_mtime_not_found(self, temp_db: Database):
        """Should return None for non-existent file mtime."""
        mtime = temp_db.get_file_mtime("nonexistent.md")
        assert mtime is None
    
    def test_delete_file(self, temp_db: Database):
        """Should delete file and return True."""
        temp_db.upsert_file("test.md", 1000.0, "Test", "Content")
        
        deleted = temp_db.delete_file("test.md")
        
        assert deleted is True
        assert temp_db.get_file_by_path("test.md") is None
    
    def test_delete_file_not_found(self, temp_db: Database):
        """Should return False for non-existent file."""
        deleted = temp_db.delete_file("nonexistent.md")
        assert deleted is False
    
    def test_delete_cascades_relations(self, temp_db: Database):
        """Deleting file should cascade to sections, tags, links."""
        # Create file with relations
        file_id = temp_db.upsert_file("test.md", 1000.0, "Test", "Content")
        temp_db.add_section(file_id, "Heading", 1, "Section content")
        tag_id = temp_db.get_or_create_tag("test-tag")
        temp_db.add_file_tag(file_id, tag_id)
        temp_db.add_link(file_id, "target.md")
        
        # Delete file
        temp_db.delete_file("test.md")
        
        # Check cascaded deletes
        with temp_db.cursor() as cur:
            cur.execute("SELECT COUNT(*) as count FROM sections WHERE file_id = ?", (file_id,))
            assert cur.fetchone()["count"] == 0
            
            cur.execute("SELECT COUNT(*) as count FROM file_tags WHERE file_id = ?", (file_id,))
            assert cur.fetchone()["count"] == 0
            
            cur.execute("SELECT COUNT(*) as count FROM links WHERE from_file_id = ?", (file_id,))
            assert cur.fetchone()["count"] == 0
    
    def test_get_all_indexed_paths(self, temp_db: Database):
        """Should return all indexed file paths."""
        temp_db.upsert_file("file1.md", 1000.0, "File 1", "Content 1")
        temp_db.upsert_file("dir/file2.md", 1000.0, "File 2", "Content 2")
        temp_db.upsert_file("dir/sub/file3.md", 1000.0, "File 3", "Content 3")
        
        paths = temp_db.get_all_indexed_paths()
        
        assert paths == {"file1.md", "dir/file2.md", "dir/sub/file3.md"}
    
    def test_get_file_content(self, temp_db: Database):
        """Should return full file content."""
        content = "# Title\n\nFull content here."
        temp_db.upsert_file("test.md", 1000.0, "Test", content)
        
        retrieved = temp_db.get_file_content("test.md")
        assert retrieved == content
    
    def test_get_file_content_not_found(self, temp_db: Database):
        """Should return None for non-existent file."""
        content = temp_db.get_file_content("nonexistent.md")
        assert content is None


class TestDatabaseSections:
    """Tests for section operations."""
    
    def test_add_section(self, temp_db: Database):
        """Should add section to file."""
        file_id = temp_db.upsert_file("test.md", 1000.0, "Test", "Content")
        
        section_id = temp_db.add_section(
            file_id=file_id,
            heading="Section One",
            level=2,
            content="Section content here."
        )
        
        assert section_id > 0
        
        with temp_db.cursor() as cur:
            cur.execute("SELECT * FROM sections WHERE id = ?", (section_id,))
            section = cur.fetchone()
            assert section["heading"] == "Section One"
            assert section["level"] == 2
    
    def test_clear_file_relations(self, temp_db: Database):
        """Should clear all sections, tags, and links for a file."""
        file_id = temp_db.upsert_file("test.md", 1000.0, "Test", "Content")
        temp_db.add_section(file_id, "Heading 1", 1, "Content 1")
        temp_db.add_section(file_id, "Heading 2", 2, "Content 2")
        tag_id = temp_db.get_or_create_tag("tag")
        temp_db.add_file_tag(file_id, tag_id)
        temp_db.add_link(file_id, "link.md")
        
        temp_db.clear_file_relations(file_id)
        
        with temp_db.cursor() as cur:
            cur.execute("SELECT COUNT(*) as count FROM sections WHERE file_id = ?", (file_id,))
            assert cur.fetchone()["count"] == 0


class TestDatabaseTags:
    """Tests for tag operations."""
    
    def test_get_or_create_tag_new(self, temp_db: Database):
        """Should create new tag and return ID."""
        tag_id = temp_db.get_or_create_tag("new-tag")
        
        assert tag_id > 0
        
        with temp_db.cursor() as cur:
            cur.execute("SELECT name FROM tags WHERE id = ?", (tag_id,))
            assert cur.fetchone()["name"] == "new-tag"
    
    def test_get_or_create_tag_existing(self, temp_db: Database):
        """Should return existing tag ID."""
        tag_id1 = temp_db.get_or_create_tag("existing-tag")
        tag_id2 = temp_db.get_or_create_tag("existing-tag")
        
        assert tag_id1 == tag_id2
    
    def test_add_file_tag(self, temp_db: Database):
        """Should associate tag with file."""
        file_id = temp_db.upsert_file("test.md", 1000.0, "Test", "Content")
        tag_id = temp_db.get_or_create_tag("test-tag")
        
        temp_db.add_file_tag(file_id, tag_id)
        
        with temp_db.cursor() as cur:
            cur.execute(
                "SELECT * FROM file_tags WHERE file_id = ? AND tag_id = ?",
                (file_id, tag_id)
            )
            assert cur.fetchone() is not None
    
    def test_add_file_tag_duplicate_ignored(self, temp_db: Database):
        """Duplicate file-tag associations should be ignored."""
        file_id = temp_db.upsert_file("test.md", 1000.0, "Test", "Content")
        tag_id = temp_db.get_or_create_tag("test-tag")
        
        temp_db.add_file_tag(file_id, tag_id)
        temp_db.add_file_tag(file_id, tag_id)  # Duplicate
        
        with temp_db.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) as count FROM file_tags WHERE file_id = ? AND tag_id = ?",
                (file_id, tag_id)
            )
            assert cur.fetchone()["count"] == 1


class TestDatabaseLinks:
    """Tests for link operations."""
    
    def test_add_link(self, temp_db: Database):
        """Should add outbound link."""
        file_id = temp_db.upsert_file("source.md", 1000.0, "Source", "Content")
        
        temp_db.add_link(file_id, "target.md")
        
        with temp_db.cursor() as cur:
            cur.execute(
                "SELECT to_path FROM links WHERE from_file_id = ?",
                (file_id,)
            )
            assert cur.fetchone()["to_path"] == "target.md"
    
    def test_multiple_links_from_file(self, temp_db: Database):
        """Should support multiple outbound links."""
        file_id = temp_db.upsert_file("source.md", 1000.0, "Source", "Content")
        
        temp_db.add_link(file_id, "target1.md")
        temp_db.add_link(file_id, "target2.md")
        temp_db.add_link(file_id, "target3.md")
        
        with temp_db.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) as count FROM links WHERE from_file_id = ?",
                (file_id,)
            )
            assert cur.fetchone()["count"] == 3


class TestDatabaseSearch:
    """Tests for full-text search."""
    
    def test_search_by_content(self, temp_db: Database):
        """Should find files by content match."""
        file_id = temp_db.upsert_file("test.md", 1000.0, "Test Note", "Content")
        temp_db.add_section(file_id, "Introduction", 1, "Python programming is fun")
        
        results = temp_db.search("Python")
        
        assert len(results) > 0
        assert any("Python" in r["snippet"] or "Python" in r["heading"] for r in results)
    
    def test_search_by_title(self, temp_db: Database):
        """Should find files by title match."""
        file_id = temp_db.upsert_file("test.md", 1000.0, "Machine Learning Guide", "Content")
        temp_db.add_section(file_id, "Machine Learning Guide", 0, "Introduction to ML")
        
        results = temp_db.search("Machine Learning")
        
        assert len(results) > 0
    
    def test_search_returns_snippets(self, temp_db: Database):
        """Search results should include snippets."""
        file_id = temp_db.upsert_file("test.md", 1000.0, "Test", "Content")
        temp_db.add_section(file_id, "Section", 1, "This is searchable content for testing")
        
        results = temp_db.search("searchable")
        
        assert len(results) > 0
        assert results[0]["snippet"] is not None
    
    def test_search_returns_file_path(self, temp_db: Database):
        """Search results should include file path."""
        file_id = temp_db.upsert_file("folder/note.md", 1000.0, "Test", "Content")
        temp_db.add_section(file_id, "Section", 1, "Unique searchable term xyz123")
        
        results = temp_db.search("xyz123")
        
        assert len(results) > 0
        assert results[0]["file_path"] == "folder/note.md"
    
    def test_search_limit(self, temp_db: Database):
        """Search should respect limit parameter."""
        # Create many matching files
        for i in range(30):
            file_id = temp_db.upsert_file(f"file{i}.md", 1000.0, f"File {i}", "Content")
            temp_db.add_section(file_id, "Section", 1, "Common search term")
        
        results = temp_db.search("Common", limit=5)
        
        assert len(results) == 5
    
    def test_search_no_results(self, temp_db: Database):
        """Should return empty list for no matches."""
        results = temp_db.search("nonexistentterm123456")
        assert results == []
    
    def test_search_ranking(self, temp_db: Database):
        """Results should be ranked by relevance."""
        # File with term in title should rank higher
        file_id1 = temp_db.upsert_file("file1.md", 1000.0, "Python Guide", "Content")
        temp_db.add_section(file_id1, "Python Guide", 0, "This is about Python programming")
        
        file_id2 = temp_db.upsert_file("file2.md", 1000.0, "Other Topic", "Content")
        temp_db.add_section(file_id2, "Section", 1, "Brief mention of Python here")
        
        results = temp_db.search("Python")
        
        assert len(results) >= 2
        # Results should have rank values
        assert all("rank" in r for r in results)


class TestDatabaseStats:
    """Tests for statistics retrieval."""
    
    def test_get_stats_empty(self, temp_db: Database):
        """Should return zero counts for empty database."""
        stats = temp_db.get_stats()
        
        assert stats["file_count"] == 0
        assert stats["section_count"] == 0
        assert stats["tag_count"] == 0
        assert stats["link_count"] == 0
    
    def test_get_stats_with_data(self, temp_db: Database):
        """Should return accurate counts."""
        # Add some data
        file_id = temp_db.upsert_file("test.md", 1000.0, "Test", "Content")
        temp_db.add_section(file_id, "Section 1", 1, "Content 1")
        temp_db.add_section(file_id, "Section 2", 2, "Content 2")
        tag_id = temp_db.get_or_create_tag("tag1")
        temp_db.add_file_tag(file_id, tag_id)
        temp_db.add_link(file_id, "link1.md")
        temp_db.add_link(file_id, "link2.md")
        
        stats = temp_db.get_stats()
        
        assert stats["file_count"] == 1
        assert stats["section_count"] == 2
        assert stats["tag_count"] == 1
        assert stats["link_count"] == 2
    
    def test_set_and_get_last_indexed(self, temp_db: Database):
        """Should store and retrieve last indexed timestamp."""
        timestamp = datetime(2024, 1, 15, 12, 30, 0)
        
        temp_db.set_last_indexed(timestamp)
        
        stats = temp_db.get_stats()
        assert stats["last_indexed"] == timestamp.isoformat()
    
    def test_last_indexed_defaults_to_none(self, temp_db: Database):
        """Last indexed should be None when not set."""
        stats = temp_db.get_stats()
        assert stats["last_indexed"] is None


class TestDatabaseConnection:
    """Tests for connection management."""
    
    def test_close_and_reconnect(self, temp_dir: Path):
        """Should handle close and reconnect."""
        db_path = temp_dir / "test.db"
        db = Database(db_path)
        db.initialize()
        
        db.upsert_file("test.md", 1000.0, "Test", "Content")
        db.close()
        
        # Reconnect
        db2 = Database(db_path)
        file = db2.get_file_by_path("test.md")
        
        assert file is not None
        assert file["title"] == "Test"
        db2.close()
    
    def test_cursor_context_manager_commits(self, temp_db: Database):
        """Cursor context manager should auto-commit."""
        with temp_db.cursor() as cur:
            cur.execute(
                "INSERT INTO files (path, mtime, title, content) VALUES (?, ?, ?, ?)",
                ("test.md", 1000.0, "Test", "Content")
            )
        
        # Should be persisted
        file = temp_db.get_file_by_path("test.md")
        assert file is not None
    
    def test_cursor_context_manager_rollback_on_error(self, temp_db: Database):
        """Cursor context manager should rollback on error."""
        try:
            with temp_db.cursor() as cur:
                cur.execute(
                    "INSERT INTO files (path, mtime, title, content) VALUES (?, ?, ?, ?)",
                    ("test.md", 1000.0, "Test", "Content")
                )
                # Force error
                raise ValueError("Test error")
        except ValueError:
            pass
        
        # Should be rolled back
        file = temp_db.get_file_by_path("test.md")
        assert file is None
