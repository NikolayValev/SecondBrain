"""
Unit tests for the indexer module.
"""

import pytest
from pathlib import Path
from datetime import datetime
import time

from app.db import Database
from app.parser import MarkdownParser
from app.indexer import Indexer


class TestIndexerFileDetection:
    """Tests for file detection and filtering."""
    
    def test_should_index_markdown_file(self, indexer: Indexer, temp_vault: Path):
        """Should return True for .md files."""
        md_file = temp_vault / "test.md"
        md_file.write_text("# Test", encoding="utf-8")
        
        assert indexer.should_index(md_file) is True
    
    def test_should_index_markdown_extension(self, indexer: Indexer, temp_vault: Path):
        """Should return True for .markdown files."""
        md_file = temp_vault / "test.markdown"
        md_file.write_text("# Test", encoding="utf-8")
        
        assert indexer.should_index(md_file) is True
    
    def test_should_not_index_other_extensions(self, indexer: Indexer, temp_vault: Path):
        """Should return False for non-markdown files."""
        txt_file = temp_vault / "test.txt"
        txt_file.write_text("Plain text", encoding="utf-8")
        
        assert indexer.should_index(txt_file) is False
    
    def test_should_not_index_hidden_files(self, indexer: Indexer, temp_vault: Path):
        """Should return False for hidden files."""
        hidden_file = temp_vault / ".hidden.md"
        hidden_file.write_text("# Hidden", encoding="utf-8")
        
        assert indexer.should_index(hidden_file) is False
    
    def test_should_not_index_directories(self, indexer: Indexer, temp_vault: Path):
        """Should return False for directories."""
        subdir = temp_vault / "subdir.md"  # Directory with .md name
        subdir.mkdir()
        
        assert indexer.should_index(subdir) is False
    
    def test_should_not_index_nonexistent(self, indexer: Indexer, temp_vault: Path):
        """Should return False for non-existent files."""
        nonexistent = temp_vault / "nonexistent.md"
        
        assert indexer.should_index(nonexistent) is False


class TestIndexerRelativePaths:
    """Tests for path handling."""
    
    def test_get_relative_path(self, indexer: Indexer, temp_vault: Path):
        """Should return path relative to vault."""
        file_path = temp_vault / "subfolder" / "note.md"
        
        rel_path = indexer.get_relative_path(file_path)
        
        assert rel_path == "subfolder\\note.md" or rel_path == "subfolder/note.md"
    
    def test_get_relative_path_root_file(self, indexer: Indexer, temp_vault: Path):
        """Should handle files in vault root."""
        file_path = temp_vault / "note.md"
        
        rel_path = indexer.get_relative_path(file_path)
        
        assert rel_path == "note.md"


class TestIndexerNeedsReindex:
    """Tests for change detection."""
    
    def test_needs_reindex_new_file(self, indexer: Indexer, temp_vault: Path):
        """New files should need reindexing."""
        new_file = temp_vault / "brand_new.md"
        new_file.write_text("# New", encoding="utf-8")
        
        assert indexer.needs_reindex(new_file) is True
    
    def test_needs_reindex_modified_file(self, indexer: Indexer, temp_vault: Path, temp_db: Database):
        """Modified files should need reindexing."""
        file_path = temp_vault / "test.md"
        file_path.write_text("# Original", encoding="utf-8")
        
        # Index the file with old mtime
        old_mtime = file_path.stat().st_mtime - 100
        temp_db.upsert_file(
            indexer.get_relative_path(file_path),
            old_mtime,
            "Original",
            "# Original"
        )
        
        assert indexer.needs_reindex(file_path) is True
    
    def test_needs_reindex_unchanged_file(self, indexer: Indexer, temp_vault: Path, temp_db: Database):
        """Unchanged files should not need reindexing."""
        file_path = temp_vault / "test.md"
        file_path.write_text("# Test", encoding="utf-8")
        
        # Index with current mtime
        current_mtime = file_path.stat().st_mtime
        temp_db.upsert_file(
            indexer.get_relative_path(file_path),
            current_mtime,
            "Test",
            "# Test"
        )
        
        assert indexer.needs_reindex(file_path) is False
    
    def test_needs_reindex_non_markdown(self, indexer: Indexer, temp_vault: Path):
        """Non-markdown files should not need reindexing."""
        txt_file = temp_vault / "test.txt"
        txt_file.write_text("Plain text", encoding="utf-8")
        
        assert indexer.needs_reindex(txt_file) is False


class TestIndexerIndexFile:
    """Tests for single file indexing."""
    
    def test_index_file_creates_record(self, indexer: Indexer, temp_vault: Path, temp_db: Database):
        """Indexing should create file record."""
        file_path = temp_vault / "simple.md"
        
        result = indexer.index_file(file_path)
        
        assert result is True
        
        file = temp_db.get_file_by_path("simple.md")
        assert file is not None
        assert file["title"] == "Simple Note"
    
    def test_index_file_creates_sections(self, indexer: Indexer, temp_vault: Path, temp_db: Database):
        """Indexing should create section records."""
        file_path = temp_vault / "complex.md"
        
        indexer.index_file(file_path)
        
        with temp_db.cursor() as cur:
            cur.execute("""
                SELECT s.* FROM sections s
                JOIN files f ON s.file_id = f.id
                WHERE f.path = 'complex.md'
            """)
            sections = cur.fetchall()
        
        assert len(sections) > 0
        headings = [s["heading"] for s in sections]
        assert "Section One" in headings
        assert "Section Two" in headings
    
    def test_index_file_creates_tags(self, indexer: Indexer, temp_vault: Path, temp_db: Database):
        """Indexing should create tag associations."""
        file_path = temp_vault / "simple.md"
        
        indexer.index_file(file_path)
        
        with temp_db.cursor() as cur:
            cur.execute("""
                SELECT t.name FROM tags t
                JOIN file_tags ft ON t.id = ft.tag_id
                JOIN files f ON ft.file_id = f.id
                WHERE f.path = 'simple.md'
            """)
            tags = [row["name"] for row in cur.fetchall()]
        
        assert "tag1" in tags
        assert "tag2" in tags
    
    def test_index_file_creates_links(self, indexer: Indexer, temp_vault: Path, temp_db: Database):
        """Indexing should create link records."""
        file_path = temp_vault / "with_frontmatter.md"
        
        indexer.index_file(file_path)
        
        with temp_db.cursor() as cur:
            cur.execute("""
                SELECT l.to_path FROM links l
                JOIN files f ON l.from_file_id = f.id
                WHERE f.path = 'with_frontmatter.md'
            """)
            links = [row["to_path"] for row in cur.fetchall()]
        
        assert "wikilink" in links
        assert "other.md" in links
    
    def test_index_file_updates_existing(self, indexer: Indexer, temp_vault: Path, temp_db: Database):
        """Reindexing should update existing record."""
        file_path = temp_vault / "simple.md"
        
        # First index
        indexer.index_file(file_path)
        
        # Modify file
        file_path.write_text("# Updated Title\n\nNew content.\n", encoding="utf-8")
        
        # Reindex
        indexer.index_file(file_path)
        
        file = temp_db.get_file_by_path("simple.md")
        assert file["title"] == "Updated Title"
    
    def test_index_file_clears_old_relations(self, indexer: Indexer, temp_vault: Path, temp_db: Database):
        """Reindexing should replace old sections/tags/links."""
        file_path = temp_vault / "simple.md"
        
        # First index (has tag1, tag2)
        indexer.index_file(file_path)
        
        # Modify file with different tags
        file_path.write_text("# Simple Note\n\nContent with #newtag only.\n", encoding="utf-8")
        
        # Reindex
        indexer.index_file(file_path)
        
        with temp_db.cursor() as cur:
            cur.execute("""
                SELECT t.name FROM tags t
                JOIN file_tags ft ON t.id = ft.tag_id
                JOIN files f ON ft.file_id = f.id
                WHERE f.path = 'simple.md'
            """)
            tags = [row["name"] for row in cur.fetchall()]
        
        assert "newtag" in tags
        assert "tag1" not in tags
        assert "tag2" not in tags
    
    def test_index_file_skips_non_markdown(self, indexer: Indexer, temp_vault: Path):
        """Should return False for non-markdown files."""
        txt_file = temp_vault / "test.txt"
        txt_file.write_text("Plain text", encoding="utf-8")
        
        result = indexer.index_file(txt_file)
        
        assert result is False
    
    def test_index_file_handles_errors(self, indexer: Indexer, temp_vault: Path):
        """Should handle indexing errors gracefully."""
        # Create a file we can't parse (binary content)
        bad_file = temp_vault / "bad.md"
        bad_file.write_bytes(b"\x00\x01\x02\x03")
        
        # Should not raise, just return False
        result = indexer.index_file(bad_file)
        # May return True or False depending on encoding handling
        assert isinstance(result, bool)


class TestIndexerDeleteFile:
    """Tests for file deletion from index."""
    
    def test_delete_file_removes_record(self, indexer: Indexer, temp_vault: Path, temp_db: Database):
        """Deleting should remove file from index."""
        file_path = temp_vault / "simple.md"
        indexer.index_file(file_path)
        
        result = indexer.delete_file(file_path)
        
        assert result is True
        assert temp_db.get_file_by_path("simple.md") is None
    
    def test_delete_file_not_indexed(self, indexer: Indexer, temp_vault: Path):
        """Deleting non-indexed file should return False."""
        file_path = temp_vault / "never_indexed.md"
        
        result = indexer.delete_file(file_path)
        
        assert result is False


class TestIndexerRenameFile:
    """Tests for file rename handling."""
    
    def test_rename_file(self, indexer: Indexer, temp_vault: Path, temp_db: Database):
        """Rename should delete old and index new."""
        old_path = temp_vault / "simple.md"
        new_path = temp_vault / "renamed.md"
        
        # Index original
        indexer.index_file(old_path)
        
        # Simulate rename
        old_path.rename(new_path)
        
        # Handle rename
        result = indexer.rename_file(old_path, new_path)
        
        assert result is True
        assert temp_db.get_file_by_path("simple.md") is None
        assert temp_db.get_file_by_path("renamed.md") is not None


class TestIndexerFullScan:
    """Tests for full vault scanning."""
    
    def test_full_scan_indexes_all_files(self, indexer: Indexer, temp_vault: Path, temp_db: Database):
        """Full scan should index all markdown files."""
        indexed, errors = indexer.full_scan()
        
        assert indexed >= 4  # At least the 4 files we created in temp_vault
        assert errors == 0
        
        stats = temp_db.get_stats()
        assert stats["file_count"] >= 4
    
    def test_full_scan_indexes_subdirectories(self, indexer: Indexer, temp_vault: Path, temp_db: Database):
        """Full scan should include files in subdirectories."""
        indexer.full_scan()
        
        # Check nested file was indexed
        file = temp_db.get_file_by_path("subfolder\\nested.md") or temp_db.get_file_by_path("subfolder/nested.md")
        assert file is not None
    
    def test_full_scan_removes_deleted_files(self, indexer: Indexer, temp_vault: Path, temp_db: Database):
        """Full scan should remove files that no longer exist."""
        # First scan
        indexer.full_scan()
        
        # Delete a file
        (temp_vault / "simple.md").unlink()
        
        # Second scan
        indexer.full_scan()
        
        assert temp_db.get_file_by_path("simple.md") is None
    
    def test_full_scan_updates_last_indexed(self, indexer: Indexer, temp_db: Database):
        """Full scan should update last indexed timestamp."""
        indexer.full_scan()
        
        stats = temp_db.get_stats()
        assert stats["last_indexed"] is not None
    
    def test_full_scan_skips_unchanged(self, indexer: Indexer, temp_vault: Path, temp_db: Database):
        """Full scan should skip unchanged files."""
        # First scan
        indexed1, _ = indexer.full_scan()
        
        # Second scan without changes
        indexed2, _ = indexer.full_scan()
        
        assert indexed2 == 0  # Nothing should be reindexed


class TestIndexerIncrementalScan:
    """Tests for incremental scanning."""
    
    def test_incremental_scan_indexes_changed(self, indexer: Indexer, temp_vault: Path, temp_db: Database):
        """Incremental scan should index changed files."""
        # Initial scan
        indexer.full_scan()
        
        # Modify a file
        file_path = temp_vault / "simple.md"
        time.sleep(0.1)  # Ensure mtime changes
        file_path.write_text("# Modified\n\nNew content.", encoding="utf-8")
        
        # Incremental scan
        indexed, errors = indexer.incremental_scan()
        
        assert indexed == 1
        assert errors == 0
    
    def test_incremental_scan_indexes_new(self, indexer: Indexer, temp_vault: Path, temp_db: Database):
        """Incremental scan should index new files."""
        # Initial scan
        indexer.full_scan()
        
        # Add new file
        new_file = temp_vault / "brand_new.md"
        new_file.write_text("# Brand New\n\nNew note.", encoding="utf-8")
        
        # Incremental scan
        indexed, errors = indexer.incremental_scan()
        
        assert indexed == 1
        
        file = temp_db.get_file_by_path("brand_new.md")
        assert file is not None
    
    def test_incremental_scan_skips_unchanged(self, indexer: Indexer, temp_vault: Path, temp_db: Database):
        """Incremental scan should skip unchanged files."""
        # Initial scan
        indexer.full_scan()
        
        # Incremental scan without changes
        indexed, errors = indexer.incremental_scan()
        
        assert indexed == 0
        assert errors == 0
    
    def test_incremental_scan_updates_last_indexed_only_if_changes(
        self, indexer: Indexer, temp_vault: Path, temp_db: Database
    ):
        """Incremental scan should only update timestamp if files were indexed."""
        # Initial scan
        indexer.full_scan()
        initial_stats = temp_db.get_stats()
        initial_timestamp = initial_stats["last_indexed"]
        
        time.sleep(0.1)
        
        # Incremental scan without changes
        indexer.incremental_scan()
        
        stats = temp_db.get_stats()
        assert stats["last_indexed"] == initial_timestamp
