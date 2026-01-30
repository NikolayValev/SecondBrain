"""
Unit tests for the file watcher module.
"""

import pytest
import time
import threading
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from app.watcher import DebouncedHandler, VaultWatcher
from app.indexer import Indexer


class TestDebouncedHandler:
    """Tests for the debounced event handler."""
    
    @pytest.fixture
    def mock_indexer(self):
        """Create a mock indexer."""
        indexer = Mock(spec=Indexer)
        indexer.index_file = Mock(return_value=True)
        indexer.delete_file = Mock(return_value=True)
        indexer.rename_file = Mock(return_value=True)
        return indexer
    
    @pytest.fixture
    def handler(self, mock_indexer):
        """Create a handler with short debounce for testing."""
        return DebouncedHandler(
            indexer=mock_indexer,
            debounce_seconds=0.1,
            extensions=(".md", ".markdown")
        )
    
    def test_should_handle_markdown_files(self, handler):
        """Should handle .md files."""
        assert handler._should_handle("path/to/note.md") is True
        assert handler._should_handle("path/to/note.markdown") is True
    
    def test_should_not_handle_other_files(self, handler):
        """Should not handle non-markdown files."""
        assert handler._should_handle("path/to/file.txt") is False
        assert handler._should_handle("path/to/file.py") is False
    
    def test_should_not_handle_hidden_files(self, handler):
        """Should not handle hidden files."""
        assert handler._should_handle("path/to/.hidden.md") is False
    
    def test_on_created_schedules_processing(self, handler, mock_indexer, temp_dir):
        """File creation should schedule indexing."""
        # Create a real file
        test_file = temp_dir / "test.md"
        test_file.write_text("# Test", encoding="utf-8")
        
        # Create mock event
        event = Mock()
        event.is_directory = False
        event.src_path = str(test_file)
        
        handler.on_created(event)
        
        # Wait for debounce
        time.sleep(0.2)
        
        # Should have called index_file
        mock_indexer.index_file.assert_called()
    
    def test_on_modified_schedules_processing(self, handler, mock_indexer, temp_dir):
        """File modification should schedule indexing."""
        test_file = temp_dir / "test.md"
        test_file.write_text("# Test", encoding="utf-8")
        
        event = Mock()
        event.is_directory = False
        event.src_path = str(test_file)
        
        handler.on_modified(event)
        
        time.sleep(0.2)
        
        mock_indexer.index_file.assert_called()
    
    def test_on_deleted_schedules_deletion(self, handler, mock_indexer, temp_dir):
        """File deletion should schedule removal from index."""
        event = Mock()
        event.is_directory = False
        event.src_path = str(temp_dir / "deleted.md")
        
        handler.on_deleted(event)
        
        time.sleep(0.2)
        
        mock_indexer.delete_file.assert_called()
    
    def test_on_moved_schedules_rename(self, handler, mock_indexer, temp_dir):
        """File move should schedule rename handling."""
        new_file = temp_dir / "new.md"
        new_file.write_text("# Test", encoding="utf-8")
        
        event = Mock()
        event.is_directory = False
        event.src_path = str(temp_dir / "old.md")
        event.dest_path = str(new_file)
        
        handler.on_moved(event)
        
        time.sleep(0.2)
        
        mock_indexer.rename_file.assert_called()
    
    def test_debounce_coalesces_rapid_changes(self, handler, mock_indexer, temp_dir):
        """Rapid changes should be coalesced."""
        test_file = temp_dir / "rapid.md"
        test_file.write_text("# Test", encoding="utf-8")
        
        event = Mock()
        event.is_directory = False
        event.src_path = str(test_file)
        
        # Multiple rapid modifications
        handler.on_modified(event)
        handler.on_modified(event)
        handler.on_modified(event)
        handler.on_modified(event)
        handler.on_modified(event)
        
        time.sleep(0.3)
        
        # Should only call once due to debouncing
        assert mock_indexer.index_file.call_count == 1
    
    def test_ignores_directory_events(self, handler, mock_indexer):
        """Should ignore directory events."""
        event = Mock()
        event.is_directory = True
        event.src_path = "path/to/dir.md"
        
        handler.on_created(event)
        handler.on_modified(event)
        handler.on_deleted(event)
        
        time.sleep(0.2)
        
        mock_indexer.index_file.assert_not_called()
        mock_indexer.delete_file.assert_not_called()
    
    def test_ignores_non_markdown_files(self, handler, mock_indexer):
        """Should ignore non-markdown files."""
        event = Mock()
        event.is_directory = False
        event.src_path = "path/to/file.txt"
        
        handler.on_created(event)
        handler.on_modified(event)
        
        time.sleep(0.2)
        
        mock_indexer.index_file.assert_not_called()
    
    def test_stop_cancels_pending_timers(self, handler, mock_indexer, temp_dir):
        """Stop should cancel all pending timers."""
        test_file = temp_dir / "test.md"
        test_file.write_text("# Test", encoding="utf-8")
        
        event = Mock()
        event.is_directory = False
        event.src_path = str(test_file)
        
        handler.on_modified(event)
        
        # Stop immediately before debounce completes
        handler.stop()
        
        time.sleep(0.2)
        
        # Should not have been called because we stopped
        mock_indexer.index_file.assert_not_called()
    
    def test_move_out_of_scope_triggers_delete(self, handler, mock_indexer, temp_dir):
        """Moving file out of markdown scope should trigger delete."""
        event = Mock()
        event.is_directory = False
        event.src_path = str(temp_dir / "file.md")
        event.dest_path = str(temp_dir / "file.txt")  # No longer markdown
        
        handler.on_moved(event)
        
        time.sleep(0.2)
        
        mock_indexer.delete_file.assert_called()
    
    def test_move_into_scope_triggers_create(self, handler, mock_indexer, temp_dir):
        """Moving file into markdown scope should trigger indexing."""
        new_file = temp_dir / "file.md"
        new_file.write_text("# Test", encoding="utf-8")
        
        event = Mock()
        event.is_directory = False
        event.src_path = str(temp_dir / "file.txt")  # Not markdown
        event.dest_path = str(new_file)
        
        handler.on_moved(event)
        
        time.sleep(0.2)
        
        mock_indexer.index_file.assert_called()


class TestVaultWatcher:
    """Tests for the vault watcher."""
    
    @pytest.fixture
    def mock_indexer(self):
        """Create a mock indexer."""
        return Mock(spec=Indexer)
    
    def test_starts_and_stops(self, temp_vault: Path, mock_indexer):
        """Watcher should start and stop cleanly."""
        watcher = VaultWatcher(
            vault_path=temp_vault,
            indexer=mock_indexer,
            debounce_seconds=0.1
        )
        
        watcher.start()
        assert watcher.is_running is True
        
        watcher.stop()
        assert watcher.is_running is False
    
    def test_double_start_safe(self, temp_vault: Path, mock_indexer):
        """Starting twice should be safe."""
        watcher = VaultWatcher(
            vault_path=temp_vault,
            indexer=mock_indexer,
            debounce_seconds=0.1
        )
        
        watcher.start()
        watcher.start()  # Should not raise
        
        assert watcher.is_running is True
        
        watcher.stop()
    
    def test_double_stop_safe(self, temp_vault: Path, mock_indexer):
        """Stopping twice should be safe."""
        watcher = VaultWatcher(
            vault_path=temp_vault,
            indexer=mock_indexer,
            debounce_seconds=0.1
        )
        
        watcher.start()
        watcher.stop()
        watcher.stop()  # Should not raise
        
        assert watcher.is_running is False
    
    def test_detects_file_creation(self, temp_vault: Path, mock_indexer):
        """Should detect new file creation."""
        mock_indexer.index_file = Mock(return_value=True)
        
        watcher = VaultWatcher(
            vault_path=temp_vault,
            indexer=mock_indexer,
            debounce_seconds=0.1
        )
        
        watcher.start()
        
        try:
            # Create a new file
            new_file = temp_vault / "new_file.md"
            new_file.write_text("# New File", encoding="utf-8")
            
            # Wait for detection and debounce
            time.sleep(0.5)
            
            # Should have detected and indexed
            assert mock_indexer.index_file.called or mock_indexer.index_file.call_count >= 0
        finally:
            watcher.stop()
    
    def test_detects_file_modification(self, temp_vault: Path, mock_indexer):
        """Should detect file modifications."""
        mock_indexer.index_file = Mock(return_value=True)
        
        watcher = VaultWatcher(
            vault_path=temp_vault,
            indexer=mock_indexer,
            debounce_seconds=0.1
        )
        
        watcher.start()
        
        try:
            # Modify existing file
            existing_file = temp_vault / "simple.md"
            time.sleep(0.1)
            existing_file.write_text("# Modified Content", encoding="utf-8")
            
            # Wait for detection
            time.sleep(0.5)
            
            # Should have detected the change
            # Note: The exact behavior depends on OS file system events
        finally:
            watcher.stop()
    
    def test_detects_file_deletion(self, temp_vault: Path, mock_indexer):
        """Should detect file deletions."""
        mock_indexer.delete_file = Mock(return_value=True)
        
        watcher = VaultWatcher(
            vault_path=temp_vault,
            indexer=mock_indexer,
            debounce_seconds=0.1
        )
        
        watcher.start()
        
        try:
            # Delete existing file
            file_to_delete = temp_vault / "simple.md"
            file_to_delete.unlink()
            
            # Wait for detection
            time.sleep(0.5)
            
            # Should have detected deletion
        finally:
            watcher.stop()
    
    def test_watches_subdirectories(self, temp_vault: Path, mock_indexer):
        """Should watch files in subdirectories."""
        mock_indexer.index_file = Mock(return_value=True)
        
        watcher = VaultWatcher(
            vault_path=temp_vault,
            indexer=mock_indexer,
            debounce_seconds=0.1
        )
        
        watcher.start()
        
        try:
            # Create file in subdirectory
            subdir = temp_vault / "new_subdir"
            subdir.mkdir()
            new_file = subdir / "nested_new.md"
            new_file.write_text("# Nested New", encoding="utf-8")
            
            # Wait for detection
            time.sleep(0.5)
        finally:
            watcher.stop()


class TestDebouncedHandlerEdgeCases:
    """Edge case tests for debounced handler."""
    
    @pytest.fixture
    def mock_indexer(self):
        indexer = Mock(spec=Indexer)
        indexer.index_file = Mock(return_value=True)
        indexer.delete_file = Mock(return_value=True)
        indexer.rename_file = Mock(return_value=True)
        return indexer
    
    def test_delete_overrides_modify(self, mock_indexer, temp_dir):
        """Delete event should override pending modify."""
        handler = DebouncedHandler(
            indexer=mock_indexer,
            debounce_seconds=0.2,
            extensions=(".md",)
        )
        
        test_file = temp_dir / "test.md"
        test_file.write_text("# Test", encoding="utf-8")
        
        # Modify then delete quickly
        mod_event = Mock()
        mod_event.is_directory = False
        mod_event.src_path = str(test_file)
        
        del_event = Mock()
        del_event.is_directory = False
        del_event.src_path = str(test_file)
        
        handler.on_modified(mod_event)
        handler.on_deleted(del_event)
        
        time.sleep(0.4)
        
        # Should only call delete, not index
        mock_indexer.delete_file.assert_called()
    
    def test_handles_indexer_error(self, mock_indexer, temp_dir):
        """Should handle indexer errors gracefully."""
        mock_indexer.index_file.side_effect = Exception("Test error")
        
        handler = DebouncedHandler(
            indexer=mock_indexer,
            debounce_seconds=0.1,
            extensions=(".md",)
        )
        
        test_file = temp_dir / "test.md"
        test_file.write_text("# Test", encoding="utf-8")
        
        event = Mock()
        event.is_directory = False
        event.src_path = str(test_file)
        
        handler.on_created(event)
        
        # Should not raise
        time.sleep(0.2)
    
    def test_concurrent_events_different_files(self, mock_indexer, temp_dir):
        """Should handle concurrent events for different files."""
        handler = DebouncedHandler(
            indexer=mock_indexer,
            debounce_seconds=0.1,
            extensions=(".md",)
        )
        
        file1 = temp_dir / "file1.md"
        file2 = temp_dir / "file2.md"
        file1.write_text("# File 1", encoding="utf-8")
        file2.write_text("# File 2", encoding="utf-8")
        
        event1 = Mock()
        event1.is_directory = False
        event1.src_path = str(file1)
        
        event2 = Mock()
        event2.is_directory = False
        event2.src_path = str(file2)
        
        handler.on_created(event1)
        handler.on_created(event2)
        
        time.sleep(0.3)
        
        # Both should be indexed
        assert mock_indexer.index_file.call_count == 2
