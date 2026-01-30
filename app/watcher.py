"""
File watcher module for Second Brain daemon.
Watches vault directory for changes with debouncing.
"""

import logging
import threading
from pathlib import Path
from typing import Optional, Callable
from collections import defaultdict
from datetime import datetime, timedelta

from watchdog.observers import Observer
from watchdog.events import (
    FileSystemEventHandler,
    FileCreatedEvent,
    FileModifiedEvent,
    FileDeletedEvent,
    FileMovedEvent,
    DirCreatedEvent,
    DirDeletedEvent,
    DirMovedEvent,
)

from app.config import config
from app.indexer import Indexer, indexer as default_indexer

logger = logging.getLogger(__name__)


class DebouncedHandler(FileSystemEventHandler):
    """
    File system event handler with debouncing.
    Aggregates rapid changes and processes them after a delay.
    """
    
    def __init__(
        self, 
        indexer: Indexer,
        debounce_seconds: float = 1.0,
        extensions: tuple = (".md", ".markdown")
    ):
        super().__init__()
        self.indexer = indexer
        self.debounce_seconds = debounce_seconds
        self.extensions = extensions
        
        # Track pending events per path
        self._pending: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._timers: dict[str, threading.Timer] = {}
    
    def _should_handle(self, path: str) -> bool:
        """Check if path should be handled."""
        p = Path(path)
        return (
            p.suffix.lower() in self.extensions and
            not p.name.startswith('.')
        )
    
    def _schedule_processing(self, path: str) -> None:
        """Schedule debounced processing for a path."""
        with self._lock:
            # Cancel existing timer for this path
            if path in self._timers:
                self._timers[path].cancel()
            
            # Schedule new timer
            timer = threading.Timer(
                self.debounce_seconds,
                self._process_path,
                args=(path,)
            )
            self._timers[path] = timer
            timer.start()
    
    def _process_path(self, path: str) -> None:
        """Process pending events for a path."""
        with self._lock:
            if path not in self._pending:
                return
            
            event_info = self._pending.pop(path)
            if path in self._timers:
                del self._timers[path]
        
        event_type = event_info.get("type")
        file_path = Path(path)
        
        try:
            if event_type == "deleted":
                self.indexer.delete_file(file_path)
            elif event_type == "moved":
                old_path = Path(event_info.get("old_path", path))
                self.indexer.rename_file(old_path, file_path)
            else:  # created or modified
                if file_path.exists():
                    self.indexer.index_file(file_path)
        except Exception as e:
            logger.error(f"Error processing {path}: {e}")
    
    def on_created(self, event: FileCreatedEvent) -> None:
        """Handle file creation."""
        if event.is_directory:
            return
        
        path = event.src_path
        if not self._should_handle(path):
            return
        
        logger.debug(f"File created: {path}")
        
        with self._lock:
            self._pending[path] = {"type": "created", "time": datetime.now()}
        
        self._schedule_processing(path)
    
    def on_modified(self, event: FileModifiedEvent) -> None:
        """Handle file modification."""
        if event.is_directory:
            return
        
        path = event.src_path
        if not self._should_handle(path):
            return
        
        logger.debug(f"File modified: {path}")
        
        with self._lock:
            # Don't override delete or move events
            if path in self._pending and self._pending[path]["type"] in ("deleted", "moved"):
                return
            self._pending[path] = {"type": "modified", "time": datetime.now()}
        
        self._schedule_processing(path)
    
    def on_deleted(self, event: FileDeletedEvent) -> None:
        """Handle file deletion."""
        if event.is_directory:
            return
        
        path = event.src_path
        if not self._should_handle(path):
            return
        
        logger.debug(f"File deleted: {path}")
        
        with self._lock:
            self._pending[path] = {"type": "deleted", "time": datetime.now()}
        
        self._schedule_processing(path)
    
    def on_moved(self, event: FileMovedEvent) -> None:
        """Handle file rename/move."""
        if event.is_directory:
            return
        
        old_path = event.src_path
        new_path = event.dest_path
        
        old_should = self._should_handle(old_path)
        new_should = self._should_handle(new_path)
        
        logger.debug(f"File moved: {old_path} -> {new_path}")
        
        with self._lock:
            # Remove old path from pending if exists
            if old_path in self._pending:
                if old_path in self._timers:
                    self._timers[old_path].cancel()
                    del self._timers[old_path]
                del self._pending[old_path]
            
            if old_should and new_should:
                # Rename within tracked files
                self._pending[new_path] = {
                    "type": "moved",
                    "old_path": old_path,
                    "time": datetime.now()
                }
                self._schedule_processing(new_path)
            elif old_should:
                # Moved out of scope - treat as delete
                self._pending[old_path] = {"type": "deleted", "time": datetime.now()}
                self._schedule_processing(old_path)
            elif new_should:
                # Moved into scope - treat as create
                self._pending[new_path] = {"type": "created", "time": datetime.now()}
                self._schedule_processing(new_path)
    
    def stop(self) -> None:
        """Stop all pending timers."""
        with self._lock:
            for timer in self._timers.values():
                timer.cancel()
            self._timers.clear()
            self._pending.clear()


class VaultWatcher:
    """
    Watches an Obsidian vault for file changes.
    Uses watchdog for cross-platform file system monitoring.
    """
    
    def __init__(
        self,
        vault_path: Optional[Path] = None,
        indexer: Optional[Indexer] = None,
        debounce_seconds: Optional[float] = None
    ):
        self.vault_path = vault_path or config.VAULT_PATH
        self.indexer = indexer or default_indexer
        self.debounce_seconds = debounce_seconds or config.DEBOUNCE_SECONDS
        
        self._observer: Optional[Observer] = None
        self._handler: Optional[DebouncedHandler] = None
        self._running = False
    
    def start(self) -> None:
        """Start watching the vault directory."""
        if self._running:
            logger.warning("Watcher already running")
            return
        
        logger.info(f"Starting vault watcher: {self.vault_path}")
        
        self._handler = DebouncedHandler(
            indexer=self.indexer,
            debounce_seconds=self.debounce_seconds,
            extensions=config.MARKDOWN_EXTENSIONS
        )
        
        self._observer = Observer()
        self._observer.schedule(
            self._handler,
            str(self.vault_path),
            recursive=True
        )
        
        self._observer.start()
        self._running = True
        
        logger.info("Vault watcher started")
    
    def stop(self) -> None:
        """Stop watching the vault directory."""
        if not self._running:
            return
        
        logger.info("Stopping vault watcher")
        
        if self._handler:
            self._handler.stop()
        
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)
        
        self._running = False
        self._observer = None
        self._handler = None
        
        logger.info("Vault watcher stopped")
    
    @property
    def is_running(self) -> bool:
        """Check if watcher is running."""
        return self._running


# Singleton watcher instance
watcher = VaultWatcher()
