"""
Inbox service: inbox file listing, processing, and contents.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.config import config
from app.inbox_processor import inbox_processor, create_default_config
from app.api.models.inbox import (
    InboxProcessRequest,
    InboxProcessResponse,
    InboxFileResult,
    InboxFileInfo,
    InboxFolderInfo,
    InboxContentsResponse,
)

logger = logging.getLogger(__name__)


class InboxService:
    """Handles inbox listing and processing."""

    async def process(self, request: InboxProcessRequest) -> InboxProcessResponse:
        """
        Run the inbox processor (classify, tag, move).

        Args:
            request: Contains dry_run flag.

        Returns:
            InboxProcessResponse with per-file results.
        """
        from app.inbox_processor import InboxProcessor

        processor_config = create_default_config()
        processor_config.dry_run = request.dry_run
        processor = InboxProcessor(processor_config)

        result = await processor.process_inbox()

        return InboxProcessResponse(
            processed=result.processed,
            moved=result.moved,
            skipped=result.skipped,
            errors=result.errors,
            duration_seconds=result.duration_seconds,
            results=[
                InboxFileResult(
                    source_path=str(r.source_path),
                    destination_path=str(r.destination_path) if r.destination_path else None,
                    action=r.action,
                    classification=r.classification,
                    added_tags=r.added_tags,
                    error=r.error,
                )
                for r in result.results
            ],
        )

    def list_files(self) -> dict:
        """Return a flat list of inbox files."""
        files = inbox_processor.get_inbox_files()
        return {
            "count": len(files),
            "files": [
                {
                    "name": f.name,
                    "path": str(f.relative_to(config.VAULT_PATH)),
                    "size_bytes": f.stat().st_size,
                    "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                }
                for f in files
            ],
        }

    def get_contents(self) -> InboxContentsResponse:
        """
        Recursively scan the inbox folder.

        Returns:
            InboxContentsResponse with hierarchical file/folder info.

        Raises:
            FileNotFoundError: If 00_Inbox doesn't exist.
        """
        inbox_path = config.VAULT_PATH / "00_Inbox"
        if not inbox_path.exists():
            raise FileNotFoundError("Inbox folder (00_Inbox) not found")

        root_files, sub_folders = self._scan_folder(inbox_path)
        nested_files, nested_folders = self._count_items(sub_folders)

        return InboxContentsResponse(
            inbox_path=str(inbox_path.relative_to(config.VAULT_PATH)),
            total_files=len(root_files) + nested_files,
            total_folders=nested_folders,
            root_files=root_files,
            folders=sub_folders,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _file_info(self, file_path: Path) -> InboxFileInfo:
        stat = file_path.stat()
        return InboxFileInfo(
            name=file_path.name,
            path=str(file_path.relative_to(config.VAULT_PATH)),
            size_bytes=stat.st_size,
            modified=datetime.fromtimestamp(stat.st_mtime).isoformat(),
        )

    def _scan_folder(
        self, folder_path: Path
    ) -> tuple[list[InboxFileInfo], list[InboxFolderInfo]]:
        files: list[InboxFileInfo] = []
        folders: list[InboxFolderInfo] = []
        try:
            for item in sorted(folder_path.iterdir()):
                if item.name.startswith("."):
                    continue
                if item.is_file() and item.suffix.lower() in (".md", ".markdown"):
                    files.append(self._file_info(item))
                elif item.is_dir():
                    sub_files, sub_folders = self._scan_folder(item)
                    folders.append(InboxFolderInfo(
                        name=item.name,
                        path=str(item.relative_to(config.VAULT_PATH)),
                        files=sub_files,
                        folders=sub_folders,
                    ))
        except PermissionError:
            logger.warning("Permission denied accessing: %s", folder_path)
        return files, folders

    @staticmethod
    def _count_items(folders: list[InboxFolderInfo]) -> tuple[int, int]:
        total_files = 0
        total_folders = len(folders)
        for folder in folders:
            total_files += len(folder.files)
            sub_files, sub_folder_count = InboxService._count_items(folder.folders)
            total_files += sub_files
            total_folders += sub_folder_count
        return total_files, total_folders


# Singleton
inbox_service = InboxService()
