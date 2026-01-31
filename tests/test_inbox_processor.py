"""
Unit tests for the inbox processor module.
"""

import pytest
import asyncio
from pathlib import Path
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from app.inbox_processor import (
    InboxProcessor,
    InboxConfig,
    ClassificationRule,
    ClassificationMethod,
    ConflictResolution,
    MetadataAction,
    LLMClassificationConfig,
    ProcessingResult,
    BatchResult,
    create_default_config,
)
from app.parser import ParsedMarkdown, Section


@pytest.fixture
def temp_inbox(temp_vault: Path) -> Path:
    """Create an inbox folder in the temp vault."""
    inbox = temp_vault / "00_Inbox"
    inbox.mkdir()
    return inbox


@pytest.fixture
def basic_config(temp_vault: Path) -> InboxConfig:
    """Create a basic inbox config for testing."""
    config = InboxConfig()
    config.inbox_folder = "00_Inbox"
    config.default_destination = "00_Inbox/Unsorted"
    config.classification_method = ClassificationMethod.RULES_ONLY
    config.dry_run = False
    config.backup_before_move = False
    config.reindex_after_move = False
    config.llm_config.enabled = False
    config.rules = []
    return config


@pytest.fixture
def inbox_processor_fixture(temp_vault: Path, basic_config: InboxConfig) -> InboxProcessor:
    """Create an inbox processor with temp vault."""
    with patch('app.inbox_processor.config') as mock_config:
        mock_config.VAULT_PATH = temp_vault
        processor = InboxProcessor(basic_config)
        processor.vault_path = temp_vault
        return processor


class TestClassificationRule:
    """Tests for ClassificationRule matching."""
    
    def test_rule_matches_title_pattern(self):
        """Should match when title matches pattern."""
        rule = ClassificationRule(
            name="test",
            destination_folder="dest",
            title_patterns=[r"meeting", r"standup"],
        )
        
        parsed = ParsedMarkdown(
            title="Weekly Team Meeting",
            content="Some content"
        )
        
        # Create processor to test matching
        config = InboxConfig()
        config.rules = [rule]
        
        with patch('app.inbox_processor.config') as mock_config:
            mock_config.VAULT_PATH = Path("/fake")
            processor = InboxProcessor(config)
        
        matched = processor._match_rule(rule, parsed, Path("test.md"))
        assert matched is True
    
    def test_rule_matches_filename_pattern(self):
        """Should match when filename matches pattern."""
        rule = ClassificationRule(
            name="journal",
            destination_folder="05_Journal",
            filename_patterns=[r"^\d{4}-\d{2}-\d{2}"],
        )
        
        parsed = ParsedMarkdown(title="Note", content="Content")
        
        config = InboxConfig()
        config.rules = [rule]
        
        with patch('app.inbox_processor.config') as mock_config:
            mock_config.VAULT_PATH = Path("/fake")
            processor = InboxProcessor(config)
        
        matched = processor._match_rule(rule, parsed, Path("2024-01-15.md"))
        assert matched is True
        
        not_matched = processor._match_rule(rule, parsed, Path("notes.md"))
        assert not_matched is False
    
    def test_rule_matches_content_pattern(self):
        """Should match when content matches pattern."""
        rule = ClassificationRule(
            name="meeting",
            destination_folder="06_Meetings",
            content_patterns=[r"attendees?:", r"agenda:"],
        )
        
        parsed = ParsedMarkdown(
            title="Note",
            content="Meeting notes\n\nAttendees: John, Jane\n\nTopics discussed..."
        )
        
        config = InboxConfig()
        with patch('app.inbox_processor.config') as mock_config:
            mock_config.VAULT_PATH = Path("/fake")
            processor = InboxProcessor(config)
        
        matched = processor._match_rule(rule, parsed, Path("test.md"))
        assert matched is True
    
    def test_rule_matches_any_tags(self):
        """Should match when document has any of the specified tags."""
        rule = ClassificationRule(
            name="fitness",
            destination_folder="14_Fitness",
            any_tags=["fitness", "workout", "exercise"],
        )
        
        parsed = ParsedMarkdown(
            title="Today's workout",
            content="Did some exercises",
            tags=["gym", "workout"]
        )
        
        config = InboxConfig()
        with patch('app.inbox_processor.config') as mock_config:
            mock_config.VAULT_PATH = Path("/fake")
            processor = InboxProcessor(config)
        
        matched = processor._match_rule(rule, parsed, Path("test.md"))
        assert matched is True
    
    def test_rule_matches_required_tags(self):
        """Should only match when document has ALL required tags."""
        rule = ClassificationRule(
            name="project",
            destination_folder="03_Projects",
            required_tags=["project", "active"],
        )
        
        parsed_match = ParsedMarkdown(
            title="Project X",
            content="...",
            tags=["project", "active", "priority"]
        )
        
        parsed_no_match = ParsedMarkdown(
            title="Project Y",
            content="...",
            tags=["project"]  # Missing "active"
        )
        
        config = InboxConfig()
        with patch('app.inbox_processor.config') as mock_config:
            mock_config.VAULT_PATH = Path("/fake")
            processor = InboxProcessor(config)
        
        assert processor._match_rule(rule, parsed_match, Path("test.md")) is True
        assert processor._match_rule(rule, parsed_no_match, Path("test.md")) is False
    
    def test_rule_matches_frontmatter(self):
        """Should match when frontmatter key matches pattern."""
        rule = ClassificationRule(
            name="draft",
            destination_folder="Drafts",
            frontmatter_matches={"status": r"draft|wip"},
        )
        
        parsed = ParsedMarkdown(
            title="Article",
            content="...",
            frontmatter={"status": "draft", "author": "me"}
        )
        
        config = InboxConfig()
        with patch('app.inbox_processor.config') as mock_config:
            mock_config.VAULT_PATH = Path("/fake")
            processor = InboxProcessor(config)
        
        matched = processor._match_rule(rule, parsed, Path("test.md"))
        assert matched is True
    
    def test_rule_priority_ordering(self):
        """Higher priority rules should be checked first."""
        low_priority = ClassificationRule(
            name="catch_all",
            destination_folder="Archive",
            content_patterns=[r".+"],  # Matches everything
            priority=0,
        )
        
        high_priority = ClassificationRule(
            name="meeting",
            destination_folder="Meetings",
            title_patterns=[r"meeting"],
            priority=10,
        )
        
        config = InboxConfig()
        config.rules = [low_priority, high_priority]
        
        with patch('app.inbox_processor.config') as mock_config:
            mock_config.VAULT_PATH = Path("/fake")
            processor = InboxProcessor(config)
        
        parsed = ParsedMarkdown(title="Team Meeting", content="Some content")
        
        matched_rule = processor.classify_by_rules(parsed, Path("test.md"))
        assert matched_rule is not None
        assert matched_rule.name == "meeting"


class TestInboxProcessorFileDiscovery:
    """Tests for inbox file discovery."""
    
    def test_get_inbox_files_returns_markdown(self, temp_vault: Path, temp_inbox: Path):
        """Should return only markdown files from inbox."""
        (temp_inbox / "note1.md").write_text("# Note 1", encoding="utf-8")
        (temp_inbox / "note2.md").write_text("# Note 2", encoding="utf-8")
        (temp_inbox / "image.png").write_bytes(b"fake image")
        (temp_inbox / "data.txt").write_text("text file", encoding="utf-8")
        
        config = InboxConfig()
        config.inbox_folder = "00_Inbox"
        
        with patch('app.inbox_processor.config') as mock_config:
            mock_config.VAULT_PATH = temp_vault
            processor = InboxProcessor(config)
            processor.vault_path = temp_vault
        
        files = processor.get_inbox_files()
        
        assert len(files) == 2
        assert all(f.suffix == ".md" for f in files)
    
    def test_get_inbox_files_ignores_hidden(self, temp_vault: Path, temp_inbox: Path):
        """Should ignore hidden files (starting with .)."""
        (temp_inbox / "visible.md").write_text("# Visible", encoding="utf-8")
        (temp_inbox / ".hidden.md").write_text("# Hidden", encoding="utf-8")
        
        config = InboxConfig()
        config.inbox_folder = "00_Inbox"
        config.ignore_patterns = [r"^\."]
        
        with patch('app.inbox_processor.config') as mock_config:
            mock_config.VAULT_PATH = temp_vault
            processor = InboxProcessor(config)
            processor.vault_path = temp_vault
        
        files = processor.get_inbox_files()
        
        assert len(files) == 1
        assert files[0].name == "visible.md"
    
    def test_get_inbox_files_ignores_templates(self, temp_vault: Path, temp_inbox: Path):
        """Should ignore template files."""
        (temp_inbox / "note.md").write_text("# Note", encoding="utf-8")
        (temp_inbox / "template_meeting.md").write_text("# Template", encoding="utf-8")
        
        config = InboxConfig()
        config.inbox_folder = "00_Inbox"
        config.ignore_patterns = [r"template"]
        
        with patch('app.inbox_processor.config') as mock_config:
            mock_config.VAULT_PATH = temp_vault
            processor = InboxProcessor(config)
            processor.vault_path = temp_vault
        
        files = processor.get_inbox_files()
        
        assert len(files) == 1
        assert files[0].name == "note.md"
    
    def test_get_inbox_files_skips_subdirectories(self, temp_vault: Path, temp_inbox: Path):
        """Should not process files in subdirectories."""
        (temp_inbox / "root_note.md").write_text("# Root", encoding="utf-8")
        
        subdir = temp_inbox / "Unsorted"
        subdir.mkdir()
        (subdir / "nested.md").write_text("# Nested", encoding="utf-8")
        
        config = InboxConfig()
        config.inbox_folder = "00_Inbox"
        
        with patch('app.inbox_processor.config') as mock_config:
            mock_config.VAULT_PATH = temp_vault
            processor = InboxProcessor(config)
            processor.vault_path = temp_vault
        
        files = processor.get_inbox_files()
        
        assert len(files) == 1
        assert files[0].name == "root_note.md"
    
    def test_get_inbox_files_empty_inbox(self, temp_vault: Path, temp_inbox: Path):
        """Should return empty list for empty inbox."""
        config = InboxConfig()
        config.inbox_folder = "00_Inbox"
        
        with patch('app.inbox_processor.config') as mock_config:
            mock_config.VAULT_PATH = temp_vault
            processor = InboxProcessor(config)
            processor.vault_path = temp_vault
        
        files = processor.get_inbox_files()
        assert files == []
    
    def test_get_inbox_files_nonexistent_inbox(self, temp_vault: Path):
        """Should return empty list if inbox doesn't exist."""
        config = InboxConfig()
        config.inbox_folder = "NonExistent_Inbox"
        
        with patch('app.inbox_processor.config') as mock_config:
            mock_config.VAULT_PATH = temp_vault
            processor = InboxProcessor(config)
            processor.vault_path = temp_vault
        
        files = processor.get_inbox_files()
        assert files == []


class TestConflictResolution:
    """Tests for file conflict resolution."""
    
    def test_resolve_no_conflict(self, temp_vault: Path):
        """Should return destination path when no conflict exists."""
        dest_folder = temp_vault / "Destination"
        dest_folder.mkdir()
        
        config = InboxConfig()
        config.conflict_resolution = ConflictResolution.RENAME
        
        with patch('app.inbox_processor.config') as mock_config:
            mock_config.VAULT_PATH = temp_vault
            processor = InboxProcessor(config)
            processor.vault_path = temp_vault
        
        result = processor._resolve_destination_path(Path("test.md"), "Destination")
        
        assert result == dest_folder / "test.md"
    
    def test_resolve_skip_on_conflict(self, temp_vault: Path):
        """Should return None when conflict_resolution is SKIP."""
        dest_folder = temp_vault / "Destination"
        dest_folder.mkdir()
        (dest_folder / "existing.md").write_text("# Existing", encoding="utf-8")
        
        config = InboxConfig()
        config.conflict_resolution = ConflictResolution.SKIP
        
        with patch('app.inbox_processor.config') as mock_config:
            mock_config.VAULT_PATH = temp_vault
            processor = InboxProcessor(config)
            processor.vault_path = temp_vault
        
        result = processor._resolve_destination_path(Path("existing.md"), "Destination")
        
        assert result is None
    
    def test_resolve_overwrite_on_conflict(self, temp_vault: Path):
        """Should return same path when conflict_resolution is OVERWRITE."""
        dest_folder = temp_vault / "Destination"
        dest_folder.mkdir()
        existing = dest_folder / "existing.md"
        existing.write_text("# Existing", encoding="utf-8")
        
        config = InboxConfig()
        config.conflict_resolution = ConflictResolution.OVERWRITE
        
        with patch('app.inbox_processor.config') as mock_config:
            mock_config.VAULT_PATH = temp_vault
            processor = InboxProcessor(config)
            processor.vault_path = temp_vault
        
        result = processor._resolve_destination_path(Path("existing.md"), "Destination")
        
        assert result == existing
    
    def test_resolve_rename_on_conflict(self, temp_vault: Path):
        """Should add timestamp suffix when conflict_resolution is RENAME."""
        dest_folder = temp_vault / "Destination"
        dest_folder.mkdir()
        (dest_folder / "existing.md").write_text("# Existing", encoding="utf-8")
        
        config = InboxConfig()
        config.conflict_resolution = ConflictResolution.RENAME
        
        with patch('app.inbox_processor.config') as mock_config:
            mock_config.VAULT_PATH = temp_vault
            processor = InboxProcessor(config)
            processor.vault_path = temp_vault
        
        result = processor._resolve_destination_path(Path("existing.md"), "Destination")
        
        assert result is not None
        assert result != dest_folder / "existing.md"
        assert result.stem.startswith("existing_")
        assert result.suffix == ".md"


class TestFrontmatterUpdate:
    """Tests for frontmatter manipulation."""
    
    def test_update_frontmatter_adds_new_tags(self, temp_vault: Path):
        """Should add new tags to frontmatter."""
        content = """---
title: Test
tags:
  - existing-tag
---

# Content
"""
        config = InboxConfig()
        config.metadata_action = MetadataAction.MERGE
        config.global_frontmatter = {}
        
        with patch('app.inbox_processor.config') as mock_config:
            mock_config.VAULT_PATH = temp_vault
            processor = InboxProcessor(config)
        
        result = processor._update_frontmatter(
            content,
            {"title": "Test", "tags": ["existing-tag"]},
            ["new-tag", "another-tag"],
            {}
        )
        
        assert "existing-tag" in result
        assert "new-tag" in result
        assert "another-tag" in result
    
    def test_update_frontmatter_preserves_existing(self, temp_vault: Path):
        """Should preserve existing frontmatter when merging."""
        content = """---
title: Original
author: Me
---

# Content
"""
        config = InboxConfig()
        config.metadata_action = MetadataAction.PRESERVE
        config.global_frontmatter = {}
        
        with patch('app.inbox_processor.config') as mock_config:
            mock_config.VAULT_PATH = temp_vault
            processor = InboxProcessor(config)
        
        result = processor._update_frontmatter(
            content,
            {"title": "Original", "author": "Me"},
            [],
            {"status": "processed"}
        )
        
        assert "title: Original" in result
        assert "author: Me" in result
        assert "status: processed" in result
    
    def test_update_frontmatter_skip(self, temp_vault: Path):
        """Should not modify content when action is SKIP."""
        content = "# No Frontmatter\n\nContent here"
        
        config = InboxConfig()
        config.metadata_action = MetadataAction.SKIP
        
        with patch('app.inbox_processor.config') as mock_config:
            mock_config.VAULT_PATH = temp_vault
            processor = InboxProcessor(config)
        
        result = processor._update_frontmatter(content, {}, ["tag"], {"key": "value"})
        
        assert result == content
    
    def test_update_frontmatter_adds_processed_date(self, temp_vault: Path):
        """Should add processed_date from global frontmatter."""
        content = "# Note\n\nContent"
        
        config = InboxConfig()
        config.metadata_action = MetadataAction.MERGE
        config.global_frontmatter = {"processed_date": None, "source": "inbox"}
        
        with patch('app.inbox_processor.config') as mock_config:
            mock_config.VAULT_PATH = temp_vault
            processor = InboxProcessor(config)
        
        result = processor._update_frontmatter(content, {}, [], {})
        
        assert "processed_date:" in result
        assert "source: inbox" in result


class TestProcessFile:
    """Tests for single file processing."""
    
    @pytest.mark.asyncio
    async def test_process_file_dry_run(self, temp_vault: Path, temp_inbox: Path):
        """Should not move file in dry run mode."""
        test_file = temp_inbox / "test.md"
        test_file.write_text("# Test\n\nContent", encoding="utf-8")
        
        config = InboxConfig()
        config.inbox_folder = "00_Inbox"
        config.default_destination = "Archive"
        config.dry_run = True
        config.classification_method = ClassificationMethod.RULES_ONLY
        config.llm_config.enabled = False
        
        with patch('app.inbox_processor.config') as mock_config:
            mock_config.VAULT_PATH = temp_vault
            processor = InboxProcessor(config)
            processor.vault_path = temp_vault
        
        result = await processor.process_file(test_file)
        
        assert result.success is True
        assert result.action == "dry_run"
        assert test_file.exists()  # File should still exist
    
    @pytest.mark.asyncio
    async def test_process_file_moves_to_destination(self, temp_vault: Path, temp_inbox: Path):
        """Should move file to classified destination."""
        test_file = temp_inbox / "test.md"
        test_file.write_text("# Test\n\nContent", encoding="utf-8")
        
        config = InboxConfig()
        config.inbox_folder = "00_Inbox"
        config.default_destination = "Archive"
        config.dry_run = False
        config.backup_before_move = False
        config.reindex_after_move = False
        config.classification_method = ClassificationMethod.RULES_ONLY
        config.metadata_action = MetadataAction.SKIP
        config.llm_config.enabled = False
        
        with patch('app.inbox_processor.config') as mock_config:
            mock_config.VAULT_PATH = temp_vault
            processor = InboxProcessor(config)
            processor.vault_path = temp_vault
        
        result = await processor.process_file(test_file)
        
        assert result.success is True
        assert result.action == "moved"
        assert not test_file.exists()  # Source should be gone
        assert (temp_vault / "Archive" / "test.md").exists()
    
    @pytest.mark.asyncio
    async def test_process_file_applies_rules(self, temp_vault: Path, temp_inbox: Path):
        """Should classify file based on matching rules."""
        test_file = temp_inbox / "team_meeting_notes.md"
        test_file.write_text("# Team Meeting\n\nAgenda:\n- Item 1", encoding="utf-8")
        
        config = InboxConfig()
        config.inbox_folder = "00_Inbox"
        config.default_destination = "Archive"
        config.dry_run = True
        config.classification_method = ClassificationMethod.RULES_ONLY
        config.llm_config.enabled = False
        config.rules = [
            ClassificationRule(
                name="meetings",
                destination_folder="06_Meetings",
                title_patterns=[r"meeting"],
                content_patterns=[r"agenda:"],
                add_tags=["meeting"],
                priority=10,
            )
        ]
        
        with patch('app.inbox_processor.config') as mock_config:
            mock_config.VAULT_PATH = temp_vault
            processor = InboxProcessor(config)
            processor.vault_path = temp_vault
        
        result = await processor.process_file(test_file)
        
        assert result.classification == "06_Meetings"
        assert "meeting" in result.added_tags
    
    @pytest.mark.asyncio
    async def test_process_file_creates_backup(self, temp_vault: Path, temp_inbox: Path):
        """Should create backup when configured."""
        test_file = temp_inbox / "test.md"
        test_file.write_text("# Test\n\nContent", encoding="utf-8")
        
        config = InboxConfig()
        config.inbox_folder = "00_Inbox"
        config.default_destination = "Archive"
        config.dry_run = False
        config.backup_before_move = True
        config.backup_folder = ".backups"
        config.reindex_after_move = False
        config.classification_method = ClassificationMethod.RULES_ONLY
        config.metadata_action = MetadataAction.SKIP
        config.llm_config.enabled = False
        
        with patch('app.inbox_processor.config') as mock_config:
            mock_config.VAULT_PATH = temp_vault
            processor = InboxProcessor(config)
            processor.vault_path = temp_vault
        
        await processor.process_file(test_file)
        
        backup_folder = temp_vault / ".backups"
        assert backup_folder.exists()
        backups = list(backup_folder.glob("test.md.*.bak"))
        assert len(backups) == 1


class TestProcessInbox:
    """Tests for batch inbox processing."""
    
    @pytest.mark.asyncio
    async def test_process_inbox_all_files(self, temp_vault: Path, temp_inbox: Path):
        """Should process all files in inbox."""
        (temp_inbox / "note1.md").write_text("# Note 1", encoding="utf-8")
        (temp_inbox / "note2.md").write_text("# Note 2", encoding="utf-8")
        (temp_inbox / "note3.md").write_text("# Note 3", encoding="utf-8")
        
        config = InboxConfig()
        config.inbox_folder = "00_Inbox"
        config.default_destination = "Archive"
        config.dry_run = True
        config.classification_method = ClassificationMethod.RULES_ONLY
        config.llm_config.enabled = False
        
        with patch('app.inbox_processor.config') as mock_config:
            mock_config.VAULT_PATH = temp_vault
            processor = InboxProcessor(config)
            processor.vault_path = temp_vault
        
        result = await processor.process_inbox()
        
        assert result.processed == 3
        assert result.skipped == 3  # All dry_run
        assert result.errors == 0
        assert result.duration_seconds >= 0
    
    @pytest.mark.asyncio
    async def test_process_inbox_returns_batch_result(self, temp_vault: Path, temp_inbox: Path):
        """Should return proper BatchResult with timing info."""
        (temp_inbox / "note.md").write_text("# Note", encoding="utf-8")
        
        config = InboxConfig()
        config.inbox_folder = "00_Inbox"
        config.default_destination = "Archive"
        config.dry_run = True
        config.classification_method = ClassificationMethod.RULES_ONLY
        config.llm_config.enabled = False
        
        with patch('app.inbox_processor.config') as mock_config:
            mock_config.VAULT_PATH = temp_vault
            processor = InboxProcessor(config)
            processor.vault_path = temp_vault
        
        result = await processor.process_inbox()
        
        assert isinstance(result, BatchResult)
        assert result.start_time is not None
        assert result.end_time is not None
        assert result.end_time >= result.start_time
        assert len(result.results) == 1


class TestClassificationMethods:
    """Tests for different classification methods."""
    
    @pytest.mark.asyncio
    async def test_rules_only_method(self, temp_vault: Path, temp_inbox: Path):
        """Should use only rules when method is RULES_ONLY."""
        test_file = temp_inbox / "meeting.md"
        test_file.write_text("# Weekly Meeting", encoding="utf-8")
        
        config = InboxConfig()
        config.inbox_folder = "00_Inbox"
        config.default_destination = "Archive"
        config.classification_method = ClassificationMethod.RULES_ONLY
        config.dry_run = True
        config.llm_config.enabled = True  # Should be ignored
        config.rules = [
            ClassificationRule(
                name="meetings",
                destination_folder="Meetings",
                title_patterns=[r"meeting"],
            )
        ]
        
        with patch('app.inbox_processor.config') as mock_config:
            mock_config.VAULT_PATH = temp_vault
            processor = InboxProcessor(config)
            processor.vault_path = temp_vault
        
        result = await processor.process_file(test_file)
        
        assert result.classification == "Meetings"
    
    @pytest.mark.asyncio
    async def test_rules_then_llm_uses_rules_first(self, temp_vault: Path, temp_inbox: Path):
        """Should try rules first, then LLM in RULES_THEN_LLM mode."""
        test_file = temp_inbox / "meeting.md"
        test_file.write_text("# Weekly Meeting", encoding="utf-8")
        
        config = InboxConfig()
        config.inbox_folder = "00_Inbox"
        config.default_destination = "Archive"
        config.classification_method = ClassificationMethod.RULES_THEN_LLM
        config.dry_run = True
        config.llm_config.enabled = True
        config.rules = [
            ClassificationRule(
                name="meetings",
                destination_folder="Meetings",
                title_patterns=[r"meeting"],
            )
        ]
        
        with patch('app.inbox_processor.config') as mock_config:
            mock_config.VAULT_PATH = temp_vault
            processor = InboxProcessor(config)
            processor.vault_path = temp_vault
        
        # Mock LLM to verify it's NOT called when rules match
        processor.classify_by_llm = AsyncMock(return_value=("LLM_Dest", []))
        
        result = await processor.process_file(test_file)
        
        assert result.classification == "Meetings"
        processor.classify_by_llm.assert_not_called()


class TestCreateDefaultConfig:
    """Tests for default configuration creation."""
    
    def test_creates_config_with_rules(self):
        """Should create config with predefined rules."""
        config = create_default_config()
        
        assert isinstance(config, InboxConfig)
        assert len(config.rules) > 0
    
    def test_default_rules_have_required_fields(self):
        """All default rules should have required fields."""
        config = create_default_config()
        
        for rule in config.rules:
            assert rule.name
            assert rule.destination_folder
            assert rule.priority >= 0
    
    def test_default_config_has_llm_config(self):
        """Should include LLM configuration."""
        config = create_default_config()
        
        assert config.llm_config is not None
        assert len(config.llm_config.available_categories) > 0


class TestDataclasses:
    """Tests for dataclass behaviors."""
    
    def test_processing_result_defaults(self):
        """ProcessingResult should have sensible defaults."""
        result = ProcessingResult(source_path=Path("test.md"))
        
        assert result.source_path == Path("test.md")
        assert result.destination_path is None
        assert result.success is False
        assert result.action == ""
        assert result.added_tags == []
        assert result.error is None
    
    def test_batch_result_duration(self):
        """BatchResult should calculate duration correctly."""
        result = BatchResult()
        result.start_time = datetime(2024, 1, 1, 10, 0, 0)
        result.end_time = datetime(2024, 1, 1, 10, 0, 30)
        
        assert result.duration_seconds == 30.0
    
    def test_batch_result_duration_no_times(self):
        """BatchResult duration should be 0 without times."""
        result = BatchResult()
        
        assert result.duration_seconds == 0.0
