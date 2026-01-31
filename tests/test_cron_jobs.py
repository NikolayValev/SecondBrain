"""
Unit tests for the cron_jobs module.
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
import yaml
import json

from cron_jobs import (
    load_config_from_file,
    _dict_to_config,
    save_sample_config,
    run_inbox_processing,
    run_full_reindex,
    run_incremental_index,
)
from app.inbox_processor import (
    InboxConfig,
    ClassificationMethod,
    ConflictResolution,
    MetadataAction,
)


class TestLoadConfigFromFile:
    """Tests for configuration file loading."""
    
    def test_load_yaml_config(self, temp_dir: Path):
        """Should load configuration from YAML file."""
        config_data = {
            "inbox_folder": "custom_inbox",
            "default_destination": "custom_dest",
            "dry_run": True,
            "classification_method": "rules_only",
        }
        
        config_path = temp_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        config = load_config_from_file(config_path)
        
        assert config.inbox_folder == "custom_inbox"
        assert config.default_destination == "custom_dest"
        assert config.dry_run is True
        assert config.classification_method == ClassificationMethod.RULES_ONLY
    
    def test_load_json_config(self, temp_dir: Path):
        """Should load configuration from JSON file."""
        config_data = {
            "inbox_folder": "json_inbox",
            "backup_before_move": False,
            "conflict_resolution": "overwrite",
        }
        
        config_path = temp_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config_data, f)
        
        config = load_config_from_file(config_path)
        
        assert config.inbox_folder == "json_inbox"
        assert config.backup_before_move is False
        assert config.conflict_resolution == ConflictResolution.OVERWRITE
    
    def test_load_nonexistent_returns_default(self, temp_dir: Path):
        """Should return default config for non-existent file."""
        config_path = temp_dir / "nonexistent.yaml"
        
        config = load_config_from_file(config_path)
        
        assert isinstance(config, InboxConfig)
        assert len(config.rules) > 0  # Default rules
    
    def test_load_config_with_rules(self, temp_dir: Path):
        """Should load custom classification rules."""
        config_data = {
            "inbox_folder": "00_Inbox",
            "rules": [
                {
                    "name": "custom_rule",
                    "destination_folder": "CustomDest",
                    "title_patterns": ["custom"],
                    "add_tags": ["custom-tag"],
                    "priority": 5,
                }
            ]
        }
        
        config_path = temp_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        config = load_config_from_file(config_path)
        
        assert len(config.rules) == 1
        assert config.rules[0].name == "custom_rule"
        assert config.rules[0].destination_folder == "CustomDest"
        assert config.rules[0].priority == 5
    
    def test_load_config_with_llm_config(self, temp_dir: Path):
        """Should load LLM configuration."""
        config_data = {
            "llm_config": {
                "enabled": False,
                "suggest_tags": True,
                "max_suggested_tags": 3,
                "available_categories": {
                    "notes": "Notes",
                    "archive": "Archive",
                }
            }
        }
        
        config_path = temp_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        config = load_config_from_file(config_path)
        
        assert config.llm_config.enabled is False
        assert config.llm_config.max_suggested_tags == 3
        assert "notes" in config.llm_config.available_categories
    
    def test_load_unsupported_format_raises(self, temp_dir: Path):
        """Should raise error for unsupported file format."""
        config_path = temp_dir / "config.txt"
        config_path.write_text("some content")
        
        with pytest.raises(ValueError, match="Unsupported config format"):
            load_config_from_file(config_path)


class TestDictToConfig:
    """Tests for dictionary to config conversion."""
    
    def test_converts_simple_fields(self):
        """Should convert simple string/bool fields."""
        data = {
            "inbox_folder": "MyInbox",
            "dry_run": True,
            "backup_before_move": False,
        }
        
        config = _dict_to_config(data)
        
        assert config.inbox_folder == "MyInbox"
        assert config.dry_run is True
        assert config.backup_before_move is False
    
    def test_converts_enum_fields(self):
        """Should convert enum string values."""
        data = {
            "classification_method": "llm_only",
            "conflict_resolution": "merge",
            "metadata_action": "overwrite",
        }
        
        config = _dict_to_config(data)
        
        assert config.classification_method == ClassificationMethod.LLM_ONLY
        assert config.conflict_resolution == ConflictResolution.MERGE
        assert config.metadata_action == MetadataAction.OVERWRITE
    
    def test_converts_file_extensions(self):
        """Should convert file extensions to tuple."""
        data = {
            "file_extensions": [".md", ".txt", ".rst"]
        }
        
        config = _dict_to_config(data)
        
        assert config.file_extensions == (".md", ".txt", ".rst")
    
    def test_converts_folder_tags(self):
        """Should convert folder_tags dict."""
        data = {
            "folder_tags": {
                "Projects": ["project", "work"],
                "Personal": ["personal"],
            }
        }
        
        config = _dict_to_config(data)
        
        assert config.folder_tags["Projects"] == ["project", "work"]
        assert config.folder_tags["Personal"] == ["personal"]
    
    def test_preserves_defaults_for_missing_fields(self):
        """Should use defaults for fields not in data."""
        data = {"inbox_folder": "Custom"}
        
        config = _dict_to_config(data)
        
        assert config.inbox_folder == "Custom"
        # These should be defaults
        assert config.backup_before_move is True  # Default
        assert config.reindex_after_move is True  # Default


class TestSaveSampleConfig:
    """Tests for sample configuration generation."""
    
    def test_creates_yaml_file(self, temp_dir: Path):
        """Should create a YAML config file."""
        output_path = temp_dir / "sample.yaml"
        
        save_sample_config(output_path)
        
        assert output_path.exists()
        content = yaml.safe_load(output_path.read_text())
        assert "inbox_folder" in content
        assert "rules" in content
    
    def test_sample_config_is_valid(self, temp_dir: Path):
        """Generated config should be loadable."""
        output_path = temp_dir / "sample.yaml"
        
        save_sample_config(output_path)
        config = load_config_from_file(output_path)
        
        assert isinstance(config, InboxConfig)
    
    def test_creates_parent_directories(self, temp_dir: Path):
        """Should create parent directories if needed."""
        output_path = temp_dir / "nested" / "dirs" / "sample.yaml"
        
        save_sample_config(output_path)
        
        assert output_path.exists()


class TestRunInboxProcessing:
    """Tests for inbox processing runner."""
    
    @pytest.mark.asyncio
    async def test_returns_success_on_completion(self, temp_vault: Path, temp_dir: Path):
        """Should return success result on completion."""
        # Create inbox
        inbox = temp_vault / "00_Inbox"
        inbox.mkdir()
        
        with patch('cron_jobs.Config') as mock_config_class, \
             patch('cron_jobs.db') as mock_db, \
             patch('cron_jobs.create_default_config') as mock_create_config:
            
            mock_config_class.validate = MagicMock()
            mock_config_class.VAULT_PATH = temp_vault
            mock_db.initialize = MagicMock()
            
            # Create a mock config
            config = InboxConfig()
            config.inbox_folder = "00_Inbox"
            config.dry_run = True
            config.classification_method = ClassificationMethod.RULES_ONLY
            config.llm_config.enabled = False
            mock_create_config.return_value = config
            
            with patch('app.inbox_processor.config') as mock_app_config:
                mock_app_config.VAULT_PATH = temp_vault
                
                result = await run_inbox_processing(dry_run=True)
        
        assert result["success"] is True
        assert "processed" in result
        assert "timestamp" in result
    
    @pytest.mark.asyncio
    async def test_returns_error_on_invalid_config(self):
        """Should return error when vault config is invalid."""
        with patch('cron_jobs.Config') as mock_config:
            mock_config.validate.side_effect = ValueError("Invalid vault path")
            
            result = await run_inbox_processing()
        
        assert result["success"] is False
        assert "error" in result
    
    @pytest.mark.asyncio
    async def test_uses_custom_config_file(self, temp_vault: Path, temp_dir: Path):
        """Should load config from file when provided."""
        config_data = {
            "inbox_folder": "CustomInbox",
            "dry_run": True,
            "classification_method": "rules_only",
        }
        
        config_path = temp_dir / "custom.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        # Create the custom inbox
        (temp_vault / "CustomInbox").mkdir()
        
        with patch('cron_jobs.Config') as mock_config_class, \
             patch('cron_jobs.db') as mock_db:
            
            mock_config_class.validate = MagicMock()
            mock_config_class.VAULT_PATH = temp_vault
            mock_db.initialize = MagicMock()
            
            with patch('app.inbox_processor.config') as mock_app_config:
                mock_app_config.VAULT_PATH = temp_vault
                
                result = await run_inbox_processing(config_path=config_path, dry_run=True)
        
        assert result["success"] is True


class TestRunReindex:
    """Tests for reindex runners."""
    
    @pytest.mark.asyncio
    async def test_full_reindex_success(self, temp_vault: Path):
        """Should return success on full reindex."""
        with patch('cron_jobs.Config') as mock_config, \
             patch('cron_jobs.db') as mock_db, \
             patch('app.indexer.indexer') as mock_indexer:
            
            mock_config.validate = MagicMock()
            mock_db.initialize = MagicMock()
            mock_indexer.full_scan.return_value = (10, 0)  # 10 indexed, 0 errors
            
            # Patch the import inside the function
            with patch.dict('sys.modules', {'app.indexer': MagicMock(indexer=mock_indexer)}):
                result = await run_full_reindex()
        
        assert result["success"] is True
        assert result["indexed"] == 10
        assert result["errors"] == 0
    
    @pytest.mark.asyncio
    async def test_full_reindex_with_errors(self):
        """Should report errors in result."""
        with patch('cron_jobs.Config') as mock_config, \
             patch('cron_jobs.db') as mock_db, \
             patch('app.indexer.indexer') as mock_indexer:
            
            mock_config.validate = MagicMock()
            mock_db.initialize = MagicMock()
            mock_indexer.full_scan.return_value = (8, 2)  # 8 indexed, 2 errors
            
            with patch.dict('sys.modules', {'app.indexer': MagicMock(indexer=mock_indexer)}):
                result = await run_full_reindex()
        
        assert result["success"] is False
        assert result["errors"] == 2
    
    @pytest.mark.asyncio
    async def test_incremental_index_success(self):
        """Should return success on incremental index."""
        with patch('cron_jobs.Config') as mock_config, \
             patch('cron_jobs.db') as mock_db, \
             patch('app.indexer.indexer') as mock_indexer:
            
            mock_config.validate = MagicMock()
            mock_db.initialize = MagicMock()
            mock_indexer.incremental_scan.return_value = (3, 0)
            
            with patch.dict('sys.modules', {'app.indexer': MagicMock(indexer=mock_indexer)}):
                result = await run_incremental_index()
        
        assert result["success"] is True
        assert result["indexed"] == 3
    
    @pytest.mark.asyncio
    async def test_reindex_handles_config_error(self):
        """Should handle configuration errors gracefully."""
        with patch('cron_jobs.Config') as mock_config:
            mock_config.validate.side_effect = ValueError("No vault")
            
            result = await run_full_reindex()
        
        assert result["success"] is False
        assert "error" in result
