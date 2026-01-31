"""
Cron Job Runner for Second Brain.
Runs scheduled tasks including inbox processing.
Can be triggered by system cron/Task Scheduler or run as a daemon.
"""

import asyncio
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import config, Config
from app.db import db
from app.inbox_processor import (
    InboxProcessor, 
    InboxConfig, 
    ClassificationRule,
    ClassificationMethod,
    ConflictResolution,
    MetadataAction,
    LLMClassificationConfig,
    create_default_config,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            Path(__file__).parent.parent / "cron.log",
            encoding="utf-8"
        ),
    ]
)

logger = logging.getLogger(__name__)


def load_config_from_file(config_path: Path) -> InboxConfig:
    """Load InboxConfig from a YAML/JSON configuration file."""
    import yaml
    import json
    
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return create_default_config()
    
    content = config_path.read_text(encoding='utf-8')
    
    if config_path.suffix in ('.yaml', '.yml'):
        data = yaml.safe_load(content)
    elif config_path.suffix == '.json':
        data = json.loads(content)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    return _dict_to_config(data)


def _dict_to_config(data: dict) -> InboxConfig:
    """Convert a dictionary to InboxConfig."""
    cfg = InboxConfig()
    
    # Simple fields
    simple_fields = [
        'inbox_folder', 'default_destination', 'dry_run', 
        'backup_before_move', 'backup_folder', 'reindex_after_move'
    ]
    for field in simple_fields:
        if field in data:
            setattr(cfg, field, data[field])
    
    # Enum fields
    if 'classification_method' in data:
        cfg.classification_method = ClassificationMethod(data['classification_method'])
    if 'conflict_resolution' in data:
        cfg.conflict_resolution = ConflictResolution(data['conflict_resolution'])
    if 'metadata_action' in data:
        cfg.metadata_action = MetadataAction(data['metadata_action'])
    
    # Tuple/list fields
    if 'file_extensions' in data:
        cfg.file_extensions = tuple(data['file_extensions'])
    if 'ignore_patterns' in data:
        cfg.ignore_patterns = data['ignore_patterns']
    
    # Dict fields
    if 'global_frontmatter' in data:
        cfg.global_frontmatter = data['global_frontmatter']
    if 'folder_tags' in data:
        cfg.folder_tags = data['folder_tags']
    
    # Classification rules
    if 'rules' in data:
        cfg.rules = []
        for rule_data in data['rules']:
            rule = ClassificationRule(
                name=rule_data.get('name', 'unnamed'),
                destination_folder=rule_data.get('destination_folder', cfg.default_destination),
                title_patterns=rule_data.get('title_patterns', []),
                content_patterns=rule_data.get('content_patterns', []),
                required_tags=rule_data.get('required_tags', []),
                any_tags=rule_data.get('any_tags', []),
                frontmatter_matches=rule_data.get('frontmatter_matches', {}),
                filename_patterns=rule_data.get('filename_patterns', []),
                add_tags=rule_data.get('add_tags', []),
                add_frontmatter=rule_data.get('add_frontmatter', {}),
                priority=rule_data.get('priority', 0),
                terminal=rule_data.get('terminal', True),
            )
            cfg.rules.append(rule)
    
    # LLM config
    if 'llm_config' in data:
        llm_data = data['llm_config']
        cfg.llm_config = LLMClassificationConfig(
            enabled=llm_data.get('enabled', True),
            available_categories=llm_data.get('available_categories', 
                                              cfg.llm_config.available_categories),
            system_prompt=llm_data.get('system_prompt', cfg.llm_config.system_prompt),
            suggest_tags=llm_data.get('suggest_tags', True),
            max_suggested_tags=llm_data.get('max_suggested_tags', 5),
            tag_prompt=llm_data.get('tag_prompt', cfg.llm_config.tag_prompt),
        )
    
    return cfg


def save_sample_config(output_path: Path) -> None:
    """Save a sample configuration file."""
    import yaml
    
    sample = {
        "inbox_folder": "00_Inbox",
        "default_destination": "00_Inbox/Unsorted",
        "classification_method": "rules_then_llm",
        "conflict_resolution": "rename",
        "metadata_action": "merge",
        "dry_run": False,
        "backup_before_move": True,
        "backup_folder": ".inbox_backup",
        "file_extensions": [".md", ".markdown"],
        "ignore_patterns": [
            r"^\.",
            r"^_",
            r"template",
        ],
        "global_frontmatter": {
            "processed_date": None,
            "source": "inbox",
        },
        "folder_tags": {
            "03_Projects": ["project"],
            "05_Journal": ["journal", "daily"],
            "06_Meetings": ["meeting"],
            "14_Fitness": ["fitness", "health"],
        },
        "reindex_after_move": True,
        "rules": [
            {
                "name": "meeting_notes",
                "destination_folder": "06_Meetings",
                "title_patterns": [r"meeting", r"standup", r"sync"],
                "content_patterns": [r"attendees?:", r"agenda:"],
                "add_tags": ["meeting"],
                "priority": 10,
            },
            {
                "name": "daily_journal",
                "destination_folder": "05_Journal",
                "filename_patterns": [r"^\d{4}-\d{2}-\d{2}"],
                "add_tags": ["journal", "daily"],
                "priority": 10,
            },
            {
                "name": "project_notes",
                "destination_folder": "03_Projects",
                "any_tags": ["project", "epic"],
                "add_tags": ["project"],
                "priority": 8,
            },
            {
                "name": "fitness_logs",
                "destination_folder": "14_Fitness",
                "any_tags": ["fitness", "workout", "exercise"],
                "title_patterns": [r"workout", r"exercise", r"training"],
                "add_tags": ["fitness"],
                "priority": 8,
            },
        ],
        "llm_config": {
            "enabled": True,
            "suggest_tags": True,
            "max_suggested_tags": 5,
            "available_categories": {
                "projects": "03_Projects",
                "reference": "04_Reference",
                "journal": "05_Journal",
                "meetings": "06_Meetings",
                "ideas": "07_Ideas",
                "ai_agent": "08_AI_Agent",
                "learning": "09_Learning",
                "resources": "10_Resources",
                "archive": "11_Archive",
                "reviews": "12_Reviews",
                "health": "13_Health",
                "fitness": "14_Fitness",
                "philosophy": "15_Philosophy",
                "professional": "16_Professional",
            },
        },
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(sample, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    logger.info(f"Sample config saved to: {output_path}")


async def run_inbox_processing(
    config_path: Optional[Path] = None,
    dry_run: bool = False,
) -> dict:
    """Run inbox processing task."""
    logger.info("=" * 60)
    logger.info(f"Starting inbox processing at {datetime.now()}")
    logger.info("=" * 60)
    
    # Validate vault config
    try:
        Config.validate()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return {"success": False, "error": str(e)}
    
    # Initialize database
    db.initialize()
    
    # Load inbox config
    if config_path:
        inbox_config = load_config_from_file(config_path)
    else:
        inbox_config = create_default_config()
    
    # Override dry_run if specified
    if dry_run:
        inbox_config.dry_run = True
    
    # Create processor and run
    processor = InboxProcessor(inbox_config)
    result = await processor.process_inbox()
    
    # Log summary
    summary = {
        "success": True,
        "processed": result.processed,
        "moved": result.moved,
        "skipped": result.skipped,
        "errors": result.errors,
        "duration_seconds": result.duration_seconds,
        "timestamp": datetime.now().isoformat(),
    }
    
    logger.info("=" * 60)
    logger.info(f"Inbox processing complete: {summary}")
    logger.info("=" * 60)
    
    return summary


async def run_full_reindex() -> dict:
    """Run a full vault re-index."""
    from app.indexer import indexer
    
    logger.info("=" * 60)
    logger.info(f"Starting full re-index at {datetime.now()}")
    logger.info("=" * 60)
    
    try:
        Config.validate()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return {"success": False, "error": str(e)}
    
    db.initialize()
    indexed, errors = indexer.full_scan()
    
    summary = {
        "success": errors == 0,
        "indexed": indexed,
        "errors": errors,
        "timestamp": datetime.now().isoformat(),
    }
    
    logger.info(f"Full re-index complete: {summary}")
    return summary


async def run_incremental_index() -> dict:
    """Run an incremental index update."""
    from app.indexer import indexer
    
    logger.info("Starting incremental index update")
    
    try:
        Config.validate()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return {"success": False, "error": str(e)}
    
    db.initialize()
    indexed, errors = indexer.incremental_scan()
    
    return {
        "success": errors == 0,
        "indexed": indexed,
        "errors": errors,
        "timestamp": datetime.now().isoformat(),
    }


def main():
    """Main entry point for cron jobs."""
    parser = argparse.ArgumentParser(
        description="Second Brain Cron Job Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cron_jobs.py inbox                    # Process inbox with default config
  python cron_jobs.py inbox --config inbox.yaml  # Use custom config
  python cron_jobs.py inbox --dry-run          # Preview without moving
  python cron_jobs.py reindex                  # Full vault re-index
  python cron_jobs.py reindex --incremental    # Incremental index update
  python cron_jobs.py generate-config          # Generate sample config file

Schedule with cron (Linux/Mac):
  0 9 * * * cd /path/to/project && python cron_jobs.py inbox

Schedule with Task Scheduler (Windows):
  schtasks /create /tn "SecondBrain Inbox" /tr "python cron_jobs.py inbox" /sc daily /st 09:00
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Inbox processing command
    inbox_parser = subparsers.add_parser("inbox", help="Process inbox documents")
    inbox_parser.add_argument(
        "--config", "-c",
        type=Path,
        help="Path to inbox configuration file (YAML or JSON)"
    )
    inbox_parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Preview changes without moving files"
    )
    
    # Reindex command
    reindex_parser = subparsers.add_parser("reindex", help="Re-index vault")
    reindex_parser.add_argument(
        "--incremental", "-i",
        action="store_true",
        help="Only index changed files"
    )
    
    # Generate config command
    genconfig_parser = subparsers.add_parser(
        "generate-config", 
        help="Generate sample configuration file"
    )
    genconfig_parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("inbox_config.yaml"),
        help="Output path for config file"
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    if args.command == "inbox":
        result = asyncio.run(run_inbox_processing(
            config_path=args.config,
            dry_run=args.dry_run,
        ))
        sys.exit(0 if result.get("success") else 1)
    
    elif args.command == "reindex":
        if args.incremental:
            result = asyncio.run(run_incremental_index())
        else:
            result = asyncio.run(run_full_reindex())
        sys.exit(0 if result.get("success") else 1)
    
    elif args.command == "generate-config":
        save_sample_config(args.output)
        sys.exit(0)


if __name__ == "__main__":
    main()
