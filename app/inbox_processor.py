"""
Inbox Processor for Second Brain.
Processes, classifies, tags, and moves documents from 00_Inbox folder.
All behavior is configurable via InboxConfig and ClassificationRules.
"""

import logging
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable
from enum import Enum

from app.config import config, Config
from app.parser import parser, ParsedMarkdown
from app.indexer import indexer
from app.llm import get_llm_provider

logger = logging.getLogger(__name__)


class ClassificationMethod(Enum):
    """How to classify documents."""
    RULES_ONLY = "rules_only"  # Use only keyword/pattern rules
    LLM_ONLY = "llm_only"  # Use only LLM classification
    RULES_THEN_LLM = "rules_then_llm"  # Try rules first, fall back to LLM
    LLM_WITH_RULES_VALIDATION = "llm_with_rules_validation"  # LLM classifies, rules validate


class ConflictResolution(Enum):
    """How to handle file name conflicts in destination."""
    SKIP = "skip"  # Don't move, leave in inbox
    OVERWRITE = "overwrite"  # Replace existing file
    RENAME = "rename"  # Add timestamp suffix
    MERGE = "merge"  # Append content to existing (for notes)


class MetadataAction(Enum):
    """What to do with metadata/frontmatter."""
    PRESERVE = "preserve"  # Keep existing, add missing
    OVERWRITE = "overwrite"  # Replace all metadata
    MERGE = "merge"  # Deep merge, new values win
    SKIP = "skip"  # Don't modify metadata


@dataclass
class ClassificationRule:
    """A rule for classifying documents based on patterns."""
    name: str
    destination_folder: str  # Relative to vault root
    
    # Matching criteria (any match triggers the rule)
    title_patterns: list[str] = field(default_factory=list)  # Regex patterns
    content_patterns: list[str] = field(default_factory=list)  # Regex patterns
    required_tags: list[str] = field(default_factory=list)  # Must have ALL these tags
    any_tags: list[str] = field(default_factory=list)  # Must have ANY of these tags
    frontmatter_matches: dict[str, str] = field(default_factory=dict)  # Key-value patterns
    filename_patterns: list[str] = field(default_factory=list)  # Regex on filename
    
    # Metadata to apply when matched
    add_tags: list[str] = field(default_factory=list)
    add_frontmatter: dict = field(default_factory=dict)
    
    # Rule priority (higher = checked first)
    priority: int = 0
    
    # Whether to stop processing after this rule matches
    terminal: bool = True


@dataclass
class LLMClassificationConfig:
    """Configuration for LLM-based classification."""
    enabled: bool = True
    
    # Categories/folders the LLM can classify into
    available_categories: dict[str, str] = field(default_factory=lambda: {
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
    })
    
    # System prompt for classification
    system_prompt: str = """You are a document classifier for a personal knowledge management system.
Analyze the document and determine the most appropriate category.

Categories available:
{categories}

Respond with ONLY the category key (e.g., "projects", "reference", etc.).
If unsure, respond with "uncategorized".

Consider:
- The document's main topic and purpose
- Any tags or metadata present
- The type of content (project notes, reference material, journal entry, etc.)
"""

    # Tags to suggest based on content
    suggest_tags: bool = True
    max_suggested_tags: int = 5
    
    tag_prompt: str = """Analyze this document and suggest up to {max_tags} relevant tags.
Tags should be:
- Lowercase, using hyphens for multi-word tags
- Specific and meaningful
- Consistent with common tagging conventions

Existing tags in the system: {existing_tags}

Respond with a JSON array of tag strings, e.g.: ["tag1", "tag2"]
If no tags are appropriate, respond with: []
"""


@dataclass  
class InboxConfig:
    """Master configuration for inbox processing."""
    
    # Source folder (relative to vault)
    inbox_folder: str = "00_Inbox"
    
    # Default destination for unclassified documents
    default_destination: str = "00_Inbox/Unsorted"
    
    # Classification settings
    classification_method: ClassificationMethod = ClassificationMethod.RULES_THEN_LLM
    
    # File handling
    conflict_resolution: ConflictResolution = ConflictResolution.RENAME
    metadata_action: MetadataAction = MetadataAction.MERGE
    
    # Processing options
    dry_run: bool = False  # If True, don't actually move files
    backup_before_move: bool = True
    backup_folder: str = ".inbox_backup"
    
    # What to process
    file_extensions: tuple[str, ...] = (".md", ".markdown")
    ignore_patterns: list[str] = field(default_factory=lambda: [
        r"^\.",  # Hidden files
        r"^_",   # Underscore prefix
        r"template",  # Template files
    ])
    
    # Metadata to add to all processed files
    global_frontmatter: dict = field(default_factory=lambda: {
        "processed_date": None,  # Will be set to current date
        "source": "inbox",
    })
    
    # Auto-tagging based on folder
    folder_tags: dict[str, list[str]] = field(default_factory=lambda: {
        "03_Projects": ["project"],
        "05_Journal": ["journal", "daily"],
        "06_Meetings": ["meeting"],
        "14_Fitness": ["fitness", "health"],
    })
    
    # Index after moving
    reindex_after_move: bool = True
    
    # Classification rules (processed in priority order)
    rules: list[ClassificationRule] = field(default_factory=list)
    
    # LLM classification config
    llm_config: LLMClassificationConfig = field(default_factory=LLMClassificationConfig)


@dataclass
class ProcessingResult:
    """Result of processing a single file."""
    source_path: Path
    destination_path: Optional[Path] = None
    success: bool = False
    action: str = ""  # "moved", "skipped", "error"
    classification: str = ""
    added_tags: list[str] = field(default_factory=list)
    added_frontmatter: dict = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class BatchResult:
    """Result of processing the entire inbox."""
    processed: int = 0
    moved: int = 0
    skipped: int = 0
    errors: int = 0
    results: list[ProcessingResult] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    @property
    def duration_seconds(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0


class InboxProcessor:
    """Processes documents in the inbox folder."""
    
    def __init__(self, inbox_config: Optional[InboxConfig] = None):
        self.config = inbox_config or InboxConfig()
        self.vault_path = config.VAULT_PATH
        self.llm = None
        self._compiled_ignore_patterns: list[re.Pattern] = []
        self._compile_patterns()
    
    def _compile_patterns(self) -> None:
        """Pre-compile regex patterns for performance."""
        self._compiled_ignore_patterns = [
            re.compile(p, re.IGNORECASE) 
            for p in self.config.ignore_patterns
        ]
    
    def _should_ignore(self, file_path: Path) -> bool:
        """Check if file should be ignored."""
        filename = file_path.name
        for pattern in self._compiled_ignore_patterns:
            if pattern.search(filename):
                return True
        return False
    
    async def _get_llm(self):
        """Lazy-load LLM provider."""
        if self.llm is None:
            self.llm = get_llm_provider()
        return self.llm
    
    def get_inbox_files(self) -> list[Path]:
        """Get all processable files from inbox."""
        inbox_path = self.vault_path / self.config.inbox_folder
        
        if not inbox_path.exists():
            logger.warning(f"Inbox folder does not exist: {inbox_path}")
            return []
        
        files = []
        for file_path in inbox_path.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in self.config.file_extensions:
                continue
            if self._should_ignore(file_path):
                continue
            # Don't process files in subfolders like Unsorted or backup
            rel_to_inbox = file_path.relative_to(inbox_path)
            if len(rel_to_inbox.parts) > 1:
                # Skip files in subdirectories
                continue
            files.append(file_path)
        
        return sorted(files)
    
    def _match_rule(self, rule: ClassificationRule, parsed: ParsedMarkdown, file_path: Path) -> bool:
        """Check if a document matches a classification rule."""
        
        # Check filename patterns
        for pattern in rule.filename_patterns:
            if re.search(pattern, file_path.name, re.IGNORECASE):
                return True
        
        # Check title patterns
        for pattern in rule.title_patterns:
            if re.search(pattern, parsed.title, re.IGNORECASE):
                return True
        
        # Check content patterns
        for pattern in rule.content_patterns:
            if re.search(pattern, parsed.content, re.IGNORECASE):
                return True
        
        # Check required tags (must have ALL)
        if rule.required_tags:
            doc_tags = set(t.lower() for t in parsed.tags)
            required = set(t.lower() for t in rule.required_tags)
            if not required.issubset(doc_tags):
                return False
            return True
        
        # Check any tags (must have at least ONE)
        if rule.any_tags:
            doc_tags = set(t.lower() for t in parsed.tags)
            any_required = set(t.lower() for t in rule.any_tags)
            if doc_tags.intersection(any_required):
                return True
        
        # Check frontmatter matches
        for key, pattern in rule.frontmatter_matches.items():
            value = parsed.frontmatter.get(key)
            if value and re.search(pattern, str(value), re.IGNORECASE):
                return True
        
        return False
    
    def classify_by_rules(self, parsed: ParsedMarkdown, file_path: Path) -> Optional[ClassificationRule]:
        """Classify document using rule-based matching."""
        # Sort rules by priority (higher first)
        sorted_rules = sorted(self.config.rules, key=lambda r: -r.priority)
        
        for rule in sorted_rules:
            if self._match_rule(rule, parsed, file_path):
                logger.debug(f"Rule '{rule.name}' matched for {file_path.name}")
                return rule
        
        return None
    
    async def classify_by_llm(self, parsed: ParsedMarkdown, file_path: Path) -> tuple[str, list[str]]:
        """Classify document using LLM."""
        llm_config = self.config.llm_config
        
        if not llm_config.enabled:
            return self.config.default_destination, []
        
        try:
            llm = await self._get_llm()
            
            # Build category description
            categories_desc = "\n".join(
                f"- {key}: {folder}" 
                for key, folder in llm_config.available_categories.items()
            )
            
            system_prompt = llm_config.system_prompt.format(categories=categories_desc)
            
            # Prepare document summary for classification
            doc_summary = f"""
Title: {parsed.title}
Tags: {', '.join(parsed.tags) if parsed.tags else 'None'}
Frontmatter: {parsed.frontmatter}

Content (first 2000 chars):
{parsed.content[:2000]}
"""
            
            # Get classification
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Classify this document:\n\n{doc_summary}"}
            ]
            
            category_response = await llm.chat(messages, temperature=0.3)
            
            if not category_response or not category_response.strip():
                logger.warning(f"LLM returned empty response for '{file_path.name}', using default destination")
                return self.config.default_destination, []
            
            category_key = category_response.strip().lower().replace('"', '').replace("'", "")
            
            # Map to destination folder
            if category_key not in llm_config.available_categories:
                logger.warning(f"LLM returned unknown category '{category_key}' for '{file_path.name}', using default. Available: {list(llm_config.available_categories.keys())}")
            
            destination = llm_config.available_categories.get(
                category_key, 
                self.config.default_destination
            )
            
            # Get tag suggestions if enabled
            suggested_tags = []
            if llm_config.suggest_tags:
                # Get existing tags from database for consistency
                from app.db import db
                existing_tags = db.get_all_tags()
                
                tag_prompt = llm_config.tag_prompt.format(
                    max_tags=llm_config.max_suggested_tags,
                    existing_tags=", ".join(existing_tags[:50])  # Limit for context
                )
                
                tag_messages = [
                    {"role": "system", "content": tag_prompt},
                    {"role": "user", "content": f"Document:\n\n{doc_summary}"}
                ]
                
                tag_response = await llm.chat(tag_messages, temperature=0.3)
                
                # Parse JSON response
                import json
                try:
                    # Clean up response
                    tag_text = tag_response.strip()
                    if tag_text.startswith("```"):
                        tag_text = tag_text.split("```")[1]
                        if tag_text.startswith("json"):
                            tag_text = tag_text[4:]
                    suggested_tags = json.loads(tag_text)
                    if not isinstance(suggested_tags, list):
                        suggested_tags = []
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse tag suggestions: {tag_response}")
                    suggested_tags = []
            
            logger.info(f"LLM classified '{file_path.name}' as '{category_key}' -> {destination}")
            return destination, suggested_tags
            
        except Exception as e:
            logger.error(f"LLM classification failed for '{file_path.name}': {type(e).__name__}: {e}")
            return self.config.default_destination, []
    
    async def classify_document(
        self, 
        parsed: ParsedMarkdown, 
        file_path: Path
    ) -> tuple[str, list[str], dict]:
        """
        Classify a document and determine destination + metadata.
        Returns (destination_folder, tags_to_add, frontmatter_to_add).
        """
        destination = self.config.default_destination
        tags_to_add = []
        frontmatter_to_add = {}
        
        method = self.config.classification_method
        
        if method == ClassificationMethod.RULES_ONLY:
            rule = self.classify_by_rules(parsed, file_path)
            if rule:
                destination = rule.destination_folder
                tags_to_add = rule.add_tags.copy()
                frontmatter_to_add = rule.add_frontmatter.copy()
        
        elif method == ClassificationMethod.LLM_ONLY:
            destination, llm_tags = await self.classify_by_llm(parsed, file_path)
            tags_to_add = llm_tags
        
        elif method == ClassificationMethod.RULES_THEN_LLM:
            rule = self.classify_by_rules(parsed, file_path)
            if rule:
                destination = rule.destination_folder
                tags_to_add = rule.add_tags.copy()
                frontmatter_to_add = rule.add_frontmatter.copy()
            else:
                destination, llm_tags = await self.classify_by_llm(parsed, file_path)
                tags_to_add = llm_tags
        
        elif method == ClassificationMethod.LLM_WITH_RULES_VALIDATION:
            destination, llm_tags = await self.classify_by_llm(parsed, file_path)
            tags_to_add = llm_tags
            # Check if any rule would match and use its additional metadata
            rule = self.classify_by_rules(parsed, file_path)
            if rule:
                tags_to_add.extend(rule.add_tags)
                frontmatter_to_add.update(rule.add_frontmatter)
        
        # Add folder-based tags
        for folder, folder_tags in self.config.folder_tags.items():
            if destination.startswith(folder):
                tags_to_add.extend(folder_tags)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_tags = []
        for tag in tags_to_add:
            tag_lower = tag.lower()
            if tag_lower not in seen:
                seen.add(tag_lower)
                unique_tags.append(tag)
        
        return destination, unique_tags, frontmatter_to_add
    
    def _update_frontmatter(
        self, 
        content: str, 
        existing_frontmatter: dict,
        new_tags: list[str],
        new_frontmatter: dict
    ) -> str:
        """Update document frontmatter with new metadata."""
        import yaml
        
        action = self.config.metadata_action
        
        if action == MetadataAction.SKIP:
            return content
        
        # Build final frontmatter
        if action == MetadataAction.OVERWRITE:
            final_fm = {}
        else:
            final_fm = existing_frontmatter.copy()
        
        # Add global frontmatter
        for key, value in self.config.global_frontmatter.items():
            if value is None:
                if key == "processed_date":
                    value = datetime.now().strftime("%Y-%m-%d")
            if action == MetadataAction.OVERWRITE or key not in final_fm:
                final_fm[key] = value
        
        # Add new frontmatter
        for key, value in new_frontmatter.items():
            if action == MetadataAction.OVERWRITE or key not in final_fm:
                final_fm[key] = value
            elif action == MetadataAction.MERGE and key in final_fm:
                # For dicts, deep merge; for lists, extend; otherwise replace
                if isinstance(final_fm[key], dict) and isinstance(value, dict):
                    final_fm[key].update(value)
                elif isinstance(final_fm[key], list) and isinstance(value, list):
                    final_fm[key].extend(value)
                else:
                    final_fm[key] = value
        
        # Handle tags
        existing_tags = final_fm.get("tags", [])
        if isinstance(existing_tags, str):
            existing_tags = [t.strip() for t in existing_tags.split(",")]
        
        all_tags = list(existing_tags)
        for tag in new_tags:
            if tag.lower() not in [t.lower() for t in all_tags]:
                all_tags.append(tag)
        
        if all_tags:
            final_fm["tags"] = all_tags
        
        # Remove old frontmatter from content
        frontmatter_pattern = re.compile(r'^---\s*\n.*?\n---\s*\n', re.DOTALL)
        body = frontmatter_pattern.sub('', content)
        
        # Build new content with frontmatter
        if final_fm:
            fm_str = yaml.dump(final_fm, default_flow_style=False, allow_unicode=True)
            new_content = f"---\n{fm_str}---\n\n{body.lstrip()}"
        else:
            new_content = body.lstrip()
        
        return new_content
    
    def _resolve_destination_path(self, file_path: Path, destination_folder: str) -> Path:
        """Resolve final destination path, handling conflicts."""
        dest_folder = self.vault_path / destination_folder
        dest_folder.mkdir(parents=True, exist_ok=True)
        
        dest_path = dest_folder / file_path.name
        
        if not dest_path.exists():
            return dest_path
        
        resolution = self.config.conflict_resolution
        
        if resolution == ConflictResolution.OVERWRITE:
            return dest_path
        
        elif resolution == ConflictResolution.SKIP:
            return None
        
        elif resolution == ConflictResolution.RENAME:
            # Add timestamp to filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stem = file_path.stem
            suffix = file_path.suffix
            new_name = f"{stem}_{timestamp}{suffix}"
            return dest_folder / new_name
        
        elif resolution == ConflictResolution.MERGE:
            # Return existing path - caller will handle merge
            return dest_path
        
        return dest_path
    
    async def process_file(self, file_path: Path) -> ProcessingResult:
        """Process a single inbox file."""
        result = ProcessingResult(source_path=file_path)
        
        try:
            # Parse the document
            parsed = parser.parse_file(file_path)
            
            # Classify and get metadata
            destination, tags, frontmatter = await self.classify_document(parsed, file_path)
            result.classification = destination
            result.added_tags = tags
            result.added_frontmatter = frontmatter
            
            # Resolve destination path
            dest_path = self._resolve_destination_path(file_path, destination)
            
            if dest_path is None:
                result.action = "skipped"
                result.success = True
                logger.info(f"Skipped (conflict): {file_path.name}")
                return result
            
            result.destination_path = dest_path
            
            # Update content with new metadata
            updated_content = self._update_frontmatter(
                parsed.content,
                parsed.frontmatter,
                tags,
                frontmatter
            )
            
            if self.config.dry_run:
                result.action = "dry_run"
                result.success = True
                logger.info(f"[DRY RUN] Would move: {file_path.name} -> {dest_path}")
                return result
            
            # Backup if configured
            if self.config.backup_before_move:
                backup_folder = self.vault_path / self.config.backup_folder
                backup_folder.mkdir(parents=True, exist_ok=True)
                backup_path = backup_folder / f"{file_path.name}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak"
                shutil.copy2(file_path, backup_path)
            
            # Handle merge case
            if self.config.conflict_resolution == ConflictResolution.MERGE and dest_path.exists():
                existing_content = dest_path.read_text(encoding='utf-8')
                merged_content = existing_content + "\n\n---\n\n" + updated_content
                dest_path.write_text(merged_content, encoding='utf-8')
                file_path.unlink()  # Delete source
            else:
                # Write updated content to destination
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                dest_path.write_text(updated_content, encoding='utf-8')
                
                # Remove source file
                file_path.unlink()
            
            result.action = "moved"
            result.success = True
            
            # Re-index if configured
            if self.config.reindex_after_move:
                indexer.index_file(dest_path)
                # Also remove old path from index
                indexer.delete_file(file_path)
            
            logger.info(f"Processed: {file_path.name} -> {dest_path.relative_to(self.vault_path)}")
            
        except Exception as e:
            result.action = "error"
            result.error = str(e)
            result.success = False
            logger.error(f"Error processing {file_path.name}: {e}")
        
        return result
    
    async def process_inbox(self) -> BatchResult:
        """Process all files in the inbox."""
        batch_result = BatchResult()
        batch_result.start_time = datetime.now()
        
        files = self.get_inbox_files()
        logger.info(f"Found {len(files)} files to process in inbox")
        
        for file_path in files:
            result = await self.process_file(file_path)
            batch_result.results.append(result)
            batch_result.processed += 1
            
            if result.action == "moved":
                batch_result.moved += 1
            elif result.action in ("skipped", "dry_run"):
                batch_result.skipped += 1
            elif result.action == "error":
                batch_result.errors += 1
        
        batch_result.end_time = datetime.now()
        
        logger.info(
            f"Inbox processing complete: {batch_result.moved} moved, "
            f"{batch_result.skipped} skipped, {batch_result.errors} errors "
            f"({batch_result.duration_seconds:.2f}s)"
        )
        
        return batch_result


def create_default_config() -> InboxConfig:
    """Create a sensible default configuration."""
    config = InboxConfig()
    
    # Add some default classification rules
    config.rules = [
        ClassificationRule(
            name="meeting_notes",
            destination_folder="06_Meetings",
            title_patterns=[r"meeting", r"standup", r"sync", r"1[:\-]1", r"one.on.one"],
            content_patterns=[r"attendees?:", r"agenda:", r"action items?:"],
            add_tags=["meeting"],
            priority=10,
        ),
        ClassificationRule(
            name="daily_journal",
            destination_folder="05_Journal",
            filename_patterns=[r"^\d{4}-\d{2}-\d{2}", r"daily", r"journal"],
            title_patterns=[r"daily", r"journal", r"today"],
            add_tags=["journal", "daily"],
            priority=10,
        ),
        ClassificationRule(
            name="project_notes",
            destination_folder="03_Projects",
            any_tags=["project", "epic", "initiative"],
            title_patterns=[r"project:", r"roadmap", r"milestone"],
            add_tags=["project"],
            priority=8,
        ),
        ClassificationRule(
            name="fitness_logs",
            destination_folder="14_Fitness",
            any_tags=["fitness", "workout", "exercise", "gym"],
            title_patterns=[r"workout", r"exercise", r"training", r"gym"],
            content_patterns=[r"sets?:", r"reps?:", r"weight:", r"cardio"],
            add_tags=["fitness"],
            priority=8,
        ),
        ClassificationRule(
            name="ai_agent_specs",
            destination_folder="08_AI_Agent",
            any_tags=["ai", "agent", "llm", "ml"],
            title_patterns=[r"agent", r"ai spec", r"model"],
            add_tags=["ai"],
            priority=7,
        ),
        ClassificationRule(
            name="ideas_and_brainstorms",
            destination_folder="07_Ideas",
            any_tags=["idea", "brainstorm", "concept"],
            title_patterns=[r"idea:", r"brainstorm", r"what if"],
            add_tags=["idea"],
            priority=5,
        ),
        ClassificationRule(
            name="reference_material",
            destination_folder="04_Reference",
            any_tags=["reference", "howto", "guide", "documentation"],
            title_patterns=[r"how to", r"guide:", r"reference:", r"cheatsheet"],
            add_tags=["reference"],
            priority=5,
        ),
    ]
    
    return config


# Singleton instance with default config
inbox_processor = InboxProcessor(create_default_config())
