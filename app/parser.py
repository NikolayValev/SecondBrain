"""
Markdown parser for Second Brain daemon.
Extracts title, headings, tags, links, and full text from markdown files.
"""

import re
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class Section:
    """Represents a section of a markdown document."""
    heading: str
    level: int
    content: str


@dataclass
class ParsedMarkdown:
    """Result of parsing a markdown file."""
    title: str
    content: str  # Full raw content
    sections: list[Section] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)  # Target paths
    frontmatter: dict = field(default_factory=dict)


class MarkdownParser:
    """Parser for Obsidian-style markdown files."""
    
    # Regex patterns
    FRONTMATTER_PATTERN = re.compile(r'^---\s*\n(.*?)\n---\s*\n', re.DOTALL)
    HEADING_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    WIKILINK_PATTERN = re.compile(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]')
    MDLINK_PATTERN = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
    INLINE_TAG_PATTERN = re.compile(r'(?<!\S)#([a-zA-Z][a-zA-Z0-9_/-]*)')
    
    def __init__(self, vault_path: Optional[Path] = None):
        self.vault_path = vault_path
    
    def parse_file(self, file_path: Path) -> ParsedMarkdown:
        """Parse a markdown file and extract all components."""
        try:
            content = file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            # Try with different encoding
            content = file_path.read_text(encoding='latin-1')
        
        return self.parse_content(content, file_path)
    
    def parse_content(self, content: str, file_path: Optional[Path] = None) -> ParsedMarkdown:
        """Parse markdown content string."""
        result = ParsedMarkdown(
            title="",
            content=content
        )
        
        # Extract frontmatter
        body = content
        frontmatter_match = self.FRONTMATTER_PATTERN.match(content)
        if frontmatter_match:
            try:
                result.frontmatter = yaml.safe_load(frontmatter_match.group(1)) or {}
                body = content[frontmatter_match.end():]
            except yaml.YAMLError as e:
                logger.warning(f"Failed to parse frontmatter: {e}")
        
        # Extract title
        result.title = self._extract_title(body, result.frontmatter, file_path)
        
        # Extract tags (from frontmatter and inline)
        result.tags = self._extract_tags(body, result.frontmatter)
        
        # Extract links
        result.links = self._extract_links(body)
        
        # Extract sections
        result.sections = self._extract_sections(body, result.title)
        
        return result
    
    def _extract_title(
        self, 
        body: str, 
        frontmatter: dict, 
        file_path: Optional[Path] = None
    ) -> str:
        """Extract title from frontmatter, first H1, or filename."""
        # Check frontmatter for title
        if frontmatter.get('title'):
            return str(frontmatter['title'])
        
        # Look for first H1
        for match in self.HEADING_PATTERN.finditer(body):
            if len(match.group(1)) == 1:  # H1
                return match.group(2).strip()
        
        # Fall back to filename
        if file_path:
            return file_path.stem
        
        return "Untitled"
    
    def _extract_tags(self, body: str, frontmatter: dict) -> list[str]:
        """Extract all tags from frontmatter and inline."""
        tags = set()
        
        # Frontmatter tags
        fm_tags = frontmatter.get('tags', [])
        if isinstance(fm_tags, str):
            # Handle comma-separated or space-separated tags
            fm_tags = re.split(r'[,\s]+', fm_tags)
        if isinstance(fm_tags, list):
            for tag in fm_tags:
                if tag:
                    # Remove leading # if present
                    tag = str(tag).lstrip('#')
                    tags.add(tag)
        
        # Inline tags (but not in code blocks)
        # Simple approach: remove code blocks first
        clean_body = self._remove_code_blocks(body)
        
        for match in self.INLINE_TAG_PATTERN.finditer(clean_body):
            tag = match.group(1)
            # Skip common non-tags
            if not tag.startswith(('!', '?')) and len(tag) > 1:
                tags.add(tag)
        
        return sorted(tags)
    
    def _extract_links(self, body: str) -> list[str]:
        """Extract all outbound links."""
        links = set()
        
        # Wikilinks [[link]] or [[link|alias]]
        for match in self.WIKILINK_PATTERN.finditer(body):
            link = match.group(1).strip()
            # Handle headings in links [[file#heading]]
            if '#' in link:
                link = link.split('#')[0]
            if link:
                links.add(link)
        
        # Markdown links [text](url)
        for match in self.MDLINK_PATTERN.finditer(body):
            url = match.group(2).strip()
            # Only include internal links (not http/https)
            if not url.startswith(('http://', 'https://', 'mailto:', '#')):
                # Remove any anchors
                if '#' in url:
                    url = url.split('#')[0]
                if url:
                    links.add(url)
        
        return sorted(links)
    
    def _extract_sections(self, body: str, title: str) -> list[Section]:
        """Extract sections based on headings."""
        sections = []
        
        # Find all headings
        headings = list(self.HEADING_PATTERN.finditer(body))
        
        if not headings:
            # No headings - treat entire content as one section
            sections.append(Section(
                heading=title,
                level=0,
                content=body.strip()
            ))
            return sections
        
        # Content before first heading
        if headings[0].start() > 0:
            pre_content = body[:headings[0].start()].strip()
            if pre_content:
                sections.append(Section(
                    heading=title,
                    level=0,
                    content=pre_content
                ))
        
        # Process each heading
        for i, match in enumerate(headings):
            level = len(match.group(1))
            heading = match.group(2).strip()
            
            # Content is from after this heading to next heading (or end)
            start = match.end()
            end = headings[i + 1].start() if i + 1 < len(headings) else len(body)
            content = body[start:end].strip()
            
            sections.append(Section(
                heading=heading,
                level=level,
                content=content
            ))
        
        return sections
    
    def _remove_code_blocks(self, body: str) -> str:
        """Remove fenced code blocks and inline code."""
        # Remove fenced code blocks
        body = re.sub(r'```[\s\S]*?```', '', body)
        # Remove inline code
        body = re.sub(r'`[^`]+`', '', body)
        return body


# Singleton parser instance
parser = MarkdownParser()
