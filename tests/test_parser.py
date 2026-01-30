"""
Unit tests for the Markdown parser module.
"""

import pytest
from pathlib import Path

from app.parser import MarkdownParser, ParsedMarkdown, Section


class TestMarkdownParserTitleExtraction:
    """Tests for title extraction logic."""
    
    def test_title_from_first_h1(self, parser: MarkdownParser):
        """Title should be extracted from first H1 heading."""
        content = "# My Title\n\nSome content."
        result = parser.parse_content(content)
        assert result.title == "My Title"
    
    def test_title_from_frontmatter(self, parser: MarkdownParser):
        """Title in frontmatter should take precedence."""
        content = """---
title: Frontmatter Title
---

# Body Title

Content here.
"""
        result = parser.parse_content(content)
        assert result.title == "Frontmatter Title"
    
    def test_title_from_filename_when_no_h1(self, parser: MarkdownParser, temp_dir: Path):
        """Title should fall back to filename when no H1 exists."""
        file_path = temp_dir / "my_note.md"
        file_path.write_text("## Only H2\n\nNo H1 heading here.", encoding="utf-8")
        
        result = parser.parse_file(file_path)
        assert result.title == "my_note"
    
    def test_title_strips_whitespace(self, parser: MarkdownParser):
        """Title should have whitespace stripped."""
        content = "#   Spaced Title   \n\nContent."
        result = parser.parse_content(content)
        assert result.title == "Spaced Title"
    
    def test_untitled_when_no_source(self, parser: MarkdownParser):
        """Should return 'Untitled' when no title source available."""
        content = "## Only H2\n\nNo H1 and no file path."
        result = parser.parse_content(content, file_path=None)
        assert result.title == "Untitled"


class TestMarkdownParserHeadings:
    """Tests for heading extraction."""
    
    def test_extracts_all_heading_levels(self, parser: MarkdownParser):
        """Should extract headings from H1 to H6."""
        content = """# H1 Heading
## H2 Heading
### H3 Heading
#### H4 Heading
##### H5 Heading
###### H6 Heading
"""
        result = parser.parse_content(content)
        
        headings = [(s.heading, s.level) for s in result.sections]
        assert ("H1 Heading", 1) in headings
        assert ("H2 Heading", 2) in headings
        assert ("H3 Heading", 3) in headings
        assert ("H4 Heading", 4) in headings
        assert ("H5 Heading", 5) in headings
        assert ("H6 Heading", 6) in headings
    
    def test_section_content_between_headings(self, parser: MarkdownParser):
        """Content between headings should be captured in sections."""
        content = """# First Section

Content for first section.

## Second Section

Content for second section.
"""
        result = parser.parse_content(content)
        
        assert len(result.sections) == 2
        assert "Content for first section" in result.sections[0].content
        assert "Content for second section" in result.sections[1].content
    
    def test_content_before_first_heading(self, parser: MarkdownParser):
        """Content before first heading should be captured."""
        content = """Intro paragraph before any heading.

# First Heading

Heading content.
"""
        result = parser.parse_content(content)
        
        # First section should contain the intro content
        assert any("Intro paragraph" in s.content for s in result.sections)
    
    def test_no_headings_single_section(self, parser: MarkdownParser):
        """Document without headings should have single section."""
        content = "Just some plain text without any headings."
        result = parser.parse_content(content)
        
        assert len(result.sections) == 1
        assert "plain text" in result.sections[0].content


class TestMarkdownParserTags:
    """Tests for tag extraction."""
    
    def test_inline_tags(self, parser: MarkdownParser):
        """Should extract inline #tags."""
        content = "# Note\n\nContent with #tag1 and #tag2 here."
        result = parser.parse_content(content)
        
        assert "tag1" in result.tags
        assert "tag2" in result.tags
    
    def test_frontmatter_tags_list(self, parser: MarkdownParser):
        """Should extract tags from frontmatter list."""
        content = """---
tags:
  - yaml-tag1
  - yaml-tag2
---

# Title
"""
        result = parser.parse_content(content)
        
        assert "yaml-tag1" in result.tags
        assert "yaml-tag2" in result.tags
    
    def test_frontmatter_tags_string(self, parser: MarkdownParser):
        """Should handle comma-separated tags in frontmatter."""
        content = """---
tags: tag1, tag2, tag3
---

# Title
"""
        result = parser.parse_content(content)
        
        assert "tag1" in result.tags
        assert "tag2" in result.tags
        assert "tag3" in result.tags
    
    def test_tags_with_slashes(self, parser: MarkdownParser):
        """Should handle nested tags with slashes."""
        content = "# Note\n\n#parent/child/grandchild tag here."
        result = parser.parse_content(content)
        
        assert "parent/child/grandchild" in result.tags
    
    def test_tags_in_code_blocks_ignored(self, parser: MarkdownParser):
        """Tags inside code blocks should be ignored."""
        content = """# Note

```python
# This #code-tag should be ignored
print("hello")
```

But this #real-tag should be extracted.
"""
        result = parser.parse_content(content)
        
        assert "real-tag" in result.tags
        assert "code-tag" not in result.tags
    
    def test_tags_in_inline_code_ignored(self, parser: MarkdownParser):
        """Tags inside inline code should be ignored."""
        content = "Use `#ignore-this` but keep #keep-this."
        result = parser.parse_content(content)
        
        assert "keep-this" in result.tags
        assert "ignore-this" not in result.tags
    
    def test_tags_deduplicated(self, parser: MarkdownParser):
        """Duplicate tags should be removed."""
        content = """---
tags:
  - duplicate
---

# Note

#duplicate #duplicate
"""
        result = parser.parse_content(content)
        
        assert result.tags.count("duplicate") == 1
    
    def test_tags_sorted(self, parser: MarkdownParser):
        """Tags should be returned sorted."""
        content = "# Note\n\n#zebra #apple #mango"
        result = parser.parse_content(content)
        
        assert result.tags == sorted(result.tags)


class TestMarkdownParserLinks:
    """Tests for link extraction."""
    
    def test_wikilinks(self, parser: MarkdownParser):
        """Should extract [[wikilinks]]."""
        content = "# Note\n\nLink to [[Another Note]] here."
        result = parser.parse_content(content)
        
        assert "Another Note" in result.links
    
    def test_wikilinks_with_alias(self, parser: MarkdownParser):
        """Should extract wikilinks with aliases."""
        content = "# Note\n\nLink to [[Actual Note|Display Text]] here."
        result = parser.parse_content(content)
        
        assert "Actual Note" in result.links
        assert "Display Text" not in result.links
    
    def test_wikilinks_with_heading(self, parser: MarkdownParser):
        """Should extract wikilinks with heading references."""
        content = "# Note\n\nLink to [[Note#Section]] here."
        result = parser.parse_content(content)
        
        assert "Note" in result.links
        assert "Note#Section" not in result.links
    
    def test_markdown_links(self, parser: MarkdownParser):
        """Should extract [markdown](links)."""
        content = "# Note\n\nLink to [display](path/to/note.md) here."
        result = parser.parse_content(content)
        
        assert "path/to/note.md" in result.links
    
    def test_external_links_ignored(self, parser: MarkdownParser):
        """External HTTP links should be ignored."""
        content = "# Note\n\nExternal [link](https://example.com) here."
        result = parser.parse_content(content)
        
        assert "https://example.com" not in result.links
        assert len([l for l in result.links if "example.com" in l]) == 0
    
    def test_mailto_links_ignored(self, parser: MarkdownParser):
        """Mailto links should be ignored."""
        content = "# Note\n\nEmail [me](mailto:test@example.com)."
        result = parser.parse_content(content)
        
        assert "mailto:test@example.com" not in result.links
    
    def test_anchor_links_ignored(self, parser: MarkdownParser):
        """Anchor-only links should be ignored."""
        content = "# Note\n\nJump to [section](#section-name)."
        result = parser.parse_content(content)
        
        assert "#section-name" not in result.links
    
    def test_links_deduplicated(self, parser: MarkdownParser):
        """Duplicate links should be removed."""
        content = "# Note\n\n[[Same Link]] and [[Same Link]] again."
        result = parser.parse_content(content)
        
        assert result.links.count("Same Link") == 1
    
    def test_links_sorted(self, parser: MarkdownParser):
        """Links should be returned sorted."""
        content = "# Note\n\n[[Zebra]] [[Apple]] [[Mango]]"
        result = parser.parse_content(content)
        
        assert result.links == sorted(result.links)


class TestMarkdownParserFrontmatter:
    """Tests for frontmatter parsing."""
    
    def test_valid_frontmatter(self, parser: MarkdownParser):
        """Should parse valid YAML frontmatter."""
        content = """---
title: Test
author: John
date: 2024-01-01
---

# Content
"""
        result = parser.parse_content(content)
        
        assert result.frontmatter["title"] == "Test"
        assert result.frontmatter["author"] == "John"
    
    def test_no_frontmatter(self, parser: MarkdownParser):
        """Should handle documents without frontmatter."""
        content = "# Title\n\nNo frontmatter here."
        result = parser.parse_content(content)
        
        assert result.frontmatter == {}
    
    def test_invalid_frontmatter_graceful(self, parser: MarkdownParser):
        """Should handle invalid YAML gracefully."""
        content = """---
invalid: yaml: content: here
  bad indentation
---

# Title
"""
        # Should not raise, just log warning
        result = parser.parse_content(content)
        assert result.title is not None


class TestMarkdownParserFullContent:
    """Tests for full content preservation."""
    
    def test_content_preserved(self, parser: MarkdownParser):
        """Full raw content should be preserved."""
        content = "# Title\n\nAll the content here."
        result = parser.parse_content(content)
        
        assert result.content == content
    
    def test_content_with_frontmatter(self, parser: MarkdownParser):
        """Content should include frontmatter."""
        content = """---
title: Test
---

# Title
"""
        result = parser.parse_content(content)
        
        assert "---" in result.content
        assert "title: Test" in result.content


class TestMarkdownParserFileReading:
    """Tests for file reading functionality."""
    
    def test_parse_file_utf8(self, parser: MarkdownParser, temp_dir: Path):
        """Should read UTF-8 encoded files."""
        file_path = temp_dir / "utf8.md"
        file_path.write_text("# Ünïcödé Tïtlé\n\nCöntént with émojis 🎉", encoding="utf-8")
        
        result = parser.parse_file(file_path)
        
        assert "Ünïcödé Tïtlé" in result.title
        assert "🎉" in result.content
    
    def test_parse_file_latin1_fallback(self, parser: MarkdownParser, temp_dir: Path):
        """Should fall back to latin-1 for non-UTF-8 files."""
        file_path = temp_dir / "latin1.md"
        file_path.write_bytes(b"# Title\n\nContent with \xe9 accent.")
        
        result = parser.parse_file(file_path)
        
        assert result.title == "Title"
