"""
Pytest configuration and shared fixtures for Second Brain tests.
"""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Generator

import pytest

from app.db import Database
from app.parser import MarkdownParser
from app.indexer import Indexer


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def temp_vault(temp_dir: Path) -> Path:
    """Create a temporary vault directory with sample markdown files."""
    vault_path = temp_dir / "vault"
    vault_path.mkdir()
    
    # Create sample markdown files
    (vault_path / "simple.md").write_text(
        "# Simple Note\n\nThis is a simple note with some content.\n\n#tag1 #tag2\n",
        encoding="utf-8"
    )
    
    (vault_path / "with_frontmatter.md").write_text(
        """---
title: Frontmatter Title
tags:
  - yaml-tag1
  - yaml-tag2
---

# Heading in Body

Some content here with [[wikilink]] and [markdown link](other.md).

## Subheading

More content with #inline-tag.
""",
        encoding="utf-8"
    )
    
    (vault_path / "complex.md").write_text(
        """# Complex Document

## Section One

Content in section one with [[Internal Link]] and [[Another Link|Alias]].

### Subsection 1.1

Nested content here.

## Section Two

- List item with #list-tag
- Another item

### Subsection 2.1

Code block example:
```python
# This #hashtag should be ignored
print("hello")
```

Final paragraph with [external](https://example.com) link.
""",
        encoding="utf-8"
    )
    
    # Create subdirectory with files
    subdir = vault_path / "subfolder"
    subdir.mkdir()
    
    (subdir / "nested.md").write_text(
        "# Nested Note\n\nA note in a subfolder.\n\n#nested-tag\n",
        encoding="utf-8"
    )
    
    return vault_path


@pytest.fixture
def temp_db(temp_dir: Path) -> Generator[Database, None, None]:
    """Create a temporary database for tests."""
    db_path = temp_dir / "test.db"
    database = Database(db_path)
    database.initialize()
    yield database
    database.close()


@pytest.fixture
def parser() -> MarkdownParser:
    """Create a markdown parser instance."""
    return MarkdownParser()


@pytest.fixture
def indexer(temp_db: Database, parser: MarkdownParser, temp_vault: Path) -> Indexer:
    """Create an indexer with temporary database and vault."""
    return Indexer(
        database=temp_db,
        md_parser=parser,
        vault_path=temp_vault
    )


@pytest.fixture
def sample_markdown_simple() -> str:
    """Simple markdown content for testing."""
    return """# Test Title

This is a paragraph with some text.

#tag1 #tag2

[[Link to Note]]
"""


@pytest.fixture
def sample_markdown_complex() -> str:
    """Complex markdown with frontmatter and multiple sections."""
    return """---
title: Custom Title
tags:
  - meta-tag1
  - meta-tag2
author: Test Author
---

# Main Heading

Introduction paragraph.

## Section A

Content for section A with [[Internal Link]] reference.

### Subsection A.1

Nested content with #inline-tag.

## Section B

Final section with [markdown link](path/to/file.md).
"""


@pytest.fixture
def sample_markdown_no_h1() -> str:
    """Markdown without H1 heading."""
    return """## Only H2 Heading

Some content without a main title.

### H3 Heading

More content.
"""
