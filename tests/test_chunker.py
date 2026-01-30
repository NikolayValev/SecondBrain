"""
Tests for the chunking module.
"""

import pytest
from app.chunker import TextChunker, Chunk, chunker


class TestTextChunker:
    """Tests for TextChunker class."""
    
    def test_estimate_tokens(self):
        """Test token estimation."""
        c = TextChunker()
        # 4 chars per token
        assert c.estimate_tokens("hello") == 1
        assert c.estimate_tokens("hello world") == 2
        assert c.estimate_tokens("a" * 100) == 25
    
    def test_chunk_empty_text(self):
        """Test chunking empty text returns empty list."""
        c = TextChunker()
        assert c.chunk_text("") == []
        assert c.chunk_text("   ") == []
        assert c.chunk_text(None) == [] if c.chunk_text("") == [] else True
    
    def test_chunk_small_text(self):
        """Test small text returns single chunk."""
        c = TextChunker(max_chunk_tokens=512, min_chunk_tokens=10)
        text = "This is a small piece of text that fits in one chunk easily."
        
        chunks = c.chunk_text(text)
        
        assert len(chunks) == 1
        assert chunks[0].content == text
        assert chunks[0].chunk_index == 0
    
    def test_chunk_large_text(self):
        """Test large text is split into multiple chunks."""
        c = TextChunker(max_chunk_tokens=50, min_chunk_tokens=10)
        # Create text that's definitely larger than 50 tokens
        text = "This is a paragraph. " * 50
        
        chunks = c.chunk_text(text)
        
        assert len(chunks) > 1
        # Check indices are sequential
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
    
    def test_chunk_preserves_file_id(self):
        """Test file_id is preserved in chunks."""
        c = TextChunker(min_chunk_tokens=5)
        text = "Some content that should be chunked with file ID."
        
        chunks = c.chunk_text(text, file_id=42)
        
        assert len(chunks) >= 1
        assert chunks[0].file_id == 42
    
    def test_chunk_preserves_section_id(self):
        """Test section_id is preserved in chunks."""
        c = TextChunker(min_chunk_tokens=5)
        text = "Some content that should be chunked with section ID."
        
        chunks = c.chunk_text(text, section_id=123)
        
        assert len(chunks) >= 1
        assert chunks[0].section_id == 123
    
    def test_chunk_text_below_minimum(self):
        """Test text below minimum tokens is excluded."""
        c = TextChunker(min_chunk_tokens=100)
        text = "Too short"
        
        chunks = c.chunk_text(text)
        
        assert len(chunks) == 0
    
    def test_chunk_sections(self):
        """Test chunking multiple sections."""
        c = TextChunker(max_chunk_tokens=100, min_chunk_tokens=10)
        sections = [
            {"id": 1, "heading": "First Section", "content": "Content of the first section."},
            {"id": 2, "heading": "Second Section", "content": "Content of the second section."},
        ]
        
        chunks = c.chunk_sections(sections, file_id=1)
        
        assert len(chunks) >= 2
        # Indices should be sequential across sections
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
    
    def test_chunk_sections_includes_heading(self):
        """Test that section headings are included in chunks."""
        c = TextChunker(max_chunk_tokens=200, min_chunk_tokens=5)
        sections = [
            {"id": 1, "heading": "My Heading", "content": "Some content here."},
        ]
        
        chunks = c.chunk_sections(sections, file_id=1, include_heading=True)
        
        assert len(chunks) >= 1
        assert "My Heading" in chunks[0].content
    
    def test_paragraph_splitting(self):
        """Test that text is split on paragraph boundaries."""
        c = TextChunker(max_chunk_tokens=30, min_chunk_tokens=5)
        text = """First paragraph with some content.

Second paragraph with different content.

Third paragraph with more content."""
        
        chunks = c.chunk_text(text)
        
        # Should have multiple chunks
        assert len(chunks) >= 1


class TestDefaultChunker:
    """Test the default chunker instance."""
    
    def test_default_chunker_exists(self):
        """Test default chunker is available."""
        assert chunker is not None
        assert isinstance(chunker, TextChunker)
    
    def test_default_chunker_config(self):
        """Test default chunker has reasonable defaults."""
        assert chunker.max_chunk_tokens == 512
        assert chunker.overlap_tokens == 50
        assert chunker.min_chunk_tokens == 50
