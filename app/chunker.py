"""
Text chunking module for Second Brain.
Splits documents into smaller chunks suitable for embedding.
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a text chunk for embedding."""
    content: str
    token_count: int
    chunk_index: int
    section_id: Optional[int] = None
    file_id: Optional[int] = None


class TextChunker:
    """
    Splits text into chunks for embedding.
    
    Uses a hierarchical approach:
    1. Try to split on paragraph boundaries
    2. Fall back to sentence boundaries
    3. Fall back to word boundaries if needed
    """
    
    # Approximate tokens per character ratio (conservative estimate)
    CHARS_PER_TOKEN = 4
    
    def __init__(
        self,
        max_chunk_tokens: int = 512,
        overlap_tokens: int = 50,
        min_chunk_tokens: int = 50,
    ):
        """
        Initialize the chunker.
        
        Args:
            max_chunk_tokens: Maximum tokens per chunk.
            overlap_tokens: Number of tokens to overlap between chunks.
            min_chunk_tokens: Minimum tokens for a valid chunk.
        """
        self.max_chunk_tokens = max_chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.min_chunk_tokens = min_chunk_tokens
        
        # Convert to approximate character counts
        self.max_chunk_chars = max_chunk_tokens * self.CHARS_PER_TOKEN
        self.overlap_chars = overlap_tokens * self.CHARS_PER_TOKEN
        self.min_chunk_chars = min_chunk_tokens * self.CHARS_PER_TOKEN
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return max(1, len(text) // self.CHARS_PER_TOKEN)
    
    def chunk_text(self, text: str, file_id: Optional[int] = None, section_id: Optional[int] = None) -> list[Chunk]:
        """
        Split text into chunks.
        
        Args:
            text: The text to chunk.
            file_id: Optional file ID for tracking.
            section_id: Optional section ID for tracking.
            
        Returns:
            List of Chunk objects.
        """
        if not text or not text.strip():
            return []
        
        text = text.strip()
        
        # If text is small enough, return as single chunk
        if len(text) <= self.max_chunk_chars:
            token_count = self.estimate_tokens(text)
            if token_count >= self.min_chunk_tokens:
                return [Chunk(
                    content=text,
                    token_count=token_count,
                    chunk_index=0,
                    section_id=section_id,
                    file_id=file_id,
                )]
            return []
        
        # Split into paragraphs first
        paragraphs = self._split_paragraphs(text)
        
        # Build chunks from paragraphs
        chunks = []
        current_chunk = ""
        chunk_index = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If adding this paragraph exceeds limit, finalize current chunk
            if current_chunk and len(current_chunk) + len(para) + 2 > self.max_chunk_chars:
                # Finalize current chunk
                chunk = self._create_chunk(current_chunk, chunk_index, file_id, section_id)
                if chunk:
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Start new chunk with overlap from previous
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + para if overlap_text else para
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
            
            # If paragraph itself is too long, split it
            if len(current_chunk) > self.max_chunk_chars:
                sub_chunks = self._split_large_text(current_chunk, chunk_index, file_id, section_id)
                if sub_chunks:
                    chunks.extend(sub_chunks[:-1])  # Add all but last
                    chunk_index += len(sub_chunks) - 1
                    # Keep last sub-chunk as current
                    current_chunk = sub_chunks[-1].content if sub_chunks else ""
        
        # Don't forget the last chunk
        if current_chunk:
            chunk = self._create_chunk(current_chunk, chunk_index, file_id, section_id)
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def chunk_sections(
        self, 
        sections: list[dict], 
        file_id: int,
        include_heading: bool = True
    ) -> list[Chunk]:
        """
        Chunk multiple sections from a document.
        
        Args:
            sections: List of section dicts with 'id', 'heading', 'content'.
            file_id: The file ID.
            include_heading: Whether to prepend heading to content.
            
        Returns:
            List of Chunk objects.
        """
        all_chunks = []
        
        for section in sections:
            content = section.get("content", "")
            heading = section.get("heading", "")
            section_id = section.get("id")
            
            # Optionally prepend heading for context
            if include_heading and heading and heading != "Untitled":
                text = f"# {heading}\n\n{content}"
            else:
                text = content
            
            chunks = self.chunk_text(text, file_id=file_id, section_id=section_id)
            all_chunks.extend(chunks)
        
        # Re-index chunks sequentially
        for i, chunk in enumerate(all_chunks):
            chunk.chunk_index = i
        
        return all_chunks
    
    def _split_paragraphs(self, text: str) -> list[str]:
        """Split text into paragraphs."""
        # Split on double newlines or more
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Simple sentence splitting on common terminators
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _split_large_text(
        self, 
        text: str, 
        start_index: int,
        file_id: Optional[int],
        section_id: Optional[int]
    ) -> list[Chunk]:
        """Split text that's too large for a single chunk."""
        chunks = []
        sentences = self._split_sentences(text)
        
        current_chunk = ""
        chunk_index = start_index
        
        for sentence in sentences:
            if current_chunk and len(current_chunk) + len(sentence) + 1 > self.max_chunk_chars:
                chunk = self._create_chunk(current_chunk, chunk_index, file_id, section_id)
                if chunk:
                    chunks.append(chunk)
                    chunk_index += 1
                
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + sentence if overlap_text else sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        # Handle remaining text
        if current_chunk:
            chunk = self._create_chunk(current_chunk, chunk_index, file_id, section_id)
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from the end of a chunk."""
        if len(text) <= self.overlap_chars:
            return text + " "
        
        # Try to break at word boundary
        overlap_start = len(text) - self.overlap_chars
        overlap = text[overlap_start:]
        
        # Find first word boundary
        space_idx = overlap.find(' ')
        if space_idx > 0:
            overlap = overlap[space_idx + 1:]
        
        return overlap + " " if overlap else ""
    
    def _create_chunk(
        self, 
        content: str, 
        chunk_index: int,
        file_id: Optional[int],
        section_id: Optional[int]
    ) -> Optional[Chunk]:
        """Create a Chunk object if content meets minimum requirements."""
        content = content.strip()
        token_count = self.estimate_tokens(content)
        
        if token_count < self.min_chunk_tokens:
            return None
        
        return Chunk(
            content=content,
            token_count=token_count,
            chunk_index=chunk_index,
            section_id=section_id,
            file_id=file_id,
        )


# Default chunker instance
chunker = TextChunker()
