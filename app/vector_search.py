"""
Vector search module for Second Brain.
Provides cosine similarity search over embeddings.
"""

import math
import struct
import logging
from dataclasses import dataclass
from typing import Optional

from app.db import db

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single vector search result."""
    chunk_id: int
    file_id: int
    file_path: str
    file_title: str
    section_heading: Optional[str]
    chunk_content: str
    similarity: float


def bytes_to_embedding(data: bytes) -> list[float]:
    """Convert bytes back to embedding list."""
    num_floats = len(data) // 4
    return list(struct.unpack(f'{num_floats}f', data))


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector.
        vec2: Second vector.
        
    Returns:
        Cosine similarity score between -1 and 1.
    """
    if len(vec1) != len(vec2):
        raise ValueError(f"Vector dimension mismatch: {len(vec1)} vs {len(vec2)}")
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def dot_product(vec1: list[float], vec2: list[float]) -> float:
    """Calculate dot product between two vectors."""
    return sum(a * b for a, b in zip(vec1, vec2))


def normalize_vector(vec: list[float]) -> list[float]:
    """Normalize a vector to unit length."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]


class VectorSearch:
    """
    Vector similarity search over stored embeddings.
    
    Uses brute-force cosine similarity search.
    For larger datasets, consider using FAISS or similar.
    """
    
    def __init__(self):
        self._cache: Optional[list[dict]] = None
        self._cache_valid = False
    
    def invalidate_cache(self) -> None:
        """Invalidate the embedding cache."""
        self._cache = None
        self._cache_valid = False
    
    def _load_embeddings(self) -> list[dict]:
        """Load all embeddings from database."""
        if self._cache_valid and self._cache is not None:
            return self._cache
        
        logger.debug("Loading embeddings from database")
        raw_embeddings = db.get_all_embeddings()
        
        # Parse embeddings from bytes
        self._cache = []
        for row in raw_embeddings:
            embedding = bytes_to_embedding(row["embedding"])
            self._cache.append({
                "embedding_id": row["embedding_id"],
                "chunk_id": row["chunk_id"],
                "embedding": embedding,
                "chunk_content": row["chunk_content"],
                "chunk_index": row["chunk_index"],
                "file_id": row["file_id"],
                "file_path": row["file_path"],
                "file_title": row["file_title"],
                "section_heading": row["section_heading"],
            })
        
        self._cache_valid = True
        logger.debug(f"Loaded {len(self._cache)} embeddings")
        return self._cache
    
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> list[SearchResult]:
        """
        Search for similar chunks using cosine similarity.
        
        Args:
            query_embedding: The query vector.
            top_k: Maximum number of results to return.
            threshold: Minimum similarity score (0-1).
            
        Returns:
            List of SearchResult objects, sorted by similarity descending.
        """
        embeddings = self._load_embeddings()
        
        if not embeddings:
            logger.warning("No embeddings in database")
            return []
        
        # Normalize query for faster comparison
        query_norm = normalize_vector(query_embedding)
        
        # Calculate similarities
        results = []
        for item in embeddings:
            stored_norm = normalize_vector(item["embedding"])
            similarity = dot_product(query_norm, stored_norm)
            
            if similarity >= threshold:
                results.append(SearchResult(
                    chunk_id=item["chunk_id"],
                    file_id=item["file_id"],
                    file_path=item["file_path"],
                    file_title=item["file_title"],
                    section_heading=item["section_heading"],
                    chunk_content=item["chunk_content"],
                    similarity=similarity,
                ))
        
        # Sort by similarity descending
        results.sort(key=lambda x: x.similarity, reverse=True)
        
        return results[:top_k]
    
    def search_by_file(
        self,
        query_embedding: list[float],
        file_path: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """
        Search within a specific file only.
        
        Args:
            query_embedding: The query vector.
            file_path: Path to limit search to.
            top_k: Maximum results.
            
        Returns:
            List of SearchResult objects.
        """
        all_results = self.search(query_embedding, top_k=top_k * 3)
        filtered = [r for r in all_results if r.file_path == file_path]
        return filtered[:top_k]
    
    def deduplicate_by_file(
        self,
        results: list[SearchResult],
        max_per_file: int = 2,
    ) -> list[SearchResult]:
        """
        Limit results to max_per_file chunks per file.
        
        Args:
            results: Search results to deduplicate.
            max_per_file: Maximum chunks per file.
            
        Returns:
            Deduplicated results.
        """
        file_counts: dict[str, int] = {}
        deduplicated = []
        
        for result in results:
            count = file_counts.get(result.file_path, 0)
            if count < max_per_file:
                deduplicated.append(result)
                file_counts[result.file_path] = count + 1
        
        return deduplicated


# Singleton instance
vector_search = VectorSearch()
