"""
Tests for vector search module.
"""

import pytest
from app.vector_search import (
    cosine_similarity, 
    dot_product, 
    normalize_vector,
    bytes_to_embedding,
    VectorSearch,
    SearchResult,
)
from app.embeddings import EmbeddingService


class TestVectorMath:
    """Tests for vector math functions."""
    
    def test_cosine_similarity_identical(self):
        """Test cosine similarity of identical vectors is 1."""
        vec = [1.0, 2.0, 3.0]
        assert cosine_similarity(vec, vec) == pytest.approx(1.0)
    
    def test_cosine_similarity_opposite(self):
        """Test cosine similarity of opposite vectors is -1."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [-1.0, 0.0, 0.0]
        assert cosine_similarity(vec1, vec2) == pytest.approx(-1.0)
    
    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity of orthogonal vectors is 0."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        assert cosine_similarity(vec1, vec2) == pytest.approx(0.0)
    
    def test_cosine_similarity_dimension_mismatch(self):
        """Test dimension mismatch raises error."""
        vec1 = [1.0, 2.0]
        vec2 = [1.0, 2.0, 3.0]
        with pytest.raises(ValueError, match="dimension mismatch"):
            cosine_similarity(vec1, vec2)
    
    def test_cosine_similarity_zero_vector(self):
        """Test zero vector returns 0."""
        vec1 = [0.0, 0.0, 0.0]
        vec2 = [1.0, 2.0, 3.0]
        assert cosine_similarity(vec1, vec2) == 0.0
    
    def test_dot_product(self):
        """Test dot product calculation."""
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [4.0, 5.0, 6.0]
        # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert dot_product(vec1, vec2) == pytest.approx(32.0)
    
    def test_normalize_vector(self):
        """Test vector normalization."""
        vec = [3.0, 4.0]  # 3-4-5 triangle
        normalized = normalize_vector(vec)
        assert normalized[0] == pytest.approx(0.6)
        assert normalized[1] == pytest.approx(0.8)
        
        # Normalized vector should have unit length
        length = sum(x * x for x in normalized) ** 0.5
        assert length == pytest.approx(1.0)
    
    def test_normalize_zero_vector(self):
        """Test normalizing zero vector returns zero vector."""
        vec = [0.0, 0.0, 0.0]
        normalized = normalize_vector(vec)
        assert normalized == vec


class TestEmbeddingConversion:
    """Tests for embedding byte conversion."""
    
    def test_embedding_roundtrip(self):
        """Test embedding to bytes and back."""
        original = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        as_bytes = EmbeddingService.embedding_to_bytes(original)
        recovered = bytes_to_embedding(as_bytes)
        
        for o, r in zip(original, recovered):
            assert o == pytest.approx(r, rel=1e-6)
    
    def test_embedding_to_bytes_size(self):
        """Test bytes size is correct (4 bytes per float)."""
        embedding = [0.1, 0.2, 0.3]
        as_bytes = EmbeddingService.embedding_to_bytes(embedding)
        assert len(as_bytes) == 12  # 3 floats * 4 bytes


class TestVectorSearch:
    """Tests for VectorSearch class."""
    
    def test_search_empty_database(self, monkeypatch):
        """Test search with no embeddings returns empty."""
        from app import db as db_module
        
        # Mock get_all_embeddings to return empty
        monkeypatch.setattr(db_module.db, "get_all_embeddings", lambda: [])
        
        vs = VectorSearch()
        vs.invalidate_cache()
        
        results = vs.search([0.1, 0.2, 0.3], top_k=5)
        assert results == []
    
    def test_deduplicate_by_file(self):
        """Test deduplication limits results per file."""
        vs = VectorSearch()
        
        results = [
            SearchResult(1, 1, "file1.md", "Title 1", None, "chunk 1", 0.9),
            SearchResult(2, 1, "file1.md", "Title 1", None, "chunk 2", 0.85),
            SearchResult(3, 1, "file1.md", "Title 1", None, "chunk 3", 0.8),
            SearchResult(4, 2, "file2.md", "Title 2", None, "chunk 4", 0.75),
        ]
        
        deduplicated = vs.deduplicate_by_file(results, max_per_file=2)
        
        # Should have 2 from file1 and 1 from file2
        assert len(deduplicated) == 3
        file1_count = sum(1 for r in deduplicated if r.file_path == "file1.md")
        assert file1_count == 2
    
    def test_invalidate_cache(self):
        """Test cache invalidation."""
        vs = VectorSearch()
        vs._cache = [{"test": "data"}]
        vs._cache_valid = True
        
        vs.invalidate_cache()
        
        assert vs._cache is None
        assert vs._cache_valid is False


class TestSearchResult:
    """Tests for SearchResult dataclass."""
    
    def test_search_result_creation(self):
        """Test SearchResult can be created."""
        result = SearchResult(
            chunk_id=1,
            file_id=2,
            file_path="test.md",
            file_title="Test Title",
            section_heading="Section",
            chunk_content="Some content",
            similarity=0.95,
        )
        
        assert result.chunk_id == 1
        assert result.file_id == 2
        assert result.file_path == "test.md"
        assert result.similarity == 0.95
