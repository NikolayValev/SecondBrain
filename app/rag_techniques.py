"""
RAG Techniques module for Second Brain.
Implements various retrieval-augmented generation strategies.
"""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from app.config import Config
from app.embeddings import embedding_service
from app.vector_search import vector_search, SearchResult
from app.db import db
from app import llm

logger = logging.getLogger(__name__)


@dataclass
class RAGContext:
    """Context retrieved for RAG."""
    chunks: list[SearchResult]
    context_text: str
    technique: str
    metadata: dict


class RAGTechnique(ABC):
    """Abstract base class for RAG techniques."""
    
    @property
    @abstractmethod
    def id(self) -> str:
        """Unique identifier for this technique."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Description of how this technique works."""
        pass
    
    @abstractmethod
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.3,
    ) -> RAGContext:
        """
        Retrieve relevant context for the query.
        
        Args:
            query: The user's question.
            top_k: Maximum chunks to return.
            threshold: Minimum similarity threshold.
            
        Returns:
            RAGContext with retrieved chunks and formatted context.
        """
        pass


class BasicRAG(RAGTechnique):
    """
    Basic RAG: Simple semantic search and retrieval.
    
    1. Embed the user query
    2. Retrieve top-k similar documents from vector store
    3. Return context
    """
    
    @property
    def id(self) -> str:
        return "basic"
    
    @property
    def name(self) -> str:
        return "Basic RAG"
    
    @property
    def description(self) -> str:
        return "Simple semantic search and retrieval"
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.3,
    ) -> RAGContext:
        # Embed the query
        query_embedding = await embedding_service.embed_text(query)
        
        # Search for similar chunks
        results = vector_search.search(
            query_embedding=query_embedding,
            top_k=top_k * 2,  # Get more for deduplication
            threshold=threshold,
        )
        
        # Deduplicate by file
        results = vector_search.deduplicate_by_file(results, max_per_file=2)[:top_k]
        
        # Build context text
        context_text = self._build_context_text(results)
        
        return RAGContext(
            chunks=results,
            context_text=context_text,
            technique=self.id,
            metadata={"query_embedded": True}
        )
    
    def _build_context_text(self, results: list[SearchResult]) -> str:
        """Build formatted context string from search results."""
        parts = []
        for i, result in enumerate(results, 1):
            header = f"[Source {i}: {result.file_title}"
            if result.section_heading:
                header += f" > {result.section_heading}"
            header += "]"
            parts.append(f"{header}\n{result.chunk_content}\n")
        return "\n".join(parts)


class HybridRAG(RAGTechnique):
    """
    Hybrid Search: Combines semantic and keyword search.
    
    1. Embed the user query (dense retrieval)
    2. Run BM25/keyword search (sparse retrieval)
    3. Combine results using Reciprocal Rank Fusion (RRF)
    """
    
    @property
    def id(self) -> str:
        return "hybrid"
    
    @property
    def name(self) -> str:
        return "Hybrid Search"
    
    @property
    def description(self) -> str:
        return "Combines semantic and keyword search (BM25 + embeddings)"
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.3,
    ) -> RAGContext:
        # Dense retrieval: semantic search
        query_embedding = await embedding_service.embed_text(query)
        semantic_results = vector_search.search(
            query_embedding=query_embedding,
            top_k=top_k * 3,
            threshold=threshold,
        )
        
        # Sparse retrieval: FTS5 keyword search
        fts_query = self._sanitize_fts_query(query)
        fts_results = db.search(fts_query, limit=top_k * 3) if fts_query else []
        
        # Combine using RRF
        combined = self._reciprocal_rank_fusion(
            semantic_results, 
            fts_results,
            top_k=top_k
        )
        
        context_text = self._build_context_text(combined)
        
        return RAGContext(
            chunks=combined,
            context_text=context_text,
            technique=self.id,
            metadata={
                "semantic_count": len(semantic_results),
                "fts_count": len(fts_results),
            }
        )
    
    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """Strip FTS5 special characters so raw user input is safe."""
        sanitized = re.sub(r'[*?"():^{}~\-]', ' ', query)
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        return sanitized

    def _reciprocal_rank_fusion(
        self,
        semantic: list[SearchResult],
        fts: list[dict],
        top_k: int,
        k: int = 60  # RRF constant
    ) -> list[SearchResult]:
        """Combine results using Reciprocal Rank Fusion."""
        scores: dict[str, float] = {}
        result_map: dict[str, SearchResult] = {}
        
        # Score semantic results
        for rank, result in enumerate(semantic):
            key = f"{result.file_path}:{result.chunk_id}"
            scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
            result_map[key] = result
        
        # Score FTS results (convert to SearchResult format)
        for rank, fts_item in enumerate(fts):
            # For FTS results, we need to find corresponding chunks
            file_path = fts_item.get("file_path", "")
            key = f"fts:{file_path}:{rank}"
            scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
            
            # Create a synthetic SearchResult for FTS results
            if key not in result_map:
                result_map[key] = SearchResult(
                    chunk_id=-1,
                    file_id=-1,
                    file_path=file_path,
                    file_title=fts_item.get("title", ""),
                    section_heading=fts_item.get("heading"),
                    chunk_content=fts_item.get("snippet", ""),
                    similarity=0.5,  # Default score for FTS
                )
        
        # Sort by combined score
        sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        # Deduplicate by file path
        seen_files: dict[str, int] = {}
        results = []
        for key in sorted_keys:
            if len(results) >= top_k:
                break
            result = result_map[key]
            file_path = result.file_path
            if seen_files.get(file_path, 0) < 2:
                results.append(result)
                seen_files[file_path] = seen_files.get(file_path, 0) + 1
        
        return results
    
    def _build_context_text(self, results: list[SearchResult]) -> str:
        """Build formatted context string from search results."""
        parts = []
        for i, result in enumerate(results, 1):
            header = f"[Source {i}: {result.file_title}"
            if result.section_heading:
                header += f" > {result.section_heading}"
            header += "]"
            parts.append(f"{header}\n{result.chunk_content}\n")
        return "\n".join(parts)


class RerankRAG(RAGTechnique):
    """
    Re-ranking RAG: Uses a cross-encoder reranker to improve results.
    
    1. Embed the user query
    2. Retrieve top-k*3 similar documents
    3. Re-rank with cross-encoder model
    4. Take top-k after re-ranking
    """
    
    def __init__(self):
        self._reranker = None
    
    @property
    def id(self) -> str:
        return "rerank"
    
    @property
    def name(self) -> str:
        return "Re-ranking"
    
    @property
    def description(self) -> str:
        return "Uses a cross-encoder reranker model to improve results"
    
    def _get_reranker(self):
        """Lazy-load the reranker model."""
        if self._reranker is None:
            try:
                from sentence_transformers import CrossEncoder
                self._reranker = CrossEncoder(Config.RERANKER_MODEL)
                logger.info(f"Loaded reranker model: {Config.RERANKER_MODEL}")
            except ImportError:
                logger.warning("sentence-transformers not installed, falling back to no reranking")
                return None
            except Exception as e:
                logger.error(f"Failed to load reranker: {e}")
                return None
        return self._reranker
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.3,
    ) -> RAGContext:
        # Get more candidates for reranking
        query_embedding = await embedding_service.embed_text(query)
        candidates = vector_search.search(
            query_embedding=query_embedding,
            top_k=top_k * 3,
            threshold=threshold,
        )
        
        if not candidates:
            return RAGContext(
                chunks=[],
                context_text="",
                technique=self.id,
                metadata={"reranked": False}
            )
        
        reranker = self._get_reranker()
        if reranker is None:
            # Fallback to basic semantic search
            results = vector_search.deduplicate_by_file(candidates, max_per_file=2)[:top_k]
            return RAGContext(
                chunks=results,
                context_text=self._build_context_text(results),
                technique=self.id,
                metadata={"reranked": False, "fallback": True}
            )
        
        # Prepare pairs for reranking
        pairs = [(query, c.chunk_content) for c in candidates]
        
        # Score with cross-encoder
        try:
            scores = reranker.predict(pairs)
            
            # Combine with original results
            scored_results = list(zip(candidates, scores))
            scored_results.sort(key=lambda x: x[1], reverse=True)
            
            # Take top-k after reranking
            reranked = [r[0] for r in scored_results[:top_k]]
            
            # Update similarity scores
            for result, (_, score) in zip(reranked, scored_results[:top_k]):
                result.similarity = float(score)
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            reranked = candidates[:top_k]
        
        results = vector_search.deduplicate_by_file(reranked, max_per_file=2)[:top_k]
        
        return RAGContext(
            chunks=results,
            context_text=self._build_context_text(results),
            technique=self.id,
            metadata={"reranked": True, "candidates": len(candidates)}
        )
    
    def _build_context_text(self, results: list[SearchResult]) -> str:
        parts = []
        for i, result in enumerate(results, 1):
            header = f"[Source {i}: {result.file_title}"
            if result.section_heading:
                header += f" > {result.section_heading}"
            header += "]"
            parts.append(f"{header}\n{result.chunk_content}\n")
        return "\n".join(parts)


class HyDERAG(RAGTechnique):
    """
    HyDE (Hypothetical Document Embeddings): Generates hypothetical answer first.
    
    1. Generate hypothetical answer using LLM (without context)
    2. Embed the hypothetical answer
    3. Retrieve documents similar to hypothetical answer
    4. Return context
    """
    
    @property
    def id(self) -> str:
        return "hyde"
    
    @property
    def name(self) -> str:
        return "HyDE"
    
    @property
    def description(self) -> str:
        return "Hypothetical Document Embeddings - generates hypothetical answer first"
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.3,
    ) -> RAGContext:
        # Generate hypothetical answer
        hypothetical = await self._generate_hypothetical(query)
        
        # Embed both the hypothetical answer and original query
        hyde_embedding = await embedding_service.embed_text(hypothetical)
        query_embedding = await embedding_service.embed_text(query)
        
        # Search using hypothetical embedding (lower threshold — HyDE
        # embeddings are naturally less similar to stored chunks)
        hyde_threshold = max(threshold - 0.15, 0.05)
        hyde_results = vector_search.search(
            query_embedding=hyde_embedding,
            top_k=top_k * 2,
            threshold=hyde_threshold,
        )
        
        # Also search with original query for fallback
        query_results = vector_search.search(
            query_embedding=query_embedding,
            top_k=top_k,
            threshold=threshold,
        )
        
        # Merge results, preferring HyDE but filling with query results
        seen_chunks: set[int] = set()
        merged: list[SearchResult] = []
        for r in hyde_results + query_results:
            if r.chunk_id not in seen_chunks:
                seen_chunks.add(r.chunk_id)
                merged.append(r)
        
        merged.sort(key=lambda x: x.similarity, reverse=True)
        results = vector_search.deduplicate_by_file(merged, max_per_file=2)[:top_k]
        
        return RAGContext(
            chunks=results,
            context_text=self._build_context_text(results),
            technique=self.id,
            metadata={"hypothetical_length": len(hypothetical)}
        )
    
    async def _generate_hypothetical(self, query: str) -> str:
        """Generate a hypothetical answer to the query."""
        prompt = f"""Generate a detailed, informative passage that would answer the following question. 
Write as if you are writing a document that contains the answer.
Do not say "I don't know" - just generate what a good answer might look like.

Question: {query}

Hypothetical document passage:"""
        
        try:
            provider = llm.get_llm_provider()
            response = await provider.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=300,
            )
            return response
        except Exception as e:
            logger.error(f"Failed to generate hypothetical: {e}")
            return query  # Fallback to original query
    
    def _build_context_text(self, results: list[SearchResult]) -> str:
        parts = []
        for i, result in enumerate(results, 1):
            header = f"[Source {i}: {result.file_title}"
            if result.section_heading:
                header += f" > {result.section_heading}"
            header += "]"
            parts.append(f"{header}\n{result.chunk_content}\n")
        return "\n".join(parts)


class MultiQueryRAG(RAGTechnique):
    """
    Multi-Query RAG: Generates multiple query variations for better coverage.
    
    1. Generate 3-5 query variations using LLM
    2. Retrieve documents for each query variation
    3. Deduplicate and merge results
    """
    
    @property
    def id(self) -> str:
        return "multi-query"
    
    @property
    def name(self) -> str:
        return "Multi-Query"
    
    @property
    def description(self) -> str:
        return "Generates multiple query variations for better coverage"
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.3,
    ) -> RAGContext:
        # Generate query variations
        variations = await self._generate_variations(query)
        variations.insert(0, query)  # Include original
        
        # Retrieve for each variation
        all_results: list[SearchResult] = []
        seen_chunks: set[int] = set()
        
        for variation in variations:
            try:
                variation_embedding = await embedding_service.embed_text(variation)
                results = vector_search.search(
                    query_embedding=variation_embedding,
                    top_k=top_k,
                    threshold=threshold,
                )
                
                for result in results:
                    if result.chunk_id not in seen_chunks:
                        all_results.append(result)
                        seen_chunks.add(result.chunk_id)
                        
            except Exception as e:
                logger.warning(f"Failed to search variation '{variation[:50]}': {e}")
        
        # Sort by similarity and deduplicate
        all_results.sort(key=lambda x: x.similarity, reverse=True)
        results = vector_search.deduplicate_by_file(all_results, max_per_file=2)[:top_k]
        
        return RAGContext(
            chunks=results,
            context_text=self._build_context_text(results),
            technique=self.id,
            metadata={
                "variations": variations,
                "total_candidates": len(all_results)
            }
        )
    
    async def _generate_variations(self, query: str, count: int = 4) -> list[str]:
        """Generate query variations."""
        prompt = f"""Generate {count} different variations of the following question.
Each variation should approach the question from a different angle or use different keywords.
Output only the variations, one per line.

Original question: {query}

Variations:"""
        
        try:
            provider = llm.get_llm_provider()
            response = await provider.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=300,
            )
            
            # Parse variations
            lines = [line.strip() for line in response.split("\n") if line.strip()]
            # Remove numbering if present
            variations = []
            for line in lines:
                # Remove common prefixes like "1.", "1)", "-", etc.
                cleaned = line.lstrip("0123456789.-) ").strip()
                if cleaned and len(cleaned) > 10:
                    variations.append(cleaned)
            
            return variations[:count]
            
        except Exception as e:
            logger.error(f"Failed to generate variations: {e}")
            return []
    
    def _build_context_text(self, results: list[SearchResult]) -> str:
        parts = []
        for i, result in enumerate(results, 1):
            header = f"[Source {i}: {result.file_title}"
            if result.section_heading:
                header += f" > {result.section_heading}"
            header += "]"
            parts.append(f"{header}\n{result.chunk_content}\n")
        return "\n".join(parts)


# Registry of available techniques
_techniques: dict[str, RAGTechnique] = {}


def register_technique(technique: RAGTechnique) -> None:
    """Register a RAG technique."""
    _techniques[technique.id] = technique


def get_technique(technique_id: str) -> RAGTechnique:
    """Get a RAG technique by ID."""
    if technique_id not in _techniques:
        raise ValueError(
            f"Unknown RAG technique: {technique_id}. "
            f"Available: {', '.join(_techniques.keys())}"
        )
    return _techniques[technique_id]


def list_techniques() -> list[dict]:
    """List all available RAG techniques."""
    return [
        {
            "id": t.id,
            "name": t.name,
            "description": t.description,
        }
        for t in _techniques.values()
    ]


# Register built-in techniques
register_technique(BasicRAG())
register_technique(HybridRAG())
register_technique(RerankRAG())
register_technique(HyDERAG())
register_technique(MultiQueryRAG())
