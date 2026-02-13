"""
Search service: full-text and semantic search logic.
"""

import logging
import time

from app.db import db
from app.rag_techniques import get_technique
from app.api.models.search import (
    SearchResult,
    SearchResponse,
    SemanticSearchResult,
    SemanticSearchResponse,
)

logger = logging.getLogger(__name__)


class SearchService:
    """Handles full-text and semantic search operations."""

    def fulltext_search(self, query: str, limit: int = 20) -> SearchResponse:
        """
        Full-text search using FTS5/BM25 ranking.

        Args:
            query: Search query string.
            limit: Maximum number of results.

        Returns:
            SearchResponse with ranked results.

        Raises:
            Exception: If the underlying search fails.
        """
        results = db.search(query, limit=limit)
        return SearchResponse(
            query=query,
            results=[
                SearchResult(
                    file_path=r["file_path"],
                    title=r["title"],
                    heading=r["heading"],
                    snippet=r["snippet"],
                    rank=r["rank"],
                )
                for r in results
            ],
            count=len(results),
        )

    async def semantic_search(
        self,
        query: str,
        limit: int = 10,
        rag_technique: str = "basic",
    ) -> SemanticSearchResponse:
        """
        Semantic search using vector embeddings.

        Args:
            query: Natural language query.
            limit: Maximum results to return.
            rag_technique: Which retrieval technique to use.

        Returns:
            SemanticSearchResponse with scored results.

        Raises:
            ValueError: If the technique is unknown.
        """
        technique = get_technique(rag_technique)

        embed_start = time.time()

        rag_context = await technique.retrieve(
            query=query,
            top_k=limit,
            threshold=0.3,
        )

        embed_time = (time.time() - embed_start) * 1000

        results = []
        for chunk in rag_context.chunks:
            file_record = db.get_file_by_path(chunk.file_path)
            metadata: dict = {}
            if file_record:
                with db.cursor() as cur:
                    cur.execute(
                        """
                        SELECT t.name FROM tags t
                        JOIN file_tags ft ON t.id = ft.tag_id
                        WHERE ft.file_id = ?
                        """,
                        (chunk.file_id,),
                    )
                    tags = [row["name"] for row in cur.fetchall()]
                    metadata["tags"] = tags
                    metadata["created_at"] = file_record.get("created_at")
                    metadata["updated_at"] = file_record.get("updated_at")

            results.append(SemanticSearchResult(
                path=chunk.file_path,
                title=chunk.file_title,
                snippet=(
                    chunk.chunk_content[:300] + "..."
                    if len(chunk.chunk_content) > 300
                    else chunk.chunk_content
                ),
                score=round(chunk.similarity, 4),
                metadata=metadata,
            ))

        return SemanticSearchResponse(
            results=results,
            query_embedding_time_ms=round(embed_time, 2),
            search_time_ms=round(embed_time, 2),
        )


# Singleton
search_service = SearchService()
