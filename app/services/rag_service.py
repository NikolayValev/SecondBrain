"""
RAG service (API layer): ask questions, manage embeddings.

This wraps the lower-level app.rag / app.embeddings modules and adds
provider selection, prompt building, and source formatting.
"""

import logging
import uuid

from app.config import Config
from app.db import db
from app.embeddings import embedding_service
from app.rag_techniques import get_technique
from app import llm
from app.api.models.rag import (
    AskRequest,
    AskResponse,
    Source,
    TokenUsage,
    EmbeddingStatsResponse,
)

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the user's personal knowledge base (Obsidian vault).

Use the provided context from the knowledge base to answer the question. If the context doesn't contain relevant information, say so honestly.

Guidelines:
- Be concise but thorough
- Reference specific notes when relevant
- If information is incomplete, acknowledge it
- Don't make up information not in the context"""


class RAGAPIService:
    """Orchestrates RAG Q&A from the API layer."""

    async def ask(self, request: AskRequest) -> AskResponse:
        """
        Answer a question using Retrieval-Augmented Generation.

        Args:
            request: The ask request with question, provider, etc.

        Returns:
            AskResponse with answer, sources, and metadata.

        Raises:
            ValueError: For bad provider / technique names.
        """
        technique = get_technique(request.rag_technique)

        rag_context = await technique.retrieve(
            query=request.question,
            top_k=5,
            threshold=0.3,
        )

        model_used = self._resolve_model(request)

        if not rag_context.chunks:
            return AskResponse(
                answer="I couldn't find any relevant information in your knowledge base to answer this question.",
                sources=[],
                conversation_id=request.conversation_id or str(uuid.uuid4()),
                model_used=model_used,
            )

        provider = llm.get_provider_by_name(request.provider)

        user_message = (
            f"Context from knowledge base:\n---\n{rag_context.context_text}\n---\n\n"
            f"Question: {request.question}\n\nAnswer based on the context above:"
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        answer = await provider.chat(messages=messages)

        sources = self._build_sources(rag_context, include=request.include_sources)
        conversation_id = request.conversation_id or str(uuid.uuid4())

        return AskResponse(
            answer=answer,
            sources=sources,
            conversation_id=conversation_id,
            model_used=model_used,
            tokens_used=TokenUsage(prompt=0, completion=0, total=0),
        )

    async def generate_embeddings(self, limit: int = 100) -> dict:
        """
        Process pending chunks and generate embeddings.

        Returns:
            Dict with processed / failed / pending counts.
        """
        success, failed = await embedding_service.process_pending_chunks(limit=limit)
        stats = db.get_embedding_stats()
        return {
            "status": "completed",
            "processed": success,
            "failed": failed,
            "pending_remaining": stats["pending_chunks"],
        }

    def get_embedding_stats(self) -> EmbeddingStatsResponse:
        """Return embedding statistics."""
        stats = db.get_embedding_stats()
        return EmbeddingStatsResponse(**stats)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_model(request: AskRequest) -> str:
        """Pick the model name from the request or provider defaults."""
        if request.model:
            return request.model
        return {
            "openai": Config.OPENAI_MODEL,
            "gemini": Config.GEMINI_MODEL,
            "ollama": Config.OLLAMA_MODEL,
            "anthropic": Config.ANTHROPIC_MODEL,
        }.get(request.provider.lower(), Config.GEMINI_MODEL)

    @staticmethod
    def _build_sources(rag_context, *, include: bool) -> list[Source]:
        """Deduplicate chunks into a list of Source models."""
        if not include:
            return []
        sources: list[Source] = []
        seen_files: set[str] = set()
        for chunk in rag_context.chunks:
            if chunk.file_path not in seen_files:
                sources.append(Source(
                    path=chunk.file_path,
                    title=chunk.file_title,
                    snippet=(
                        chunk.chunk_content[:200] + "..."
                        if len(chunk.chunk_content) > 200
                        else chunk.chunk_content
                    ),
                    score=round(chunk.similarity, 3),
                ))
                seen_files.add(chunk.file_path)
        return sources


# Singleton
rag_api_service = RAGAPIService()
