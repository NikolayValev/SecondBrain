"""
RAG service (API layer): ask questions, manage embeddings.

This wraps the lower-level app.rag / app.embeddings modules and adds
provider selection, prompt building, and source formatting.
"""

import json as _json
import logging
import uuid
from typing import AsyncIterator, Optional

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
- Use file paths to understand context (e.g. folder structure reveals projects, categories, and relationships)
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

        # Build messages — include conversation history when available
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
        ]

        history = self._get_history(request.conversation_id)
        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": user_message})

        chat_result = await provider.chat_with_usage(messages=messages)

        sources = self._build_sources(rag_context, include=request.include_sources)
        conversation_id = request.conversation_id or str(uuid.uuid4())

        # Persist the exchange
        self._save_exchange(
            conversation_id=conversation_id,
            question=request.question,
            answer=chat_result.content,
            sources=sources,
        )

        return AskResponse(
            answer=chat_result.content,
            sources=sources,
            conversation_id=conversation_id,
            model_used=model_used,
            tokens_used=TokenUsage(
                prompt=chat_result.usage.prompt,
                completion=chat_result.usage.completion,
                total=chat_result.usage.total,
            ),
        )

    async def ask_stream(self, request: AskRequest) -> AsyncIterator[str]:
        """
        Stream an answer as Server-Sent Events (SSE).

        Yields ``data: <json>\n\n`` strings suitable for an SSE response.
        Events:
        * ``{"type":"source", ...}``  – one per source (sent first)
        * ``{"type":"token", "content":"..."}``  – streamed tokens
        * ``{"type":"done", "conversation_id":"..."}`` – final event
        """
        technique = get_technique(request.rag_technique)

        rag_context = await technique.retrieve(
            query=request.question,
            top_k=5,
            threshold=0.3,
        )

        model_used = self._resolve_model(request)
        conversation_id = request.conversation_id or str(uuid.uuid4())

        if not rag_context.chunks:
            yield self._sse({"type": "token", "content": "I couldn't find any relevant information in your knowledge base to answer this question."})
            yield self._sse({"type": "done", "conversation_id": conversation_id, "model_used": model_used})
            return

        # Send sources first
        sources = self._build_sources(rag_context, include=request.include_sources)
        for src in sources:
            yield self._sse({"type": "source", **src.model_dump()})

        # Build messages
        provider = llm.get_provider_by_name(request.provider)
        user_message = (
            f"Context from knowledge base:\n---\n{rag_context.context_text}\n---\n\n"
            f"Question: {request.question}\n\nAnswer based on the context above:"
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
        ]
        history = self._get_history(request.conversation_id)
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        # Stream tokens
        full_answer_parts: list[str] = []
        async for token in provider.stream_chat(messages=messages):
            full_answer_parts.append(token)
            yield self._sse({"type": "token", "content": token})

        full_answer = "".join(full_answer_parts)

        # Persist
        self._save_exchange(
            conversation_id=conversation_id,
            question=request.question,
            answer=full_answer,
            sources=sources,
        )

        yield self._sse({"type": "done", "conversation_id": conversation_id, "model_used": model_used})

    @staticmethod
    def _sse(data: dict) -> str:
        """Format a dict as an SSE data line."""
        return f"data: {_json.dumps(data)}\n\n"

    async def generate_embeddings(self, limit: int = 100) -> dict:
        """
        Process pending chunks and generate embeddings.

        If no chunks exist yet, auto-generate them from indexed files first.

        Returns:
            Dict with processed / failed / pending counts.
        """
        # Auto-create chunks if none exist yet
        stats = db.get_embedding_stats()
        if stats["chunk_count"] == 0:
            logger.info("No chunks found — auto-generating from indexed files")
            self._create_chunks_for_all_files()

        success, failed = await embedding_service.process_pending_chunks(limit=limit)
        stats = db.get_embedding_stats()
        return {
            "status": "completed",
            "processed": success,
            "failed": failed,
            "pending_remaining": stats["pending_chunks"],
        }

    @staticmethod
    def _create_chunks_for_all_files() -> int:
        """Create chunks for every indexed file that has no chunks yet."""
        all_files = db.get_all_files()
        total_chunks = 0
        for f in all_files:
            file_id = f["id"]
            existing = db.get_chunks_by_file(file_id)
            if existing:
                continue
            sections = db.get_sections_by_file(file_id)
            if sections:
                chunk_ids = embedding_service.create_chunks_for_file(file_id, sections)
                total_chunks += len(chunk_ids)
        logger.info("Auto-created %d chunks across %d files", total_chunks, len(all_files))
        return total_chunks

    def get_embedding_stats(self) -> EmbeddingStatsResponse:
        """Return embedding statistics."""
        stats = db.get_embedding_stats()
        return EmbeddingStatsResponse(**stats)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_history(conversation_id: Optional[str]) -> list[dict[str, str]]:
        """Fetch prior messages for the conversation from SQLite."""
        if not conversation_id:
            return []
        try:
            cid = int(conversation_id)
        except (ValueError, TypeError):
            return []

        messages = db.get_conversation_messages(cid, limit=10)
        # Return only role + content (strip sources / ids)
        return [
            {"role": m["role"], "content": m["content"]}
            for m in messages
            if m["role"] in ("user", "assistant")
        ]

    @staticmethod
    def _save_exchange(
        conversation_id: str,
        question: str,
        answer: str,
        sources: list[Source],
    ) -> None:
        """Persist the user question and assistant answer in SQLite."""
        try:
            cid = int(conversation_id)
        except (ValueError, TypeError):
            # First time — create conversation with generated id
            cid = db.create_conversation(title=question[:80])
        else:
            # Ensure the conversation row exists
            if db.get_conversation(cid) is None:
                cid = db.create_conversation(title=question[:80])

        db.add_message(conversation_id=cid, role="user", content=question)
        source_dicts = [s.model_dump() for s in sources] if sources else None
        db.add_message(
            conversation_id=cid,
            role="assistant",
            content=answer,
            sources=source_dicts,
        )

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
