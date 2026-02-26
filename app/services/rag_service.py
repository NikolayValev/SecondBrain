"""
RAG service (API layer): ask questions, manage embeddings.

This wraps the lower-level app.rag / app.embeddings modules and adds
provider selection, prompt building, and source formatting.
"""

import json as _json
import logging
from typing import AsyncIterator, Optional

from app import llm
from app.api.models.rag import (
    AskRequest,
    AskResponse,
    EmbeddingStatsResponse,
    Source,
    TokenUsage,
)
from app.config import Config
from app.db import db
from app.embeddings import embedding_service
from app.rag_techniques import get_technique

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
        conversation_id = self._resolve_or_create_conversation_id(
            request.conversation_id,
            title=request.question[:80],
        )

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
                conversation_id=str(conversation_id),
                model_used=model_used,
            )

        provider = llm.get_provider_by_name(request.provider)
        if request.system_prompt and request.system_prompt.strip():
            self._save_system_prompt(conversation_id, request.system_prompt.strip())
        system_prompt = self._resolve_system_prompt(conversation_id)

        user_message = (
            f"Context from knowledge base:\n---\n{rag_context.context_text}\n---\n\n"
            f"Question: {request.question}\n\nAnswer based on the context above:"
        )

        # Build messages and include conversation history when available.
        messages = [
            {"role": "system", "content": system_prompt},
        ]

        history = self._get_history(conversation_id)
        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": user_message})

        chat_result = await provider.chat_with_usage(
            messages=messages,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            top_k=request.top_k,
        )

        sources = self._build_sources(rag_context, include=request.include_sources)

        self._save_exchange(
            conversation_id=conversation_id,
            question=request.question,
            answer=chat_result.content,
            sources=sources,
        )

        return AskResponse(
            answer=chat_result.content,
            sources=sources,
            conversation_id=str(conversation_id),
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
        * ``{"type":"source", ...}``  - one per source (sent first)
        * ``{"type":"token", "content":"..."}``  - streamed tokens
        * ``{"type":"done", "conversation_id":"..."}`` - final event
        """
        conversation_id = self._resolve_or_create_conversation_id(
            request.conversation_id,
            title=request.question[:80],
        )

        technique = get_technique(request.rag_technique)

        rag_context = await technique.retrieve(
            query=request.question,
            top_k=5,
            threshold=0.3,
        )

        model_used = self._resolve_model(request)

        if not rag_context.chunks:
            yield self._sse({"type": "token", "content": "I couldn't find any relevant information in your knowledge base to answer this question."})
            yield self._sse({"type": "done", "conversation_id": str(conversation_id), "model_used": model_used})
            return

        # Send sources first.
        sources = self._build_sources(rag_context, include=request.include_sources)
        for src in sources:
            yield self._sse({"type": "source", **src.model_dump()})

        provider = llm.get_provider_by_name(request.provider)
        if request.system_prompt and request.system_prompt.strip():
            self._save_system_prompt(conversation_id, request.system_prompt.strip())
        system_prompt = self._resolve_system_prompt(conversation_id)
        user_message = (
            f"Context from knowledge base:\n---\n{rag_context.context_text}\n---\n\n"
            f"Question: {request.question}\n\nAnswer based on the context above:"
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
        ]
        history = self._get_history(conversation_id)
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        full_answer_parts: list[str] = []
        async for token in provider.stream_chat(
            messages=messages,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            top_k=request.top_k,
        ):
            full_answer_parts.append(token)
            yield self._sse({"type": "token", "content": token})

        full_answer = "".join(full_answer_parts)

        self._save_exchange(
            conversation_id=conversation_id,
            question=request.question,
            answer=full_answer,
            sources=sources,
        )

        yield self._sse({"type": "done", "conversation_id": str(conversation_id), "model_used": model_used})

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
        stats = db.get_embedding_stats()
        if stats["chunk_count"] == 0:
            logger.info("No chunks found - auto-generating from indexed files")
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
        for file_record in all_files:
            file_id = file_record["id"]
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

    def _resolve_or_create_conversation_id(
        self,
        requested_conversation_id: Optional[str],
        *,
        title: str,
    ) -> int:
        """
        Resolve request conversation identifier to an existing SQLite conversation id.

        Rules:
        - Missing id: create a new conversation.
        - Numeric id: use it when it exists; otherwise create a new conversation.
        - Non-numeric id: treat as a legacy/session key and map by session_id.
        """
        if not requested_conversation_id:
            return db.create_conversation(title=title)

        candidate = requested_conversation_id.strip()
        if not candidate:
            return db.create_conversation(title=title)

        try:
            conversation_id = int(candidate)
        except (ValueError, TypeError):
            existing = db.get_recent_conversations(session_id=candidate, limit=1)
            if existing:
                return int(existing[0]["id"])
            return db.create_conversation(session_id=candidate, title=title)

        if db.get_conversation(conversation_id) is not None:
            return conversation_id
        return db.create_conversation(title=title)

    def _resolve_system_prompt(self, conversation_id: Optional[int]) -> str:
        """Resolve the effective system prompt for a conversation."""
        custom_prompt = self._get_latest_system_prompt(conversation_id)
        if not custom_prompt:
            return _SYSTEM_PROMPT
        return f"{_SYSTEM_PROMPT}\n\nConversation-specific instructions:\n{custom_prompt}"

    def _get_latest_system_prompt(self, conversation_id: Optional[int]) -> Optional[str]:
        """Return the latest non-empty system prompt stored for a conversation."""
        if not conversation_id:
            return None
        messages = db.get_conversation_messages(conversation_id, limit=50)
        for message in reversed(messages):
            if message.get("role") != "system":
                continue
            content = str(message.get("content", "")).strip()
            if content:
                return content
        return None

    def _save_system_prompt(self, conversation_id: int, system_prompt: str) -> None:
        """Persist a conversation system prompt as a dedicated system message."""
        prompt = system_prompt.strip()
        if not prompt:
            return
        latest = self._get_latest_system_prompt(conversation_id)
        if latest == prompt:
            return
        db.add_message(conversation_id=conversation_id, role="system", content=prompt)

    @staticmethod
    def _get_history(conversation_id: Optional[int]) -> list[dict[str, str]]:
        """Fetch prior messages for the conversation from SQLite."""
        if not conversation_id:
            return []

        messages = db.get_conversation_messages(conversation_id, limit=10)
        return [
            {"role": message["role"], "content": message["content"]}
            for message in messages
            if message["role"] in ("user", "assistant")
        ]

    @staticmethod
    def _save_exchange(
        conversation_id: int,
        question: str,
        answer: str,
        sources: list[Source],
    ) -> None:
        """Persist the user question and assistant answer in SQLite."""
        cid = conversation_id
        if db.get_conversation(cid) is None:
            cid = db.create_conversation(title=question[:80])

        db.add_message(conversation_id=cid, role="user", content=question)
        source_dicts = [source.model_dump() for source in sources] if sources else None
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
            if chunk.file_path in seen_files:
                continue

            snippet = chunk.chunk_content
            if len(snippet) > 200:
                snippet = snippet[:200] + "..."

            sources.append(
                Source(
                    path=chunk.file_path,
                    title=chunk.file_title,
                    snippet=snippet,
                    score=round(chunk.similarity, 3),
                )
            )
            seen_files.add(chunk.file_path)

        return sources


# Singleton
rag_api_service = RAGAPIService()
