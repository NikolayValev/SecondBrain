"""
RAG (Retrieval-Augmented Generation) service for Second Brain.
Orchestrates retrieval and LLM generation for Q&A.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from app.config import Config
from app.embeddings import embedding_service
from app.vector_search import vector_search, SearchResult
from app import llm

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Response from the RAG pipeline."""
    answer: str
    sources: list[dict]
    query: str
    context_used: str


class RAGService:
    """
    Retrieval-Augmented Generation service.
    
    Pipeline:
    1. Embed the user's question
    2. Search for relevant chunks
    3. Build context from retrieved chunks
    4. Generate answer using LLM with context
    """
    
    DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the user's personal knowledge base (Obsidian vault).

Use the provided context from the knowledge base to answer the question. If the context doesn't contain relevant information, say so honestly.

Guidelines:
- Be concise but thorough
- Reference specific notes when relevant
- If information is incomplete, acknowledge it
- Don't make up information not in the context"""

    def __init__(
        self,
        top_k: int = 5,
        max_context_tokens: int = 3000,
        similarity_threshold: float = 0.3,
    ):
        """
        Initialize the RAG service.
        
        Args:
            top_k: Number of chunks to retrieve.
            max_context_tokens: Maximum tokens for context.
            similarity_threshold: Minimum similarity for retrieval.
        """
        self.top_k = top_k
        self.max_context_tokens = max_context_tokens
        self.similarity_threshold = similarity_threshold
    
    async def ask(
        self,
        question: str,
        system_prompt: Optional[str] = None,
        include_sources: bool = True,
    ) -> RAGResponse:
        """
        Answer a question using RAG.
        
        Args:
            question: The user's question.
            system_prompt: Optional custom system prompt.
            include_sources: Whether to include source references.
            
        Returns:
            RAGResponse with answer and sources.
        """
        logger.info(f"RAG query: {question[:100]}...")
        
        # Step 1: Embed the question
        query_embedding = await embedding_service.embed_text(question)
        
        # Step 2: Search for relevant chunks
        search_results = vector_search.search(
            query_embedding=query_embedding,
            top_k=self.top_k * 2,  # Get more, then filter
            threshold=self.similarity_threshold,
        )
        
        if not search_results:
            logger.warning("No relevant chunks found")
            return RAGResponse(
                answer="I couldn't find any relevant information in your knowledge base to answer this question.",
                sources=[],
                query=question,
                context_used="",
            )
        
        # Deduplicate by file
        search_results = vector_search.deduplicate_by_file(
            search_results, 
            max_per_file=2
        )[:self.top_k]
        
        # Step 3: Build context
        context = self._build_context(search_results)
        
        # Step 4: Generate answer
        answer = await self._generate_answer(
            question=question,
            context=context,
            system_prompt=system_prompt or self.DEFAULT_SYSTEM_PROMPT,
        )
        
        # Build sources list
        sources = []
        if include_sources:
            seen_files = set()
            for result in search_results:
                if result.file_path not in seen_files:
                    sources.append({
                        "file_path": result.file_path,
                        "file_title": result.file_title,
                        "section": result.section_heading,
                        "similarity": round(result.similarity, 3),
                    })
                    seen_files.add(result.file_path)
        
        return RAGResponse(
            answer=answer,
            sources=sources,
            query=question,
            context_used=context,
        )
    
    def _build_context(self, results: list[SearchResult]) -> str:
        """
        Build context string from search results.
        
        Args:
            results: Search results to include.
            
        Returns:
            Formatted context string.
        """
        context_parts = []
        estimated_tokens = 0
        chars_per_token = 4
        
        for i, result in enumerate(results, 1):
            # Build chunk context
            header = f"[Source {i}: {result.file_title}"
            if result.section_heading:
                header += f" > {result.section_heading}"
            header += "]"
            
            chunk_text = f"{header}\n{result.chunk_content}\n"
            chunk_tokens = len(chunk_text) // chars_per_token
            
            if estimated_tokens + chunk_tokens > self.max_context_tokens:
                # Truncate if we're over limit
                remaining_tokens = self.max_context_tokens - estimated_tokens
                if remaining_tokens > 100:
                    truncated_len = remaining_tokens * chars_per_token
                    chunk_text = chunk_text[:truncated_len] + "...\n"
                    context_parts.append(chunk_text)
                break
            
            context_parts.append(chunk_text)
            estimated_tokens += chunk_tokens
        
        return "\n".join(context_parts)
    
    async def _generate_answer(
        self,
        question: str,
        context: str,
        system_prompt: str,
    ) -> str:
        """
        Generate answer using LLM.
        
        Args:
            question: The user's question.
            context: Retrieved context.
            system_prompt: System instructions.
            
        Returns:
            Generated answer.
        """
        user_message = f"""Context from knowledge base:
---
{context}
---

Question: {question}

Answer based on the context above:"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        
        try:
            answer = await llm.chat(
                messages=messages,
                temperature=Config.LLM_TEMPERATURE,
                max_tokens=Config.LLM_MAX_TOKENS,
            )
            return answer
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise
    
    async def ask_with_history(
        self,
        question: str,
        history: list[dict],
        system_prompt: Optional[str] = None,
    ) -> RAGResponse:
        """
        Answer a question with conversation history.
        
        Args:
            question: Current question.
            history: List of previous messages.
            system_prompt: Optional custom system prompt.
            
        Returns:
            RAGResponse with answer.
        """
        # Combine question with recent history for better retrieval
        history_context = ""
        if history:
            recent = history[-4:]  # Last 2 exchanges
            history_context = " ".join(
                m["content"] for m in recent if m["role"] == "user"
            )
        
        search_query = f"{history_context} {question}".strip()
        
        # Get relevant chunks
        query_embedding = await embedding_service.embed_text(search_query)
        search_results = vector_search.search(
            query_embedding=query_embedding,
            top_k=self.top_k * 2,
            threshold=self.similarity_threshold,
        )
        
        search_results = vector_search.deduplicate_by_file(
            search_results, 
            max_per_file=2
        )[:self.top_k]
        
        context = self._build_context(search_results)
        
        # Build messages with history
        messages = [
            {"role": "system", "content": system_prompt or self.DEFAULT_SYSTEM_PROMPT},
        ]
        
        # Add history
        for msg in history:
            messages.append(msg)
        
        # Add current question with context
        user_message = f"""Context from knowledge base:
---
{context}
---

Question: {question}

Answer based on the context above:"""
        
        messages.append({"role": "user", "content": user_message})
        
        try:
            answer = await llm.chat(
                messages=messages,
                temperature=Config.LLM_TEMPERATURE,
                max_tokens=Config.LLM_MAX_TOKENS,
            )
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise
        
        sources = [
            {
                "file_path": r.file_path,
                "file_title": r.file_title,
                "section": r.section_heading,
                "similarity": round(r.similarity, 3),
            }
            for r in search_results
        ]
        
        return RAGResponse(
            answer=answer,
            sources=sources,
            query=question,
            context_used=context,
        )


# Singleton instance
rag_service = RAGService()
