"""
Embedding service for Second Brain.
Generates and manages vector embeddings for document chunks.
"""

import json
import logging
import struct
from typing import Optional

from app.config import Config
from app.db import db
from app.chunker import chunker, Chunk
from app import llm

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Manages embedding generation and storage.
    Uses the configured LLM provider for embedding generation.
    """
    
    def __init__(self, batch_size: int = 20):
        """
        Initialize the embedding service.
        
        Args:
            batch_size: Number of chunks to embed in one batch.
        """
        self.batch_size = batch_size
        self._provider = None
    
    @property
    def provider(self):
        """Lazy-load the LLM provider."""
        if self._provider is None:
            self._provider = llm.get_llm_provider()
        return self._provider
    
    def get_model_name(self) -> str:
        """Get the embedding model name for the current provider."""
        provider_name = Config.LLM_PROVIDER.lower()
        if provider_name == "openai":
            return Config.OPENAI_EMBEDDING_MODEL
        elif provider_name == "gemini":
            return Config.GEMINI_EMBEDDING_MODEL
        elif provider_name == "ollama":
            return Config.OLLAMA_EMBEDDING_MODEL
        return "unknown"
    
    @staticmethod
    def embedding_to_bytes(embedding: list[float]) -> bytes:
        """Convert embedding list to bytes for storage."""
        return struct.pack(f'{len(embedding)}f', *embedding)
    
    @staticmethod
    def bytes_to_embedding(data: bytes) -> list[float]:
        """Convert bytes back to embedding list."""
        num_floats = len(data) // 4  # 4 bytes per float
        return list(struct.unpack(f'{num_floats}f', data))
    
    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        return await self.provider.embed(text)
    
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        return await self.provider.embed_batch(texts)
    
    async def embed_chunk(self, chunk: Chunk) -> Optional[int]:
        """
        Generate and store embedding for a single chunk.
        
        Args:
            chunk: The chunk to embed.
            
        Returns:
            The embedding ID, or None if failed.
        """
        try:
            embedding = await self.embed_text(chunk.content)
            
            embedding_bytes = self.embedding_to_bytes(embedding)
            model = self.get_model_name()
            
            embedding_id = db.add_embedding(
                chunk_id=chunk.chunk_index,  # This should be the DB chunk ID
                embedding=embedding_bytes,
                model=model,
                dimensions=len(embedding)
            )
            
            return embedding_id
            
        except Exception as e:
            logger.error(f"Failed to embed chunk: {e}")
            return None
    
    async def embed_file_chunks(self, file_id: int) -> tuple[int, int]:
        """
        Generate embeddings for all chunks of a file.
        
        Args:
            file_id: The file ID to process.
            
        Returns:
            Tuple of (successful_count, failed_count).
        """
        chunks = db.get_chunks_by_file(file_id)
        
        if not chunks:
            logger.debug(f"No chunks found for file {file_id}")
            return 0, 0
        
        success = 0
        failed = 0
        model = self.get_model_name()
        
        # Process in batches
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            texts = [c["content"] for c in batch]
            
            try:
                embeddings = await self.embed_texts(texts)
                
                for chunk_data, embedding in zip(batch, embeddings):
                    try:
                        embedding_bytes = self.embedding_to_bytes(embedding)
                        db.add_embedding(
                            chunk_id=chunk_data["id"],
                            embedding=embedding_bytes,
                            model=model,
                            dimensions=len(embedding)
                        )
                        success += 1
                    except Exception as e:
                        logger.error(f"Failed to store embedding: {e}")
                        failed += 1
                        
            except Exception as e:
                logger.error(f"Failed to generate embeddings batch: {e}")
                failed += len(batch)
        
        return success, failed
    
    async def process_pending_chunks(self, limit: int = 100) -> tuple[int, int]:
        """
        Process chunks that don't have embeddings yet.
        
        Args:
            limit: Maximum chunks to process.
            
        Returns:
            Tuple of (successful_count, failed_count).
        """
        pending = db.get_chunks_without_embeddings(limit)
        
        if not pending:
            logger.debug("No pending chunks to embed")
            return 0, 0
        
        logger.info(f"Processing {len(pending)} pending chunks")
        
        success = 0
        failed = 0
        model = self.get_model_name()
        
        # Process in batches
        for i in range(0, len(pending), self.batch_size):
            batch = pending[i:i + self.batch_size]
            texts = [c["content"] for c in batch]
            
            try:
                embeddings = await self.embed_texts(texts)
                
                for chunk_data, embedding in zip(batch, embeddings):
                    try:
                        embedding_bytes = self.embedding_to_bytes(embedding)
                        db.add_embedding(
                            chunk_id=chunk_data["id"],
                            embedding=embedding_bytes,
                            model=model,
                            dimensions=len(embedding)
                        )
                        success += 1
                    except Exception as e:
                        logger.error(f"Failed to store embedding: {e}")
                        failed += 1
                        
            except Exception as e:
                logger.error(f"Failed to generate embeddings batch: {e}")
                failed += len(batch)
        
        logger.info(f"Embedded {success} chunks, {failed} failed")
        return success, failed
    
    def create_chunks_for_file(
        self, 
        file_id: int, 
        sections: list[dict],
        clear_existing: bool = True
    ) -> list[int]:
        """
        Create chunks for a file from its sections.
        
        Args:
            file_id: The file ID.
            sections: List of section dicts with 'id', 'heading', 'content'.
            clear_existing: Whether to clear existing chunks first.
            
        Returns:
            List of created chunk IDs.
        """
        if clear_existing:
            db.clear_file_chunks(file_id)
        
        chunks = chunker.chunk_sections(sections, file_id)
        
        chunk_ids = []
        for chunk in chunks:
            chunk_id = db.add_chunk(
                file_id=file_id,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                token_count=chunk.token_count,
                section_id=chunk.section_id
            )
            chunk_ids.append(chunk_id)
        
        logger.debug(f"Created {len(chunk_ids)} chunks for file {file_id}")
        return chunk_ids


# Singleton instance
embedding_service = EmbeddingService()
