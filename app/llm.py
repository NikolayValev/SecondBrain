"""
LLM Service module for Second Brain.
Provides a unified interface for OpenAI, Gemini, and Ollama providers.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

import httpx

from app.config import Config

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Send a chat completion request.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            temperature: Override default temperature.
            max_tokens: Override default max tokens.
            
        Returns:
            The assistant's response text.
        """
        pass
    
    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """
        Generate embeddings for the given text.
        
        Args:
            text: The text to embed.
            
        Returns:
            List of floats representing the embedding vector.
        """
        pass
    
    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            List of embedding vectors.
        """
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider implementation."""
    
    def __init__(self):
        from openai import AsyncOpenAI
        
        self.client = AsyncOpenAI(
            api_key=Config.OPENAI_API_KEY,
            base_url=Config.OPENAI_BASE_URL,
        )
        self.model = Config.OPENAI_MODEL
        self.embedding_model = Config.OPENAI_EMBEDDING_MODEL
        self.default_temperature = Config.LLM_TEMPERATURE
        self.default_max_tokens = Config.LLM_MAX_TOKENS
    
    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature or self.default_temperature,
            max_tokens=max_tokens or self.default_max_tokens,
        )
        return response.choices[0].message.content or ""
    
    async def embed(self, text: str) -> list[float]:
        response = await self.client.embeddings.create(
            model=self.embedding_model,
            input=text,
        )
        return response.data[0].embedding
    
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = await self.client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        return [item.embedding for item in response.data]


class GeminiProvider(LLMProvider):
    """Google Gemini LLM provider implementation."""
    
    def __init__(self):
        from google import genai
        from google.genai import types
        
        self.client = genai.Client(api_key=Config.GEMINI_API_KEY)
        self.types = types
        self.model_name = Config.GEMINI_MODEL
        self.embedding_model = Config.GEMINI_EMBEDDING_MODEL
        self.default_temperature = Config.LLM_TEMPERATURE
        self.default_max_tokens = Config.LLM_MAX_TOKENS
    
    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        # Convert messages to Gemini format
        gemini_messages = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            if msg["role"] == "system":
                # Gemini doesn't have system role, prepend to first user message
                continue
            gemini_messages.append({
                "role": role,
                "parts": [msg["content"]]
            })
        
        # Handle system message
        system_msg = next((m["content"] for m in messages if m["role"] == "system"), None)
        
        # Build contents for the API
        contents = []
        for msg in gemini_messages:
            contents.append(self.types.Content(
                role=msg["role"],
                parts=[self.types.Part.from_text(msg["parts"][0])]
            ))
        
        # Include system prompt in the last user message if present
        if system_msg and contents:
            last_content = contents[-1]
            original_text = last_content.parts[0].text
            contents[-1] = self.types.Content(
                role=last_content.role,
                parts=[self.types.Part.from_text(f"{system_msg}\n\n{original_text}")]
            )
        
        config = self.types.GenerateContentConfig(
            temperature=temperature or self.default_temperature,
            max_output_tokens=max_tokens or self.default_max_tokens,
        )
        
        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config,
        )
        return response.text
    
    async def embed(self, text: str) -> list[float]:
        response = await self.client.aio.models.embed_content(
            model=self.embedding_model,
            contents=text,
        )
        return response.embeddings[0].values
    
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = await self.client.aio.models.embed_content(
            model=self.embedding_model,
            contents=texts,
        )
        return [emb.values for emb in response.embeddings]


class OllamaProvider(LLMProvider):
    """Ollama local LLM provider implementation."""
    
    def __init__(self):
        self.base_url = Config.OLLAMA_BASE_URL
        self.model = Config.OLLAMA_MODEL
        self.embedding_model = Config.OLLAMA_EMBEDDING_MODEL
        self.default_temperature = Config.LLM_TEMPERATURE
        self.default_max_tokens = Config.LLM_MAX_TOKENS
        self.client = httpx.AsyncClient(timeout=120.0)
    
    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        response = await self.client.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature or self.default_temperature,
                    "num_predict": max_tokens or self.default_max_tokens,
                },
            },
        )
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "")
    
    async def embed(self, text: str) -> list[float]:
        response = await self.client.post(
            f"{self.base_url}/api/embeddings",
            json={
                "model": self.embedding_model,
                "prompt": text,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data.get("embedding", [])
    
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            embedding = await self.embed(text)
            embeddings.append(embedding)
        return embeddings
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


# Provider registry
_providers: dict[str, type[LLMProvider]] = {
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
    "ollama": OllamaProvider,
}

# Singleton instance
_llm_instance: Optional[LLMProvider] = None


def get_llm_provider() -> LLMProvider:
    """
    Get the configured LLM provider instance (singleton).
    
    Returns:
        The LLM provider instance based on LLM_PROVIDER config.
        
    Raises:
        ValueError: If the provider is not supported.
    """
    global _llm_instance
    
    if _llm_instance is None:
        provider_name = Config.LLM_PROVIDER.lower()
        
        if provider_name not in _providers:
            raise ValueError(
                f"Unsupported LLM provider: {provider_name}. "
                f"Supported: {', '.join(_providers.keys())}"
            )
        
        Config.validate_llm_config()
        _llm_instance = _providers[provider_name]()
        logger.info(f"Initialized LLM provider: {provider_name}")
    
    return _llm_instance


async def chat(
    messages: list[dict[str, str]],
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """
    Convenience function for chat completion.
    
    Args:
        messages: List of message dicts with 'role' and 'content' keys.
        temperature: Override default temperature.
        max_tokens: Override default max tokens.
        
    Returns:
        The assistant's response text.
    """
    provider = get_llm_provider()
    return await provider.chat(messages, temperature, max_tokens)


async def embed(text: str) -> list[float]:
    """
    Convenience function for generating embeddings.
    
    Args:
        text: The text to embed.
        
    Returns:
        The embedding vector.
    """
    provider = get_llm_provider()
    return await provider.embed(text)


async def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Convenience function for batch embeddings.
    
    Args:
        texts: List of texts to embed.
        
    Returns:
        List of embedding vectors.
    """
    provider = get_llm_provider()
    return await provider.embed_batch(texts)
