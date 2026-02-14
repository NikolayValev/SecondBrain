"""
LLM Service module for Second Brain.
Provides a unified interface for OpenAI, Gemini, Ollama, and Anthropic providers.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

import httpx

from app.config import Config

logger = logging.getLogger(__name__)


@dataclass
class TokenUsage:
    """Token usage information from an LLM call."""
    prompt: int = 0
    completion: int = 0
    total: int = 0


@dataclass
class ChatResult:
    """Result from a chat completion, including token usage."""
    content: str
    usage: TokenUsage = field(default_factory=TokenUsage)


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

    async def chat_with_usage(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ChatResult:
        """
        Send a chat completion and return both text and token usage.

        Default implementation calls :meth:`chat` and returns zero usage.
        Providers should override to extract real token counts.
        """
        content = await self.chat(messages, temperature, max_tokens)
        return ChatResult(content=content)

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """
        Stream a chat completion token-by-token.

        Default implementation falls back to non-streaming :meth:`chat`
        and yields the full response as one chunk.
        """
        full = await self.chat(messages, temperature, max_tokens)
        yield full
    
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
        result = await self.chat_with_usage(messages, temperature, max_tokens)
        return result.content

    async def chat_with_usage(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ChatResult:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature or self.default_temperature,
            max_tokens=max_tokens or self.default_max_tokens,
        )
        content = response.choices[0].message.content or ""
        usage = TokenUsage()
        if response.usage:
            usage = TokenUsage(
                prompt=response.usage.prompt_tokens or 0,
                completion=response.usage.completion_tokens or 0,
                total=response.usage.total_tokens or 0,
            )
        return ChatResult(content=content, usage=usage)

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature or self.default_temperature,
            max_tokens=max_tokens or self.default_max_tokens,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content
    
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
        result = await self.chat_with_usage(messages, temperature, max_tokens)
        return result.content

    async def chat_with_usage(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ChatResult:
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
                parts=[self.types.Part(text=msg["parts"][0])]
            ))
        
        # Include system prompt in the last user message if present
        if system_msg and contents:
            last_content = contents[-1]
            original_text = last_content.parts[0].text
            contents[-1] = self.types.Content(
                role=last_content.role,
                parts=[self.types.Part(text=f"{system_msg}\n\n{original_text}")]
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
        usage = TokenUsage()
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            um = response.usage_metadata
            usage = TokenUsage(
                prompt=getattr(um, "prompt_token_count", 0) or 0,
                completion=getattr(um, "candidates_token_count", 0) or 0,
                total=getattr(um, "total_token_count", 0) or 0,
            )
        return ChatResult(content=response.text, usage=usage)
    
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

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        import json as _json

        async with self.client.stream(
            "POST",
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": temperature or self.default_temperature,
                    "num_predict": max_tokens or self.default_max_tokens,
                },
            },
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                try:
                    data = _json.loads(line)
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield content
                except _json.JSONDecodeError:
                    continue
    
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


class AnthropicProvider(LLMProvider):
    """Anthropic Claude LLM provider implementation."""
    
    def __init__(self):
        from anthropic import AsyncAnthropic
        
        self.client = AsyncAnthropic(api_key=Config.ANTHROPIC_API_KEY)
        self.model = Config.ANTHROPIC_MODEL
        self.default_temperature = Config.LLM_TEMPERATURE
        self.default_max_tokens = Config.LLM_MAX_TOKENS
        self._embedding_provider: Optional[LLMProvider] = None
    
    def _get_embedding_provider(self) -> LLMProvider:
        """Get or create a fallback provider for embeddings.

        Anthropic does not offer an embedding API, so we delegate to
        another provider.  The provider is chosen from:
        1. ``EMBEDDING_PROVIDER`` env var (e.g. "openai", "gemini", "ollama")
        2. Auto-detect: first available provider that supports embeddings.
        """
        if self._embedding_provider is not None:
            return self._embedding_provider

        fallback = Config.EMBEDDING_PROVIDER.lower().strip()
        if not fallback:
            # Auto-detect: prefer openai > gemini > ollama
            for candidate in ("openai", "gemini", "ollama"):
                try:
                    provider = _providers[candidate]()
                    self._embedding_provider = provider
                    logger.info(
                        "Anthropic: using '%s' as embedding fallback (auto-detected)",
                        candidate,
                    )
                    return provider
                except Exception:
                    continue
            raise RuntimeError(
                "Anthropic does not provide an embedding API and no fallback "
                "provider could be initialised.  Set EMBEDDING_PROVIDER to "
                "'openai', 'gemini', or 'ollama' and supply the required keys."
            )

        if fallback not in _providers:
            raise ValueError(
                f"EMBEDDING_PROVIDER='{fallback}' is not a known provider. "
                f"Supported: {', '.join(_providers.keys())}"
            )
        self._embedding_provider = _providers[fallback]()
        logger.info("Anthropic: using '%s' as embedding provider", fallback)
        return self._embedding_provider
    
    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        result = await self.chat_with_usage(messages, temperature, max_tokens)
        return result.content

    async def chat_with_usage(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ChatResult:
        # Extract system message
        system_msg = None
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                chat_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        kwargs = {
            "model": self.model,
            "messages": chat_messages,
            "temperature": temperature or self.default_temperature,
            "max_tokens": max_tokens or self.default_max_tokens,
        }
        if system_msg:
            kwargs["system"] = system_msg
        
        response = await self.client.messages.create(**kwargs)
        content = response.content[0].text
        usage = TokenUsage()
        if hasattr(response, "usage") and response.usage:
            usage = TokenUsage(
                prompt=getattr(response.usage, "input_tokens", 0) or 0,
                completion=getattr(response.usage, "output_tokens", 0) or 0,
                total=(
                    (getattr(response.usage, "input_tokens", 0) or 0)
                    + (getattr(response.usage, "output_tokens", 0) or 0)
                ),
            )
        return ChatResult(content=content, usage=usage)
    
    async def embed(self, text: str) -> list[float]:
        return await self._get_embedding_provider().embed(text)
    
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return await self._get_embedding_provider().embed_batch(texts)


# Provider registry
_providers: dict[str, type[LLMProvider]] = {
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
    "ollama": OllamaProvider,
    "anthropic": AnthropicProvider,
}

# Singleton instance
_llm_instance: Optional[LLMProvider] = None

# Cache for dynamically created providers
_provider_cache: dict[str, LLMProvider] = {}


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


def get_provider_by_name(provider_name: str) -> LLMProvider:
    """
    Get a specific LLM provider instance by name.
    
    Args:
        provider_name: The provider name (openai, gemini, ollama, anthropic).
        
    Returns:
        An instance of the requested provider.
        
    Raises:
        ValueError: If the provider is not supported or not configured.
    """
    provider_name = provider_name.lower()
    
    if provider_name not in _providers:
        raise ValueError(
            f"Unsupported LLM provider: {provider_name}. "
            f"Supported: {', '.join(_providers.keys())}"
        )
    
    # Check if already cached
    if provider_name in _provider_cache:
        return _provider_cache[provider_name]
    
    # Create and cache the provider
    try:
        provider = _providers[provider_name]()
        _provider_cache[provider_name] = provider
        logger.info(f"Created LLM provider: {provider_name}")
        return provider
    except Exception as e:
        raise ValueError(f"Failed to create provider {provider_name}: {e}")


async def check_provider_availability(provider_name: str) -> dict:
    """
    Check if a provider is available and configured.
    
    Returns:
        Dict with 'available' bool and optional 'error' message.
    """
    provider_name = provider_name.lower()
    
    if provider_name not in _providers:
        return {"available": False, "error": f"Unknown provider: {provider_name}"}
    
    # Check configuration
    if provider_name == "openai":
        if not Config.OPENAI_API_KEY:
            return {"available": False, "error": "OPENAI_API_KEY not configured"}
    elif provider_name == "gemini":
        if not Config.GEMINI_API_KEY:
            return {"available": False, "error": "GEMINI_API_KEY not configured"}
    elif provider_name == "anthropic":
        if not Config.ANTHROPIC_API_KEY:
            return {"available": False, "error": "ANTHROPIC_API_KEY not configured"}
    elif provider_name == "ollama":
        # Check if Ollama is running
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{Config.OLLAMA_BASE_URL}/api/tags")
                if response.status_code != 200:
                    return {"available": False, "error": "Ollama not responding"}
        except Exception as e:
            return {"available": False, "error": f"Cannot connect to Ollama: {e}"}
    
    return {"available": True}


async def list_ollama_models() -> list[str]:
    """List available Ollama models."""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{Config.OLLAMA_BASE_URL}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
    except Exception as e:
        logger.error(f"Failed to list Ollama models: {e}")
    return []


def list_providers() -> list[str]:
    """List all registered provider names."""
    return list(_providers.keys())
