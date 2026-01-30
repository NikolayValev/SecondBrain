"""
Tests for LLM configuration and service module.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import os


class TestLLMConfig:
    """Tests for LLM configuration validation."""
    
    def test_validate_llm_config_openai_missing_key(self, monkeypatch):
        """Test validation fails when OpenAI key is missing."""
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "")
        
        # Reload config to pick up env changes
        from app import config
        config.Config.LLM_PROVIDER = "openai"
        config.Config.OPENAI_API_KEY = ""
        
        with pytest.raises(ValueError, match="OPENAI_API_KEY must be set"):
            config.Config.validate_llm_config()
    
    def test_validate_llm_config_gemini_missing_key(self, monkeypatch):
        """Test validation fails when Gemini key is missing."""
        from app import config
        config.Config.LLM_PROVIDER = "gemini"
        config.Config.GEMINI_API_KEY = ""
        
        with pytest.raises(ValueError, match="GEMINI_API_KEY must be set"):
            config.Config.validate_llm_config()
    
    def test_validate_llm_config_ollama_no_key_needed(self, monkeypatch):
        """Test validation passes for Ollama without API key."""
        from app import config
        config.Config.LLM_PROVIDER = "ollama"
        
        # Should not raise
        config.Config.validate_llm_config()
    
    def test_validate_llm_config_invalid_provider(self, monkeypatch):
        """Test validation fails for unknown provider."""
        from app import config
        config.Config.LLM_PROVIDER = "unknown_provider"
        
        with pytest.raises(ValueError, match="LLM_PROVIDER must be one of"):
            config.Config.validate_llm_config()
    
    def test_get_llm_config_openai(self, monkeypatch):
        """Test get_llm_config returns correct OpenAI config."""
        from app import config
        config.Config.LLM_PROVIDER = "openai"
        config.Config.OPENAI_API_KEY = "test-key"
        config.Config.OPENAI_MODEL = "gpt-4o"
        config.Config.OPENAI_BASE_URL = "https://api.openai.com/v1"
        config.Config.OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
        config.Config.LLM_TEMPERATURE = 0.7
        config.Config.LLM_MAX_TOKENS = 4096
        
        result = config.Config.get_llm_config()
        
        assert result["provider"] == "openai"
        assert result["api_key"] == "test-key"
        assert result["model"] == "gpt-4o"
        assert result["base_url"] == "https://api.openai.com/v1"
        assert result["embedding_model"] == "text-embedding-3-small"
        assert result["temperature"] == 0.7
        assert result["max_tokens"] == 4096
    
    def test_get_llm_config_gemini(self, monkeypatch):
        """Test get_llm_config returns correct Gemini config."""
        from app import config
        config.Config.LLM_PROVIDER = "gemini"
        config.Config.GEMINI_API_KEY = "gemini-test-key"
        config.Config.GEMINI_MODEL = "gemini-1.5-pro"
        config.Config.GEMINI_EMBEDDING_MODEL = "text-embedding-004"
        
        result = config.Config.get_llm_config()
        
        assert result["provider"] == "gemini"
        assert result["api_key"] == "gemini-test-key"
        assert result["model"] == "gemini-1.5-pro"
        assert result["embedding_model"] == "text-embedding-004"
    
    def test_get_llm_config_ollama(self, monkeypatch):
        """Test get_llm_config returns correct Ollama config."""
        from app import config
        config.Config.LLM_PROVIDER = "ollama"
        config.Config.OLLAMA_BASE_URL = "http://localhost:11434"
        config.Config.OLLAMA_MODEL = "llama3"
        config.Config.OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
        
        result = config.Config.get_llm_config()
        
        assert result["provider"] == "ollama"
        assert result["base_url"] == "http://localhost:11434"
        assert result["model"] == "llama3"
        assert result["embedding_model"] == "nomic-embed-text"
        assert "api_key" not in result


class TestLLMProviders:
    """Tests for LLM provider implementations."""
    
    @pytest.fixture(autouse=True)
    def reset_llm_singleton(self):
        """Reset the LLM singleton before each test."""
        from app import llm
        llm._llm_instance = None
        yield
        llm._llm_instance = None
    
    @pytest.mark.asyncio
    async def test_openai_provider_chat(self, monkeypatch):
        """Test OpenAI provider chat method."""
        from app import config
        config.Config.LLM_PROVIDER = "openai"
        config.Config.OPENAI_API_KEY = "test-key"
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello, I'm an AI assistant."
        
        with patch("openai.AsyncOpenAI") as mock_client_class:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client
            
            from app.llm import OpenAIProvider
            provider = OpenAIProvider()
            
            result = await provider.chat([
                {"role": "user", "content": "Hello"}
            ])
            
            assert result == "Hello, I'm an AI assistant."
            mock_client.chat.completions.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_openai_provider_embed(self, monkeypatch):
        """Test OpenAI provider embed method."""
        from app import config
        config.Config.LLM_PROVIDER = "openai"
        config.Config.OPENAI_API_KEY = "test-key"
        
        mock_response = MagicMock()
        mock_response.data = [MagicMock()]
        mock_response.data[0].embedding = [0.1, 0.2, 0.3]
        
        with patch("openai.AsyncOpenAI") as mock_client_class:
            mock_client = MagicMock()
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client
            
            from app.llm import OpenAIProvider
            provider = OpenAIProvider()
            
            result = await provider.embed("test text")
            
            assert result == [0.1, 0.2, 0.3]
    
    @pytest.mark.asyncio
    async def test_ollama_provider_chat(self, monkeypatch):
        """Test Ollama provider chat method."""
        from app import config
        config.Config.LLM_PROVIDER = "ollama"
        config.Config.OLLAMA_BASE_URL = "http://localhost:11434"
        config.Config.OLLAMA_MODEL = "llama3"
        
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": "Hello from Ollama!"}
        }
        mock_response.raise_for_status = MagicMock()
        
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client
            
            from app.llm import OllamaProvider
            provider = OllamaProvider()
            
            result = await provider.chat([
                {"role": "user", "content": "Hello"}
            ])
            
            assert result == "Hello from Ollama!"
    
    @pytest.mark.asyncio
    async def test_ollama_provider_embed(self, monkeypatch):
        """Test Ollama provider embed method."""
        from app import config
        config.Config.LLM_PROVIDER = "ollama"
        config.Config.OLLAMA_BASE_URL = "http://localhost:11434"
        config.Config.OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
        
        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": [0.4, 0.5, 0.6]}
        mock_response.raise_for_status = MagicMock()
        
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client
            
            from app.llm import OllamaProvider
            provider = OllamaProvider()
            
            result = await provider.embed("test text")
            
            assert result == [0.4, 0.5, 0.6]


class TestLLMConvenienceFunctions:
    """Tests for LLM convenience functions."""
    
    @pytest.fixture(autouse=True)
    def reset_llm_singleton(self):
        """Reset the LLM singleton before each test."""
        from app import llm
        llm._llm_instance = None
        yield
        llm._llm_instance = None
    
    def test_get_llm_provider_unsupported(self, monkeypatch):
        """Test get_llm_provider raises for unsupported provider."""
        from app import config, llm
        config.Config.LLM_PROVIDER = "unsupported"
        
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            llm.get_llm_provider()
    
    def test_get_llm_provider_singleton(self, monkeypatch):
        """Test get_llm_provider returns singleton instance."""
        from app import config
        config.Config.LLM_PROVIDER = "openai"
        config.Config.OPENAI_API_KEY = "test-key"
        
        with patch("openai.AsyncOpenAI"):
            from app import llm
            
            provider1 = llm.get_llm_provider()
            provider2 = llm.get_llm_provider()
            
            assert provider1 is provider2
    
    @pytest.mark.asyncio
    async def test_chat_convenience_function(self, monkeypatch):
        """Test chat convenience function."""
        from app import config
        config.Config.LLM_PROVIDER = "openai"
        config.Config.OPENAI_API_KEY = "test-key"
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        
        with patch("openai.AsyncOpenAI") as mock_client_class:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client
            
            from app import llm
            result = await llm.chat([{"role": "user", "content": "Test"}])
            
            assert result == "Test response"
    
    @pytest.mark.asyncio
    async def test_embed_convenience_function(self, monkeypatch):
        """Test embed convenience function."""
        from app import config
        config.Config.LLM_PROVIDER = "openai"
        config.Config.OPENAI_API_KEY = "test-key"
        
        mock_response = MagicMock()
        mock_response.data = [MagicMock()]
        mock_response.data[0].embedding = [0.1, 0.2, 0.3]
        
        with patch("openai.AsyncOpenAI") as mock_client_class:
            mock_client = MagicMock()
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client
            
            from app import llm
            result = await llm.embed("test")
            
            assert result == [0.1, 0.2, 0.3]
