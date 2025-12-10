"""Tests for provider implementations."""

import os

import pytest

# Set dummy API keys for testing
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("GEMINI_API_KEY", "test-key")

from src.providers.base import ProviderCapabilities
from src.providers.gemini_provider import GeminiProvider
from src.providers.openai_provider import OpenAIProvider
from src.providers.registry import ProviderRegistry


class TestOpenAIProvider:
    """Tests for OpenAI provider."""

    def test_provider_name(self):
        """Provider should have correct name."""
        provider = OpenAIProvider()
        assert provider.name == "openai"

    def test_capabilities(self):
        """Provider should report correct capabilities."""
        provider = OpenAIProvider()
        caps = provider.capabilities

        assert isinstance(caps, ProviderCapabilities)
        assert caps.name == "openai"
        assert caps.display_name == "OpenAI GPT-Image-1"
        assert "1024x1024" in caps.supported_sizes
        assert caps.supports_reference_images is False
        assert caps.supports_real_time_data is False

    def test_supported_sizes(self):
        """Provider should support expected sizes."""
        provider = OpenAIProvider()
        expected_sizes = ["1024x1024", "1024x1536", "1536x1024"]
        for size in expected_sizes:
            assert size in provider.capabilities.supported_sizes


class TestGeminiProvider:
    """Tests for Gemini provider."""

    def test_provider_name(self):
        """Provider should have correct name."""
        provider = GeminiProvider()
        assert provider.name == "gemini"

    def test_capabilities(self):
        """Provider should report correct capabilities."""
        provider = GeminiProvider()
        caps = provider.capabilities

        assert isinstance(caps, ProviderCapabilities)
        assert caps.name == "gemini"
        assert "Gemini" in caps.display_name
        assert caps.supports_reference_images is True
        assert caps.supports_real_time_data is True
        assert caps.supports_thinking_mode is True

    def test_supported_sizes(self):
        """Provider should support expected sizes."""
        provider = GeminiProvider()
        expected_sizes = ["1K", "2K", "4K"]
        for size in expected_sizes:
            assert size in provider.capabilities.supported_sizes

    def test_supported_aspect_ratios(self):
        """Provider should support multiple aspect ratios."""
        provider = GeminiProvider()
        caps = provider.capabilities
        assert len(caps.supported_aspect_ratios) >= 10
        assert "1:1" in caps.supported_aspect_ratios
        assert "16:9" in caps.supported_aspect_ratios


class TestProviderRegistry:
    """Tests for provider registry."""

    def test_list_all_providers(self):
        """Registry should list all supported providers."""
        registry = ProviderRegistry()
        providers = registry.list_all_providers()
        assert "openai" in providers
        assert "gemini" in providers

    def test_get_provider_by_name(self):
        """Registry should return correct provider by name."""
        registry = ProviderRegistry()

        openai = registry.get_provider("openai")
        assert openai.name == "openai"

        gemini = registry.get_provider("gemini")
        assert gemini.name == "gemini"

    def test_get_unknown_provider_raises(self):
        """Registry should raise for unknown provider."""
        registry = ProviderRegistry()
        with pytest.raises(ValueError, match="Unknown provider"):
            registry.get_provider("unknown")

    def test_provider_info(self):
        """Registry should return provider info."""
        registry = ProviderRegistry()

        info = registry.get_provider_info("openai")
        assert info["name"] == "openai"
        assert "supported_sizes" in info

        info = registry.get_provider_info("gemini")
        assert info["name"] == "gemini"
        assert info["supports_reference_images"] is True

    def test_provider_comparison(self):
        """Registry should generate comparison table."""
        registry = ProviderRegistry()
        comparison = registry.get_comparison()

        assert "OpenAI" in comparison
        assert "Gemini" in comparison
        assert "Text Rendering" in comparison
