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
        assert "gpt-image-2" in caps.display_name
        assert "1024x1024" in caps.supported_sizes
        assert caps.supports_reference_images is False
        assert caps.supports_real_time_data is False
        # gpt-image-2 is dramatically faster than 1.x
        assert caps.typical_latency_seconds <= 15.0
        # Max resolution is now 1792x1024 on 2.0 (was 1536x1024 on 1.x)
        assert caps.max_resolution == "1792x1024"

    def test_supported_sizes(self):
        """Provider should support the full 2.0-era size set."""
        provider = OpenAIProvider()
        expected_sizes = [
            "1024x1024",
            "1024x1536",
            "1536x1024",
            "1792x1024",  # new in 2.0-era
            "1024x1792",  # new in 2.0-era
            "auto",
        ]
        for size in expected_sizes:
            assert size in provider.capabilities.supported_sizes, f"Missing size {size}"

    def test_resolve_model_default(self):
        """_resolve_model should default to gpt-image-2 when nothing is passed."""
        from src.config.constants import DEFAULT_OPENAI_IMAGE_MODEL

        provider = OpenAIProvider()
        assert provider._resolve_model(None) == DEFAULT_OPENAI_IMAGE_MODEL
        assert provider._resolve_model("") == DEFAULT_OPENAI_IMAGE_MODEL

    def test_resolve_model_explicit(self):
        """_resolve_model should pass through known aliases and arbitrary names."""
        provider = OpenAIProvider()
        assert provider._resolve_model("gpt-image-2") == "gpt-image-2"
        assert provider._resolve_model("gpt-image-1") == "gpt-image-1"
        # Unknown names pass through so tests can target future models
        assert provider._resolve_model("gpt-image-future") == "gpt-image-future"

    @pytest.mark.asyncio
    async def test_validate_params_accepts_new_options(self):
        """validate_params should accept and echo back 2.0 parameters."""
        provider = OpenAIProvider()
        validated = await provider.validate_params(
            "A sunset",
            size="1792x1024",
            quality="high",
            openai_output_format="webp",
            openai_output_compression=80,
            background="transparent",
            moderation="low",
            n=2,
        )
        assert validated["size"] == "1792x1024"
        assert validated["quality"] == "high"
        assert validated["output_format"] == "webp"
        assert validated["output_compression"] == 80
        assert validated["background"] == "transparent"
        assert validated["moderation"] == "low"
        assert validated["n"] == 2

    @pytest.mark.asyncio
    async def test_validate_params_rejects_invalid_quality(self):
        provider = OpenAIProvider()
        with pytest.raises(ValueError, match="Invalid quality"):
            await provider.validate_params("A sunset", quality="super")


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

    def test_default_is_nano_banana_2(self):
        """Default Gemini model should be Nano Banana 2 (current Google default)."""
        from src.config.constants import DEFAULT_GEMINI_IMAGE_MODEL

        assert DEFAULT_GEMINI_IMAGE_MODEL == "gemini-3.1-flash-image-preview"

    def test_resolve_model_id_handles_aliases(self):
        """Friendly aliases should resolve to canonical Nano Banana model IDs."""
        provider = GeminiProvider()
        assert provider._resolve_model_id("nano-banana-2") == "gemini-3.1-flash-image-preview"
        assert provider._resolve_model_id("nano-banana-pro") == "gemini-3-pro-image-preview"

    def test_imagen_aliases_no_longer_exist(self):
        """Imagen 4 aliases were removed in v0.3.0 — they should fall back to the default."""
        from src.config.constants import DEFAULT_GEMINI_IMAGE_MODEL

        provider = GeminiProvider()
        # Any imagen-* string is now "unknown" and should fall back
        assert provider._resolve_model_id("imagen-4") == DEFAULT_GEMINI_IMAGE_MODEL
        assert provider._resolve_model_id("imagen-4-ultra") == DEFAULT_GEMINI_IMAGE_MODEL
        assert provider._resolve_model_id("imagen-4-fast") == DEFAULT_GEMINI_IMAGE_MODEL

    def test_resolve_model_id_passes_through_canonical_ids(self):
        """Canonical model IDs in GEMINI_MODELS should pass through unchanged."""
        provider = GeminiProvider()
        canonical = "gemini-3-pro-image-preview"
        assert provider._resolve_model_id(canonical) == canonical

    def test_resolve_model_id_falls_back_on_unknown(self):
        """Unknown model names should warn and fall back to the default."""
        from src.config.constants import DEFAULT_GEMINI_IMAGE_MODEL

        provider = GeminiProvider()
        assert provider._resolve_model_id("bogus-model-xyz") == DEFAULT_GEMINI_IMAGE_MODEL
        assert provider._resolve_model_id(None) == DEFAULT_GEMINI_IMAGE_MODEL

    def test_imagen_models_removed(self):
        """Imagen 4 models were removed in v0.3.0 and must not appear in the registry."""
        from src.config.constants import GEMINI_MODELS

        for removed_id in (
            "imagen-4.0-generate-001",
            "imagen-4.0-ultra-generate-001",
            "imagen-4.0-fast-generate-001",
        ):
            assert removed_id not in GEMINI_MODELS, (
                f"{removed_id} should have been removed in v0.3.0"
            )


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

    def test_request_scoped_openai_key_enables_provider(self, monkeypatch):
        """Per-call OpenAI keys should count for selection without polluting the cache."""
        from src.config.settings import get_settings

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        get_settings.cache_clear()

        try:
            registry = ProviderRegistry()

            assert registry.is_provider_available("openai") is False
            assert registry.is_provider_available("openai", api_key="request-key") is True
            assert registry.list_providers(openai_api_key="request-key") == ["openai"]

            provider, rec = registry.get_provider_for_prompt(
                "Create a poster with headline text",
                explicit_provider="openai",
                openai_api_key="request-key",
            )

            assert provider.name == "openai"
            assert rec.provider == "openai"

            # Request-scoped credentials must not make future no-key lookups succeed.
            with pytest.raises(ValueError, match="OpenAI provider not available"):
                registry.get_provider("openai")
        finally:
            get_settings.cache_clear()

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
