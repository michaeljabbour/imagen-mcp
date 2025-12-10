"""Tests for MCP server."""

import os
import pytest

# Set dummy API keys for testing
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("GEMINI_API_KEY", "test-key")


class TestServerImports:
    """Tests for server module imports."""

    def test_server_imports(self):
        """Server module should import without errors."""
        from src.server import mcp
        assert mcp is not None

    def test_config_imports(self):
        """Config modules should import without errors."""
        from src.config.constants import OPENAI_SIZES, GEMINI_SIZES
        from src.config.settings import Settings, get_settings

        assert len(OPENAI_SIZES) == 3
        assert len(GEMINI_SIZES) == 3
        assert get_settings() is not None

    def test_provider_imports(self):
        """Provider modules should import without errors."""
        from src.providers import (
            ImageProvider,
            ImageResult,
            ProviderCapabilities,
            ProviderRegistry,
            get_provider_registry,
        )

        assert ImageProvider is not None
        assert ImageResult is not None
        assert get_provider_registry() is not None

    def test_model_imports(self):
        """Model modules should import without errors."""
        from src.models import ImageGenerationInput, ConversationalImageInput

        assert ImageGenerationInput is not None
        assert ConversationalImageInput is not None


class TestSettings:
    """Tests for settings configuration."""

    def test_settings_from_env(self):
        """Settings should load from environment."""
        from src.config.settings import Settings

        settings = Settings.from_env()
        assert settings.default_provider == "auto"
        assert settings.default_openai_size == "1024x1024"
        assert settings.default_gemini_size == "2K"

    def test_settings_has_keys(self):
        """Settings should detect API keys."""
        from src.config.settings import get_settings

        settings = get_settings()
        # With test keys set, both should be available
        assert settings.has_openai_key()
        assert settings.has_gemini_key()

    def test_available_providers(self):
        """Settings should list available providers."""
        from src.config.settings import get_settings

        settings = get_settings()
        providers = settings.available_providers()
        assert "openai" in providers
        assert "gemini" in providers


class TestInputModels:
    """Tests for Pydantic input models."""

    def test_image_generation_input(self):
        """ImageGenerationInput should validate correctly."""
        from src.models import ImageGenerationInput

        # Valid input
        input_data = ImageGenerationInput(prompt="A sunset over mountains")
        assert input_data.prompt == "A sunset over mountains"
        assert input_data.provider.value == "auto"  # default is auto
        assert input_data.size is None  # default

    def test_image_generation_with_provider(self):
        """ImageGenerationInput should accept provider."""
        from src.models import ImageGenerationInput

        input_data = ImageGenerationInput(
            prompt="A sunset",
            provider="gemini",
            size="4K"
        )
        assert input_data.provider == "gemini"
        assert input_data.size == "4K"

    def test_conversational_input(self):
        """ConversationalImageInput should validate correctly."""
        from src.models import ConversationalImageInput

        input_data = ConversationalImageInput(
            prompt="Make it more colorful",
            conversation_id="test-123"
        )
        assert input_data.prompt == "Make it more colorful"
        assert input_data.conversation_id == "test-123"
