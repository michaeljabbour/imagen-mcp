"""Tests for provider selection logic."""

import os

import pytest

# Set dummy API keys for testing
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("GEMINI_API_KEY", "test-key")

from src.providers.selector import ProviderSelector


@pytest.fixture
def selector():
    """Create a provider selector instance."""
    return ProviderSelector()


class TestProviderSelection:
    """Tests for auto provider selection."""

    def test_text_rendering_selects_openai(self, selector):
        """Text-heavy prompts should select OpenAI."""
        prompts = [
            "Create a menu card for an Italian restaurant with prices",
            "Design an infographic about climate change",
            "Make a poster with headline text",
            "Create a comic strip with dialogue bubbles",
        ]
        for prompt in prompts:
            rec = selector.suggest_provider(prompt)
            assert rec.provider == "openai", f"Expected OpenAI for: {prompt}"

    def test_photorealistic_selects_gemini(self, selector):
        """Photorealistic prompts should select Gemini."""
        prompts = [
            "Professional headshot with studio lighting",
            "Product photography of a perfume bottle",
            "Photorealistic portrait of a woman",
            "4K landscape photo of mountains",
        ]
        for prompt in prompts:
            rec = selector.suggest_provider(prompt)
            assert rec.provider == "gemini", f"Expected Gemini for: {prompt}"

    def test_reference_images_require_gemini(self, selector):
        """Reference images should force Gemini."""
        rec = selector.suggest_provider("A simple cat", reference_images=["base64encodedimage"])
        assert rec.provider == "gemini"
        assert rec.confidence == 1.0

    def test_4k_requires_gemini(self, selector):
        """4K resolution should force Gemini."""
        rec = selector.suggest_provider("A sunset", size="4K")
        assert rec.provider == "gemini"
        assert rec.confidence == 1.0

    def test_google_search_requires_gemini(self, selector):
        """Google Search grounding should force Gemini."""
        rec = selector.suggest_provider("Show me the weather", enable_google_search=True)
        assert rec.provider == "gemini"
        assert rec.confidence == 1.0

    def test_explicit_provider_override(self, selector):
        """Explicit provider should override auto-selection."""
        # Force OpenAI even for portrait
        rec = selector.suggest_provider("Professional headshot", explicit_provider="openai")
        assert rec.provider == "openai"
        assert rec.confidence == 1.0

    def test_realtime_keywords_require_gemini(self, selector):
        """Real-time data keywords should force Gemini."""
        prompts = [
            "Show the current weather in NYC",
            "What's today's stock price for AAPL",
            "Real-time traffic map",
        ]
        for prompt in prompts:
            rec = selector.suggest_provider(prompt)
            assert rec.provider == "gemini", f"Expected Gemini for: {prompt}"
            assert rec.confidence == 1.0


class TestProviderRecommendation:
    """Tests for recommendation metadata."""

    def test_recommendation_includes_reasoning(self, selector):
        """Recommendations should include reasoning."""
        rec = selector.suggest_provider("Create a menu with prices")
        assert rec.reasoning is not None
        assert len(rec.reasoning) > 0

    def test_recommendation_includes_confidence(self, selector):
        """Recommendations should include confidence score."""
        rec = selector.suggest_provider("A beautiful sunset")
        assert 0.0 <= rec.confidence <= 1.0

    def test_recommendation_detects_image_type(self, selector):
        """Recommendations should detect image type when applicable."""
        rec = selector.suggest_provider("Professional portrait photo")
        assert rec.detected_image_type == "portrait"

    def test_recommendation_includes_alternative(self, selector):
        """Recommendations should include alternative provider."""
        rec = selector.suggest_provider("A simple landscape")
        assert rec.alternative in ["openai", "gemini", None]
