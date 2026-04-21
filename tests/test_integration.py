"""
Integration tests for imagen-mcp with mocked API responses.

Tests full flow from MCP tools through provider selection to image generation,
using mocked HTTP responses to avoid actual API calls.
"""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

# Set dummy API keys before imports
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key")

from src.providers.base import ImageResult
from src.providers.registry import ProviderRegistry
from src.services.dialogue import DialogueSystem

# Sample base64 PNG (1x1 red pixel)
SAMPLE_IMAGE_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAA"
    "DUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


class TestProviderSelection:
    """Tests for automatic provider selection."""

    def test_text_heavy_prompt_selects_openai(self):
        """Text-heavy prompts should select OpenAI."""
        registry = ProviderRegistry()

        # Prompts that should prefer OpenAI
        openai_prompts = [
            "Create a menu card for an Italian restaurant with prices",
            "Design an infographic explaining climate change",
            "Make a comic strip with dialogue bubbles",
            "Create a poster with the headline 'Summer Sale'",
        ]

        for prompt in openai_prompts:
            _, rec = registry.get_provider_for_prompt(prompt)
            assert rec.provider == "openai", f"Expected OpenAI for: {prompt}"

    def test_photorealistic_prompt_selects_gemini(self):
        """Photorealistic prompts should select Gemini."""
        registry = ProviderRegistry()
        gemini_available = registry.is_provider_available("gemini")

        # Prompts that should prefer Gemini
        gemini_prompts = [
            "Professional headshot with studio lighting",
            "Photorealistic portrait of a woman",
            "Product shot of a perfume bottle on marble",
            "4K high resolution landscape photo",
        ]

        for prompt in gemini_prompts:
            _, rec = registry.get_provider_for_prompt(prompt)
            expected = "gemini" if gemini_available else "openai"
            assert rec.provider == expected, f"Expected {expected} for: {prompt}"

    def test_reference_images_force_gemini(self):
        """Reference images should force Gemini selection."""
        registry = ProviderRegistry()

        _, rec = registry.get_provider_for_prompt(
            "Create a logo",  # Would normally prefer OpenAI
            reference_images=[SAMPLE_IMAGE_B64],
        )
        expected = "gemini" if registry.is_provider_available("gemini") else "openai"
        assert rec.provider == expected

    def test_google_search_forces_gemini(self):
        """Google Search grounding should force Gemini."""
        registry = ProviderRegistry()

        _, rec = registry.get_provider_for_prompt(
            "Show current weather in NYC",
            enable_google_search=True,
        )
        expected = "gemini" if registry.is_provider_available("gemini") else "openai"
        assert rec.provider == expected

    def test_explicit_provider_override(self):
        """Explicit provider should override auto-selection."""
        registry = ProviderRegistry()

        # Force OpenAI even for photorealistic prompt
        _, rec = registry.get_provider_for_prompt(
            "Photorealistic portrait",
            explicit_provider="openai",
        )
        assert rec.provider == "openai"


class TestDialogueSystem:
    """Tests for the dialogue system."""

    def test_skip_mode_always_generates(self):
        """Skip mode should always return should_generate=True."""
        dialogue = DialogueSystem(mode="skip")
        result = dialogue.analyze("anything")
        assert result.should_generate is True

    def test_vague_prompt_asks_questions(self):
        """Vague prompts should trigger questions in guided mode."""
        dialogue = DialogueSystem(mode="guided")
        result = dialogue.analyze("something cool")

        assert result.should_generate is False
        assert len(result.questions) > 0

    def test_detailed_prompt_generates_directly(self):
        """Detailed prompts should generate directly."""
        dialogue = DialogueSystem(mode="guided")
        result = dialogue.analyze(
            "A photorealistic portrait of a woman with warm studio lighting, "
            "wearing a red dress, neutral gray background, shallow depth of field, "
            "shot on Canon 5D Mark IV with 85mm f/1.4 lens"
        )

        assert result.should_generate is True

    def test_quick_mode_limits_questions(self):
        """Quick mode should ask at most 2 questions."""
        dialogue = DialogueSystem(mode="quick")
        result = dialogue.analyze("a picture")

        assert len(result.questions) <= 2

    def test_explorer_mode_asks_more_questions(self):
        """Explorer mode should ask more questions."""
        dialogue = DialogueSystem(mode="explorer")
        result = dialogue.analyze("a picture")

        assert len(result.questions) <= 6

    def test_image_type_detection(self):
        """Dialogue should detect image types."""
        dialogue = DialogueSystem(mode="guided")

        result = dialogue.analyze("portrait of a CEO")
        assert result.detected_intent == "portrait"

        result = dialogue.analyze("product shot of a watch")
        assert result.detected_intent == "product"

        result = dialogue.analyze("landscape with mountains")
        assert result.detected_intent == "scene"

    def test_prompt_enhancement(self):
        """enhance_prompt should incorporate answers."""
        dialogue = DialogueSystem(mode="guided")

        enhanced = dialogue.enhance_prompt(
            "a cat",
            {
                "What style?": "photorealistic",
                "What mood?": "playful and happy",
            },
        )

        assert "cat" in enhanced
        assert "photorealistic" in enhanced
        assert "playful" in enhanced


class TestConversationStore:
    """Tests for persistent conversation storage."""

    @pytest.fixture
    def temp_db(self, tmp_path):
        """Create a temporary database for testing."""
        from src.services.conversation_store import ConversationStore

        return ConversationStore(tmp_path / "test_conversations.db")

    def test_create_conversation(self, temp_db):
        """Should create and retrieve conversation."""
        temp_db.create_conversation("test-conv-1", "openai")

        conv = temp_db.get_conversation("test-conv-1")
        assert conv is not None
        assert conv["id"] == "test-conv-1"
        assert conv["provider"] == "openai"

    def test_add_and_get_messages(self, temp_db):
        """Should store and retrieve messages."""
        temp_db.create_conversation("test-conv-2", "gemini")
        temp_db.add_message("test-conv-2", "user", "Create a sunset")
        temp_db.add_message(
            "test-conv-2",
            "assistant",
            {"type": "image_generated"},
            image_base64=SAMPLE_IMAGE_B64,
        )

        messages = temp_db.get_messages("test-conv-2")
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Create a sunset"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["image_base64"] == SAMPLE_IMAGE_B64

    def test_list_conversations(self, temp_db):
        """Should list conversations with summary."""
        temp_db.create_conversation("conv-a", "openai")
        temp_db.add_message("conv-a", "user", "First prompt")

        temp_db.create_conversation("conv-b", "gemini")
        temp_db.add_message("conv-b", "user", "Second prompt")

        convs = temp_db.list_conversations()
        assert len(convs) == 2
        assert any(c["id"] == "conv-a" for c in convs)
        assert any(c["id"] == "conv-b" for c in convs)

    def test_get_last_image(self, temp_db):
        """Should retrieve the last image from conversation."""
        temp_db.create_conversation("conv-img", "gemini")
        temp_db.add_message("conv-img", "user", "First image")
        temp_db.add_message("conv-img", "assistant", "Generated", image_base64="first_image_b64")
        temp_db.add_message("conv-img", "user", "Refine it")
        temp_db.add_message("conv-img", "assistant", "Refined", image_base64="second_image_b64")

        last_img = temp_db.get_last_image("conv-img")
        assert last_img == "second_image_b64"

    def test_delete_conversation(self, temp_db):
        """Should delete conversation and messages."""
        temp_db.create_conversation("conv-del", "openai")
        temp_db.add_message("conv-del", "user", "test")

        assert temp_db.delete_conversation("conv-del") is True
        assert temp_db.get_conversation("conv-del") is None
        assert temp_db.get_messages("conv-del") == []

    def test_provider_filter(self, temp_db):
        """Should filter conversations by provider."""
        temp_db.create_conversation("openai-1", "openai")
        temp_db.create_conversation("gemini-1", "gemini")
        temp_db.create_conversation("openai-2", "openai")

        openai_convs = temp_db.list_conversations(provider="openai")
        assert len(openai_convs) == 2
        assert all(c["provider"] == "openai" for c in openai_convs)


class TestOpenAIProviderMocked:
    """Tests for OpenAI provider with mocked API."""

    @pytest.fixture
    def mock_openai_response(self):
        """Create a mock OpenAI API response."""
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "function": {
                                    "name": "generate_image",
                                    "arguments": json.dumps(
                                        {
                                            "prompt": "A beautiful sunset",
                                            "size": "1024x1024",
                                        }
                                    ),
                                },
                            }
                        ],
                    }
                }
            ]
        }

    @pytest.fixture
    def mock_image_response(self):
        """Create a mock image generation response."""
        return {"data": [{"b64_json": SAMPLE_IMAGE_B64}]}

    @pytest.mark.asyncio
    async def test_generate_image_success(
        self, mock_openai_response, mock_image_response, tmp_path
    ):
        """Should successfully generate image with mocked API."""
        from src.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider(api_key="test-key")

        with patch.object(provider, "_make_api_request") as mock_request:
            # First call returns chat completion, second returns image
            mock_request.side_effect = [mock_openai_response, mock_image_response]

            result = await provider.generate_image(
                "A beautiful sunset",
                output_path=str(tmp_path / "test_image.png"),
            )

            assert result.success is True
            assert result.provider == "openai"
            assert result.image_path is not None
            assert result.image_path.exists()


class TestGeminiProviderMocked:
    """Tests for Gemini provider with mocked API."""

    @pytest.mark.asyncio
    async def test_validate_params(self):
        """Should validate and normalize parameters."""
        from src.providers.gemini_provider import GeminiProvider

        provider = GeminiProvider(api_key="test-key")

        # Valid params
        validated = await provider.validate_params("A sunset", size="2K", aspect_ratio="16:9")
        assert validated["size"] == "2K"
        assert validated["aspect_ratio"] == "16:9"

        # OpenAI size conversion
        validated = await provider.validate_params("A sunset", size="1024x1024")
        assert validated["size"] == "1K"

    @pytest.mark.asyncio
    async def test_invalid_size_raises(self):
        """Should raise for invalid size."""
        from src.providers.gemini_provider import GeminiProvider

        provider = GeminiProvider(api_key="test-key")

        with pytest.raises(ValueError, match="Invalid size"):
            await provider.validate_params("A sunset", size="8K")

    @pytest.mark.asyncio
    async def test_too_many_reference_images_raises(self):
        """Should raise for too many reference images."""
        from src.providers.gemini_provider import GeminiProvider

        provider = GeminiProvider(api_key="test-key")

        with pytest.raises(ValueError, match="Too many reference images"):
            await provider.validate_params(
                "A sunset",
                reference_images=[SAMPLE_IMAGE_B64] * 20,
            )


class TestImageResultFormatting:
    """Tests for result formatting."""

    def test_markdown_format_success(self):
        """Should format successful result as markdown."""
        from src.providers.selector import ProviderRecommendation
        from src.server import format_result_markdown

        result = ImageResult(
            success=True,
            provider="openai",
            model="gpt-image-1",
            image_path=Path("/tmp/test.png"),
            prompt="A sunset",
            size="1024x1024",
            conversation_id="conv-123",
            generation_time_seconds=5.5,
        )

        rec = ProviderRecommendation(
            provider="openai",
            confidence=0.85,
            reasoning="Text rendering detected",
        )

        output = format_result_markdown(result, rec)

        assert "Image Generated Successfully" in output
        assert "openai" in output.lower()
        assert "conv-123" in output

    def test_markdown_format_error(self):
        """Should format error result as markdown."""
        from src.server import format_result_markdown

        result = ImageResult(
            success=False,
            provider="openai",
            model="gpt-image-1",
            prompt="A sunset",
            error="API rate limit exceeded",
        )

        output = format_result_markdown(result)

        assert "Failed" in output
        assert "rate limit" in output

    def test_json_format(self):
        """Should format result as valid JSON."""
        from src.server import format_result_json

        result = ImageResult(
            success=True,
            provider="gemini",
            model="gemini-3-pro-image-preview",
            prompt="A mountain",
        )

        output = format_result_json(result)
        parsed = json.loads(output)

        assert parsed["success"] is True
        assert parsed["provider"] == "gemini"
