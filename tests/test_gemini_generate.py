"""Tests for GeminiProvider.generate_image with a mocked google-genai SDK."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.providers import gemini_provider
from src.providers.gemini_provider import GeminiProvider


class FakePart:
    """Mimics a google-genai response part."""

    def __init__(
        self, *, data: bytes | None = None, text: str | None = None, thought: bool = False
    ):
        self.thought = thought
        self.inline_data = MagicMock(data=data) if data is not None else None
        self.text = text


class FakeResponse:
    def __init__(self, parts):
        self.parts = parts


@pytest.fixture
def mocked_sdk(monkeypatch):
    """Stub out the lazily-imported google-genai globals."""
    fake_genai = MagicMock()
    fake_types = MagicMock()
    fake_image = MagicMock()

    monkeypatch.setattr(gemini_provider, "genai", fake_genai)
    monkeypatch.setattr(gemini_provider, "types", fake_types)
    monkeypatch.setattr(gemini_provider, "Image", fake_image)
    # Prevent _import_dependencies from overwriting our stubs.
    monkeypatch.setattr(gemini_provider, "_import_dependencies", lambda: None)
    # Avoid real backoff sleeps in error paths.
    import src.providers.base as base

    async def _no_sleep(_seconds):
        return None

    monkeypatch.setattr(base.asyncio, "sleep", _no_sleep)
    return fake_genai


def _set_response(fake_genai, response):
    client = MagicMock()
    client.models.generate_content.return_value = response
    fake_genai.Client.return_value = client
    return client


PNG_BYTES = b"\x89PNG\r\n\x1a\nfakeimagedata"


class TestGeminiGenerate:
    @pytest.mark.asyncio
    async def test_single_image_success(self, mocked_sdk, tmp_path):
        _set_response(mocked_sdk, FakeResponse([FakePart(data=PNG_BYTES)]))

        provider = GeminiProvider(api_key="test-key")
        result = await provider.generate_image(
            "A photorealistic portrait", size="2K", output_path=str(tmp_path)
        )

        assert result.success is True
        assert result.provider == "gemini"
        assert result.image_path is not None
        assert result.image_path.is_file()
        assert result.additional_paths is None

    @pytest.mark.asyncio
    async def test_batch_multiple_images_saved(self, mocked_sdk, tmp_path):
        _set_response(
            mocked_sdk,
            FakeResponse([FakePart(data=PNG_BYTES), FakePart(data=PNG_BYTES)]),
        )

        provider = GeminiProvider(api_key="test-key")
        result = await provider.generate_image("Two variants", size="1K", output_path=str(tmp_path))

        assert result.success is True
        assert result.additional_paths is not None
        assert len(result.additional_paths) == 1
        assert result.additional_paths[0].is_file()

    @pytest.mark.asyncio
    async def test_text_and_thought_parts_ignored_for_image(self, mocked_sdk, tmp_path):
        _set_response(
            mocked_sdk,
            FakeResponse(
                [
                    FakePart(text="some reasoning", thought=True),
                    FakePart(text="caption"),
                    FakePart(data=PNG_BYTES),
                ]
            ),
        )

        provider = GeminiProvider(api_key="test-key")
        result = await provider.generate_image("A scene", output_path=str(tmp_path))
        assert result.success is True
        assert result.image_path is not None

    @pytest.mark.asyncio
    async def test_no_image_in_response_fails(self, mocked_sdk, tmp_path):
        _set_response(mocked_sdk, FakeResponse([FakePart(text="only text")]))

        provider = GeminiProvider(api_key="test-key")
        result = await provider.generate_image("A scene", output_path=str(tmp_path))
        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_api_exception_returns_failure(self, mocked_sdk, tmp_path):
        client = MagicMock()
        client.models.generate_content.side_effect = RuntimeError("API down")
        mocked_sdk.Client.return_value = client

        provider = GeminiProvider(api_key="test-key")
        result = await provider.generate_image("A scene", output_path=str(tmp_path))
        assert result.success is False
        assert "API down" in (result.error or "")

    @pytest.mark.asyncio
    async def test_invalid_size_returns_failure(self, mocked_sdk, tmp_path):
        _set_response(mocked_sdk, FakeResponse([FakePart(data=PNG_BYTES)]))
        provider = GeminiProvider(api_key="test-key")
        result = await provider.generate_image("A scene", size="8K", output_path=str(tmp_path))
        assert result.success is False
