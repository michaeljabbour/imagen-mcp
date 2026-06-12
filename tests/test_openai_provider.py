"""Tests for OpenAIProvider generate/edit paths with mocked HTTP."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.providers.openai_provider import OpenAIProvider

SAMPLE_IMAGE_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAA"
    "DUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _img_response(count: int = 1) -> dict:
    return {
        "data": [{"b64_json": SAMPLE_IMAGE_B64} for _ in range(count)],
        "usage": {"input_tokens": 5, "output_tokens": 100, "total_tokens": 105},
    }


class TestGenerateDirect:
    @pytest.mark.asyncio
    async def test_direct_path_success(self, tmp_path):
        provider = OpenAIProvider(api_key="k")
        with patch.object(
            provider, "_make_api_request", AsyncMock(return_value=_img_response())
        ) as mock_req:
            result = await provider.generate_image(
                "A poster",
                enable_enhancement=False,  # direct /images/generations path
                output_path=str(tmp_path),
            )
        assert result.success is True
        assert result.image_path is not None
        assert result.image_path.is_file()
        assert result.usage_tokens == {
            "input_tokens": 5,
            "output_tokens": 100,
            "total_tokens": 105,
        }
        # Direct path makes exactly one API call.
        assert mock_req.await_count == 1

    @pytest.mark.asyncio
    async def test_batch_n_populates_additional_paths(self, tmp_path):
        provider = OpenAIProvider(api_key="k")
        with patch.object(provider, "_make_api_request", AsyncMock(return_value=_img_response(3))):
            result = await provider.generate_image(
                "A poster", enable_enhancement=False, n=3, output_path=str(tmp_path)
            )
        assert result.success is True
        assert result.additional_paths is not None
        assert len(result.additional_paths) == 2

    @pytest.mark.asyncio
    async def test_api_error_returns_failure(self, tmp_path):
        provider = OpenAIProvider(api_key="k")
        with patch.object(
            provider, "_make_api_request", AsyncMock(side_effect=ValueError("OpenAI API error"))
        ):
            result = await provider.generate_image(
                "A poster", enable_enhancement=False, output_path=str(tmp_path)
            )
        assert result.success is False
        assert "OpenAI API error" in (result.error or "")


class TestValidateParams:
    @pytest.mark.asyncio
    async def test_rejects_invalid_background(self):
        provider = OpenAIProvider(api_key="k")
        with pytest.raises(ValueError, match="Invalid background"):
            await provider.validate_params("x", background="rainbow")

    @pytest.mark.asyncio
    async def test_rejects_invalid_output_format(self):
        provider = OpenAIProvider(api_key="k")
        with pytest.raises(ValueError, match="Invalid output_format"):
            await provider.validate_params("x", openai_output_format="gif")

    @pytest.mark.asyncio
    async def test_rejects_invalid_moderation(self):
        provider = OpenAIProvider(api_key="k")
        with pytest.raises(ValueError, match="Invalid moderation"):
            await provider.validate_params("x", moderation="strict")

    @pytest.mark.asyncio
    async def test_rejects_n_out_of_range(self):
        provider = OpenAIProvider(api_key="k")
        with pytest.raises(ValueError, match="n must be between"):
            await provider.validate_params("x", n=99)

    @pytest.mark.asyncio
    async def test_gemini_size_mapped(self):
        provider = OpenAIProvider(api_key="k")
        validated = await provider.validate_params("x", size="4K")
        assert validated["size"] == "1792x1024"

    @pytest.mark.asyncio
    async def test_aspect_ratio_to_size(self):
        provider = OpenAIProvider(api_key="k")
        validated = await provider.validate_params("x", aspect_ratio="16:9")
        assert validated["size"] == "1792x1024"


class TestEditImage:
    @pytest.mark.asyncio
    async def test_edit_success(self, tmp_path):
        src = tmp_path / "in.png"
        src.write_bytes(b"\x89PNG\r\n\x1a\nfake")

        provider = OpenAIProvider(api_key="k")
        with patch.object(provider, "_make_api_request", AsyncMock(return_value=_img_response())):
            result = await provider.edit_image(
                prompt="change the sky to sunset",
                image_path=str(src),
                output_path=str(tmp_path / "out"),
            )
        assert result.success is True
        assert result.image_path is not None
        assert result.image_path.is_file()

    @pytest.mark.asyncio
    async def test_edit_with_mask(self, tmp_path):
        src = tmp_path / "in.png"
        src.write_bytes(b"\x89PNG\r\n\x1a\nfake")
        mask = tmp_path / "mask.png"
        mask.write_bytes(b"\x89PNG\r\n\x1a\nmask")

        provider = OpenAIProvider(api_key="k")
        with patch.object(
            provider, "_make_api_request", AsyncMock(return_value=_img_response())
        ) as mock_req:
            result = await provider.edit_image(
                prompt="inpaint",
                image_path=str(src),
                mask_path=str(mask),
                output_path=str(tmp_path / "out"),
            )
        assert result.success is True
        # The multipart files dict should include the mask.
        _, kwargs = mock_req.call_args
        assert "mask" in kwargs["files"]

    @pytest.mark.asyncio
    async def test_edit_missing_source_fails(self, tmp_path):
        provider = OpenAIProvider(api_key="k")
        result = await provider.edit_image(prompt="x", image_path=str(tmp_path / "missing.png"))
        assert result.success is False
        assert "not found" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_edit_invalid_size_fails(self, tmp_path):
        src = tmp_path / "in.png"
        src.write_bytes(b"fake")
        provider = OpenAIProvider(api_key="k")
        result = await provider.edit_image(
            prompt="x",
            image_path=str(src),
            size="1792x1024",  # not allowed for edits
        )
        assert result.success is False
        assert "Invalid size" in (result.error or "")


class TestClientLifecycle:
    @pytest.mark.asyncio
    async def test_close_is_safe_without_client(self):
        provider = OpenAIProvider(api_key="k")
        await provider.close()  # no client created yet — must not raise

    def test_ensure_client_reused(self):
        provider = OpenAIProvider(api_key="k")
        c1 = provider._ensure_client()
        c2 = provider._ensure_client()
        assert c1 is c2
