"""Tests for the MCP tool handlers in src/server.py.

These exercise each tool function directly (the FastMCP decorator returns
the original callable), with the provider registry mocked so no network
calls happen.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src import server
from src.models.input_models import (
    ConversationalImageInput,
    CostEstimateInput,
    EditImageInput,
    ImageGenerationInput,
    ListConversationsInput,
    OutputFormat,
    Provider,
)
from src.providers.base import ImageResult
from src.providers.openai_provider import OpenAIProvider
from src.providers.selector import ProviderRecommendation


def make_result(**kw) -> ImageResult:
    base = ImageResult(
        success=True,
        provider="openai",
        model="gpt-image-2",
        image_path=Path("/tmp/x.png"),
        prompt="p",
        size="1024x1024",
        generation_time_seconds=1.2,
    )
    return dataclasses.replace(base, **kw)


@pytest.fixture
def fake_registry(monkeypatch):
    """Patch server.get_provider_registry with a controllable mock."""
    reg = MagicMock()
    rec = ProviderRecommendation(provider="openai", confidence=0.9, reasoning="text rendering")
    provider = MagicMock()
    provider.generate_image = AsyncMock(return_value=make_result())
    reg.get_provider_for_prompt.return_value = (provider, rec)
    reg.is_provider_available.return_value = True
    monkeypatch.setattr(server, "get_provider_registry", lambda: reg)
    return reg, provider, rec


# --------------------------------------------------------------------------
# generate_image
# --------------------------------------------------------------------------


class TestGenerateImage:
    @pytest.mark.asyncio
    async def test_markdown_success(self, fake_registry):
        out = await server.generate_image(ImageGenerationInput(prompt="A poster with text"))
        assert "Image Generated Successfully" in out
        assert "Openai" in out

    @pytest.mark.asyncio
    async def test_json_output(self, fake_registry):
        out = await server.generate_image(
            ImageGenerationInput(prompt="A poster", output_format=OutputFormat.JSON)
        )
        data = json.loads(out)
        assert data["success"] is True
        assert data["provider"] == "openai"

    @pytest.mark.asyncio
    async def test_failure_result_formatted(self, fake_registry):
        _, provider, _ = fake_registry
        provider.generate_image.return_value = make_result(
            success=False, image_path=None, error="boom"
        )
        out = await server.generate_image(ImageGenerationInput(prompt="x"))
        assert "Failed" in out
        assert "boom" in out

    @pytest.mark.asyncio
    async def test_unexpected_exception_sanitized(self, fake_registry):
        _, provider, _ = fake_registry
        provider.generate_image.side_effect = RuntimeError("token sk-ABCDEFGHIJKLMNOP")
        out = await server.generate_image(ImageGenerationInput(prompt="x"))
        assert "Failed" in out

    @pytest.mark.asyncio
    async def test_progress_reported_when_ctx_present(self, fake_registry):
        ctx = MagicMock()
        ctx.report_progress = AsyncMock()
        await server.generate_image(ImageGenerationInput(prompt="A poster"), ctx=ctx)
        assert ctx.report_progress.await_count >= 2

    @pytest.mark.asyncio
    async def test_explicit_provider_forwarded(self, fake_registry):
        reg, _, _ = fake_registry
        await server.generate_image(ImageGenerationInput(prompt="x", provider=Provider.OPENAI))
        _, kwargs = reg.get_provider_for_prompt.call_args
        assert kwargs["explicit_provider"] == "openai"


# --------------------------------------------------------------------------
# conversational_image
# --------------------------------------------------------------------------


class TestConversationalImage:
    @pytest.mark.asyncio
    async def test_skip_dialogue_generates(self, fake_registry):
        out = await server.conversational_image(
            ConversationalImageInput(prompt="a cat", skip_dialogue=True)
        )
        assert "Image Generated Successfully" in out

    @pytest.mark.asyncio
    async def test_vague_prompt_returns_questions(self, fake_registry):
        # No ctx → elicitation unavailable → text dialogue questions.
        out = await server.conversational_image(
            ConversationalImageInput(prompt="something cool", dialogue_mode="guided")
        )
        assert "?" in out  # contains dialogue questions

    @pytest.mark.asyncio
    async def test_elicitation_accepted_enriches_prompt(self, fake_registry):
        _, provider, _ = fake_registry
        ctx = MagicMock()
        ctx.elicit = AsyncMock(
            return_value=MagicMock(
                action="accept",
                data=MagicMock(style="oil-painting", mood="warm", additional_details=None),
            )
        )
        out = await server.conversational_image(
            ConversationalImageInput(prompt="something cool", dialogue_mode="guided"),
            ctx=ctx,
        )
        assert "Image Generated Successfully" in out
        # The enriched prompt is what reached the provider.
        called_prompt = provider.generate_image.call_args.args[0]
        assert "oil-painting" in called_prompt
        assert "warm" in called_prompt

    @pytest.mark.asyncio
    async def test_elicitation_declined_falls_back_to_questions(self, fake_registry):
        ctx = MagicMock()
        ctx.elicit = AsyncMock(return_value=MagicMock(action="decline", data=None))
        out = await server.conversational_image(
            ConversationalImageInput(prompt="something cool", dialogue_mode="guided"),
            ctx=ctx,
        )
        assert "?" in out


# --------------------------------------------------------------------------
# edit_image
# --------------------------------------------------------------------------


class TestEditImage:
    @pytest.mark.asyncio
    async def test_edit_success(self, monkeypatch):
        reg = MagicMock()
        reg.is_provider_available.return_value = True
        op = OpenAIProvider(api_key="k")
        op.edit_image = AsyncMock(return_value=make_result(provider="openai"))
        reg.get_provider.return_value = op
        monkeypatch.setattr(server, "get_provider_registry", lambda: reg)

        out = await server.edit_image(EditImageInput(prompt="change sky", image_path="/tmp/in.png"))
        assert "Image Generated Successfully" in out
        op.edit_image.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_edit_unavailable_when_no_openai(self, monkeypatch):
        reg = MagicMock()
        reg.is_provider_available.return_value = False
        monkeypatch.setattr(server, "get_provider_registry", lambda: reg)

        out = await server.edit_image(EditImageInput(prompt="change sky", image_path="/tmp/in.png"))
        assert "Unavailable" in out


# --------------------------------------------------------------------------
# read-only tools (use the real registry)
# --------------------------------------------------------------------------


class TestReadOnlyTools:
    @pytest.mark.asyncio
    async def test_list_providers(self):
        out = await server.list_providers()
        assert "Provider Comparison" in out

    @pytest.mark.asyncio
    async def test_list_conversations_empty(self):
        out = await server.list_conversations(ListConversationsInput(limit=5))
        assert "Conversations" in out


# --------------------------------------------------------------------------
# estimate_cost
# --------------------------------------------------------------------------


class TestEstimateCost:
    @pytest.mark.asyncio
    async def test_markdown(self):
        out = await server.estimate_cost(
            CostEstimateInput(prompt="A poster with bold text", quality="medium")
        )
        assert "Cost Estimate" in out
        assert "$" in out

    @pytest.mark.asyncio
    async def test_json(self):
        out = await server.estimate_cost(
            CostEstimateInput(
                prompt="A poster",
                provider=Provider.OPENAI,
                quality="high",
                size="1024x1024",
                n=2,
                output_format=OutputFormat.JSON,
            )
        )
        data = json.loads(out)
        assert data["provider"] == "openai"
        assert data["n"] == 2
        assert data["total_usd"] is not None
