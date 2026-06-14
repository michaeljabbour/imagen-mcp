#!/usr/bin/env python3
"""
imagen-mcp: Multi-Provider Image Generation MCP Server

An MCP server that provides intelligent image generation using multiple providers:
- OpenAI GPT-Image-1: Best for text rendering, infographics, comics
- Google Gemini 3 Pro Image (Nano Banana Pro): Best for portraits, products, 4K

The server automatically selects the best provider based on prompt analysis,
or allows explicit provider selection.
"""

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from hashlib import sha256
from typing import Any
from uuid import uuid4

from mcp.server.fastmcp import Context, FastMCP
from mcp.types import ToolAnnotations

from .config.pricing import estimate_generation_cost, format_cost_estimate
from .config.settings import get_settings
from .exceptions import ImagenError, _sanitize_message
from .models.input_models import (
    BatchGenerationInput,
    BatchItem,
    ConversationalImageInput,
    CostEstimateInput,
    EditImageInput,
    ImageGenerationInput,
    ImageRefinementElicitation,
    ListConversationsInput,
    OutputFormat,
    Provider,
)
from .providers import ImageResult, ProviderRecommendation, get_provider_registry
from .services.dialogue import DialogueSystem, create_dialogue_response
from .services.logging_config import configure_logging, log_event

logger = logging.getLogger(__name__)


async def _report_progress(
    ctx: Context | None, progress: float, total: float, message: str
) -> None:
    """Report progress to the client if a context is available.

    No-ops safely when running without a live MCP session (e.g. direct
    unit-test calls) or when the client did not supply a progress token.
    """
    if ctx is None:
        return
    try:
        await ctx.report_progress(progress, total, message)
    except Exception:  # pragma: no cover - depends on live client capability
        logger.debug("Progress reporting unavailable; continuing", exc_info=True)


async def _elicit_refinement(ctx: Context | None, prompt: str) -> str | None:
    """Ask the client for structured refinement via MCP Elicitation.

    Returns an enriched prompt string when the user accepts, or ``None``
    when elicitation is unavailable (no context / client doesn't support
    it) or the user declines — in which case the caller falls back to the
    text-based dialogue questions.
    """
    if ctx is None:
        return None
    try:
        result = await ctx.elicit(
            message="Add a little detail to refine your image (optional).",
            schema=ImageRefinementElicitation,
        )
    except Exception:  # pragma: no cover - depends on live client capability
        logger.debug("Elicitation unavailable; falling back to dialogue", exc_info=True)
        return None

    if result.action != "accept" or result.data is None:
        return None

    extras: list[str] = []
    if result.data.style:
        extras.append(f"style: {result.data.style}")
    if result.data.mood:
        extras.append(f"mood: {result.data.mood}")
    if result.data.additional_details:
        extras.append(result.data.additional_details)

    if not extras:
        return prompt
    return f"{prompt} ({', '.join(extras)})"


# ============================
# Lifespan (startup / shutdown)
# ============================


@asynccontextmanager
async def _lifespan(_app: FastMCP) -> AsyncGenerator[None, None]:
    """Manage server-wide resources across the process lifetime."""
    # --- startup ---
    configure_logging()
    logger.info("imagen-mcp server starting")
    yield
    # --- shutdown ---
    logger.info("imagen-mcp server shutting down — closing providers")
    try:
        registry = get_provider_registry()
        await registry.close_all()
    except Exception:
        logger.exception("Error during provider cleanup")

    # Close the persistent conversation store (checkpoints the WAL). Only touch
    # the existing singleton — don't create one just to close it.
    try:
        from .services import conversation_store as cs

        if cs._store is not None:
            cs._store.close()
    except Exception:
        logger.exception("Error closing conversation store")


# Initialize MCP server
mcp = FastMCP("imagen_mcp", lifespan=_lifespan)

# ============================
# Helper Functions
# ============================


def format_result_markdown(
    result: ImageResult,
    recommendation: ProviderRecommendation | None = None,
) -> str:
    """Format image generation result as markdown."""
    if not result.success:
        # Include fallback notice even on failure so user knows why
        error_lines = [f"## ❌ Image Generation Failed\n\n**Error:** {result.error}"]
        if recommendation and recommendation.fallback_notice:
            error_lines.append("")
            error_lines.append(f"> ⚠️ {recommendation.fallback_notice}")
        return "\n".join(error_lines)

    lines = [
        "## ✅ Image Generated Successfully",
        "",
    ]

    # Fallback warning (shown prominently before provider info)
    if recommendation and recommendation.fallback_notice:
        lines.extend(
            [
                f"> ⚠️ **Provider Fallback:** {recommendation.fallback_notice}",
                "",
            ]
        )
        if recommendation.missing_features:
            lines.append("> **Missing features:** " + ", ".join(recommendation.missing_features))
            lines.append("")

    # Provider info
    if recommendation:
        label = "(fallback)" if recommendation.is_fallback else "(auto-selected)"
        lines.extend(
            [
                f"**Provider:** {result.provider.title()} {label}",
                f"**Reasoning:** {recommendation.reasoning}",
                "",
            ]
        )
    else:
        lines.append(f"**Provider:** {result.provider.title()}")
        lines.append("")

    # Image info
    if result.image_path:
        lines.extend(
            [
                f"📁 **Saved to:** `{result.image_path}`",
                "",
            ]
        )

    # Metadata
    lines.extend(
        [
            f"**Model:** {result.model}",
            f"**Size:** {result.size or 'default'}",
        ]
    )

    if result.aspect_ratio:
        lines.append(f"**Aspect Ratio:** {result.aspect_ratio}")

    if result.generation_time_seconds:
        lines.append(f"**Generation Time:** {result.generation_time_seconds:.1f}s")

    # Conversation ID for refinement
    if result.conversation_id:
        lines.extend(
            [
                "",
                "## 🔄 Continue Refining",
                f"**Conversation ID:** `{result.conversation_id}`",
                "*Use this ID to refine this image further.*",
            ]
        )

    # Gemini-specific: thinking mode
    if result.thoughts:
        lines.extend(
            [
                "",
                "## 💭 Model Reasoning",
                f"*{len(result.thoughts)} thought steps processed*",
            ]
        )

    # Gemini-specific: grounding
    if result.grounding_metadata:
        lines.extend(
            [
                "",
                "## 🔍 Real-time Data Sources",
                "*Used Google Search for current information*",
            ]
        )

    return "\n".join(lines)


def format_result_json(
    result: ImageResult, recommendation: ProviderRecommendation | None = None
) -> str:
    """Format image generation result as JSON."""
    data = result.to_dict()
    if recommendation:
        data["recommendation"] = {
            "provider": recommendation.provider,
            "confidence": recommendation.confidence,
            "reasoning": recommendation.reasoning,
            "alternative": recommendation.alternative,
            "is_fallback": recommendation.is_fallback,
            "fallback_notice": recommendation.fallback_notice,
            "preferred_provider": recommendation.preferred_provider,
            "missing_features": recommendation.missing_features,
        }
    return json.dumps(data, indent=2, default=str)


# ============================
# MCP Tools
# ============================


@mcp.tool(
    name="generate_image",
    annotations=ToolAnnotations(
        title="Generate Image",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,  # makes external API calls
    ),
)
async def generate_image(params: ImageGenerationInput, ctx: Context | None = None) -> str:
    """Generate an image using the best available provider.

    **Automatic Provider Selection:**
    The server analyzes your prompt and automatically selects the best provider:

    - **OpenAI GPT-Image-1** is auto-selected for:
      - Text-heavy images (menus, posters, infographics)
      - Comics with dialogue or speech bubbles
      - Technical diagrams with labels
      - Marketing materials requiring precise text

    - **Gemini Nano Banana Pro** is auto-selected for:
      - Photorealistic portraits and headshots
      - Product photography
      - High resolution (4K) output
      - Images using reference images for consistency
      - Real-time data visualization (weather, stocks)

    **Examples:**
    - "Create a menu card for an Italian restaurant" → OpenAI (text rendering)
    - "Professional headshot with studio lighting" → Gemini (photorealism)
    - "Infographic explaining photosynthesis" → OpenAI (diagram + text)
    - "Product shot of perfume floating on water" → Gemini (product photography)

    **Override Selection:**
    Set `provider` to 'openai' or 'gemini' to override auto-selection.

    Args:
        params: Image generation parameters including prompt and optional settings.

    Returns:
        Formatted response with image path and metadata.
    """
    request_id = uuid4().hex[:12]
    try:
        registry = get_provider_registry()
        settings = get_settings()

        prompt_hash = sha256(params.prompt.encode("utf-8")).hexdigest()
        start_event: dict[str, object] = {
            "request_id": request_id,
            "tool": "generate_image",
            "provider_requested": params.provider.value if params.provider else None,
            "prompt_length": len(params.prompt),
            "prompt_sha256": prompt_hash,
            "size": params.size,
            "aspect_ratio": params.aspect_ratio,
            "reference_images_count": len(params.reference_images or []),
            "enable_google_search": bool(params.enable_google_search),
            "enhance_prompt": bool(params.enhance_prompt),
            "output_path": params.output_path,
            "output_format": params.output_format.value if params.output_format else None,
        }
        if settings.log_prompts:
            start_event["prompt"] = params.prompt
        log_event("image.generate.start", **start_event)

        await _report_progress(ctx, 5, 100, "Selecting provider...")

        # Determine explicit provider if specified
        explicit_provider = None
        if params.provider and params.provider != Provider.AUTO:
            explicit_provider = params.provider.value

        # Get best provider for this prompt
        provider, recommendation = registry.get_provider_for_prompt(
            params.prompt,
            size=params.size,
            reference_images=params.reference_images,
            enable_google_search=params.enable_google_search or False,
            explicit_provider=explicit_provider,
            openai_api_key=params.openai_api_key,
            gemini_api_key=params.gemini_api_key,
        )

        logger.info(
            f"Selected provider: {recommendation.provider} "
            f"(confidence: {recommendation.confidence:.0%})"
        )
        log_event(
            "image.provider.selected",
            request_id=request_id,
            provider=recommendation.provider,
            confidence=recommendation.confidence,
            reasoning=recommendation.reasoning,
            alternative=recommendation.alternative,
            alternative_reasoning=recommendation.alternative_reasoning,
            detected_image_type=recommendation.detected_image_type,
        )

        await _report_progress(ctx, 20, 100, f"Generating with {recommendation.provider}...")

        # Generate image — forward OpenAI gpt-image-2 params only when OpenAI is
        # the selected provider to avoid polluting Gemini's signature.
        openai_kwargs: dict[str, Any] = {}
        if recommendation.provider == "openai":
            openai_kwargs = {
                "openai_model": params.openai_model,
                "quality": params.quality,
                "openai_output_format": params.openai_output_format,
                "openai_output_compression": params.openai_output_compression,
                "background": params.background,
                "moderation": params.moderation,
                "n": params.n,
            }

        result = await provider.generate_image(
            params.prompt,
            size=params.size,
            aspect_ratio=params.aspect_ratio,
            reference_images=params.reference_images,
            enable_enhancement=params.enhance_prompt
            if params.enhance_prompt is not None
            else False,
            enable_google_search=params.enable_google_search or False,
            api_key=params.openai_api_key
            if recommendation.provider == "openai"
            else params.gemini_api_key,
            model=params.gemini_model if recommendation.provider == "gemini" else None,
            output_path=params.output_path,
            **openai_kwargs,
        )

        log_event(
            "image.generate.result",
            request_id=request_id,
            success=result.success,
            provider=result.provider,
            model=result.model,
            image_path=str(result.image_path) if result.image_path else None,
            conversation_id=result.conversation_id,
            generation_time_seconds=result.generation_time_seconds,
            error=result.error,
        )

        await _report_progress(ctx, 100, 100, "Done")

        # Format response
        if params.output_format == OutputFormat.JSON:
            return format_result_json(result, recommendation)
        else:
            return format_result_markdown(result, recommendation)

    except ImagenError as e:
        logger.exception("Image generation failed")
        log_event("image.generate.error", request_id=request_id, error=str(e))
        error_response = {
            "success": False,
            "error": e.user_message,
        }
        if params.output_format == OutputFormat.JSON:
            return json.dumps(error_response, indent=2)
        else:
            return (
                f"## ❌ Image Generation Failed\n\n**Error ({type(e).__name__}):** {e.user_message}"
            )

    except Exception as e:
        logger.exception("Image generation failed")
        sanitized = _sanitize_message(str(e))
        log_event("image.generate.error", request_id=request_id, error=sanitized)
        error_response = {
            "success": False,
            "error": sanitized,
        }
        if params.output_format == OutputFormat.JSON:
            return json.dumps(error_response, indent=2)
        else:
            return f"## ❌ Image Generation Failed\n\n**Error:** {sanitized}"


@mcp.tool(
    name="conversational_image",
    annotations=ToolAnnotations(
        title="Conversational Image",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    ),
)
async def conversational_image(params: ConversationalImageInput, ctx: Context | None = None) -> str:
    """Generate images conversationally with iterative refinement.

    **USE THIS TOOL when:**
    - User gives a vague/incomplete prompt that needs refinement
    - User wants iterative refinement across multiple messages
    - User explicitly asks for guidance or suggestions

    **Dialogue Modes:**
    - "quick": 1-2 questions, fast path
    - "guided": 3-5 questions, balanced (DEFAULT)
    - "explorer": Deep exploration with 6+ questions
    - "skip": Direct generation, no dialogue

    **Provider Selection:**
    Same auto-selection logic as generate_image. Provider is locked for
    the duration of a conversation (cannot switch mid-conversation).

    **Usage Pattern:**
    1. Initial: "A cozy coffee shop" → System asks refinement questions
    2. User answers questions
    3. Image generated with refined prompt
    4. Refine: "Add more plants" (with same conversation_id)
    5. Continue refining as needed

    Args:
        params: Conversational image parameters including prompt and dialogue options.

    Returns:
        Either dialogue questions or generated image with metadata.
    """
    request_id = uuid4().hex[:12]
    try:
        registry = get_provider_registry()
        settings = get_settings()

        prompt_hash = sha256(params.prompt.encode("utf-8")).hexdigest()
        start_event: dict[str, object] = {
            "request_id": request_id,
            "tool": "conversational_image",
            "provider_requested": params.provider.value if params.provider else None,
            "conversation_id": params.conversation_id,
            "prompt_length": len(params.prompt),
            "prompt_sha256": prompt_hash,
            "size": params.size,
            "aspect_ratio": params.aspect_ratio,
            "reference_images_count": len(params.reference_images or []),
            "enable_google_search": bool(params.enable_google_search),
            "skip_dialogue": bool(params.skip_dialogue),
            "dialogue_mode": params.dialogue_mode,
            "output_path": params.output_path,
            "output_format": params.output_format.value if params.output_format else None,
        }
        if settings.log_prompts:
            start_event["prompt"] = params.prompt
        log_event("image.conversation.start", **start_event)

        # Determine provider
        explicit_provider = None
        if params.provider and params.provider != Provider.AUTO:
            explicit_provider = params.provider.value

        # Get provider
        provider, recommendation = registry.get_provider_for_prompt(
            params.prompt,
            size=params.size,
            reference_images=params.reference_images,
            enable_google_search=params.enable_google_search or False,
            explicit_provider=explicit_provider,
            openai_api_key=params.openai_api_key,
            gemini_api_key=params.gemini_api_key,
        )
        log_event(
            "image.provider.selected",
            request_id=request_id,
            provider=recommendation.provider,
            confidence=recommendation.confidence,
            reasoning=recommendation.reasoning,
            alternative=recommendation.alternative,
            alternative_reasoning=recommendation.alternative_reasoning,
            detected_image_type=recommendation.detected_image_type,
        )

        # Integrate dialogue system
        dialogue = DialogueSystem(mode=params.dialogue_mode or "guided")
        dialogue_result = dialogue.analyze(params.prompt)

        # If dialogue needs more info, gather it (unless skip_dialogue).
        effective_prompt = params.prompt
        if not params.skip_dialogue and not dialogue_result.should_generate:
            # Phase 9: prefer native MCP Elicitation when the client supports
            # it — the client renders proper form fields. Fall back to the
            # text-based dialogue questions when elicitation is unavailable or
            # the user declines.
            elicited = await _elicit_refinement(ctx, params.prompt)
            if elicited is None:
                log_event(
                    "image.conversation.dialogue",
                    request_id=request_id,
                    questions_count=len(dialogue_result.questions),
                    detected_intent=dialogue_result.detected_intent,
                )
                return create_dialogue_response(dialogue_result, recommendation.provider)
            effective_prompt = elicited
            log_event(
                "image.conversation.elicited",
                request_id=request_id,
                prompt_length=len(effective_prompt),
            )

        # Generate the image
        result = await provider.generate_image(
            effective_prompt,
            size=params.size,
            aspect_ratio=params.aspect_ratio,
            conversation_id=params.conversation_id,
            reference_images=params.reference_images,
            enable_enhancement=not params.skip_dialogue,
            enable_google_search=params.enable_google_search or False,
            api_key=params.openai_api_key
            if recommendation.provider == "openai"
            else params.gemini_api_key,
            model=params.gemini_model if recommendation.provider == "gemini" else None,
            output_path=params.output_path,
            # OpenAI-specific conversational fields
            assistant_model=params.assistant_model or "gpt-4o",
            input_image_file_id=params.input_image_file_id,
        )

        log_event(
            "image.conversation.result",
            request_id=request_id,
            success=result.success,
            provider=result.provider,
            model=result.model,
            image_path=str(result.image_path) if result.image_path else None,
            conversation_id=result.conversation_id,
            generation_time_seconds=result.generation_time_seconds,
            error=result.error,
        )

        if params.output_format == OutputFormat.JSON:
            return format_result_json(result, recommendation)
        else:
            return format_result_markdown(result, recommendation)

    except ImagenError as e:
        logger.exception("Conversational image generation failed")
        log_event("image.conversation.error", request_id=request_id, error=str(e))
        return f"## ❌ Generation Failed\n\n**Error ({type(e).__name__}):** {e.user_message}"

    except Exception as e:
        logger.exception("Conversational image generation failed")
        sanitized = _sanitize_message(str(e))
        log_event("image.conversation.error", request_id=request_id, error=sanitized)
        return f"## ❌ Generation Failed\n\n**Error:** {sanitized}"


@mcp.tool(
    name="edit_image",
    annotations=ToolAnnotations(
        title="Edit Image",
        readOnlyHint=False,
        destructiveHint=False,  # writes a new file; never overwrites the source
        idempotentHint=False,
        openWorldHint=True,
    ),
)
async def edit_image(params: EditImageInput) -> str:
    """Edit an existing image using OpenAI gpt-image-2's /images/edits endpoint.

    This is the right tool for:
    - Image-to-image refinement (OpenAI's answer to reference images)
    - Inpainting with a mask (paint over regions while preserving the rest)
    - Sequential/cumulative edits that preserve unchanged pixels
    - Brand-accurate modifications to existing images

    **Key features of gpt-image-2 editing:**
    - `input_fidelity='high'` (default) keeps unchanged pixels constant —
      critical for multi-step refinement where each edit should build on
      the last without drift.
    - Full control over quality, background, output_format, and compression.
    - Supports optional PNG mask (transparent pixels are the edit region).

    **Typical workflow:**
    1. Generate or obtain a base image (path on disk)
    2. Call edit_image with prompt='change the sky to sunset'
    3. Take the output path, call edit_image again with next instruction
    4. Repeat — each step preserves pixels outside the described change

    Args:
        params: Edit parameters including prompt, image_path, and options.

    Returns:
        Formatted response with edited image path and metadata.
    """
    request_id = uuid4().hex[:12]
    try:
        registry = get_provider_registry()
        settings = get_settings()

        if not registry.is_provider_available("openai", api_key=params.openai_api_key):
            msg = (
                "## ❌ edit_image Unavailable\n\n"
                "OpenAI provider is not configured. "
                "Set `OPENAI_API_KEY` or pass `openai_api_key` to enable image editing "
                "via gpt-image-2."
            )
            return msg

        prompt_hash = sha256(params.prompt.encode("utf-8")).hexdigest()
        start_event: dict[str, object] = {
            "request_id": request_id,
            "tool": "edit_image",
            "prompt_length": len(params.prompt),
            "prompt_sha256": prompt_hash,
            "image_path": params.image_path,
            "mask_path": params.mask_path,
            "size": params.size,
            "quality": params.quality,
            "background": params.background,
            "output_format_encoding": params.openai_output_format,
            "output_compression": params.openai_output_compression,
            "input_fidelity": params.input_fidelity,
            "n": params.n,
            "openai_model": params.openai_model,
            "output_path": params.output_path,
            "output_format": params.output_format.value if params.output_format else None,
        }
        if settings.log_prompts:
            start_event["prompt"] = params.prompt
        log_event("image.edit.start", **start_event)

        provider = registry.get_provider("openai", api_key=params.openai_api_key)
        # edit_image is OpenAI-specific; cast to the concrete type for mypy.
        from .providers.openai_provider import OpenAIProvider

        assert isinstance(provider, OpenAIProvider)

        result = await provider.edit_image(
            prompt=params.prompt,
            image_path=params.image_path,
            mask_path=params.mask_path,
            size=params.size,
            quality=params.quality,
            background=params.background,
            openai_output_format=params.openai_output_format,
            openai_output_compression=params.openai_output_compression,
            input_fidelity=params.input_fidelity,
            n=params.n,
            openai_model=params.openai_model,
            api_key=params.openai_api_key,
            output_path=params.output_path,
        )

        log_event(
            "image.edit.result",
            request_id=request_id,
            success=result.success,
            provider=result.provider,
            model=result.model,
            image_path=str(result.image_path) if result.image_path else None,
            generation_time_seconds=result.generation_time_seconds,
            error=result.error,
        )

        if params.output_format == OutputFormat.JSON:
            return format_result_json(result)
        return format_result_markdown(result)

    except ImagenError as e:
        logger.exception("Image edit failed")
        log_event("image.edit.error", request_id=request_id, error=str(e))
        error_response = {"success": False, "error": e.user_message}
        if params.output_format == OutputFormat.JSON:
            return json.dumps(error_response, indent=2)
        return f"## ❌ Image Edit Failed\n\n**Error ({type(e).__name__}):** {e.user_message}"

    except Exception as e:
        logger.exception("Image edit failed")
        sanitized = _sanitize_message(str(e))
        log_event("image.edit.error", request_id=request_id, error=sanitized)
        error_response = {"success": False, "error": sanitized}
        if params.output_format == OutputFormat.JSON:
            return json.dumps(error_response, indent=2)
        return f"## ❌ Image Edit Failed\n\n**Error:** {sanitized}"


@mcp.tool(
    name="list_providers",
    annotations=ToolAnnotations(
        title="List Providers",
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def list_providers() -> str:
    """List available image generation providers and their capabilities.

    Returns a comparison of available providers including:
    - Which providers have API keys configured
    - Best use cases for each provider
    - Feature comparison (text rendering, resolution, etc.)

    Use this to understand which provider to choose for your task.
    """
    registry = get_provider_registry()
    return registry.get_comparison()


@mcp.tool(
    name="list_conversations",
    annotations=ToolAnnotations(
        title="List Conversations",
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def list_conversations(params: ListConversationsInput) -> str:
    """List saved image generation conversations.

    Returns recent conversations that can be continued for refinement.
    Each conversation tracks the provider used and generation history.

    Args:
        params: Options for filtering and formatting the list.

    Returns:
        List of conversations with metadata.
    """
    try:
        registry = get_provider_registry()

        # Read straight from the persistent store — no provider/network client
        # construction (or API key) needed just to list conversations.
        conversations = registry.list_conversations(
            limit=params.limit or 10, provider_filter=params.provider
        )

        if not conversations:
            return "## 📝 Conversations\n\n*No active conversations found.*"

        lines = ["## 📝 Conversations", ""]

        for conv in conversations:
            lines.append(f"### `{conv['id']}` ({conv['provider']})")
            lines.append(f"- **Messages:** {conv['message_count']}")
            lines.append(f"- **Last:** {conv['last_message']}")
            lines.append("")

        return "\n".join(lines)

    except ImagenError as e:
        logger.exception("Failed to list conversations")
        return f"## ❌ Error\n\nFailed to list conversations ({type(e).__name__}): {e.user_message}"

    except Exception as e:
        logger.exception("Failed to list conversations")
        sanitized = _sanitize_message(str(e))
        return f"## ❌ Error\n\nFailed to list conversations: {sanitized}"


@mcp.tool(
    name="list_gemini_models",
    annotations=ToolAnnotations(
        title="List Gemini Models",
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=True,  # queries the Gemini API
    ),
)
async def list_gemini_models() -> str:
    """List available Gemini models that support image generation.

    Queries the Gemini API to show which models are available for image
    generation with your API key. Useful for troubleshooting or choosing
    alternative models.

    Returns:
        List of available Gemini image models with their capabilities.
    """
    import httpx

    settings = get_settings()

    if not settings.has_gemini_key():
        return (
            "## ❌ No Gemini API Key\n\nSet GEMINI_API_KEY or GOOGLE_API_KEY environment variable."
        )

    api_key: str = settings.gemini_api_key  # type: ignore[assignment]  # guarded by has_gemini_key()
    api_base = "https://generativelanguage.googleapis.com/v1beta"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{api_base}/models",
                headers={"x-goog-api-key": api_key},
            )
            response.raise_for_status()
            data = response.json()

        image_models = []
        for model in data.get("models", []):
            name = model.get("name", "")
            methods = model.get("supportedGenerationMethods", [])

            # Check if model might support images
            if any(x in name.lower() for x in ["image", "imagen", "flash-exp", "nano-banana"]):
                image_models.append(
                    {
                        "name": name.replace("models/", ""),
                        "methods": methods,
                        "description": model.get("description", "")[:100],
                    }
                )

        if not image_models:
            return (
                "## No Image Models Found\n\n"
                "No image generation models available with current API key."
            )

        lines = ["## Available Gemini Image Models\n"]
        for m in image_models:
            lines.append(f"### {m['name']}")
            lines.append(f"- **Methods:** {', '.join(m['methods'])}")
            if m["description"]:
                lines.append(f"- {m['description']}")
            lines.append("")

        return "\n".join(lines)

    except ImagenError as e:
        logger.exception("Error listing Gemini models")
        return f"## ❌ Error\n\nFailed to list models ({type(e).__name__}): {e.user_message}"

    except Exception as e:
        logger.exception("Error listing Gemini models")
        sanitized = _sanitize_message(str(e))
        return f"## ❌ Error\n\nFailed to list models: {sanitized}"


@mcp.tool(
    name="estimate_cost",
    annotations=ToolAnnotations(
        title="Estimate Cost",
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=False,  # local pricing table; no network call
    ),
)
async def estimate_cost(params: CostEstimateInput) -> str:
    """Estimate the cost of generating an image *without* generating it.

    Runs the same provider auto-selection as ``generate_image`` (unless you
    pin a provider) and looks up an approximate price from a local pricing
    table. Useful for comparing providers/qualities before committing.

    The figure is a ballpark — real cost depends on live provider pricing
    and, for OpenAI, actual image output tokens.

    Args:
        params: Prompt plus optional provider/quality/size/n.

    Returns:
        A formatted cost estimate.
    """
    try:
        registry = get_provider_registry()

        explicit_provider = None
        if params.provider and params.provider != Provider.AUTO:
            explicit_provider = params.provider.value

        # Resolve which provider would be used (auto-selection or explicit).
        try:
            _, recommendation = registry.get_provider_for_prompt(
                params.prompt,
                size=params.size,
                explicit_provider=explicit_provider,
            )
            provider_name = recommendation.provider
        except Exception:
            # No provider configured — still estimate for the requested/auto
            # provider so callers get a number without needing API keys.
            provider_name = explicit_provider or "openai"

        est = estimate_generation_cost(
            provider_name,
            quality=params.quality,
            size=params.size,
            n=params.n,
        )

        log_event(
            "image.cost.estimate",
            provider=est.provider,
            quality=est.quality,
            size=est.size,
            n=est.n,
            total_usd=est.total_usd,
        )

        if params.output_format == OutputFormat.JSON:
            return json.dumps(
                {
                    "provider": est.provider,
                    "model": est.model,
                    "quality": est.quality,
                    "size": est.size,
                    "n": est.n,
                    "per_image_usd": est.per_image_usd,
                    "total_usd": est.total_usd,
                    "approximate": est.approximate,
                    "note": est.note,
                },
                indent=2,
            )
        return format_cost_estimate(est)

    except Exception as e:
        logger.exception("Cost estimation failed")
        sanitized = _sanitize_message(str(e))
        return f"## ❌ Cost Estimation Failed\n\n**Error:** {sanitized}"


@mcp.tool(
    name="generate_image_batch",
    annotations=ToolAnnotations(
        title="Generate Image Batch",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    ),
)
async def generate_image_batch(params: BatchGenerationInput, ctx: Context | None = None) -> str:
    """Generate many images concurrently from a list of prompts.

    Each item runs through the same auto provider selection as ``generate_image``,
    bounded by ``max_concurrency``. Per-item failures are isolated — one bad
    prompt does not fail the whole batch. Returns every result (saved paths plus
    any per-item errors).

    Use this instead of calling ``generate_image`` in a loop: 8 prompts that
    would take ~4 minutes serially complete in roughly one generation's time
    (subject to max_concurrency and provider rate limits).

    Args:
        params: The batch (items + concurrency + optional default provider).

    Returns:
        A formatted summary of all results.
    """
    batch_id = uuid4().hex[:12]
    registry = get_provider_registry()
    total = len(params.items)
    semaphore = asyncio.Semaphore(params.max_concurrency)
    progress_lock = asyncio.Lock()
    done = 0

    default_provider = (
        params.default_provider.value
        if params.default_provider and params.default_provider != Provider.AUTO
        else None
    )

    log_event(
        "image.batch.start",
        batch_id=batch_id,
        items=total,
        max_concurrency=params.max_concurrency,
    )

    async def _run_one(index: int, item: BatchItem) -> dict[str, Any]:
        nonlocal done
        async with semaphore:
            try:
                explicit = (
                    item.provider.value
                    if item.provider and item.provider != Provider.AUTO
                    else default_provider
                )
                provider, recommendation = registry.get_provider_for_prompt(
                    item.prompt,
                    size=item.size,
                    reference_images=item.reference_images,
                    enable_google_search=item.enable_google_search or False,
                    explicit_provider=explicit,
                    openai_api_key=params.openai_api_key,
                    gemini_api_key=params.gemini_api_key,
                )
                openai_kwargs: dict[str, Any] = {}
                if recommendation.provider == "openai":
                    openai_kwargs = {
                        "openai_model": item.openai_model,
                        "quality": item.quality,
                        "n": item.n,
                    }
                result = await provider.generate_image(
                    item.prompt,
                    size=item.size,
                    aspect_ratio=item.aspect_ratio,
                    reference_images=item.reference_images,
                    enable_enhancement=False,
                    enable_google_search=item.enable_google_search or False,
                    api_key=params.openai_api_key
                    if recommendation.provider == "openai"
                    else params.gemini_api_key,
                    model=item.gemini_model if recommendation.provider == "gemini" else None,
                    output_path=item.output_path,
                    **openai_kwargs,
                )
                return {
                    "index": index,
                    "prompt": item.prompt,
                    "provider": recommendation.provider,
                    "result": result,
                }
            except Exception as e:
                sanitized = _sanitize_message(str(e))
                log_event("image.batch.item.error", batch_id=batch_id, index=index, error=sanitized)
                return {
                    "index": index,
                    "prompt": item.prompt,
                    "provider": "unknown",
                    "result": ImageResult(
                        success=False,
                        provider="unknown",
                        model="unknown",
                        prompt=item.prompt,
                        error=sanitized,
                    ),
                }
            finally:
                async with progress_lock:
                    done += 1
                    await _report_progress(ctx, done, total, f"Generated {done}/{total}")

    results = await asyncio.gather(
        *(_run_one(i, item) for i, item in enumerate(params.items)),
        return_exceptions=False,
    )
    results.sort(key=lambda r: r["index"])
    succeeded = sum(1 for r in results if r["result"].success)

    log_event("image.batch.result", batch_id=batch_id, total=total, succeeded=succeeded)

    if params.output_format == OutputFormat.JSON:
        data = {
            "batch_id": batch_id,
            "total": total,
            "succeeded": succeeded,
            "results": [
                {
                    "index": r["index"],
                    "prompt": r["prompt"],
                    "provider": r["provider"],
                    **r["result"].to_dict(),
                }
                for r in results
            ],
        }
        return json.dumps(data, indent=2, default=str)

    lines = [f"## 🧺 Batch Generation — {succeeded}/{total} succeeded", ""]
    for r in results:
        res: ImageResult = r["result"]
        snippet = r["prompt"][:60]
        if res.success:
            paths = [res.image_path, *(res.additional_paths or [])]
            lines.append(f"### ✅ [{r['index']}] {snippet} ({r['provider']})")
            lines.extend(f"- `{p}`" for p in paths if p)
        else:
            lines.append(f"### ❌ [{r['index']}] {snippet}")
            lines.append(f"- **Error:** {res.error}")
        lines.append("")
    return "\n".join(lines)


# ============================
# Server Entry Point
# ============================


def main() -> None:
    """Console-script entry point.

    Transport is selected via ``IMAGEN_MCP_TRANSPORT`` (default ``stdio``):

    - ``stdio``           — local subprocess transport (Claude Desktop default)
    - ``streamable-http`` — HTTP service; honors ``IMAGEN_MCP_HOST`` /
      ``IMAGEN_MCP_PORT`` (default ``127.0.0.1:8000``)
    - ``sse``             — legacy Server-Sent Events transport

    Streamable HTTP (MCP spec 2025-03-26) lets the server run as a web
    service that multiple clients can connect to, behind load balancers
    and proxies, instead of only as a local stdio subprocess.
    """
    import os

    from .config.dotenv import load_dotenv

    load_dotenv(override=False)

    transport = os.getenv("IMAGEN_MCP_TRANSPORT", "stdio").strip().lower()

    if transport in ("streamable-http", "http", "sse"):
        # Host/port live on FastMCP.settings; only override when provided so
        # the SDK defaults (127.0.0.1:8000) still apply otherwise.
        host = os.getenv("IMAGEN_MCP_HOST")
        port = os.getenv("IMAGEN_MCP_PORT")
        if host:
            mcp.settings.host = host
        if port:
            mcp.settings.port = int(port)
        if transport == "sse":
            logger.info("Starting imagen-mcp over sse transport")
            mcp.run(transport="sse")
        else:
            logger.info("Starting imagen-mcp over streamable-http transport")
            mcp.run(transport="streamable-http")
    else:
        mcp.run()


if __name__ == "__main__":
    main()
