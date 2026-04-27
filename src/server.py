#!/usr/bin/env python3
"""
imagen-mcp: Multi-Provider Image Generation MCP Server

An MCP server that provides intelligent image generation using multiple providers:
- OpenAI GPT-Image-1: Best for text rendering, infographics, comics
- Google Gemini 3 Pro Image (Nano Banana Pro): Best for portraits, products, 4K

The server automatically selects the best provider based on prompt analysis,
or allows explicit provider selection.
"""

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from hashlib import sha256
from typing import Any
from uuid import uuid4

from mcp.server.fastmcp import FastMCP

from .config.settings import get_settings
from .exceptions import ImagenError, _sanitize_message
from .models.input_models import (
    ConversationalImageInput,
    EditImageInput,
    ImageGenerationInput,
    ListConversationsInput,
    OutputFormat,
    Provider,
)
from .providers import ImageResult, ProviderRecommendation, get_provider_registry
from .services.dialogue import DialogueSystem, create_dialogue_response
from .services.logging_config import configure_logging, log_event

logger = logging.getLogger(__name__)


# ============================
# Lifespan (startup / shutdown)
# ============================


@asynccontextmanager
async def _lifespan(_app: FastMCP) -> AsyncIterator[None]:
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


@mcp.tool(name="generate_image")
async def generate_image(params: ImageGenerationInput) -> str:
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
            enable_enhancement=params.enhance_prompt if params.enhance_prompt is not None else True,
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


@mcp.tool(name="conversational_image")
async def conversational_image(params: ConversationalImageInput) -> str:
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

        # If dialogue needs more info, return questions (unless skip_dialogue)
        if not params.skip_dialogue and not dialogue_result.should_generate:
            log_event(
                "image.conversation.dialogue",
                request_id=request_id,
                questions_count=len(dialogue_result.questions),
                detected_intent=dialogue_result.detected_intent,
            )
            return create_dialogue_response(dialogue_result, recommendation.provider)

        # Generate the image
        result = await provider.generate_image(
            params.prompt,
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


@mcp.tool(name="edit_image")
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


@mcp.tool(name="list_providers")
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


@mcp.tool(name="list_conversations")
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

        # Ensure providers are initialized to get their conversations
        # If we filter by provider, only initialize that one
        if params.provider:
            if registry.is_provider_available(params.provider):
                registry.get_provider(params.provider)
        else:
            # Initialize all available
            for p in registry.list_providers():
                try:
                    registry.get_provider(p)
                except Exception:
                    pass

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


@mcp.tool(name="list_gemini_models")
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


# ============================
# Server Entry Point
# ============================


if __name__ == "__main__":
    from .config.dotenv import load_dotenv

    load_dotenv(override=False)
    mcp.run()
