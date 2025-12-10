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
from datetime import datetime
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

from .config.settings import get_settings
from .models.input_models import (
    ImageGenerationInput,
    ConversationalImageInput,
    ListConversationsInput,
    Provider,
    OutputFormat,
)
from .providers import get_provider_registry, ProviderRecommendation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("imagen_mcp")

# ============================
# Helper Functions
# ============================


def format_result_markdown(
    result,
    recommendation: Optional[ProviderRecommendation] = None,
) -> str:
    """Format image generation result as markdown."""
    if not result.success:
        return f"## ❌ Image Generation Failed\n\n**Error:** {result.error}"

    lines = [
        "## ✅ Image Generated Successfully",
        "",
    ]

    # Provider info
    if recommendation:
        lines.extend([
            f"**Provider:** {result.provider.title()} (auto-selected)",
            f"**Reasoning:** {recommendation.reasoning}",
            "",
        ])
    else:
        lines.append(f"**Provider:** {result.provider.title()}")
        lines.append("")

    # Image info
    if result.image_path:
        lines.extend([
            f"📁 **Saved to:** `{result.image_path}`",
            "",
        ])

    # Metadata
    lines.extend([
        f"**Model:** {result.model}",
        f"**Size:** {result.size or 'default'}",
    ])

    if result.aspect_ratio:
        lines.append(f"**Aspect Ratio:** {result.aspect_ratio}")

    if result.generation_time_seconds:
        lines.append(f"**Generation Time:** {result.generation_time_seconds:.1f}s")

    # Conversation ID for refinement
    if result.conversation_id:
        lines.extend([
            "",
            "## 🔄 Continue Refining",
            f"**Conversation ID:** `{result.conversation_id}`",
            "*Use this ID to refine this image further.*",
        ])

    # Gemini-specific: thinking mode
    if result.thoughts:
        lines.extend([
            "",
            "## 💭 Model Reasoning",
            f"*{len(result.thoughts)} thought steps processed*",
        ])

    # Gemini-specific: grounding
    if result.grounding_metadata:
        lines.extend([
            "",
            "## 🔍 Real-time Data Sources",
            "*Used Google Search for current information*",
        ])

    return "\n".join(lines)


def format_result_json(result, recommendation: Optional[ProviderRecommendation] = None) -> str:
    """Format image generation result as JSON."""
    data = result.to_dict()
    if recommendation:
        data["recommendation"] = {
            "provider": recommendation.provider,
            "confidence": recommendation.confidence,
            "reasoning": recommendation.reasoning,
            "alternative": recommendation.alternative,
        }
    return json.dumps(data, indent=2, default=str)


# ============================
# MCP Tools
# ============================


@mcp.tool(name="generate_image")
async def generate_image(params: ImageGenerationInput):
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
    try:
        registry = get_provider_registry()

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
        )

        logger.info(
            f"Selected provider: {recommendation.provider} "
            f"(confidence: {recommendation.confidence:.0%})"
        )

        # Generate image
        result = await provider.generate_image(
            params.prompt,
            size=params.size,
            aspect_ratio=params.aspect_ratio,
            reference_images=params.reference_images,
            enable_enhancement=params.enhance_prompt if params.enhance_prompt is not None else True,
            enable_google_search=params.enable_google_search or False,
            api_key=params.openai_api_key if recommendation.provider == "openai" else params.gemini_api_key,
        )

        # Format response
        if params.output_format == OutputFormat.JSON:
            return format_result_json(result, recommendation)
        else:
            return format_result_markdown(result, recommendation)

    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        error_response = {
            "success": False,
            "error": str(e),
        }
        if params.output_format == OutputFormat.JSON:
            return json.dumps(error_response, indent=2)
        else:
            return f"## ❌ Image Generation Failed\n\n**Error:** {str(e)}"


@mcp.tool(name="conversational_image")
async def conversational_image(params: ConversationalImageInput):
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
    try:
        registry = get_provider_registry()

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
        )

        # For now, skip dialogue and generate directly
        # TODO: Integrate dialogue system with provider awareness
        result = await provider.generate_image(
            params.prompt,
            size=params.size,
            aspect_ratio=params.aspect_ratio,
            conversation_id=params.conversation_id,
            reference_images=params.reference_images,
            enable_enhancement=not params.skip_dialogue,
            enable_google_search=params.enable_google_search or False,
            api_key=params.openai_api_key if recommendation.provider == "openai" else params.gemini_api_key,
        )

        if params.output_format == OutputFormat.JSON:
            return format_result_json(result, recommendation)
        else:
            return format_result_markdown(result, recommendation)

    except Exception as e:
        logger.error(f"Conversational image generation failed: {e}")
        return f"## ❌ Generation Failed\n\n**Error:** {str(e)}"


@mcp.tool(name="list_providers")
async def list_providers():
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
async def list_conversations(params: ListConversationsInput):
    """List saved image generation conversations.

    Returns recent conversations that can be continued for refinement.
    Each conversation tracks the provider used and generation history.

    Args:
        params: Options for filtering and formatting the list.

    Returns:
        List of conversations with metadata.
    """
    # TODO: Implement conversation listing from storage
    return "## 📝 Conversations\n\n*Conversation listing coming soon.*"


# ============================
# Server Entry Point
# ============================


def create_app():
    """Create the MCP server application."""
    return mcp


if __name__ == "__main__":
    mcp.run()
