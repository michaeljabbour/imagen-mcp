"""
Pydantic input models for imagen-mcp tools.

These models define the parameters accepted by MCP tools
with rich descriptions for Claude to understand how to use them.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Provider(str, Enum):
    """Available image generation providers."""

    AUTO = "auto"  # Auto-select based on prompt analysis
    OPENAI = "openai"  # OpenAI GPT-Image-1
    GEMINI = "gemini"  # Google Gemini 3 Pro Image (Nano Banana Pro)


class OutputFormat(str, Enum):
    """Output format for tool responses."""

    MARKDOWN = "markdown"
    JSON = "json"


class ImageGenerationInput(BaseModel):
    """
    Input model for unified image generation.

    This model supports both OpenAI and Gemini providers with
    intelligent auto-selection based on prompt analysis.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    prompt: str = Field(
        ...,
        description=(
            "Text description of the desired image. Be specific about subject, "
            "composition, style, lighting, and mood. "
            "For text-heavy images (menus, infographics), OpenAI is auto-selected. "
            "For portraits and products, Gemini is auto-selected."
        ),
        min_length=1,
        max_length=4000,
    )

    provider: Provider | None = Field(
        default=Provider.AUTO,
        description=(
            "Image generation provider to use:\n"
            "- 'auto' (default): Automatically selects best provider based on prompt\n"
            "- 'openai': OpenAI GPT-Image-1 - best for text rendering, infographics\n"
            "- 'gemini': Gemini Nano Banana Pro - best for portraits, products, 4K"
        ),
    )

    # Size/resolution (provider-specific)
    size: str | None = Field(
        default=None,
        description=(
            "Image size. Format depends on provider:\n"
            "- OpenAI: '1024x1024' (square), '1024x1536' (portrait), '1536x1024' (landscape)\n"
            "- Gemini: '1K' (fast), '2K' (default), '4K' (max quality)\n"
            "Auto-detected from prompt if not specified."
        ),
    )

    aspect_ratio: str | None = Field(
        default=None,
        description=(
            "Aspect ratio (Gemini only). Options: "
            "'1:1', '2:3', '3:2', '3:4', '4:3', '4:5', '5:4', '9:16', '16:9', '21:9'. "
            "For OpenAI, this is converted to the nearest supported size."
        ),
    )

    # Gemini-specific features
    reference_images: list[str] | None = Field(
        default=None,
        description=(
            "Base64-encoded reference images for style/character consistency (Gemini only). "
            "Up to 14 images: 6 for objects, 5 for human portraits. "
            "If provided, Gemini provider is automatically selected."
        ),
    )

    enable_google_search: bool | None = Field(
        default=False,
        description=(
            "Enable Google Search grounding for real-time data (Gemini only). "
            "Use for current weather, stock prices, live events, etc. "
            "If enabled, Gemini provider is automatically selected."
        ),
    )

    gemini_model: str | None = Field(
        default=None,
        description=(
            "Specific Gemini model to use (Gemini only). Options:\n"
            "- 'gemini-3-pro-image-preview': Nano Banana Pro, highest quality (default)\n"
            "- 'gemini-2.0-flash-exp-image-generation': Fast experimental model\n"
            "- 'imagen-3.0-generate-002': Imagen 3.0 model"
        ),
    )

    # Common options
    enhance_prompt: bool | None = Field(
        default=True,
        description="Whether to enhance the prompt for better results.",
    )

    output_path: str | None = Field(
        default=None,
        description=(
            "Optional path to save the generated image. "
            "If a directory, saves with generated filename. "
            "If a file path, saves to that exact path. "
            "Supports `~` and environment variables; defaults to `OUTPUT_DIR/{provider}` or "
            "`~/Downloads/images/{provider}`."
        ),
    )

    output_format: OutputFormat | None = Field(
        default=OutputFormat.MARKDOWN,
        description="Output format for the tool response.",
    )

    # API keys (optional overrides)
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key override (uses OPENAI_API_KEY env var if not provided).",
    )

    gemini_api_key: str | None = Field(
        default=None,
        description="Gemini API key override (uses GEMINI_API_KEY env var if not provided).",
    )


class ConversationalImageInput(BaseModel):
    """
    Input model for conversational image generation with multi-turn refinement.

    Supports iterative refinement where each prompt builds on previous results.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="allow",  # Allow dialogue_responses
    )

    prompt: str = Field(
        ...,
        description=(
            "Text description for new image or refinement instruction. "
            "For refinements, use natural language like 'make it darker' or 'add more detail'."
        ),
        min_length=1,
        max_length=4000,
    )

    conversation_id: str | None = Field(
        default=None,
        description=(
            "Conversation ID from previous generation to continue refining. "
            "Omit to start a new conversation. Auto-generated if not provided."
        ),
    )

    provider: Provider | None = Field(
        default=Provider.AUTO,
        description=(
            "Provider to use. Note: Cannot switch providers mid-conversation. "
            "The provider from the first message in a conversation is used throughout."
        ),
    )

    # Dialogue system options
    dialogue_mode: str | None = Field(
        default="guided",
        description=(
            "Dialogue depth for pre-generation refinement:\n"
            "- 'quick': 1-2 questions, fast path\n"
            "- 'guided': 3-5 questions, balanced (default)\n"
            "- 'explorer': Deep exploration with 6+ questions\n"
            "- 'skip': Direct generation, no dialogue"
        ),
    )

    skip_dialogue: bool | None = Field(
        default=False,
        description="Set to true to skip dialogue and generate immediately.",
    )

    dialogue_responses: dict[str, Any] | None = Field(
        default=None,
        description="User responses to dialogue questions (internal use).",
    )

    # Size/resolution
    size: str | None = Field(
        default=None,
        description="Image size (provider-specific format). Auto-detected if not specified.",
    )

    aspect_ratio: str | None = Field(
        default=None,
        description="Aspect ratio (Gemini only).",
    )

    # Input image for refinement
    input_image_file_id: str | None = Field(
        default=None,
        description=(
            "File ID from previous generation to refine (OpenAI only). "
            "Obtained from prior tool responses."
        ),
    )

    input_image_path: str | None = Field(
        default=None,
        description="Absolute path to local image file to upload and refine.",
    )

    # Reference images (Gemini only)
    reference_images: list[str] | None = Field(
        default=None,
        description="Base64-encoded reference images (Gemini only, up to 14).",
    )

    enable_google_search: bool | None = Field(
        default=False,
        description="Enable Google Search grounding (Gemini only).",
    )

    # Gemini-specific
    gemini_model: str | None = Field(
        default=None,
        description=(
            "Specific Gemini model (Gemini only):\n"
            "- 'gemini-3-pro-image-preview': Nano Banana Pro (default)\n"
            "- 'gemini-2.0-flash-exp-image-generation': Fast experimental\n"
            "- 'imagen-3.0-generate-002': Imagen 3.0"
        ),
    )

    # OpenAI-specific
    assistant_model: str | None = Field(
        default="gpt-4o",
        description="GPT model for understanding refinement instructions (OpenAI only).",
    )

    # Output options
    output_path: str | None = Field(
        default=None,
        description=(
            "Optional path to save the generated image. "
            "If a directory, saves with generated filename. "
            "If a file path, saves to that exact path. "
            "Supports `~` and environment variables; defaults to `OUTPUT_DIR/{provider}` or "
            "`~/Downloads/images/{provider}`."
        ),
    )

    output_format: OutputFormat | None = Field(
        default=OutputFormat.MARKDOWN,
        description="Output format for the tool response.",
    )

    # API keys
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key override.",
    )

    gemini_api_key: str | None = Field(
        default=None,
        description="Gemini API key override.",
    )


class ListConversationsInput(BaseModel):
    """Input model for listing saved conversations."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    limit: int | None = Field(
        default=10,
        description="Maximum number of conversations to return.",
        ge=1,
        le=100,
    )

    provider: str | None = Field(
        default=None,
        description="Filter by provider ('openai' or 'gemini').",
    )

    output_format: OutputFormat | None = Field(
        default=OutputFormat.MARKDOWN,
        description="Output format for the tool response.",
    )
