"""
Pydantic input models for imagen-mcp tools.

These models define the parameters accepted by MCP tools
with rich descriptions for Claude to understand how to use them.
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class Provider(str, Enum):
    """Available image generation providers."""

    AUTO = "auto"  # Auto-select based on prompt analysis
    OPENAI = "openai"  # OpenAI gpt-image-2 (ChatGPT Images 2.0)
    GEMINI = "gemini"  # Google Gemini 3 Pro Image (Nano Banana Pro)


class OutputFormat(str, Enum):
    """Output format for tool responses."""

    MARKDOWN = "markdown"
    JSON = "json"


class ImageGenerationInput(BaseModel):
    """
    Input model for unified image generation.

    This model supports both OpenAI (gpt-image-2) and Gemini providers with
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
            "For text-heavy images (menus, infographics, UI mockups), OpenAI is "
            "auto-selected. For portraits, products, and 4K output, Gemini is "
            "auto-selected."
        ),
        min_length=1,
        max_length=4000,
    )

    provider: Provider | None = Field(
        default=Provider.AUTO,
        description=(
            "Image generation provider to use:\n"
            "- 'auto' (default): Automatically selects best provider based on prompt\n"
            "- 'openai': OpenAI gpt-image-2 - best for text, UI mockups, diagrams\n"
            "- 'gemini': Gemini Nano Banana Pro - best for portraits, products, 4K"
        ),
    )

    # Size/resolution (provider-specific)
    size: str | None = Field(
        default=None,
        description=(
            "Image size. Format depends on provider:\n"
            "- OpenAI: 'auto', '1024x1024' (square), '1024x1536' (portrait), "
            "'1536x1024' (landscape), '1792x1024' (widescreen), "
            "'1024x1792' (tall), '512x512', '256x256'\n"
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

    # --- Gemini-specific features ---
    reference_images: list[str] | None = Field(
        default=None,
        description=(
            "Base64-encoded reference images for style/character consistency "
            "(Gemini only). Up to 14 images: 6 for objects, 5 for human portraits. "
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
            "Specific Google image model (Gemini only). Accepts canonical "
            "API IDs or friendly aliases.\n"
            "Nano Banana family (conversational, reference images, search):\n"
            "- 'gemini-3.1-flash-image-preview' / alias 'nano-banana-2' "
            "(default, fast, current default across Gemini/Search/Flow)\n"
            "- 'gemini-3-pro-image-preview' / alias 'nano-banana-pro' "
            "(highest fidelity, 4K, Thinking mode)\n"
            "Imagen 4 family (text-to-image only, no editing/references, "
            "DEPRECATED — shutdown 2026-06-24):\n"
            "- 'imagen-4.0-generate-001' / alias 'imagen-4' (standard)\n"
            "- 'imagen-4.0-ultra-generate-001' / alias 'imagen-4-ultra'\n"
            "- 'imagen-4.0-fast-generate-001' / alias 'imagen-4-fast'"
        ),
    )

    # --- OpenAI gpt-image-2 specific features ---
    openai_model: str | None = Field(
        default=None,
        description=(
            "Specific OpenAI image model to use (OpenAI only). Options:\n"
            "- 'gpt-image-2' (default): ChatGPT Images 2.0 — fast, 99% text accuracy\n"
            "- 'gpt-image-1.5': Interim Dec 2025 model\n"
            "- 'gpt-image-1': Legacy Apr 2025 model"
        ),
    )

    quality: str | None = Field(
        default=None,
        description=(
            "Image quality tier (OpenAI only):\n"
            "- 'auto' (default): let the model choose\n"
            "- 'low' / 'medium' / 'high': explicit tier\n"
            "Higher quality costs more output tokens but produces sharper images."
        ),
    )

    openai_output_format: str | None = Field(
        default=None,
        description=(
            "Image file encoding (OpenAI only). Options: 'png' (default, supports "
            "transparency), 'jpeg' (smaller, lossy), 'webp' (best compression)."
        ),
    )

    openai_output_compression: int | None = Field(
        default=None,
        description=(
            "Compression level 0-100 for jpeg/webp (OpenAI only). "
            "Higher = better quality, larger file. Ignored for png."
        ),
        ge=0,
        le=100,
    )

    background: str | None = Field(
        default=None,
        description=(
            "Background treatment (OpenAI only):\n"
            "- 'auto' (default): model decides\n"
            "- 'transparent': requires png or webp format\n"
            "- 'opaque': solid background"
        ),
    )

    moderation: str | None = Field(
        default=None,
        description=(
            "Content moderation strictness (OpenAI only):\n"
            "- 'auto' (default): standard filtering\n"
            "- 'low': more permissive (may still refuse unsafe content)"
        ),
    )

    n: int | None = Field(
        default=None,
        description=(
            "Number of images to generate in a single request (OpenAI only, 1-10). "
            "Default is 1. Gemini always returns 1 per call."
        ),
        ge=1,
        le=10,
    )

    # --- Common options ---
    enhance_prompt: bool | None = Field(
        default=True,
        description=(
            "Whether to enhance the prompt for better results. For OpenAI, "
            "enables the multi-turn Responses-API flow (richer context). "
            "When False, uses a direct /images/generations call for speed."
        ),
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
        description="Output format for the tool response (markdown or json).",
    )

    # --- API keys (optional overrides — hidden from repr/serialization) ---
    openai_api_key: str | None = Field(
        default=None,
        repr=False,
        exclude=True,
        description="OpenAI API key override (uses OPENAI_API_KEY env var if not provided).",
    )

    gemini_api_key: str | None = Field(
        default=None,
        repr=False,
        exclude=True,
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
        extra="ignore",  # Silently drop unknown fields
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
            "- 'gemini-2.5-flash-preview-image-generation': Gemini 2.5 Flash (default)\n"
            "- 'gemini-3-pro-image-preview': Nano Banana Pro, highest quality"
        ),
    )

    # OpenAI-specific
    openai_model: str | None = Field(
        default=None,
        description=("Specific OpenAI image model (OpenAI only). Default 'gpt-image-2'."),
    )

    assistant_model: str | None = Field(
        default="gpt-4o",
        description="GPT model for understanding refinement instructions (OpenAI only).",
    )

    quality: str | None = Field(
        default=None,
        description="Image quality tier: 'auto' / 'low' / 'medium' / 'high' (OpenAI only).",
    )

    background: str | None = Field(
        default=None,
        description="Background: 'auto' / 'transparent' / 'opaque' (OpenAI only).",
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

    # API keys (hidden from repr/serialization)
    openai_api_key: str | None = Field(
        default=None,
        repr=False,
        exclude=True,
        description="OpenAI API key override.",
    )

    gemini_api_key: str | None = Field(
        default=None,
        repr=False,
        exclude=True,
        description="Gemini API key override.",
    )


class EditImageInput(BaseModel):
    """
    Input model for image editing with gpt-image-2.

    Uses the /images/edits endpoint with input_fidelity=high by default,
    which preserves unchanged pixels more faithfully than direct generation.
    Supports inpainting via optional mask.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    prompt: str = Field(
        ...,
        description=(
            "Instruction describing the edit. "
            "Examples: 'add red roses to the ad frames', "
            "'change the sky to sunset', 'remove the person in the background'."
        ),
        min_length=1,
        max_length=4000,
    )

    image_path: str = Field(
        ...,
        description=(
            "Absolute or ~-expanded path to the source image on disk (png / jpeg / webp)."
        ),
    )

    mask_path: str | None = Field(
        default=None,
        description=(
            "Optional path to a PNG mask. Transparent pixels indicate regions "
            "the model should edit; opaque pixels remain unchanged."
        ),
    )

    size: str | None = Field(
        default=None,
        description=(
            "Output size: 'auto', '1024x1024', '1024x1536', '1536x1024', "
            "'512x512', '256x256'. Defaults to 'auto'."
        ),
    )

    quality: str | None = Field(
        default=None,
        description="Quality tier: 'auto' / 'low' / 'medium' / 'high'.",
    )

    background: str | None = Field(
        default=None,
        description="Background: 'auto' / 'transparent' / 'opaque'.",
    )

    openai_output_format: str | None = Field(
        default=None,
        description="Output encoding: 'png' / 'jpeg' / 'webp'.",
    )

    openai_output_compression: int | None = Field(
        default=None,
        description="Compression 0-100 for jpeg/webp.",
        ge=0,
        le=100,
    )

    input_fidelity: str | None = Field(
        default=None,
        description=(
            "How faithfully to preserve the source image: 'high' (default) or 'low'. "
            "gpt-image-2 performs best at 'high', which keeps unchanged pixels constant."
        ),
    )

    n: int | None = Field(
        default=None,
        description="Number of edited variants to generate (1-10).",
        ge=1,
        le=10,
    )

    openai_model: str | None = Field(
        default=None,
        description="OpenAI image model to use. Default 'gpt-image-2'.",
    )

    output_path: str | None = Field(
        default=None,
        description="Optional path to save the edited image.",
    )

    output_format: OutputFormat | None = Field(
        default=OutputFormat.MARKDOWN,
        description="Output format for the tool response.",
    )

    openai_api_key: str | None = Field(
        default=None,
        repr=False,
        exclude=True,
        description="OpenAI API key override.",
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
