"""
Constants for imagen-mcp providers.

This module defines provider-specific constants including supported sizes,
aspect ratios, quality tiers, and model identifiers.
"""

# ============================
# OpenAI gpt-image-2 Constants
# ============================
#
# Verified against openai>=2.31.0 SDK (images.generate / images.edit signatures).
# The SDK is the source of truth; these constants mirror the typed literal
# options so validate_params() can reject anything the API will reject.

OPENAI_API_BASE_URL = "https://api.openai.com/v1"

# Image generation models
OPENAI_MODELS = {
    "gpt-image-2": "gpt-image-2",  # ChatGPT Images 2.0 (GA April 2026) - default
    "gpt-image-1": "gpt-image-1",  # Legacy dedicated image model (April 2025)
    "gpt-image-1.5": "gpt-image-1.5",  # Interim model (Dec 2025)
    # Conversation orchestration models (used by the Responses-API multi-turn path)
    "gpt-5.1": "gpt-5.1",
    "gpt-4o": "gpt-4o",
}

# Default OpenAI model for image generation (ChatGPT Images 2.0)
DEFAULT_OPENAI_IMAGE_MODEL = "gpt-image-2"

# Sizes supported by the /images/generations endpoint (per SDK 2.31.0 literal).
# gpt-image-2 adds ultrawide (1792x1024) and ultratall (1024x1792) over 1.x.
OPENAI_SIZES = [
    "auto",  # Let the model choose
    "1024x1024",  # Square (baseline)
    "1536x1024",  # Landscape 3:2
    "1024x1536",  # Portrait 2:3
    "1792x1024",  # Widescreen 16:9-ish (new in 2.0-era)
    "1024x1792",  # Tall 9:16-ish (new in 2.0-era)
    "512x512",  # Small square (legacy)
    "256x256",  # Thumbnail (legacy)
]

# Sizes supported by the /images/edits endpoint (per SDK 2.31.0 literal).
# Note: edits do NOT support 1792x1024 / 1024x1792.
OPENAI_EDIT_SIZES = [
    "auto",
    "1024x1024",
    "1536x1024",
    "1024x1536",
    "512x512",
    "256x256",
]

# Quality tiers for gpt-image-2 (low/medium/high/auto are current;
# standard/hd are legacy DALL-E-era values still accepted by the SDK).
OPENAI_QUALITY_OPTIONS = ["auto", "low", "medium", "high", "standard", "hd"]
DEFAULT_OPENAI_QUALITY = "auto"

# Output image encoding formats
OPENAI_OUTPUT_FORMATS = ["png", "jpeg", "webp"]
DEFAULT_OPENAI_OUTPUT_FORMAT = "png"

# Background treatment — transparent requires png or webp
OPENAI_BACKGROUND_OPTIONS = ["auto", "transparent", "opaque"]
DEFAULT_OPENAI_BACKGROUND = "auto"

# Content moderation strictness
OPENAI_MODERATION_OPTIONS = ["auto", "low"]
DEFAULT_OPENAI_MODERATION = "auto"

# DALL-E-era style hint (kept for backward compat; largely a no-op on gpt-image-2)
OPENAI_STYLES = ["vivid", "natural"]

# Input fidelity for /images/edits — gpt-image-2 works best at high
OPENAI_INPUT_FIDELITY_OPTIONS = ["high", "low"]
DEFAULT_OPENAI_INPUT_FIDELITY = "high"

# Max number of images per request (SDK caps; OpenAI may return fewer)
OPENAI_MAX_N = 10

# ============================
# Google Gemini / Imagen Constants
# ============================
#
# Two distinct model families with different API shapes:
#
# - **Nano Banana** (`gemini-*-image*`) — uses the ``generateContent``
#   endpoint; supports multi-turn editing, reference images (up to 4),
#   Google Search grounding, and Thinking mode on Pro.
# - **Imagen 4** (`imagen-4.0-*`) — uses the ``:predict`` endpoint via
#   ``client.models.generate_images()``; text-to-image only, no
#   conversational editing, no reference images, no search grounding.
#
# Deprecation watch (as of 2026-04-21):
# - `gemini-2.5-flash-preview-image-generation` shut down Jan 15 2026
# - `gemini-2.5-flash-image`                     shuts down Oct 2 2026
# - All three `imagen-4.0-*-001` models          shut down Jun 24 2026
# Google's migration guidance points at Nano Banana 2 (speed) or Pro
# (quality) for everything.

# Nano Banana endpoint identifiers
GEMINI_ENDPOINT_GENERATECONTENT = "generateContent"  # Nano Banana
GEMINI_ENDPOINT_PREDICT = "predict"  # Imagen 4

GEMINI_MODELS: dict[str, dict[str, object]] = {
    # --- Nano Banana family (generateContent) -------------------------
    "gemini-3.1-flash-image-preview": {
        "marketing_name": "Nano Banana 2",
        "description": (
            "Nano Banana 2 (Gemini 3.1 Flash Image) — current default across "
            "Gemini app / Search / Flow. Fast, efficient, full feature set."
        ),
        "endpoint": GEMINI_ENDPOINT_GENERATECONTENT,
        "speed": "fast",
        "quality": "good",
        "max_resolution": "4K",
        "supports_conversational_edit": True,
        "supports_reference_images": True,
        "supports_google_search": True,
        "supports_thinking_mode": False,
    },
    "gemini-3-pro-image-preview": {
        "marketing_name": "Nano Banana Pro",
        "description": (
            "Nano Banana Pro (Gemini 3 Pro Image) — highest fidelity, 4K, "
            "Thinking mode for precise text rendering. Top-3 on LMArena."
        ),
        "endpoint": GEMINI_ENDPOINT_GENERATECONTENT,
        "speed": "slow",
        "quality": "best",
        "max_resolution": "4K",
        "supports_conversational_edit": True,
        "supports_reference_images": True,
        "supports_google_search": True,
        "supports_thinking_mode": True,
    },
    # --- Imagen 4 family (predict) ------------------------------------
    # DEPRECATED — shutdown 2026-06-24. Text-to-image only.
    "imagen-4.0-generate-001": {
        "marketing_name": "Imagen 4",
        "description": (
            "Imagen 4 Standard — text-to-image, mid-tier quality. "
            "DEPRECATED: shutdown 2026-06-24; migrate to Nano Banana 2 or Pro."
        ),
        "endpoint": GEMINI_ENDPOINT_PREDICT,
        "speed": "standard",
        "quality": "good",
        "max_resolution": "2K",
        "supports_conversational_edit": False,
        "supports_reference_images": False,
        "supports_google_search": False,
        "supports_thinking_mode": False,
        "deprecated": True,
        "shutdown_date": "2026-06-24",
    },
    "imagen-4.0-ultra-generate-001": {
        "marketing_name": "Imagen 4 Ultra",
        "description": (
            "Imagen 4 Ultra — highest quality text-to-image. "
            "DEPRECATED: shutdown 2026-06-24; migrate to Nano Banana Pro."
        ),
        "endpoint": GEMINI_ENDPOINT_PREDICT,
        "speed": "slow",
        "quality": "best",
        "max_resolution": "2K",
        "supports_conversational_edit": False,
        "supports_reference_images": False,
        "supports_google_search": False,
        "supports_thinking_mode": False,
        "deprecated": True,
        "shutdown_date": "2026-06-24",
    },
    "imagen-4.0-fast-generate-001": {
        "marketing_name": "Imagen 4 Fast",
        "description": (
            "Imagen 4 Fast — fastest text-to-image, 1K. "
            "DEPRECATED: shutdown 2026-06-24; migrate to Nano Banana 2."
        ),
        "endpoint": GEMINI_ENDPOINT_PREDICT,
        "speed": "very fast",
        "quality": "good",
        "max_resolution": "1K",
        "supports_conversational_edit": False,
        "supports_reference_images": False,
        "supports_google_search": False,
        "supports_thinking_mode": False,
        "deprecated": True,
        "shutdown_date": "2026-06-24",
    },
    # --- Legacy (kept for backward compat; shut down or soon-to-be) ---
    "gemini-2.5-flash-preview-image-generation": {
        "marketing_name": "Gemini 2.5 Flash Image (legacy)",
        "description": ("LEGACY — retired 2026-01-15. Prefer gemini-3.1-flash-image-preview."),
        "endpoint": GEMINI_ENDPOINT_GENERATECONTENT,
        "speed": "fast",
        "quality": "good",
        "max_resolution": "4K",
        "supports_conversational_edit": True,
        "supports_reference_images": True,
        "supports_google_search": True,
        "supports_thinking_mode": False,
        "deprecated": True,
        "shutdown_date": "2026-01-15",
    },
}

# Friendly aliases — users can pass these human-readable names and we map
# them to canonical API identifiers.  Extend as new marketing names land.
GEMINI_MODEL_ALIASES: dict[str, str] = {
    "nano-banana-2": "gemini-3.1-flash-image-preview",
    "nano-banana-pro": "gemini-3-pro-image-preview",
    "imagen-4": "imagen-4.0-generate-001",
    "imagen-4-ultra": "imagen-4.0-ultra-generate-001",
    "imagen-4-fast": "imagen-4.0-fast-generate-001",
}

# Fast lookups
GEMINI_IMAGEN_MODELS: set[str] = {
    mid for mid, meta in GEMINI_MODELS.items() if meta.get("endpoint") == GEMINI_ENDPOINT_PREDICT
}
GEMINI_NANO_BANANA_MODELS: set[str] = {
    mid
    for mid, meta in GEMINI_MODELS.items()
    if meta.get("endpoint") == GEMINI_ENDPOINT_GENERATECONTENT and not meta.get("deprecated", False)
}

# Default Gemini model — Nano Banana 2, which Google uses across their
# own surfaces (Gemini app, Search, Flow).
DEFAULT_GEMINI_IMAGE_MODEL = "gemini-3.1-flash-image-preview"

# Gemini supports 10 aspect ratios
GEMINI_ASPECT_RATIOS = [
    "1:1",  # Square
    "2:3",  # Portrait (phone)
    "3:2",  # Landscape (photo)
    "3:4",  # Portrait (social)
    "4:3",  # Landscape (classic)
    "4:5",  # Portrait (Instagram)
    "5:4",  # Landscape (social)
    "9:16",  # Portrait (Stories/Reels)
    "16:9",  # Landscape (video)
    "21:9",  # Ultra-wide
]

# Gemini resolution options (use uppercase K)
GEMINI_SIZES = [
    "1K",  # Fast generation
    "2K",  # High quality (default)
    "4K",  # Maximum resolution
]

# Reference image limits
GEMINI_MAX_REFERENCE_IMAGES = 14
GEMINI_MAX_OBJECT_IMAGES = 6
GEMINI_MAX_HUMAN_IMAGES = 5

# ============================
# Shared Constants
# ============================

MAX_PROMPT_LENGTH = 4000  # Shared limit (OpenAI: 4000, Gemini: 8192)
MAX_RETRIES = 3
DEFAULT_TIMEOUT = 120  # seconds

# ============================
# Provider Selection Keywords
# ============================

# Keywords that suggest OpenAI is better (text-heavy tasks)
OPENAI_PREFERRED_KEYWORDS = [
    "text",
    "label",
    "menu",
    "infographic",
    "diagram",
    "comic",
    "dialogue",
    "speech bubble",
    "caption",
    "title",
    "headline",
    "poster",
    "flyer",
    "certificate",
    "badge",
    "logo with text",
    "banner",
    "sign",
    "watermark",
    # 2.0-era strengths: UI mockups, brand-accurate rendering, world knowledge
    "ui mockup",
    "screenshot",
    "app interface",
    "webpage",
    "dashboard",
]

# Keywords that suggest Gemini is better
GEMINI_PREFERRED_KEYWORDS = [
    "portrait",
    "headshot",
    "photo",
    "photorealistic",
    "realistic",
    "product shot",
    "product photography",
    "studio lighting",
    "4k",
    "high resolution",
    "character consistency",
    "reference image",
    "weather",
    "stock",
    "current",
    "today",
    "real-time",
    "selfie",
    "person",
    "face",
    "beauty",
    "fashion",
    "magazine",
]

# Keywords that require Gemini (real-time data)
GEMINI_REQUIRED_KEYWORDS = [
    "current weather",
    "today's",
    "real-time",
    "stock price",
    "latest news",
    "live",
]
