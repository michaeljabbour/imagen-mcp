"""
Constants for imagen-mcp providers.

This module defines provider-specific constants including supported sizes,
aspect ratios, and model identifiers.
"""

# ============================
# OpenAI GPT-Image-1 Constants
# ============================

OPENAI_API_BASE_URL = "https://api.openai.com/v1"

# OpenAI only supports these 3 sizes (verified from API)
OPENAI_SIZES = [
    "1024x1024",  # Square
    "1024x1536",  # Portrait
    "1536x1024",  # Landscape
]

OPENAI_MODELS = {
    "gpt-image-1": "gpt-image-1",
    "gpt-4o": "gpt-4o",  # For conversation orchestration
    "gpt-4-turbo": "gpt-4-turbo",
}

# ============================
# Google Gemini Constants
# ============================

# Gemini model identifiers
GEMINI_MODELS = {
    "gemini-3-pro-image-preview": "gemini-3-pro-image-preview",  # Nano Banana Pro
    "gemini-flash-latest": "gemini-flash-latest",  # For prompt enhancement
}

# Gemini supports 10 aspect ratios
GEMINI_ASPECT_RATIOS = [
    "1:1",    # Square
    "2:3",    # Portrait (phone)
    "3:2",    # Landscape (photo)
    "3:4",    # Portrait (social)
    "4:3",    # Landscape (classic)
    "4:5",    # Portrait (Instagram)
    "5:4",    # Landscape (social)
    "9:16",   # Portrait (Stories/Reels)
    "16:9",   # Landscape (video)
    "21:9",   # Ultra-wide
]

# Gemini resolution options (use uppercase K)
GEMINI_SIZES = [
    "1K",   # Fast generation
    "2K",   # High quality (default)
    "4K",   # Maximum resolution
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
