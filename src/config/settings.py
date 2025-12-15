"""
Settings management for imagen-mcp.

Handles API keys and configuration from environment variables.
"""

import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass
class Settings:
    """Application settings loaded from environment."""

    # API Keys
    openai_api_key: str | None = None
    gemini_api_key: str | None = None  # Also checks GOOGLE_API_KEY

    # Defaults
    default_provider: str = "auto"  # auto, openai, gemini
    default_openai_size: str = "1024x1024"
    default_gemini_size: str = "2K"
    default_gemini_aspect_ratio: str = "1:1"

    # Feature flags
    enable_prompt_enhancement: bool = True
    enable_google_search: bool = False  # Gemini-only feature

    # Timeouts
    request_timeout: int = 120

    # Output
    output_dir: str | None = None

    # Logging
    log_dir: str | None = None
    log_level: str = "INFO"
    log_max_bytes: int = 5_242_880  # 5 MiB
    log_backup_count: int = 3
    log_prompts: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables."""
        # Support both GEMINI_API_KEY and GOOGLE_API_KEY for Gemini
        gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        log_dir = os.getenv("IMAGEN_MCP_LOG_DIR") or os.getenv("LOG_DIR")
        log_level = os.getenv("IMAGEN_MCP_LOG_LEVEL") or os.getenv("LOG_LEVEL", "INFO")

        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            gemini_api_key=gemini_key,
            default_provider=os.getenv("DEFAULT_PROVIDER", "auto"),
            default_openai_size=os.getenv("DEFAULT_OPENAI_SIZE", "1024x1024"),
            default_gemini_size=os.getenv("DEFAULT_GEMINI_SIZE", "2K"),
            default_gemini_aspect_ratio=os.getenv("DEFAULT_GEMINI_ASPECT_RATIO", "1:1"),
            enable_prompt_enhancement=(
                os.getenv("ENABLE_PROMPT_ENHANCEMENT", "true").lower() == "true"
            ),
            enable_google_search=os.getenv("ENABLE_GOOGLE_SEARCH", "false").lower() == "true",
            request_timeout=int(os.getenv("REQUEST_TIMEOUT", "120")),
            output_dir=os.getenv("OUTPUT_DIR"),
            log_dir=log_dir,
            log_level=log_level,
            log_max_bytes=int(
                os.getenv("IMAGEN_MCP_LOG_MAX_BYTES") or os.getenv("LOG_MAX_BYTES", "5242880")
            ),
            log_backup_count=int(
                os.getenv("IMAGEN_MCP_LOG_BACKUP_COUNT") or os.getenv("LOG_BACKUP_COUNT", "3")
            ),
            log_prompts=(
                os.getenv("IMAGEN_MCP_LOG_PROMPTS") or os.getenv("LOG_PROMPTS", "false")
            ).lower()
            == "true",
        )

    def get_openai_api_key(self, provided_key: str | None = None) -> str:
        """Get OpenAI API key from provided value or settings."""
        api_key = provided_key or self.openai_api_key
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or provide api_key parameter."
            )
        return api_key

    def get_gemini_api_key(self, provided_key: str | None = None) -> str:
        """Get Gemini API key from provided value or settings."""
        api_key = provided_key or self.gemini_api_key
        if not api_key:
            raise ValueError(
                "Gemini API key not found. Set GEMINI_API_KEY environment variable "
                "or provide api_key parameter."
            )
        return api_key

    def has_openai_key(self) -> bool:
        """Check if OpenAI API key is available."""
        return bool(self.openai_api_key)

    def has_gemini_key(self) -> bool:
        """Check if Gemini API key is available."""
        return bool(self.gemini_api_key)

    def available_providers(self) -> list[str]:
        """Return list of providers with valid API keys."""
        providers = []
        if self.has_openai_key():
            providers.append("openai")
        if self.has_gemini_key():
            providers.append("gemini")
        return providers


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings.from_env()
