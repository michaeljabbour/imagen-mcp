"""
Provider registry for imagen-mcp.

Manages provider instances and provides factory methods
for creating and accessing providers.
"""

import logging
from functools import lru_cache
from typing import Optional

from ..config.settings import get_settings
from .base import ImageProvider
from .openai_provider import OpenAIProvider
from .gemini_provider import GeminiProvider
from .selector import ProviderSelector, ProviderRecommendation

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """
    Registry for image generation providers.

    Manages provider instances and provides methods for:
    - Getting specific providers by name
    - Auto-selecting providers based on prompt analysis
    - Listing available providers
    """

    def __init__(self):
        """Initialize the provider registry."""
        self._providers: dict[str, ImageProvider] = {}
        self._selector = ProviderSelector()
        self._settings = get_settings()

    def get_provider(self, name: str) -> ImageProvider:
        """
        Get a provider by name.

        Args:
            name: Provider name ("openai" or "gemini")

        Returns:
            ImageProvider instance

        Raises:
            ValueError: If provider not found or not configured
        """
        name = name.lower()

        # Return cached instance if available
        if name in self._providers:
            return self._providers[name]

        # Create new instance
        if name == "openai":
            if not self._settings.has_openai_key():
                raise ValueError(
                    "OpenAI provider not available. Set OPENAI_API_KEY environment variable."
                )
            provider = OpenAIProvider()
        elif name == "gemini":
            if not self._settings.has_gemini_key():
                raise ValueError(
                    "Gemini provider not available. Set GEMINI_API_KEY environment variable."
                )
            provider = GeminiProvider()
        else:
            raise ValueError(f"Unknown provider: {name}. Available: openai, gemini")

        self._providers[name] = provider
        return provider

    def get_provider_for_prompt(
        self,
        prompt: str,
        *,
        size: Optional[str] = None,
        reference_images: Optional[list[str]] = None,
        enable_google_search: bool = False,
        explicit_provider: Optional[str] = None,
    ) -> tuple[ImageProvider, ProviderRecommendation]:
        """
        Get the best provider for a given prompt.

        Uses the ProviderSelector to analyze the prompt and choose
        the most appropriate provider.

        Args:
            prompt: Image generation prompt
            size: Requested size
            reference_images: Reference images (requires Gemini)
            enable_google_search: Enable Google Search grounding
            explicit_provider: User-specified provider override

        Returns:
            Tuple of (ImageProvider, ProviderRecommendation)
        """
        # Get recommendation
        recommendation = self._selector.suggest_provider(
            prompt,
            size=size,
            reference_images=reference_images,
            enable_google_search=enable_google_search,
            explicit_provider=explicit_provider,
        )

        # Get the provider
        provider = self.get_provider(recommendation.provider)

        return provider, recommendation

    def list_providers(self) -> list[str]:
        """List all available provider names."""
        return self._settings.available_providers()

    def list_all_providers(self) -> list[str]:
        """List all supported provider names (including unavailable)."""
        return ["openai", "gemini"]

    def is_provider_available(self, name: str) -> bool:
        """Check if a provider is available (has API key configured)."""
        name = name.lower()
        if name == "openai":
            return self._settings.has_openai_key()
        elif name == "gemini":
            return self._settings.has_gemini_key()
        return False

    def get_provider_info(self, name: str) -> dict:
        """Get information about a provider."""
        name = name.lower()

        if name == "openai":
            caps = OpenAIProvider().capabilities
        elif name == "gemini":
            caps = GeminiProvider().capabilities
        else:
            raise ValueError(f"Unknown provider: {name}")

        return {
            "name": caps.name,
            "display_name": caps.display_name,
            "available": self.is_provider_available(name),
            "supported_sizes": caps.supported_sizes,
            "supported_aspect_ratios": caps.supported_aspect_ratios,
            "max_resolution": caps.max_resolution,
            "text_rendering_quality": caps.text_rendering_quality,
            "supports_reference_images": caps.supports_reference_images,
            "supports_real_time_data": caps.supports_real_time_data,
            "supports_thinking_mode": caps.supports_thinking_mode,
            "best_for": caps.best_for,
            "not_recommended_for": caps.not_recommended_for,
        }

    def get_comparison(self) -> str:
        """Get a formatted comparison of providers."""
        return self._selector.get_provider_comparison()

    async def close_all(self) -> None:
        """Close all provider instances."""
        for provider in self._providers.values():
            await provider.close()
        self._providers.clear()


@lru_cache(maxsize=1)
def get_provider_registry() -> ProviderRegistry:
    """Get the singleton provider registry."""
    return ProviderRegistry()
