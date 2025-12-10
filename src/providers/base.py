"""
Abstract base class for image generation providers.

This module defines the interface that all providers must implement,
ensuring consistent behavior across OpenAI, Gemini, and future providers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class ProviderCapabilities:
    """Describes what a provider can do."""

    name: str
    display_name: str

    # Size/resolution support
    supported_sizes: list[str]
    supported_aspect_ratios: list[str]
    max_resolution: str

    # Feature support
    supports_text_rendering: bool = True
    text_rendering_quality: str = "good"  # excellent, good, fair

    supports_reference_images: bool = False
    max_reference_images: int = 0

    supports_real_time_data: bool = False
    supports_thinking_mode: bool = False
    supports_multi_turn: bool = True

    # Performance characteristics
    typical_latency_seconds: float = 30.0
    cost_tier: str = "standard"  # low, standard, premium

    # Best use cases
    best_for: list[str] = field(default_factory=list)
    not_recommended_for: list[str] = field(default_factory=list)


@dataclass
class ImageResult:
    """Result from image generation."""

    success: bool
    provider: str
    model: str

    # Image data (one of these will be set)
    image_path: Path | None = None
    image_base64: str | None = None
    image_url: str | None = None

    # Metadata
    prompt: str = ""
    enhanced_prompt: str | None = None
    size: str | None = None
    aspect_ratio: str | None = None

    # Conversation tracking
    conversation_id: str | None = None
    file_id: str | None = None  # Provider-specific file reference

    # Timestamps
    timestamp: datetime = field(default_factory=datetime.now)
    generation_time_seconds: float | None = None

    # Additional data
    thoughts: list[dict[str, Any]] | None = None  # Gemini thinking mode
    grounding_metadata: dict[str, Any] | None = None  # Gemini search grounding
    verification_result: dict[str, Any] | None = None

    # Error info
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "provider": self.provider,
            "model": self.model,
            "image_path": str(self.image_path) if self.image_path else None,
            "prompt": self.prompt,
            "enhanced_prompt": self.enhanced_prompt,
            "size": self.size,
            "aspect_ratio": self.aspect_ratio,
            "conversation_id": self.conversation_id,
            "file_id": self.file_id,
            "timestamp": self.timestamp.isoformat(),
            "generation_time_seconds": self.generation_time_seconds,
            "thoughts": self.thoughts,
            "grounding_metadata": self.grounding_metadata,
            "verification_result": self.verification_result,
            "error": self.error,
        }


class ImageProvider(ABC):
    """
    Abstract base class for image generation providers.

    All providers must implement these methods to ensure consistent
    behavior across the MCP server.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier (e.g., 'openai', 'gemini')."""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable provider name."""
        pass

    @property
    @abstractmethod
    def capabilities(self) -> ProviderCapabilities:
        """Provider capabilities and feature support."""
        pass

    @abstractmethod
    async def generate_image(
        self,
        prompt: str,
        *,
        size: str | None = None,
        aspect_ratio: str | None = None,
        conversation_id: str | None = None,
        reference_images: list[str] | None = None,
        enable_enhancement: bool = True,
        output_path: str | None = None,
        **kwargs: Any,
    ) -> ImageResult:
        """
        Generate an image from a text prompt.

        Args:
            prompt: Text description of desired image
            size: Image size (provider-specific format)
            aspect_ratio: Aspect ratio (provider-specific format)
            conversation_id: ID for multi-turn conversation
            reference_images: List of base64-encoded reference images
            enable_enhancement: Whether to enhance the prompt
            output_path: Optional path to save the image
            **kwargs: Provider-specific parameters

        Returns:
            ImageResult with generated image data
        """
        pass

    @abstractmethod
    async def validate_params(
        self,
        prompt: str,
        size: str | None = None,
        aspect_ratio: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Validate and normalize parameters for this provider.

        Args:
            prompt: Text prompt to validate
            size: Size to validate
            aspect_ratio: Aspect ratio to validate
            **kwargs: Additional parameters

        Returns:
            Dictionary of validated/normalized parameters

        Raises:
            ValueError: If parameters are invalid
        """
        pass

    def get_best_size_for_type(self, image_type: str) -> str:
        """
        Get recommended size for a specific image type.

        Args:
            image_type: Type of image (portrait, landscape, logo, etc.)

        Returns:
            Recommended size string for this provider
        """
        # Default implementation - subclasses can override
        caps = self.capabilities
        sizes = caps.supported_sizes

        if image_type in ["portrait", "headshot", "person", "selfie"]:
            # Prefer portrait/vertical sizes
            for s in sizes:
                if "x" in s:
                    w, h = s.split("x")
                    if int(h) > int(w):
                        return s
            return sizes[0] if sizes else "1024x1024"

        elif image_type in ["landscape", "scene", "banner", "panorama"]:
            # Prefer landscape/horizontal sizes
            for s in sizes:
                if "x" in s:
                    w, h = s.split("x")
                    if int(w) > int(h):
                        return s
            return sizes[0] if sizes else "1024x1024"

        else:
            # Default to square
            for s in sizes:
                if "x" in s:
                    w, h = s.split("x")
                    if w == h:
                        return s
            return sizes[0] if sizes else "1024x1024"

    def supports_feature(self, feature: str) -> bool:
        """Check if provider supports a specific feature."""
        caps = self.capabilities
        feature_map = {
            "text_rendering": caps.supports_text_rendering,
            "reference_images": caps.supports_reference_images,
            "real_time_data": caps.supports_real_time_data,
            "thinking_mode": caps.supports_thinking_mode,
            "multi_turn": caps.supports_multi_turn,
        }
        return feature_map.get(feature, False)

    async def close(self) -> None:  # noqa: B027
        """Clean up provider resources."""
        pass  # Default implementation does nothing

    def get_conversations(self, limit: int = 10) -> list[dict[str, Any]]:
        """
        Get list of recent conversations.

        Args:
            limit: Maximum number of conversations to return

        Returns:
            List of conversation summaries
        """
        return []
