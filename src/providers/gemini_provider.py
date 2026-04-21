"""
Google Gemini 3 Pro Image (Nano Banana Pro) provider implementation.

This provider wraps Google's Gemini 3 Pro Image model for image generation,
featuring advanced reasoning, high-resolution output, reference images,
Google Search grounding, and thinking mode.
"""

import asyncio
import base64
import io
import logging
import time
from datetime import datetime
from functools import partial
from typing import Any
from uuid import uuid4

from ..config.constants import (
    DEFAULT_GEMINI_IMAGE_MODEL,
    GEMINI_ASPECT_RATIOS,
    GEMINI_MAX_REFERENCE_IMAGES,
    GEMINI_MODEL_ALIASES,
    GEMINI_MODELS,
    GEMINI_SIZES,
    MAX_PROMPT_LENGTH,
)
from ..config.settings import get_settings
from .base import ImageProvider, ImageResult, ProviderCapabilities

logger = logging.getLogger(__name__)

# Lazy import for google-genai (may not be installed)
genai: Any = None
types: Any = None
Image: Any = None


def _import_dependencies() -> None:
    """Lazily import Gemini dependencies."""
    global genai, types, Image
    if genai is None:
        try:
            from google import genai as _genai  # type: ignore[attr-defined]
            from google.genai import types as _types  # type: ignore[import-untyped]
            from PIL import Image as _Image  # type: ignore[import-untyped]

            genai = _genai
            types = _types
            Image = _Image
            # Explicit default — guards against decompression bombs from
            # untrusted reference images or conversation-history payloads.
            Image.MAX_IMAGE_PIXELS = 89_478_485
        except ImportError as e:
            raise ImportError(
                "Gemini provider requires google-genai and pillow packages. "
                "Install with: pip install google-genai pillow"
            ) from e


class GeminiProvider(ImageProvider):
    """
    Google Gemini 3 Pro Image (Nano Banana Pro) provider.

    Best for:
    - Photorealistic portraits and headshots
    - Product photography
    - High resolution (4K) output
    - Character consistency with reference images
    - Real-time data visualization (weather, stocks, events)
    - Multi-turn iterative refinement

    Features:
    - Up to 14 reference images (6 objects + 5 humans)
    - Google Search grounding for real-time data
    - Thinking mode for complex prompts
    - 10 aspect ratio options
    - 1K, 2K, 4K resolution support
    """

    def __init__(self, api_key: str | None = None):
        """Initialize Gemini provider."""
        self._api_key = api_key
        self._client = None
        self._active_api_key: str | None = None  # Track which key the client was created with

    def _ensure_initialized(self, api_key: str | None = None) -> None:
        """Ensure dependencies are imported and client is initialized."""
        _import_dependencies()

        resolved_key = api_key or self._api_key
        if not resolved_key:
            settings = get_settings()
            resolved_key = settings.get_gemini_api_key()

        # Reinitialize if key changed (e.g. per-request override)
        if self._client is not None and resolved_key != self._active_api_key:
            self._client = None

        if self._client is None:
            self._client = genai.Client(api_key=resolved_key)
            self._active_api_key = resolved_key

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def display_name(self) -> str:
        return "Google Gemini — Nano Banana 2 (default) / Nano Banana Pro"

    # Cached capabilities — constant across all instances, no need to rebuild per access.
    # Describes the Nano Banana family (Imagen 4 support was removed in
    # v0.3.0 — see the module docstring for rationale).
    _capabilities = ProviderCapabilities(
        name="gemini",
        display_name="Gemini Nano Banana 2 / Pro",
        supported_sizes=GEMINI_SIZES,
        supported_aspect_ratios=GEMINI_ASPECT_RATIOS,
        max_resolution="4K",
        supports_text_rendering=True,
        text_rendering_quality="good",  # "excellent" when routed to Nano Banana Pro (Thinking)
        supports_reference_images=True,
        max_reference_images=GEMINI_MAX_REFERENCE_IMAGES,
        supports_real_time_data=True,
        supports_thinking_mode=True,
        supports_multi_turn=True,
        typical_latency_seconds=8.0,  # Nano Banana 2 is fast; Pro ~15-20s
        cost_tier="standard",
        best_for=[
            "Photorealistic portraits and headshots",
            "Product photography",
            "High resolution (4K) output",
            "Character consistency with reference images (up to 14)",
            "Real-time data visualization (weather, stocks)",
            "Multi-turn iterative refinement",
            "Complex compositions with multiple subjects",
        ],
        not_recommended_for=[
            "Precise text rendering (OpenAI gpt-image-2 is better)",
            "Technical diagrams with detailed labels",
        ],
    )

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._capabilities

    async def validate_params(
        self,
        prompt: str,
        size: str | None = None,
        aspect_ratio: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Validate and normalize parameters for Gemini."""
        # Validate prompt length (Gemini supports 8192 but we use shared limit)
        if len(prompt) > MAX_PROMPT_LENGTH:
            raise ValueError(f"Prompt too long. Maximum {MAX_PROMPT_LENGTH} characters.")

        # Validate/normalize size (must be uppercase K)
        if size:
            size = size.upper()
            # Convert OpenAI-style sizes to Gemini equivalents
            openai_to_gemini = {
                "1024X1024": "1K",
                "1024X1536": "2K",
                "1536X1024": "2K",
            }
            if size in openai_to_gemini:
                logger.info(f"Converting OpenAI size '{size}' to Gemini: {openai_to_gemini[size]}")
                size = openai_to_gemini[size]
            if size not in GEMINI_SIZES:
                raise ValueError(
                    f"Invalid size '{size}' for Gemini. Supported sizes: {', '.join(GEMINI_SIZES)}"
                )
        else:
            size = "2K"  # Default

        # Validate aspect ratio
        if aspect_ratio:
            if aspect_ratio not in GEMINI_ASPECT_RATIOS:
                raise ValueError(
                    f"Invalid aspect ratio '{aspect_ratio}' for Gemini. "
                    f"Supported ratios: {', '.join(GEMINI_ASPECT_RATIOS)}"
                )
        else:
            aspect_ratio = "1:1"  # Default

        # Validate reference images count
        # Use `or []` because dict.get() returns None (not default) when key exists with None value
        reference_images = kwargs.get("reference_images") or []
        if len(reference_images) > GEMINI_MAX_REFERENCE_IMAGES:
            raise ValueError(f"Too many reference images. Maximum {GEMINI_MAX_REFERENCE_IMAGES}.")

        return {
            "prompt": prompt,
            "size": size,
            "aspect_ratio": aspect_ratio,
        }

    def _resolve_model_id(self, model: str | None) -> str:
        """Resolve a user-provided model name/alias to a canonical Gemini
        model identifier.

        Accepts either:
        - a canonical model id from ``GEMINI_MODELS`` keys (e.g.
          ``"gemini-3.1-flash-image-preview"``),
        - a friendly alias from ``GEMINI_MODEL_ALIASES`` (e.g.
          ``"nano-banana-2"``, ``"imagen-4-ultra"``),
        - ``None`` (returns the default).

        Unknown names fall back to ``DEFAULT_GEMINI_IMAGE_MODEL`` with a
        warning so callers targeting a brand-new model that hasn't been
        added to the registry yet don't silently misroute.
        """
        if not model:
            return DEFAULT_GEMINI_IMAGE_MODEL

        # Try alias first
        if model in GEMINI_MODEL_ALIASES:
            return GEMINI_MODEL_ALIASES[model]

        if model in GEMINI_MODELS:
            return model

        logger.warning(
            "Unknown Gemini model '%s', falling back to '%s'",
            model,
            DEFAULT_GEMINI_IMAGE_MODEL,
        )
        return DEFAULT_GEMINI_IMAGE_MODEL

    async def generate_image(
        self,
        prompt: str,
        *,
        size: str | None = None,
        aspect_ratio: str | None = None,
        conversation_id: str | None = None,
        reference_images: list[str] | None = None,
        enable_enhancement: bool = True,
        enable_google_search: bool = False,
        api_key: str | None = None,
        model: str | None = None,
        output_path: str | None = None,
        **kwargs: Any,
    ) -> ImageResult:
        """Generate an image using Gemini.

        Routes to one of two endpoints based on the resolved model:

        - **Nano Banana** models (``gemini-*-image*``) use the
          ``generateContent`` endpoint with full feature support
          (conversational editing, reference images, Google Search
          grounding, Thinking mode on Pro).

        Imagen 4 (``imagen-4.0-*``) support was removed in v0.3.0 — those
        models are text-to-image only (no conversational editing, no
        reference images, no Google Search) and shut down 2026-06-24.
        Google's own guidance is to migrate to Nano Banana 2 or Pro.
        """
        model_id = self._resolve_model_id(model)
        start_time = time.time()

        try:
            # Ensure initialized
            self._ensure_initialized(api_key)
            assert self._client is not None, "Gemini client not initialized"

            # Validate parameters (Nano Banana shape)
            validated = await self.validate_params(
                prompt, size, aspect_ratio, reference_images=reference_images, **kwargs
            )
            size = validated["size"]
            aspect_ratio = validated["aspect_ratio"]

            # Generate conversation ID if not provided
            conversation_id = conversation_id or f"gemini_{uuid4().hex[:12]}"

            # Build contents list; track PIL images for cleanup
            contents: list[Any] = []
            pil_images_to_close: list[Any] = []

            try:
                # Add previous image from conversation history if available
                last_image_b64 = self._get_last_image_from_conversation(conversation_id)
                if last_image_b64:
                    try:
                        last_image_bytes = base64.b64decode(last_image_b64)
                        last_pil_image = Image.open(io.BytesIO(last_image_bytes))
                        contents.append(last_pil_image)
                        pil_images_to_close.append(last_pil_image)
                        logger.info(f"Added prev image from conv {conversation_id} as context")
                    except Exception as e:
                        logger.warning(f"Failed to load previous image from history: {e}")

                # Add reference images if provided (up to 14)
                if reference_images:
                    for ref_image_b64 in reference_images[:GEMINI_MAX_REFERENCE_IMAGES]:
                        try:
                            image_bytes = base64.b64decode(ref_image_b64)
                            pil_image = Image.open(io.BytesIO(image_bytes))
                            contents.append(pil_image)
                            pil_images_to_close.append(pil_image)
                        except Exception as e:
                            logger.warning(f"Failed to process reference image: {e}")

                # Add prompt
                contents.append(prompt)

                # Build config
                image_config = types.ImageConfig(
                    aspect_ratio=aspect_ratio,
                    image_size=size,
                )

                config_args: dict[str, Any] = {
                    "response_modalities": ["TEXT", "IMAGE"],
                    "image_config": image_config,
                }

                # Add Google Search grounding if enabled
                if enable_google_search:
                    config_args["tools"] = [{"google_search": {}}]

                config = types.GenerateContentConfig(**config_args)

                logger.info(
                    f"Generating image with Gemini model={model_id}, size={size}, "
                    f"aspect_ratio={aspect_ratio}"
                )

                # Acquire rate limit before making request
                await self._acquire_rate_limit()

                # Generate content (SDK is synchronous, run in executor)
                # Wrap with timeout and retry to handle transient failures
                settings = get_settings()

                client = self._client  # Bind for closure (already asserted non-None)

                async def _do_generate() -> Any:
                    loop = asyncio.get_running_loop()
                    return await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            partial(
                                client.models.generate_content,  # type: ignore[union-attr]
                                model=model_id,
                                contents=contents,
                                config=config,
                            ),
                        ),
                        timeout=settings.request_timeout,
                    )

                response = await self._retry_with_backoff(_do_generate)
            finally:
                # Release PIL image memory immediately after API call
                for img in pil_images_to_close:
                    try:
                        img.close()
                    except Exception:
                        pass

            # Extract content from response
            extraction = self._extract_content(response)

            if not extraction["images"]:
                raise ValueError("No image data found in Gemini API response")

            # Save first image using base class method
            image_b64 = extraction["images"][0]
            image_path = await self._save_image(image_b64, prompt, output_path)

            # Store in persistent conversation store
            self._store_conversation_message(
                conversation_id,
                "user",
                prompt,
            )
            self._store_conversation_message(
                conversation_id,
                "assistant",
                {"type": "image_generated", "prompt": prompt},
                image_base64=image_b64,
                metadata={"size": size, "aspect_ratio": aspect_ratio, "model": model_id},
            )

            generation_time = time.time() - start_time

            # Don't carry image_base64 on the result when we already
            # persisted the file — avoids keeping ~4-12 MB in memory
            # for the lifetime of the result object.
            return ImageResult(
                success=True,
                provider=self.name,
                model=model_id,
                image_path=image_path,
                prompt=prompt,
                size=size,
                aspect_ratio=aspect_ratio,
                conversation_id=conversation_id,
                timestamp=datetime.now(),
                generation_time_seconds=generation_time,
                thoughts=extraction.get("thoughts"),
                grounding_metadata=extraction.get("grounding_metadata"),
            )

        except Exception as e:
            logger.exception("Gemini image generation failed")
            return ImageResult(
                success=False,
                provider=self.name,
                model=model or DEFAULT_GEMINI_IMAGE_MODEL,
                prompt=prompt,
                error=str(e),
            )

    def _extract_content(self, response: Any) -> dict[str, Any]:
        """Extract images, text, and thoughts from Gemini response.

        Encodes raw image bytes directly to base64 without an unnecessary
        PIL decode → re-encode round-trip.  PIL is only used as a fallback
        when the raw bytes cannot be encoded directly.
        """
        images: list[str] = []
        text_parts: list[str] = []
        thoughts: list[dict[str, Any]] = []

        try:
            for idx, part in enumerate(response.parts):
                is_thought = getattr(part, "thought", False)

                # Extract image data
                if hasattr(part, "inline_data") and part.inline_data:
                    try:
                        inline_data = part.inline_data
                        image_bytes = inline_data.data

                        # Encode raw bytes directly — avoids PIL decode/re-encode
                        # overhead (~200-500 ms per 4K image).
                        image_b64 = base64.b64encode(image_bytes).decode()

                        if is_thought:
                            thoughts.append(
                                {
                                    "type": "image",
                                    "data": image_b64,
                                    "index": len(thoughts),
                                }
                            )
                        else:
                            images.append(image_b64)
                    except Exception as e:
                        logger.error(f"Could not extract image from part {idx}: {e}")

                # Extract text
                if hasattr(part, "text") and part.text:
                    if is_thought:
                        thoughts.append(
                            {
                                "type": "text",
                                "data": part.text,
                                "index": len(thoughts),
                            }
                        )
                    else:
                        text_parts.append(part.text)

        except Exception as e:
            logger.error(f"Error extracting content from response: {e}")

        result: dict[str, Any] = {
            "images": images,
            "text": text_parts,
            "thoughts": thoughts if thoughts else None,
        }

        # Include grounding metadata if available
        if hasattr(response, "grounding_metadata"):
            result["grounding_metadata"] = response.grounding_metadata

        return result

    async def close(self) -> None:
        """Clean up resources."""
        # genai SDK handles cleanup automatically
        self._client = None
