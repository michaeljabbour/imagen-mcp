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
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..config.constants import (
    DEFAULT_GEMINI_IMAGE_MODEL,
    GEMINI_ASPECT_RATIOS,
    GEMINI_MAX_REFERENCE_IMAGES,
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
            from google import genai as _genai
            from google.genai import types as _types
            from PIL import Image as _Image

            genai = _genai
            types = _types
            Image = _Image
        except ImportError as e:
            raise ImportError(
                "Gemini provider requires google-genai and pillow packages. "
                "Install with: pip install google-genai pillow"
            ) from e

def get_downloads_directory() -> Path:
    """Get the appropriate downloads directory for images."""
    downloads_base = Path.home() / "Downloads"
    images_dir = downloads_base / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    return images_dir


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
        self._conversation_store: dict[str, list[dict[str, Any]]] = {}

    def _ensure_initialized(self, api_key: str | None = None) -> None:
        """Ensure dependencies are imported and client is initialized."""
        _import_dependencies()

        if self._client is None:
            api_key = api_key or self._api_key
            if not api_key:
                settings = get_settings()
                api_key = settings.get_gemini_api_key()
            # We know genai is not None after _import_dependencies
            self._client = genai.Client(api_key=api_key)

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def display_name(self) -> str:
        return "Google Gemini 3 Pro Image (Nano Banana Pro)"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            name="gemini",
            display_name="Gemini 3 Pro Image (Nano Banana Pro)",
            supported_sizes=GEMINI_SIZES,
            supported_aspect_ratios=GEMINI_ASPECT_RATIOS,
            max_resolution="4K",
            supports_text_rendering=True,
            text_rendering_quality="good",  # Improved in Nano Banana Pro
            supports_reference_images=True,
            max_reference_images=GEMINI_MAX_REFERENCE_IMAGES,
            supports_real_time_data=True,
            supports_thinking_mode=True,
            supports_multi_turn=True,
            typical_latency_seconds=15.0,  # Faster than OpenAI
            cost_tier="standard",
            best_for=[
                "Photorealistic portraits and headshots",
                "Product photography",
                "High resolution (4K) output",
                "Character consistency with reference images",
                "Real-time data visualization (weather, stocks)",
                "Multi-turn iterative refinement",
                "Complex compositions with multiple subjects",
            ],
            not_recommended_for=[
                "Text-heavy images (menus, infographics)",
                "Precise text rendering with specific fonts",
                "Technical diagrams with detailed labels",
            ],
        )

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
        reference_images = kwargs.get("reference_images", [])
        if len(reference_images) > GEMINI_MAX_REFERENCE_IMAGES:
            raise ValueError(f"Too many reference images. Maximum {GEMINI_MAX_REFERENCE_IMAGES}.")

        return {
            "prompt": prompt,
            "size": size,
            "aspect_ratio": aspect_ratio,
        }

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
        """Generate an image using Gemini."""
        start_time = time.time()

        try:
            # Ensure initialized
            self._ensure_initialized(api_key)
            assert self._client is not None, "Gemini client not initialized"

            # Validate parameters
            validated = await self.validate_params(
                prompt, size, aspect_ratio, reference_images=reference_images, **kwargs
            )
            size = validated["size"]
            aspect_ratio = validated["aspect_ratio"]

            # Select model (default to Nano Banana Pro)
            model_id = model or DEFAULT_GEMINI_IMAGE_MODEL
            if model_id in GEMINI_MODELS:
                model_id = GEMINI_MODELS[model_id]

            # Generate conversation ID if not provided
            conversation_id = conversation_id or f"gemini_{uuid4().hex[:12]}"

            # Build contents list
            contents: list[Any] = []

            # Add previous image from conversation history if available
            if conversation_id in self._conversation_store:
                history = self._conversation_store[conversation_id]
                if history:
                    # Find last generated image
                    last_entry = history[-1]
                    if "image_base64" in last_entry:
                        try:
                            last_image_bytes = base64.b64decode(last_entry["image_base64"])
                            last_pil_image = Image.open(io.BytesIO(last_image_bytes))
                            contents.append(last_pil_image)
                            logger.info(
                                f"Added prev image from conv {conversation_id} as context"
                            )
                        except Exception as e:
                            logger.warning(f"Failed to load previous image from history: {e}")

            # Add reference images if provided (up to 14)
            if reference_images:
                for ref_image_b64 in reference_images[:GEMINI_MAX_REFERENCE_IMAGES]:
                    try:
                        image_bytes = base64.b64decode(ref_image_b64)
                        pil_image = Image.open(io.BytesIO(image_bytes))
                        contents.append(pil_image)
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

            # Generate content (SDK is synchronous, run in executor)
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                partial(
                    self._client.models.generate_content,
                    model=model_id,
                    contents=contents,
                    config=config,
                ),
            )

            # Extract content from response
            extraction = self._extract_content(response)

            if not extraction["images"]:
                raise ValueError("No image data found in Gemini API response")

            # Save first image
            image_b64 = extraction["images"][0]
            image_path = self._save_image(image_b64, prompt, output_path=output_path)

            # Store in conversation history
            if conversation_id not in self._conversation_store:
                self._conversation_store[conversation_id] = []

            self._conversation_store[conversation_id].append({
                "prompt": prompt,
                "image_base64": image_b64,
                "timestamp": datetime.now(),
                "model": model_id
            })

            generation_time = time.time() - start_time

            return ImageResult(
                success=True,
                provider=self.name,
                model=model_id,
                image_path=image_path,
                image_base64=image_b64,
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
            logger.error(f"Gemini image generation failed: {e}")
            return ImageResult(
                success=False,
                provider=self.name,
                model=model or DEFAULT_GEMINI_IMAGE_MODEL,
                prompt=prompt,
                error=str(e),
            )

    def _extract_content(self, response: Any) -> dict[str, Any]:
        """Extract images, text, and thoughts from Gemini response."""
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

                        # Convert to PIL Image to validate
                        pil_image = Image.open(io.BytesIO(image_bytes))

                        # Convert to base64
                        buffer = io.BytesIO()
                        pil_image.save(buffer, format="PNG")
                        image_b64 = base64.b64encode(buffer.getvalue()).decode()

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

    def _save_image(
        self, b64_json: str, prompt: str, output_path: str | None = None
    ) -> Path:
        """Save base64 image to path or Downloads folder."""
        # Decode image
        image_bytes = base64.b64decode(b64_json)

        # Generate default filename parts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_id = uuid4().hex[:8]
        prompt_snippet = "".join(c for c in prompt[:30] if c.isalnum() or c == " ").strip()
        prompt_snippet = prompt_snippet.replace(" ", "_")[:20]
        filename = f"gemini_{timestamp}_{prompt_snippet}_{short_id}.png"

        # Determine save location
        if output_path:
            path_obj = Path(output_path)
            # Check if it looks like a directory or file
            if path_obj.suffix:  # Has extension, treat as file
                save_path = path_obj
                # Ensure parent exists
                save_path.parent.mkdir(parents=True, exist_ok=True)
            else:  # Treat as directory
                save_path = path_obj / filename
                save_path.mkdir(parents=True, exist_ok=True)
        else:
            save_path = get_downloads_directory() / filename

        with open(save_path, "wb") as f:
            f.write(image_bytes)

        logger.info(f"Image saved to: {save_path}")
        return save_path

    def get_best_size_for_type(self, image_type: str) -> str:
        """Get best Gemini size for image type."""
        # Gemini uses resolution (1K, 2K, 4K) rather than pixel dimensions
        if image_type in ["portrait", "headshot", "product", "professional"]:
            return "2K"  # High quality for important content
        elif image_type in ["draft", "concept", "quick"]:
            return "1K"  # Faster for iteration
        elif image_type in ["print", "production", "high_quality"]:
            return "4K"  # Maximum quality
        else:
            return "2K"  # Default to balanced

    def get_best_aspect_ratio_for_type(self, image_type: str) -> str:
        """Get best Gemini aspect ratio for image type."""
        type_to_ratio = {
            "portrait": "2:3",
            "headshot": "4:5",
            "landscape": "16:9",
            "square": "1:1",
            "social": "4:5",
            "story": "9:16",
            "video": "16:9",
            "panorama": "21:9",
            "photo": "3:2",
            "product": "1:1",
        }
        return type_to_ratio.get(image_type, "1:1")

    async def enhance_prompt(self, prompt: str) -> str:
        """Enhance prompt using Gemini Flash."""
        try:
            self._ensure_initialized()
            assert self._client is not None, "Gemini client not initialized"

            system_instruction = """You are an expert prompt engineer for image generation.
Enhance the given prompt to produce better images by:
1. Adding specific details about composition, lighting, and style
2. Including camera/lens specifications if photorealistic
3. Specifying materials, textures, and colors
4. Adding mood and atmosphere descriptors
5. Keeping the core intent while making it more vivid

Return ONLY the enhanced prompt, no explanation."""

            config = types.GenerateContentConfig(system_instruction=system_instruction)

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                partial(
                    self._client.models.generate_content,
                    model=GEMINI_MODELS["gemini-flash-latest"],
                    contents=prompt,
                    config=config,
                ),
            )

            return response.text or prompt

        except Exception as e:
            logger.warning(f"Prompt enhancement failed, using original: {e}")
            return prompt

    async def close(self) -> None:
        """Clean up resources."""
        # genai SDK handles cleanup automatically
        self._client = None

    def get_conversations(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get list of recent conversations."""
        conversations = []
        for conv_id, history in self._conversation_store.items():
            if not history:
                continue

            last_item = history[-1]
            last_content = "Image generated"
            if isinstance(last_item, dict) and "prompt" in last_item:
                last_content = last_item["prompt"][:50]

            conversations.append(
                {
                    "id": conv_id,
                    "provider": "gemini",
                    "message_count": len(history),
                    "last_message": last_content,
                    "updated": datetime.now(),
                }
            )

        # Sort by ID descending
        conversations.sort(key=lambda x: str(x["id"]), reverse=True)
        return conversations[:limit]
