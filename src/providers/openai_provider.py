"""
OpenAI gpt-image-2 (ChatGPT Images 2.0) provider implementation.

Two code paths are exposed:

- **Direct path** — a plain POST to ``/images/generations`` with the full
  gpt-image-2 parameter surface (quality, output_format, background,
  moderation, n, style, output_compression). Used when
  ``enable_enhancement=False`` for speed (~3-8s on gpt-image-2).

- **Responses-API path** — a two-stage flow where ``gpt-4o`` (or a
  user-chosen assistant model) first refines the prompt through a
  forced function call, then the refined prompt is passed to
  ``/images/generations``. Used when ``enable_enhancement=True`` and
  for the conversational tool (preserves multi-turn context).

An ``edit_image`` entry point targets ``/images/edits`` with
``input_fidelity=high`` by default so gpt-image-2 can preserve
unchanged pixels between sequential edits.
"""

from __future__ import annotations

import base64
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from ..config.constants import (
    DEFAULT_OPENAI_BACKGROUND,
    DEFAULT_OPENAI_IMAGE_MODEL,
    DEFAULT_OPENAI_INPUT_FIDELITY,
    DEFAULT_OPENAI_MODERATION,
    DEFAULT_OPENAI_OUTPUT_FORMAT,
    DEFAULT_OPENAI_QUALITY,
    MAX_PROMPT_LENGTH,
    OPENAI_API_BASE_URL,
    OPENAI_BACKGROUND_OPTIONS,
    OPENAI_EDIT_SIZES,
    OPENAI_INPUT_FIDELITY_OPTIONS,
    OPENAI_MAX_N,
    OPENAI_MODELS,
    OPENAI_MODERATION_OPTIONS,
    OPENAI_OUTPUT_FORMATS,
    OPENAI_QUALITY_OPTIONS,
    OPENAI_SIZES,
    OPENAI_STYLES,
)
from ..config.settings import get_settings
from .base import ImageProvider, ImageResult, ProviderCapabilities

logger = logging.getLogger(__name__)


class OpenAIProvider(ImageProvider):
    """
    OpenAI gpt-image-2 (ChatGPT Images 2.0) provider.

    Best for:
    - Text rendering (menus, infographics, comics) — ~99% character accuracy
    - UI mockups and screenshot-style renders
    - Technical diagrams and labeled illustrations
    - Marketing materials with exact text
    - Precise instruction following and world knowledge

    Limitations vs Gemini:
    - Max 1792x1024 (no native 4K)
    - No reference image support on /images/generations
      (use the edit_image tool with input_fidelity='high' instead)
    - No real-time data grounding
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key
        self._client: httpx.AsyncClient | None = None
        self._capabilities: ProviderCapabilities | None = None

    # ------------------------------------------------------------------
    # HTTP client lifecycle
    # ------------------------------------------------------------------

    def _ensure_client(self) -> httpx.AsyncClient:
        """Return the shared httpx client, creating it lazily."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(120.0),
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            )
        return self._client

    async def close(self) -> None:
        """Close the shared HTTP client and release connections."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    # ------------------------------------------------------------------
    # Identity / capabilities
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "openai"

    @property
    def display_name(self) -> str:
        return "OpenAI gpt-image-2 (ChatGPT Images 2.0)"

    @property
    def capabilities(self) -> ProviderCapabilities:
        if self._capabilities is None:
            self._capabilities = ProviderCapabilities(
                name="openai",
                display_name="OpenAI gpt-image-2",
                supported_sizes=OPENAI_SIZES,
                supported_aspect_ratios=["1:1", "2:3", "3:2", "16:9", "9:16"],
                max_resolution="1792x1024",
                supports_text_rendering=True,
                text_rendering_quality="excellent",  # ~99% char accuracy on 2.0
                supports_reference_images=False,  # via edit_image only
                max_reference_images=0,
                supports_real_time_data=False,
                supports_thinking_mode=False,
                supports_multi_turn=True,
                typical_latency_seconds=8.0,  # gpt-image-2 is ~3-8s
                cost_tier="standard",
                best_for=[
                    "Text rendering (menus, posters, infographics)",
                    "UI mockups and labeled screenshots",
                    "Comics with dialogue and speech bubbles",
                    "Technical diagrams with precise labels",
                    "Marketing materials with text",
                    "Multi-step sequential edits (preserve-pixel editing)",
                ],
                not_recommended_for=[
                    "Native 4K output (use Gemini)",
                    "Multi-reference character consistency (use Gemini)",
                    "Real-time data visualization (use Gemini)",
                ],
            )
        return self._capabilities

    # ------------------------------------------------------------------
    # API key + HTTP helper
    # ------------------------------------------------------------------

    def _get_api_key(self, provided_key: str | None = None) -> str:
        """Get API key from provided value, instance, or settings."""
        api_key = provided_key or self._api_key
        if not api_key:
            settings = get_settings()
            api_key = settings.get_openai_api_key()
        return api_key

    def _resolve_model(self, openai_model: str | None) -> str:
        """Resolve a user-provided model alias/name to a canonical model id."""
        if not openai_model:
            return DEFAULT_OPENAI_IMAGE_MODEL
        # Accept both alias keys and canonical ids
        return OPENAI_MODELS.get(openai_model, openai_model)

    async def _make_api_request(
        self,
        endpoint: str,
        api_key: str,
        json_data: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        method: str = "POST",
    ) -> dict[str, Any]:
        """POST or multipart-POST to OpenAI with retry + rate limiting."""
        await self._acquire_rate_limit()

        url = f"{OPENAI_API_BASE_URL}{endpoint}"
        headers: dict[str, str] = {"Authorization": f"Bearer {api_key}"}
        if files is None:
            headers["Content-Type"] = "application/json"

        client = self._ensure_client()

        async def _do_request() -> dict[str, Any]:
            if files is not None:
                # Multipart upload (for /images/edits)
                response = await client.post(url, headers=headers, files=files, data=data)
            elif method == "POST":
                response = await client.post(url, headers=headers, json=json_data)
            else:
                response = await client.request(method, url, headers=headers, json=json_data)
            response.raise_for_status()
            return dict(response.json())

        try:
            return await self._retry_with_backoff(_do_request)
        except httpx.HTTPStatusError as e:
            error_detail = e.response.text
            raise ValueError(f"OpenAI API error ({e.response.status_code}): {error_detail}") from e
        except Exception as e:
            raise ValueError(f"API request failed: {e!s}") from e

    # ------------------------------------------------------------------
    # Direct /images/generations path
    # ------------------------------------------------------------------

    def _build_generate_payload(
        self,
        *,
        model: str,
        prompt: str,
        size: str,
        quality: str | None,
        output_format: str | None,
        output_compression: int | None,
        background: str | None,
        moderation: str | None,
        style: str | None,
        n: int | None,
    ) -> dict[str, Any]:
        """Build the JSON body for /images/generations.

        Only includes keys that were explicitly provided so the API applies
        its own defaults for anything omitted.
        """
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "response_format": "b64_json",  # we save to disk locally
        }
        if quality is not None:
            payload["quality"] = quality
        if output_format is not None:
            payload["output_format"] = output_format
        if output_compression is not None:
            payload["output_compression"] = output_compression
        if background is not None:
            payload["background"] = background
        if moderation is not None:
            payload["moderation"] = moderation
        if style is not None:
            payload["style"] = style
        if n is not None and n > 1:
            payload["n"] = n
        return payload

    async def _call_images_generate_direct(
        self,
        *,
        api_key: str,
        model: str,
        prompt: str,
        size: str,
        quality: str | None,
        output_format: str | None,
        output_compression: int | None,
        background: str | None,
        moderation: str | None,
        style: str | None,
        n: int | None,
    ) -> dict[str, Any]:
        """Call /images/generations directly (no Responses-API pre-stage)."""
        payload = self._build_generate_payload(
            model=model,
            prompt=prompt,
            size=size,
            quality=quality,
            output_format=output_format,
            output_compression=output_compression,
            background=background,
            moderation=moderation,
            style=style,
            n=n,
        )
        logger.info("Calling /images/generations directly with model=%s", model)
        return await self._make_api_request(
            endpoint="/images/generations",
            api_key=api_key,
            json_data=payload,
        )

    # ------------------------------------------------------------------
    # Responses-API (multi-turn / enhanced) path
    # ------------------------------------------------------------------

    async def _call_responses_api(
        self,
        *,
        prompt: str,
        api_key: str,
        conversation_id: str,
        assistant_model: str,
        image_model: str,
        input_image_file_id: str | None,
        size: str,
        quality: str | None,
        output_format: str | None,
        output_compression: int | None,
        background: str | None,
        moderation: str | None,
        style: str | None,
        n: int | None,
    ) -> dict[str, Any]:
        """Two-stage call: prompt refinement via chat -> image generation."""
        messages: list[dict[str, Any]] = []

        # Replay history from persistent store
        history = self._get_conversation_history(conversation_id)
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})

        # Current user turn
        current_message: dict[str, Any] = {"role": "user", "content": []}
        if input_image_file_id:
            current_message["content"].append(
                {
                    "type": "image_file",
                    "image_file": {"file_id": input_image_file_id},
                }
            )
        current_message["content"].append({"type": "text", "text": prompt})
        messages.append(current_message)

        # Forced function-calling tool schema — lets the assistant model refine
        # the prompt and choose a size. We still own the actual image call.
        tool_schema_properties: dict[str, Any] = {
            "prompt": {
                "type": "string",
                "description": "Refined prompt to send to the image model",
            },
            "size": {
                "type": "string",
                "enum": list(OPENAI_SIZES),
                "default": "1024x1024",
            },
        }
        if quality is None:
            tool_schema_properties["quality"] = {
                "type": "string",
                "enum": list(OPENAI_QUALITY_OPTIONS),
            }

        payload = {
            "model": assistant_model,
            "messages": messages,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "generate_image",
                        "description": (
                            f"Generate an image using {image_model}. "
                            "Refine the user's prompt before calling."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": tool_schema_properties,
                            "required": ["prompt"],
                        },
                    },
                }
            ],
            "tool_choice": {
                "type": "function",
                "function": {"name": "generate_image"},
            },
            "parallel_tool_calls": False,
            "max_tokens": 1000,
        }

        response = await self._make_api_request(
            endpoint="/chat/completions",
            api_key=api_key,
            json_data=payload,
        )

        # Persist user message
        self._store_conversation_message(conversation_id, "user", current_message["content"])

        if not ("choices" in response and response["choices"]):
            return response

        choice = response["choices"][0]
        if "message" not in choice:
            return response

        assistant_message = choice["message"]
        self._store_conversation_message(
            conversation_id, "assistant", assistant_message.get("content", "")
        )

        if "tool_calls" not in assistant_message:
            return response

        for tool_call in assistant_message["tool_calls"]:
            if tool_call["function"]["name"] != "generate_image":
                continue

            raw_args = tool_call["function"].get("arguments", {})
            tool_args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
            logger.info("Assistant refined prompt via tool call: %s", tool_args)

            image_response = await self._call_images_generate_direct(
                api_key=api_key,
                model=image_model,
                prompt=tool_args.get("prompt", prompt),
                size=tool_args.get("size", size),
                quality=tool_args.get("quality", quality),
                output_format=output_format,
                output_compression=output_compression,
                background=background,
                moderation=moderation,
                style=style,
                n=n,
            )

            return {
                "conversation_id": conversation_id,
                "chat_response": response,
                "image_response": image_response,
                "tool_calls": assistant_message["tool_calls"],
                "refined_prompt": tool_args.get("prompt", prompt),
            }

        return response

    # ------------------------------------------------------------------
    # Param validation
    # ------------------------------------------------------------------

    async def validate_params(
        self,
        prompt: str,
        size: str | None = None,
        aspect_ratio: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Validate and normalize parameters for OpenAI."""
        if len(prompt) > MAX_PROMPT_LENGTH:
            raise ValueError(f"Prompt too long. Maximum {MAX_PROMPT_LENGTH} characters.")

        # --- Size ---
        if size:
            size = size.replace("X", "x")
            # Accept Gemini-style sizes by mapping to the closest OpenAI size.
            gemini_to_openai = {
                "1K": "1024x1024",
                "2K": "1536x1024",
                "4K": "1792x1024",  # gpt-image-2 max, was 1536x1024 on 1.x
            }
            normalized = size.upper()
            if normalized in gemini_to_openai:
                mapped = gemini_to_openai[normalized]
                logger.info(
                    "Converting Gemini size '%s' to OpenAI size '%s' (max supported).",
                    size,
                    mapped,
                )
                size = mapped
            if size not in OPENAI_SIZES:
                raise ValueError(
                    f"Invalid size '{size}' for OpenAI. Supported: {', '.join(OPENAI_SIZES)}"
                )
        else:
            size = self._size_from_aspect_ratio(aspect_ratio) if aspect_ratio else "1024x1024"

        # --- Enum validation (pass through to API if valid, else raise) ---
        validated: dict[str, Any] = {"prompt": prompt, "size": size}

        quality = kwargs.get("quality")
        if quality is not None:
            if quality not in OPENAI_QUALITY_OPTIONS:
                raise ValueError(
                    f"Invalid quality '{quality}'. Supported: {', '.join(OPENAI_QUALITY_OPTIONS)}"
                )
            validated["quality"] = quality

        output_format = kwargs.get("openai_output_format")
        if output_format is not None:
            if output_format not in OPENAI_OUTPUT_FORMATS:
                raise ValueError(
                    f"Invalid output_format '{output_format}'. "
                    f"Supported: {', '.join(OPENAI_OUTPUT_FORMATS)}"
                )
            validated["output_format"] = output_format

        output_compression = kwargs.get("openai_output_compression")
        if output_compression is not None:
            if not (0 <= int(output_compression) <= 100):
                raise ValueError("output_compression must be between 0 and 100.")
            validated["output_compression"] = int(output_compression)

        background = kwargs.get("background")
        if background is not None:
            if background not in OPENAI_BACKGROUND_OPTIONS:
                raise ValueError(
                    f"Invalid background '{background}'. "
                    f"Supported: {', '.join(OPENAI_BACKGROUND_OPTIONS)}"
                )
            validated["background"] = background

        moderation = kwargs.get("moderation")
        if moderation is not None:
            if moderation not in OPENAI_MODERATION_OPTIONS:
                raise ValueError(
                    f"Invalid moderation '{moderation}'. "
                    f"Supported: {', '.join(OPENAI_MODERATION_OPTIONS)}"
                )
            validated["moderation"] = moderation

        style = kwargs.get("style")
        if style is not None:
            if style not in OPENAI_STYLES:
                raise ValueError(f"Invalid style '{style}'. Supported: {', '.join(OPENAI_STYLES)}")
            validated["style"] = style

        n = kwargs.get("n")
        if n is not None:
            n = int(n)
            if not (1 <= n <= OPENAI_MAX_N):
                raise ValueError(f"n must be between 1 and {OPENAI_MAX_N}.")
            validated["n"] = n

        return validated

    def _size_from_aspect_ratio(self, aspect_ratio: str) -> str:
        """Convert an aspect ratio string to the closest OpenAI size."""
        ratio_to_size = {
            "1:1": "1024x1024",
            "2:3": "1024x1536",
            "3:2": "1536x1024",
            "9:16": "1024x1792",
            "16:9": "1792x1024",
            "portrait": "1024x1536",
            "landscape": "1536x1024",
            "square": "1024x1024",
        }
        return ratio_to_size.get(aspect_ratio, "1024x1024")

    # ------------------------------------------------------------------
    # Response parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_image_data(image_response: dict[str, Any]) -> list[dict[str, Any]]:
        """Return a list of {'b64_json'|'url', 'revised_prompt'} entries."""
        entries: list[dict[str, Any]] = []
        for item in image_response.get("data") or []:
            entry: dict[str, Any] = {}
            if "b64_json" in item and item["b64_json"]:
                entry["b64_json"] = item["b64_json"]
            elif "url" in item and item["url"]:
                entry["url"] = item["url"]
            if item.get("revised_prompt"):
                entry["revised_prompt"] = item["revised_prompt"]
            if entry:
                entries.append(entry)
        return entries

    @staticmethod
    def _extract_usage(image_response: dict[str, Any]) -> dict[str, int] | None:
        """Return the usage block from an images response, if present."""
        usage = image_response.get("usage")
        if not usage or not isinstance(usage, dict):
            return None
        # Normalize to plain ints so the result is JSON-serializable.
        normalized: dict[str, int] = {}
        for key, value in usage.items():
            try:
                normalized[key] = int(value)
            except (TypeError, ValueError):
                continue
        return normalized or None

    # ------------------------------------------------------------------
    # Public generate / edit entry points
    # ------------------------------------------------------------------

    async def generate_image(
        self,
        prompt: str,
        *,
        size: str | None = None,
        aspect_ratio: str | None = None,
        conversation_id: str | None = None,
        reference_images: list[str] | None = None,
        enable_enhancement: bool = True,
        api_key: str | None = None,
        assistant_model: str = "gpt-4o",
        input_image_file_id: str | None = None,
        output_path: str | None = None,
        # New gpt-image-2 params (kwargs so base.ImageProvider.generate_image sig stays stable)
        openai_model: str | None = None,
        quality: str | None = None,
        openai_output_format: str | None = None,
        openai_output_compression: int | None = None,
        background: str | None = None,
        moderation: str | None = None,
        style: str | None = None,
        n: int | None = None,
        **kwargs: Any,
    ) -> ImageResult:
        """Generate an image using OpenAI gpt-image-2."""
        start_time = time.time()
        image_model = self._resolve_model(openai_model)

        try:
            api_key = self._get_api_key(api_key)

            # Validate and normalize
            validated = await self.validate_params(
                prompt,
                size,
                aspect_ratio,
                quality=quality,
                openai_output_format=openai_output_format,
                openai_output_compression=openai_output_compression,
                background=background,
                moderation=moderation,
                style=style,
                n=n,
            )
            size = str(validated["size"])
            quality = validated.get("quality", quality)
            output_format = validated.get("output_format", openai_output_format)
            output_compression = validated.get("output_compression", openai_output_compression)
            background = validated.get("background", background)
            moderation = validated.get("moderation", moderation)
            style = validated.get("style", style)
            n = validated.get("n", n)

            conversation_id = conversation_id or self._generate_conversation_id()

            if reference_images:
                logger.warning(
                    "OpenAI provider does not accept reference_images on /images/generations. "
                    "Use the edit_image tool for image-to-image editing."
                )

            # Pick the code path
            refined_prompt: str | None = None
            if enable_enhancement:
                result = await self._call_responses_api(
                    prompt=prompt,
                    api_key=api_key,
                    conversation_id=conversation_id,
                    assistant_model=assistant_model,
                    image_model=image_model,
                    input_image_file_id=input_image_file_id,
                    size=size,
                    quality=quality,
                    output_format=output_format,
                    output_compression=output_compression,
                    background=background,
                    moderation=moderation,
                    style=style,
                    n=n,
                )
                image_response = result.get("image_response", {})
                refined_prompt = result.get("refined_prompt")
            else:
                image_response = await self._call_images_generate_direct(
                    api_key=api_key,
                    model=image_model,
                    prompt=prompt,
                    size=size,
                    quality=quality,
                    output_format=output_format,
                    output_compression=output_compression,
                    background=background,
                    moderation=moderation,
                    style=style,
                    n=n,
                )

            # Parse images
            entries = self._extract_image_data(image_response)
            usage = self._extract_usage(image_response)

            image_path: Path | None = None
            additional_paths: list[Path] = []
            response_revised: str | None = None

            for i, entry in enumerate(entries):
                if "b64_json" not in entry:
                    continue
                save_path = await self._save_image(entry["b64_json"], prompt, output_path)
                if image_path is None:
                    image_path = save_path
                    if entry.get("revised_prompt"):
                        response_revised = entry["revised_prompt"]
                else:
                    additional_paths.append(save_path)
                # Persist first image to conversation store for multi-turn
                if i == 0:
                    self._store_conversation_message(
                        conversation_id,
                        "assistant",
                        {"type": "image_generated", "prompt": prompt},
                        image_base64=entry["b64_json"],
                        metadata={
                            "size": size,
                            "model": image_model,
                            "quality": quality,
                            "output_format": output_format,
                        },
                    )

            generation_time = time.time() - start_time

            return ImageResult(
                success=True,
                provider=self.name,
                model=image_model,
                image_path=image_path,
                additional_paths=additional_paths or None,
                prompt=prompt,
                enhanced_prompt=refined_prompt or response_revised,
                size=image_response.get("size") or size,
                aspect_ratio=aspect_ratio,
                quality=image_response.get("quality") or quality,
                output_format=image_response.get("output_format") or output_format,
                background=image_response.get("background") or background,
                conversation_id=conversation_id,
                timestamp=datetime.now(),
                generation_time_seconds=generation_time,
                usage_tokens=usage,
            )

        except Exception as e:
            logger.exception("OpenAI image generation failed")
            return ImageResult(
                success=False,
                provider=self.name,
                model=image_model,
                prompt=prompt,
                error=str(e),
            )

    async def edit_image(
        self,
        *,
        prompt: str,
        image_path: str,
        mask_path: str | None = None,
        size: str | None = None,
        quality: str | None = None,
        background: str | None = None,
        openai_output_format: str | None = None,
        openai_output_compression: int | None = None,
        input_fidelity: str | None = None,
        n: int | None = None,
        openai_model: str | None = None,
        api_key: str | None = None,
        output_path: str | None = None,
    ) -> ImageResult:
        """Edit an image via /images/edits with gpt-image-2.

        This is the right entry point for image-to-image workflows on
        OpenAI (reference-image-style consistency). Uses
        ``input_fidelity='high'`` by default so unchanged pixels are
        preserved.
        """
        start_time = time.time()
        image_model = self._resolve_model(openai_model)

        try:
            api_key = self._get_api_key(api_key)

            # Validate size against the edits endpoint's narrower size list.
            if size:
                size = size.replace("X", "x")
                if size not in OPENAI_EDIT_SIZES:
                    raise ValueError(
                        f"Invalid size '{size}' for /images/edits. "
                        f"Supported: {', '.join(OPENAI_EDIT_SIZES)}"
                    )
            else:
                size = "auto"

            if quality is not None and quality not in OPENAI_QUALITY_OPTIONS:
                raise ValueError(
                    f"Invalid quality '{quality}'. Supported: {', '.join(OPENAI_QUALITY_OPTIONS)}"
                )
            if background is not None and background not in OPENAI_BACKGROUND_OPTIONS:
                raise ValueError(
                    f"Invalid background '{background}'. "
                    f"Supported: {', '.join(OPENAI_BACKGROUND_OPTIONS)}"
                )
            if (
                openai_output_format is not None
                and openai_output_format not in OPENAI_OUTPUT_FORMATS
            ):
                raise ValueError(
                    f"Invalid output_format '{openai_output_format}'. "
                    f"Supported: {', '.join(OPENAI_OUTPUT_FORMATS)}"
                )
            if input_fidelity is None:
                input_fidelity = DEFAULT_OPENAI_INPUT_FIDELITY
            if input_fidelity not in OPENAI_INPUT_FIDELITY_OPTIONS:
                raise ValueError(
                    f"Invalid input_fidelity '{input_fidelity}'. "
                    f"Supported: {', '.join(OPENAI_INPUT_FIDELITY_OPTIONS)}"
                )
            if n is not None:
                n = int(n)
                if not (1 <= n <= OPENAI_MAX_N):
                    raise ValueError(f"n must be between 1 and {OPENAI_MAX_N}.")

            # Resolve + read files (async to avoid blocking the event loop)
            import asyncio

            img_path = Path(image_path).expanduser().resolve()
            if not img_path.is_file():
                raise ValueError(f"Source image not found: {img_path}")

            def _read_bytes(p: Path) -> bytes:
                return p.read_bytes()

            image_bytes = await asyncio.to_thread(_read_bytes, img_path)

            mask_bytes: bytes | None = None
            if mask_path:
                m_path = Path(mask_path).expanduser().resolve()
                if not m_path.is_file():
                    raise ValueError(f"Mask not found: {m_path}")
                mask_bytes = await asyncio.to_thread(_read_bytes, m_path)

            # Build multipart form
            files: dict[str, Any] = {
                "image": (img_path.name, image_bytes, "application/octet-stream"),
            }
            if mask_bytes is not None:
                files["mask"] = ("mask.png", mask_bytes, "image/png")

            form_data: dict[str, Any] = {
                "model": image_model,
                "prompt": prompt,
                "size": size,
                "input_fidelity": input_fidelity,
                "response_format": "b64_json",
            }
            if quality is not None:
                form_data["quality"] = quality
            if background is not None:
                form_data["background"] = background
            if openai_output_format is not None:
                form_data["output_format"] = openai_output_format
            if openai_output_compression is not None:
                form_data["output_compression"] = int(openai_output_compression)
            if n is not None and n > 1:
                form_data["n"] = str(n)

            logger.info(
                "Calling /images/edits model=%s size=%s fidelity=%s n=%s",
                image_model,
                size,
                input_fidelity,
                n or 1,
            )
            image_response = await self._make_api_request(
                endpoint="/images/edits",
                api_key=api_key,
                files=files,
                data=form_data,
            )

            entries = self._extract_image_data(image_response)
            usage = self._extract_usage(image_response)

            saved_path: Path | None = None
            additional_paths: list[Path] = []
            response_revised: str | None = None
            for entry in entries:
                if "b64_json" not in entry:
                    continue
                save_path = await self._save_image(entry["b64_json"], prompt, output_path)
                if saved_path is None:
                    saved_path = save_path
                    if entry.get("revised_prompt"):
                        response_revised = entry["revised_prompt"]
                else:
                    additional_paths.append(save_path)

            generation_time = time.time() - start_time

            return ImageResult(
                success=True,
                provider=self.name,
                model=image_model,
                image_path=saved_path,
                additional_paths=additional_paths or None,
                prompt=prompt,
                enhanced_prompt=response_revised,
                size=image_response.get("size") or size,
                quality=image_response.get("quality") or quality,
                output_format=image_response.get("output_format") or openai_output_format,
                background=image_response.get("background") or background,
                timestamp=datetime.now(),
                generation_time_seconds=generation_time,
                usage_tokens=usage,
            )

        except Exception as e:
            logger.exception("OpenAI image edit failed")
            return ImageResult(
                success=False,
                provider=self.name,
                model=image_model,
                prompt=prompt,
                error=str(e),
            )


# Keep legacy default export path used by tests / downstream code
__all__ = [
    "OpenAIProvider",
    "DEFAULT_OPENAI_BACKGROUND",
    "DEFAULT_OPENAI_MODERATION",
    "DEFAULT_OPENAI_OUTPUT_FORMAT",
    "DEFAULT_OPENAI_QUALITY",
]

# Module-level marker: base64 is imported for potential in-memory workflows
# that the edit path might grow (stream decoded bytes into files directly
# without saving base64 to disk first). Currently unused; keep available.
_ = base64
