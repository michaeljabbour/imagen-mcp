"""
OpenAI GPT-Image-1 provider implementation.

This provider wraps OpenAI's GPT-Image-1 model for image generation,
using the Responses API for conversational workflows.
"""

import asyncio
import base64
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

from ..config.constants import (
    MAX_PROMPT_LENGTH,
    MAX_RETRIES,
    OPENAI_API_BASE_URL,
    OPENAI_SIZES,
)
from ..config.paths import resolve_output_path
from ..config.settings import get_settings
from .base import ImageProvider, ImageResult, ProviderCapabilities

logger = logging.getLogger(__name__)


class OpenAIProvider(ImageProvider):
    """
    OpenAI GPT-Image-1 provider.

    Best for:
    - Text rendering (menus, infographics, comics)
    - Precise instruction following
    - Technical diagrams with labels
    - Marketing materials with text

    Limitations:
    - Only 3 size options
    - No reference image support
    - No real-time data grounding
    - Slower generation (~60s)
    """

    def __init__(self, api_key: str | None = None):
        """Initialize OpenAI provider."""
        self._api_key = api_key
        self._conversation_store: dict[str, list[dict[str, Any]]] = {}

    @property
    def name(self) -> str:
        return "openai"

    @property
    def display_name(self) -> str:
        return "OpenAI GPT-Image-1"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            name="openai",
            display_name="OpenAI GPT-Image-1",
            supported_sizes=OPENAI_SIZES,
            supported_aspect_ratios=["1:1", "2:3", "3:2"],  # Derived from sizes
            max_resolution="1536x1024",
            supports_text_rendering=True,
            text_rendering_quality="excellent",
            supports_reference_images=False,
            max_reference_images=0,
            supports_real_time_data=False,
            supports_thinking_mode=False,
            supports_multi_turn=True,
            typical_latency_seconds=60.0,
            cost_tier="standard",
            best_for=[
                "Text rendering (menus, posters, infographics)",
                "Comics and sequential art with dialogue",
                "Technical diagrams with labels",
                "Marketing materials with text",
                "Precise instruction following",
            ],
            not_recommended_for=[
                "Photorealistic portraits",
                "Product photography",
                "Real-time data visualization",
                "High resolution (4K) output",
                "Multi-reference consistency",
            ],
        )

    def _get_api_key(self, provided_key: str | None = None) -> str:
        """Get API key from provided value, instance, or settings."""
        api_key = provided_key or self._api_key
        if not api_key:
            settings = get_settings()
            api_key = settings.get_openai_api_key()
        return api_key

    def _generate_conversation_id(self) -> str:
        """Generate a unique conversation ID."""
        return f"conv_{uuid4().hex[:12]}"

    async def _make_api_request(
        self,
        endpoint: str,
        api_key: str,
        json_data: dict[str, Any] | None = None,
        method: str = "POST",
    ) -> dict[str, Any]:
        """Make an API request to OpenAI with retry logic."""
        url = f"{OPENAI_API_BASE_URL}{endpoint}"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            for attempt in range(MAX_RETRIES):
                try:
                    if method == "POST":
                        response = await client.post(url, headers=headers, json=json_data)
                    else:
                        response = await client.request(
                            method, url, headers=headers, json=json_data
                        )

                    response.raise_for_status()
                    return dict(response.json())

                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 429:  # Rate limit
                        if attempt < MAX_RETRIES - 1:
                            await asyncio.sleep(2**attempt)
                            continue
                    error_detail = e.response.text
                    raise ValueError(
                        f"OpenAI API error ({e.response.status_code}): {error_detail}"
                    ) from e
                except Exception as e:
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(1)
                        continue
                    raise ValueError(f"API request failed: {str(e)}") from e

        raise ValueError("API request failed after all retries")

    async def _call_responses_api(
        self,
        prompt: str,
        api_key: str,
        conversation_id: str,
        assistant_model: str = "gpt-4o",
        input_image_file_id: str | None = None,
        size: str = "1024x1024",
    ) -> dict[str, Any]:
        """Call OpenAI Responses API for conversational image generation."""
        messages = []

        # Retrieve conversation history if exists
        if conversation_id in self._conversation_store:
            messages.extend(self._conversation_store[conversation_id])

        # Build the current message
        current_message: dict[str, Any] = {"role": "user", "content": []}

        # Add image input if provided
        if input_image_file_id:
            current_message["content"].append(
                {
                    "type": "image_file",
                    "image_file": {"file_id": input_image_file_id},
                }
            )

        # Add text prompt
        current_message["content"].append(
            {
                "type": "text",
                "text": prompt,
            }
        )

        messages.append(current_message)

        # Build the request payload
        payload = {
            "model": assistant_model,
            "messages": messages,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "generate_image",
                        "description": "Generate an image based on a text prompt using gpt-image-1",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "model": {
                                    "type": "string",
                                    "enum": ["gpt-image-1"],
                                    "default": "gpt-image-1",
                                },
                                "prompt": {
                                    "type": "string",
                                    "description": "The prompt for image generation",
                                },
                                "size": {
                                    "type": "string",
                                    "enum": OPENAI_SIZES,
                                    "default": "1024x1024",
                                },
                            },
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

        # Make the API call
        response = await self._make_api_request(
            endpoint="/chat/completions",
            api_key=api_key,
            json_data=payload,
        )

        # Store conversation
        if conversation_id not in self._conversation_store:
            self._conversation_store[conversation_id] = []

        self._conversation_store[conversation_id].append(current_message)

        # Process response
        if "choices" in response and response["choices"]:
            choice = response["choices"][0]
            if "message" in choice:
                assistant_message = choice["message"]
                self._conversation_store[conversation_id].append(assistant_message)

                # Check for tool calls
                if "tool_calls" in assistant_message:
                    for tool_call in assistant_message["tool_calls"]:
                        if tool_call["function"]["name"] == "generate_image":
                            tool_args = (
                                json.loads(tool_call["function"]["arguments"])
                                if isinstance(tool_call["function"].get("arguments"), str)
                                else tool_call["function"].get("arguments", {})
                            )

                            logger.info(f"Tool call arguments: {tool_args}")

                            # Call the actual image generation endpoint
                            image_payload = {
                                "model": "gpt-image-1",
                                "prompt": tool_args.get("prompt", prompt),
                                "size": tool_args.get("size", size),
                                "n": 1,
                            }

                            logger.info(f"Calling Images API with payload: {image_payload}")
                            image_response = await self._make_api_request(
                                endpoint="/images/generations",
                                api_key=api_key,
                                json_data=image_payload,
                            )

                            # Extract image data
                            image_data = None
                            if "data" in image_response and image_response["data"]:
                                first_image = image_response["data"][0]
                                if "b64_json" in first_image:
                                    image_data = {
                                        "b64_json": first_image["b64_json"],
                                    }
                                elif "url" in first_image:
                                    image_data = {"url": first_image["url"]}

                            return {
                                "conversation_id": conversation_id,
                                "response": response,
                                "image_response": image_response,
                                "image_data": image_data,
                                "tool_calls": assistant_message["tool_calls"],
                                "full_message": assistant_message,
                            }

        return response

    async def validate_params(
        self,
        prompt: str,
        size: str | None = None,
        aspect_ratio: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Validate and normalize parameters for OpenAI."""
        # Validate prompt length
        if len(prompt) > MAX_PROMPT_LENGTH:
            raise ValueError(f"Prompt too long. Maximum {MAX_PROMPT_LENGTH} characters.")

        # Validate/normalize size
        if size:
            if size not in OPENAI_SIZES:
                raise ValueError(
                    f"Invalid size '{size}' for OpenAI. Supported sizes: {', '.join(OPENAI_SIZES)}"
                )
        else:
            # Default based on aspect ratio if provided
            if aspect_ratio:
                size = self._size_from_aspect_ratio(aspect_ratio)
            else:
                size = "1024x1024"

        return {
            "prompt": prompt,
            "size": size,
        }

    def _size_from_aspect_ratio(self, aspect_ratio: str) -> str:
        """Convert aspect ratio to OpenAI size."""
        ratio_to_size = {
            "1:1": "1024x1024",
            "2:3": "1024x1536",
            "3:2": "1536x1024",
            "portrait": "1024x1536",
            "landscape": "1536x1024",
            "square": "1024x1024",
        }
        return ratio_to_size.get(aspect_ratio, "1024x1024")

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
        **kwargs: Any,
    ) -> ImageResult:
        """Generate an image using OpenAI GPT-Image-1."""
        start_time = time.time()

        try:
            # Get API key
            api_key = self._get_api_key(api_key)

            # Validate parameters
            validated = await self.validate_params(prompt, size, aspect_ratio, **kwargs)
            size = validated["size"]

            # Generate conversation ID if not provided
            conversation_id = conversation_id or self._generate_conversation_id()

            # Note: OpenAI doesn't support reference images
            if reference_images:
                logger.warning("OpenAI provider does not support reference images. Ignoring.")

            # Call Responses API
            result = await self._call_responses_api(
                prompt=prompt,
                api_key=api_key,
                conversation_id=conversation_id,
                assistant_model=assistant_model,
                input_image_file_id=input_image_file_id,
                size=size,
            )

            # Extract and save image
            image_path = None
            if "image_data" in result and result["image_data"]:
                image_data = result["image_data"]
                if "b64_json" in image_data:
                    # Save to file
                    image_path = self._save_image(
                        image_data["b64_json"], prompt, output_path=output_path
                    )

            generation_time = time.time() - start_time

            return ImageResult(
                success=True,
                provider=self.name,
                model="gpt-image-1",
                image_path=image_path,
                image_base64=result.get("image_data", {}).get("b64_json")
                if result.get("image_data")
                else None,
                prompt=prompt,
                size=size,
                conversation_id=conversation_id,
                timestamp=datetime.now(),
                generation_time_seconds=generation_time,
            )

        except Exception as e:
            logger.exception("OpenAI image generation failed")
            return ImageResult(
                success=False,
                provider=self.name,
                model="gpt-image-1",
                prompt=prompt,
                error=str(e),
            )

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
        filename = f"openai_{timestamp}_{prompt_snippet}_{short_id}.png"

        save_path = resolve_output_path(output_path, default_filename=filename, provider=self.name)

        with open(save_path, "wb") as f:
            f.write(image_bytes)

        logger.info(f"Image saved to: {save_path}")
        return save_path

    def get_best_size_for_type(self, image_type: str) -> str:
        """Get best OpenAI size for image type."""
        if image_type in ["portrait", "headshot", "person", "selfie", "phone"]:
            return "1024x1536"
        elif image_type in ["landscape", "scene", "banner", "panorama"]:
            return "1536x1024"
        else:
            return "1024x1024"

    def get_conversation_history(self, conversation_id: str) -> list[dict[str, Any]]:
        """Get conversation history for a conversation ID."""
        return self._conversation_store.get(conversation_id, [])

    def clear_conversation(self, conversation_id: str) -> None:
        """Clear conversation history."""
        if conversation_id in self._conversation_store:
            del self._conversation_store[conversation_id]

    def get_conversations(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get list of recent conversations."""
        conversations = []
        for conv_id, messages in self._conversation_store.items():
            if not messages:
                continue

            last_msg = messages[-1]
            last_content = "No content"

            # Try to extract meaningful content from the last message
            if "content" in last_msg:
                if isinstance(last_msg["content"], str):
                    last_content = last_msg["content"][:50]
                elif isinstance(last_msg["content"], list):
                    for part in last_msg["content"]:
                        if part.get("type") == "text":
                            last_content = part.get("text", "")[:50]
                            break

            conversations.append(
                {
                    "id": conv_id,
                    "provider": "openai",
                    "message_count": len(messages),
                    "last_message": last_content,
                    "updated": datetime.now(),  # Approximate time
                }
            )

        # Sort by ID (approximate time) descending
        conversations.sort(key=lambda x: str(x["id"]), reverse=True)
        return conversations[:limit]
