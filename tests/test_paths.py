"""Tests for output path handling utilities."""

import os
from pathlib import Path

# Set dummy API keys for testing
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("GEMINI_API_KEY", "test-key")


TINY_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7+9pQAAAAA"
    "SUVORK5CYII="
)


def test_resolve_output_path_expands_user_directory(tmp_path: Path, monkeypatch):
    from src.config.paths import resolve_output_path

    monkeypatch.setenv("HOME", str(tmp_path))

    resolved = resolve_output_path("~/images/", default_filename="test.png")
    assert resolved == tmp_path / "images" / "test.png"
    assert resolved.parent.is_dir()


def test_resolve_output_path_existing_directory_with_suffix_is_directory(
    tmp_path: Path,
):
    from src.config.paths import resolve_output_path

    output_dir = tmp_path / "my.images"
    output_dir.mkdir()

    resolved = resolve_output_path(str(output_dir), default_filename="test.png")
    assert resolved == output_dir / "test.png"


def test_default_output_directory_uses_output_dir_env(tmp_path: Path, monkeypatch):
    from src.config.paths import get_base_output_directory, get_provider_output_directory
    from src.config.settings import get_settings

    output_dir = tmp_path / "custom-output"
    monkeypatch.setenv("OUTPUT_DIR", str(output_dir))
    get_settings.cache_clear()

    resolved = get_base_output_directory()
    assert resolved == output_dir
    assert resolved.is_dir()

    provider_dir = get_provider_output_directory("openai")
    assert provider_dir == output_dir / "openai"
    assert provider_dir.is_dir()


def test_provider_save_image_uses_output_dir_env(tmp_path: Path, monkeypatch):
    from src.config.settings import get_settings
    from src.providers.openai_provider import OpenAIProvider

    output_dir = tmp_path / "custom-output"
    monkeypatch.setenv("OUTPUT_DIR", str(output_dir))
    get_settings.cache_clear()

    provider = OpenAIProvider()
    saved_path = provider._save_image(TINY_PNG_BASE64, "Test prompt")
    assert saved_path.parent == output_dir / "openai"
    assert saved_path.is_file()


def test_provider_save_image_output_path_directory_creates_file(tmp_path: Path):
    from src.providers.gemini_provider import GeminiProvider

    output_dir = tmp_path / "outputs"
    provider = GeminiProvider()
    saved_path = provider._save_image(TINY_PNG_BASE64, "Test prompt", output_path=str(output_dir))

    assert saved_path.parent == output_dir
    assert output_dir.is_dir()
    assert saved_path.is_file()
