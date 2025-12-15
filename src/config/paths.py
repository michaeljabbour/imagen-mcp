"""
Path utilities for saving generated images.

Centralizes output directory behavior across providers:
- Expands `~` and environment variables for user-supplied paths
- Supports `OUTPUT_DIR` as the default save directory when `output_path` is omitted
- Creates parent directories as needed
"""

from __future__ import annotations

import os
from pathlib import Path

from .settings import get_settings


def expand_path(path: str) -> Path:
    """Expand `~` and environment variables in a path string."""
    expanded = os.path.expandvars(os.path.expanduser(path))
    return Path(expanded)


def get_base_output_directory() -> Path:
    """Get the base directory for saving images (provider subdirs are created under this)."""
    settings = get_settings()
    if settings.output_dir:
        output_dir = expand_path(settings.output_dir)
    else:
        output_dir = Path.home() / "Downloads" / "images"

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_provider_output_directory(provider: str) -> Path:
    """Get the default provider-specific directory for saving images."""
    provider_dir = get_base_output_directory() / provider
    provider_dir.mkdir(parents=True, exist_ok=True)
    return provider_dir


def get_log_directory() -> Path:
    """Get the directory for server logs."""
    settings = get_settings()
    if settings.log_dir:
        log_dir = expand_path(settings.log_dir)
    else:
        log_dir = get_base_output_directory() / "logs"

    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def resolve_output_path(
    output_path: str | None, *, default_filename: str, provider: str | None = None
) -> Path:
    """
    Resolve an `output_path` (directory or file path) into a concrete file path.

    Rules:
    - If `output_path` is omitted, saves under `~/Downloads/images/{provider}` by default
      (or under `OUTPUT_DIR/{provider}` when `OUTPUT_DIR` is set)
    - If `output_path` ends with a path separator, treat it as a directory
    - If `output_path` exists and is a directory, treat it as a directory
    - If `output_path` has no suffix, treat it as a directory
    - Otherwise treat it as a file path
    """
    if not default_filename:
        raise ValueError("default_filename must not be empty")

    if output_path and output_path.strip():
        raw = output_path.strip()
        path_obj = expand_path(raw)

        is_directory = (
            raw.endswith(("/", "\\"))
            or (path_obj.exists() and path_obj.is_dir())
            or not path_obj.suffix
        )
        if is_directory:
            save_path = path_obj / default_filename
        else:
            save_path = path_obj
    else:
        base_dir = get_provider_output_directory(provider) if provider else get_base_output_directory()
        save_path = base_dir / default_filename

    save_path.parent.mkdir(parents=True, exist_ok=True)
    return save_path
