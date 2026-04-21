#!/usr/bin/env python3
"""
Slim interactive CLI for imagen-mcp.

Usage:
    python -m src.cli                    # Interactive mode
    python -m src.cli "A sunset"         # One-shot generation
    python -m src.cli -p openai "Menu"   # Explicit provider
"""

import argparse
import asyncio
import importlib
import os
import subprocess
import sys
from getpass import getpass
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from .config.dotenv import get_dotenv_path, load_dotenv, upsert_dotenv
from .config.paths import expand_path
from .config.settings import Settings, get_settings
from .providers import ProviderRegistry, get_provider_registry
from .services.logging_config import configure_logging


def _module_available(module_name: str) -> bool:
    try:
        return find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def _in_virtual_env() -> bool:
    return bool(os.environ.get("VIRTUAL_ENV")) or sys.prefix != getattr(
        sys, "base_prefix", sys.prefix
    )


def _is_interactive_session() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _install_packages(packages: list[str]) -> bool:
    if not packages:
        return True

    cmd = [sys.executable, "-m", "pip", "install", *packages]
    print(f"Installing: {' '.join(packages)}")
    try:
        proc = subprocess.run(cmd, check=False)
    except FileNotFoundError:
        print("pip not available in this Python environment.")
        return False

    if proc.returncode != 0:
        print("Dependency install failed. You can install manually with:")
        print(f"  {' '.join(cmd)}")
        return False

    importlib.invalidate_caches()
    return True


def _ensure_runtime_dependencies(*, interactive: bool) -> None:
    missing_packages: list[str] = []

    if not _module_available("httpx"):
        missing_packages.append("httpx")
    if not _module_available("google.genai"):
        missing_packages.append("google-genai")
    if not _module_available("PIL"):
        missing_packages.append("pillow")

    if not missing_packages:
        return

    if not _in_virtual_env():
        print("Warning: not running in a virtual environment.")
        if interactive:
            answer = (
                input("Install missing dependencies into this Python anyway? [y/N] ")
                .strip()
                .lower()
            )
            if answer not in ("y", "yes"):
                return

    _install_packages(missing_packages)


def _configure_api_keys(dotenv_path: Path, *, interactive: bool, force: bool) -> None:
    if not interactive:
        return

    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_gemini = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))

    if not force and (has_openai or has_gemini):
        return

    if not has_openai and not has_gemini:
        print("No API keys found (OPENAI_API_KEY / GEMINI_API_KEY).")
        print("Paste at least one key to continue (leave blank to skip).")

    updates: dict[str, str] = {}

    if not has_openai:
        openai_key = getpass("OPENAI_API_KEY: ").strip()
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key
            updates["OPENAI_API_KEY"] = openai_key

    if not has_gemini:
        gemini_key = getpass("GEMINI_API_KEY: ").strip()
        if gemini_key:
            os.environ["GEMINI_API_KEY"] = gemini_key
            updates["GEMINI_API_KEY"] = gemini_key

    if updates:
        upsert_dotenv(dotenv_path, updates)
        print(f"Saved to: {dotenv_path}")

        # Refresh cached singletons that read env vars.
        get_settings.cache_clear()
        get_provider_registry.cache_clear()


def bootstrap_cli(*, interactive: bool) -> Path:
    dotenv_path = get_dotenv_path()
    load_dotenv(dotenv_path, override=False)
    _ensure_runtime_dependencies(interactive=interactive)
    _configure_api_keys(dotenv_path, interactive=interactive, force=False)
    return dotenv_path


def _provider_status(settings: Settings) -> dict[str, dict[str, Any]]:
    providers: dict[str, dict[str, Any]] = {}

    openai_missing: list[str] = []
    if not settings.has_openai_key():
        openai_missing.append("OPENAI_API_KEY")
    if not _module_available("httpx"):
        openai_missing.append("dependency:httpx")
    providers["openai"] = {
        "available": len(openai_missing) == 0,
        "missing": openai_missing,
        "install_hint": "pip install httpx",
    }

    gemini_missing: list[str] = []
    if not settings.has_gemini_key():
        gemini_missing.append("GEMINI_API_KEY (or GOOGLE_API_KEY)")
    if not _module_available("google.genai"):
        gemini_missing.append("dependency:google-genai")
    if not _module_available("PIL"):
        gemini_missing.append("dependency:pillow")
    providers["gemini"] = {
        "available": len(gemini_missing) == 0,
        "missing": gemini_missing,
        "install_hint": "pip install google-genai pillow",
    }

    return providers


def _print_provider_status(registry: ProviderRegistry, settings: Settings) -> None:
    status = _provider_status(settings)
    available = registry.list_providers()

    print("Providers:")
    for name in registry.list_all_providers():
        if name in available:
            print(f"- {name}: available")
            continue

        missing = status.get(name, {}).get("missing") or ["unknown"]
        install_hint = status.get(name, {}).get("install_hint")
        missing_str = ", ".join(missing)
        print(f"- {name}: unavailable ({missing_str})")
        if install_hint:
            print(f"  install: {install_hint}")


def _print_intro(registry: ProviderRegistry, settings: Settings) -> None:
    print("imagen-mcp | interactive")
    _print_provider_status(registry, settings)
    output_dir = (
        expand_path(settings.output_dir)
        if settings.output_dir
        else Path.home() / "Downloads" / "images"
    )
    print(f"Default output dir: {output_dir}")
    print("Override: `OUTPUT_DIR`, `-o/--output`, or `/o`")
    print("Tip: /? help  |  /providers  |  /status  |  /reset  |  /setup")
    print()


def get_available_provider(registry: ProviderRegistry, explicit: str | None = None) -> str | None:
    """Get an available provider, respecting explicit choice."""
    available = registry.list_providers()

    if explicit and explicit != "auto":
        if explicit in available:
            return explicit
        print(f"Warning: {explicit} not available, falling back...")

    # Prefer openai, then gemini
    for p in ["openai", "gemini"]:
        if p in available:
            return p
    return None


async def generate(
    prompt: str,
    *,
    provider: str | None = None,
    show_reasoning: bool = False,
    **kwargs: Any,
) -> tuple[bool, str | None]:
    """Generate an image and print result."""
    configure_logging()
    registry = get_provider_registry()

    # Get available provider
    available_provider = get_available_provider(registry, provider)
    if not available_provider:
        print("No providers available. Set OPENAI_API_KEY or GEMINI_API_KEY.")
        return False, None

    explicit_provider = None
    if provider and provider != "auto" and provider == available_provider:
        explicit_provider = provider

    try:
        prov, rec = registry.get_provider_for_prompt(
            prompt,
            explicit_provider=explicit_provider,
            **{k: v for k, v in kwargs.items() if v is not None},
        )
    except ValueError as e:
        print(f"Error: {e}")
        return False, None

    print(f"[{rec.provider}] Generating...")

    if show_reasoning and rec.reasoning:
        print(f"  reasoning: {rec.reasoning}")
        if rec.alternative and rec.alternative_reasoning:
            print(f"  alternative: {rec.alternative} ({rec.alternative_reasoning})")

    result = await prov.generate_image(prompt, **kwargs)

    if result.success:
        print(f"✓ {result.image_path}")
        if result.generation_time_seconds:
            print(f"  {result.generation_time_seconds:.1f}s")
        return True, str(result.image_path) if result.image_path else None
    else:
        print(f"✗ {result.error}")
        return False, None


async def interactive() -> None:
    """Run interactive REPL."""
    registry = get_provider_registry()
    settings = get_settings()

    # Check what's available
    available = registry.list_providers()
    if not available:
        _print_intro(registry, settings)
        print("No providers available. Set OPENAI_API_KEY or GEMINI_API_KEY.")
        return

    _print_intro(registry, settings)

    current_provider: str | None = None
    current_size: str | None = None
    conv_id: str | None = None
    output_path: str | None = None
    enable_enhancement: bool = settings.enable_prompt_enhancement
    enable_google_search: bool = settings.enable_google_search
    show_reasoning: bool = False
    last_image_path: str | None = None

    while True:
        try:
            prompt = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not prompt:
            continue

        # Commands
        if prompt.startswith("/"):
            cmd = prompt[1:].split(maxsplit=1)
            c = cmd[0].lower() if cmd else ""
            arg = cmd[1] if len(cmd) > 1 else ""

            if c in ("q", "quit", "exit"):
                break
            elif c in ("p", "provider"):
                if not arg:
                    print(f"Current: {current_provider or 'auto'}")
                else:
                    value = arg.lower()
                    if value == "auto":
                        current_provider = None
                        print("Provider: auto")
                    elif value in registry.list_all_providers():
                        current_provider = value
                        if value in registry.list_providers():
                            print(f"Provider: {value}")
                        else:
                            print(f"Provider: {value} (currently unavailable; will fall back)")
                    else:
                        print(f"Unknown provider: {arg}")
            elif c in ("s", "size"):
                current_size = arg if arg and arg.lower() not in ("default", "none") else None
                print(f"Size: {current_size or 'default'}")
            elif c in ("c", "conv"):
                conv_id = arg if arg and arg.lower() not in ("default", "none") else None
                print(f"Conversation: {conv_id or 'none'}")
            elif c in ("o", "output"):
                output_path = arg if arg and arg.lower() not in ("default", "none") else None
                print(f"Output: {output_path or 'default'}")
            elif c in ("e", "enhance"):
                if not arg:
                    print(f"Enhancement: {'on' if enable_enhancement else 'off'}")
                else:
                    enable_enhancement = arg.lower() in ("1", "true", "on", "yes", "y")
                    print(f"Enhancement: {'on' if enable_enhancement else 'off'}")
            elif c in ("g", "google", "search"):
                if not arg:
                    print(f"Google search: {'on' if enable_google_search else 'off'}")
                else:
                    enable_google_search = arg.lower() in ("1", "true", "on", "yes", "y")
                    print(f"Google search: {'on' if enable_google_search else 'off'}")
            elif c in ("why", "reasoning"):
                show_reasoning = not show_reasoning
                print(f"Show reasoning: {'on' if show_reasoning else 'off'}")
            elif c in ("providers",):
                _print_provider_status(registry, settings)
            elif c in ("status",):
                print(f"Provider: {current_provider or 'auto'}")
                print(f"Size: {current_size or 'default'}")
                print(f"Conversation: {conv_id or 'none'}")
                print(f"Output: {output_path or 'default'}")
                print(f"Enhancement: {'on' if enable_enhancement else 'off'}")
                print(f"Google search: {'on' if enable_google_search else 'off'}")
                print(f"Show reasoning: {'on' if show_reasoning else 'off'}")
                if last_image_path:
                    print(f"Last image: {last_image_path}")
            elif c in ("reset",):
                current_provider = None
                current_size = None
                conv_id = None
                output_path = None
                enable_enhancement = settings.enable_prompt_enhancement
                enable_google_search = settings.enable_google_search
                show_reasoning = False
                last_image_path = None
                print("Reset: provider/size/conv/output + toggles")
            elif c in ("setup",):
                dotenv_path = get_dotenv_path()
                load_dotenv(dotenv_path, override=False)
                _ensure_runtime_dependencies(interactive=True)
                _configure_api_keys(dotenv_path, interactive=True, force=True)
                registry = get_provider_registry()
                settings = get_settings()
            elif c in ("last",):
                print(last_image_path or "(no images yet)")
            elif c in ("?", "h", "help"):
                print("/q        - quit")
                print("/p [name] - set/show provider (auto, openai, gemini)")
                print("/s [size] - set/show size (or 'default')")
                print("/c [id]   - set/show conversation id")
                print("/o [path] - set/show output path/dir")
                print("/e [on|off]      - prompt enhancement")
                print("/search [on|off] - Gemini Google Search grounding")
                print("/why      - toggle selection reasoning")
                print("/providers - show provider availability")
                print("/status    - show current settings")
                print("/reset     - reset settings to defaults")
                print("/setup     - install deps + configure keys")
                print("/last      - print last image path")
            else:
                print(f"Unknown command: /{c}")
            continue

        # Generate image
        ok, image_path = await generate(
            prompt,
            provider=current_provider,
            size=current_size,
            conversation_id=conv_id,
            output_path=output_path,
            enable_enhancement=enable_enhancement,
            enable_google_search=enable_google_search,
            show_reasoning=show_reasoning,
        )
        if ok and image_path:
            last_image_path = image_path


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="imagen-mcp CLI")
    parser.add_argument("prompt", nargs="?", help="Image prompt (omit for interactive)")
    parser.add_argument("-p", "--provider", choices=["auto", "openai", "gemini"])
    parser.add_argument("-s", "--size", help="Image size")
    parser.add_argument("-o", "--output", help="Output path")
    parser.add_argument(
        "--search", action="store_true", help="Enable Gemini Google Search grounding"
    )
    parser.add_argument(
        "--no-enhance", action="store_true", help="Disable prompt enhancement (if supported)"
    )
    parser.add_argument("--why", action="store_true", help="Print provider selection reasoning")
    parser.add_argument("--setup", action="store_true", help="Install deps and configure keys")
    parser.add_argument("-i", "--interactive", action="store_true", help="Force interactive")
    args = parser.parse_args()

    interactive_session = _is_interactive_session()
    dotenv_path = bootstrap_cli(interactive=interactive_session)
    if args.setup:
        _ensure_runtime_dependencies(interactive=interactive_session)
        _configure_api_keys(dotenv_path, interactive=interactive_session, force=True)

    # Interactive mode if no prompt
    if args.interactive or not args.prompt:
        asyncio.run(interactive())
        return 0

    # One-shot generation
    asyncio.run(
        generate(
            args.prompt,
            provider=args.provider,
            size=args.size,
            output_path=args.output,
            enable_google_search=args.search,
            enable_enhancement=not args.no_enhance,
            show_reasoning=args.why,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
