# imagen-mcp Modernization Plan

> **Comprehensive plan for research, modernization, and upgrade of imagen-mcp**
>
> Generated: 2026-02-26
> Based on: Deep codebase audit (22 source files, 5,060 LOC), API research, MCP spec analysis

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current State Assessment](#2-current-state-assessment)
3. [Phase 0: Critical Fixes (Day 1)](#3-phase-0-critical-fixes-day-1)
4. [Phase 1: Security Hardening (Week 1)](#4-phase-1-security-hardening-week-1)
5. [Phase 2: Performance Fixes (Week 1)](#5-phase-2-performance-fixes-week-1)
6. [Phase 3: Provider Modernization (Week 2)](#6-phase-3-provider-modernization-week-2)
7. [Phase 4: Error Handling & Robustness (Week 2)](#7-phase-4-error-handling--robustness-week-2)
8. [Phase 5: Architecture Modernization (Week 3)](#8-phase-5-architecture-modernization-week-3)
9. [Phase 6: Testing & CI/CD (Week 3-4)](#9-phase-6-testing--cicd-week-3-4)
10. [Phase 7: Dependency & Packaging (Week 4)](#10-phase-7-dependency--packaging-week-4)
11. [Phase 8: FastMCP 3.0 Migration (Week 5)](#11-phase-8-fastmcp-30-migration-week-5)
12. [Phase 9: MCP Protocol Advancement (Week 6+)](#12-phase-9-mcp-protocol-advancement-week-6)
13. [Phase 10: New Features & Providers (Week 7+)](#13-phase-10-new-features--providers-week-7)
14. [Dead Code Removal Inventory](#14-dead-code-removal-inventory)
15. [API Research Reference](#15-api-research-reference)
16. [Timeline & Effort Summary](#16-timeline--effort-summary)

---

## 1. Executive Summary

imagen-mcp has a clean architecture (4-layer design, no circular deps, good type hints) but
suffers from **2 critical security vulnerabilities**, **3 critical performance bottlenecks**,
**outdated/dead provider models**, **38 dead code items**, and **zero test coverage on all 5
MCP tool handlers**. The provider APIs have moved significantly since the codebase was written.

### Current Scorecard

| Area | Grade | Key Finding |
|------|:-----:|-------------|
| Security | **D** | API key in URL query param, path traversal, `str(e)` leak |
| Performance | **D+** | Sync I/O in async, PIL round-trip, per-request HTTP clients |
| Test Coverage | **D+** | ~49% symbol coverage; all 5 MCP handlers untested |
| Dead Code | **C** | 38 dead items including 3 bugs masquerading as dead code |
| Code Quality | **C** | God object in base.py, 7 functions >50 lines, 28+ magic numbers |
| Consistency | **C-** | Version 0.1.0 vs 4.0.0, stale README, duplicate logic |
| Configuration | **C+** | 7 undocumented env vars, 5 settings loaded but never used |
| Dependencies | **B+** | Clean -- no missing or phantom deps |
| Architecture | **B-** | Clear 4-layer structure but base.py violates SRP |
| Error Handling | **C-** | No custom exceptions, raw errors returned to users |

### Target Scorecard (Post-Modernization)

| Area | Target | How |
|------|:------:|-----|
| Security | **A** | Fix path traversal, header-based auth, sanitized errors |
| Performance | **A-** | Connection pooling, async I/O, eliminated round-trips |
| Test Coverage | **A-** | >80% coverage, all MCP handlers tested, CI enforcement |
| Dead Code | **A** | All 38 items resolved (removed or wired up) |
| Code Quality | **A-** | Extracted services, data-driven patterns, named constants |
| Consistency | **A** | Single version source, accurate README, unified patterns |
| Dependencies | **A** | Modern models, pinned versions, consolidated config |
| Architecture | **A-** | Thin server, service layer, pluggable providers |
| Error Handling | **A** | Custom hierarchy, MCP error codes, sanitized messages |

---

## 2. Current State Assessment

### 2.1 File Inventory (22 source files)

| File | Lines | Purpose |
|------|------:|---------|
| `src/__init__.py` | 9 | Package init, `__version__ = "4.0.0"` |
| `src/cli.py` | 445 | Interactive CLI + one-shot generation mode |
| `src/server.py` | 610 | MCP server -- 5 tools, event logging, response formatting |
| `src/config/__init__.py` | 26 | Re-exports constants + settings |
| `src/config/constants.py` | 148 | Provider constants, sizes, keywords, limits |
| `src/config/dotenv.py` | 149 | Zero-dep `.env` file loader/writer |
| `src/config/paths.py` | 93 | Output directory resolution, path expansion |
| `src/config/settings.py` | 119 | `Settings` dataclass loaded from env vars |
| `src/models/__init__.py` | 17 | Re-exports Pydantic models |
| `src/models/input_models.py` | 315 | Pydantic schemas for all 5 MCP tools |
| `src/providers/__init__.py` | 15 | Re-exports provider classes |
| `src/providers/base.py` | 418 | ABC + `ImageResult` + shared infrastructure |
| `src/providers/gemini_provider.py` | 484 | Google Gemini implementation |
| `src/providers/openai_provider.py` | 460 | OpenAI GPT-Image-1 implementation |
| `src/providers/registry.py` | 228 | Provider factory + cache |
| `src/providers/selector.py` | 446 | Auto-selection scoring engine |
| `src/services/__init__.py` | 1 | Empty package marker |
| `src/services/conversation_store.py` | 408 | SQLite conversation persistence |
| `src/services/dialogue.py` | 370 | Pre-generation dialogue/question system |
| `src/services/logging_config.py` | 119 | Dual-output logging (rotating file + JSONL) |
| `src/services/rate_limiter.py` | 179 | Client-side rate limiting with burst detection |
| `src/tools/__init__.py` | 1 | Empty package marker (unused) |

### 2.2 Test Inventory

| File | Lines | Focus |
|------|------:|-------|
| `tests/test_integration.py` | 419 | E2E integration tests (mocked APIs) |
| `tests/test_paths.py` | 79 | Output path resolution tests |
| `tests/test_providers.py` | 157 | Provider capability unit tests |
| `tests/test_selector.py` | 237 | Auto-selection algorithm tests |
| `tests/test_server.py` | 108 | Import smoke tests + model validation |

### 2.3 Test Coverage Matrix

| Module | Grade | Tested | Untested | Notes |
|--------|:-----:|-------:|---------:|-------|
| `selector.py` | **A** | 11/11 | 0 | 19 focused tests |
| `conversation_store.py` | **B+** | 9/11 | 2 | Missing: cleanup, singleton |
| `dialogue.py` | **B** | 6/9 | 3 | Missing: `create_dialogue_response`, suggestions |
| `registry.py` | **B** | 9/12 | 3 | Missing: `close_all`, `list_conversations` |
| `paths.py` | **B** | 4/5 | 1 | Missing: `get_log_directory` |
| `settings.py` | **C+** | 5/7 | 2 | Missing: key retrieval error paths |
| `openai_provider.py` | **C** | 6/13 | 7 | 1 mocked test; no retry/error tests |
| `gemini_provider.py` | **C** | 5/13 | 8 | `generate_image` itself has 0 tests |
| `input_models.py` | **C** | 3/5 | 2 | Only happy-path validation |
| `server.py` | **D** | 2/10 | 8 | All 5 MCP handlers untested |
| `cli.py` | **F** | 0/14 | 14 | 445 lines, zero tests |
| `dotenv.py` | **F** | 0/6 | 6 | .env parser untested |
| `rate_limiter.py` | **F** | 0/6 | 6 | Entire module untested |
| `logging_config.py` | **F** | 0/4 | 4 | Entire module untested |

---

## 3. Phase 0: Critical Fixes (Day 1)

> **These are broken things that affect users RIGHT NOW.**
> Estimated effort: ~3 hours

### 3.1 Remove Dead Gemini Models

Two models referenced in the codebase are **shut down**:

| Dead Model | Shutdown Date | Status |
|------------|:------------:|--------|
| `gemini-2.0-flash-exp-image-generation` | Nov 14, 2025 | API returns 404 |
| `imagen-3.0-generate-002` | Nov 10, 2025 | API returns 404 |

**Action:** Remove from `src/config/constants.py`, `src/providers/gemini_provider.py`, and all
references. Replace with current models (see [Phase 3](#6-phase-3-provider-modernization-week-2)
for the complete model matrix).

**Files:** `src/config/constants.py`, `src/providers/gemini_provider.py`, `src/server.py`

### 3.2 Fix Version Inconsistency

| Location | Version |
|----------|---------|
| `pyproject.toml:7` | `"0.1.0"` |
| `src/__init__.py:9` | `"4.0.0"` |

These are **40x apart**. Decide on the canonical version and single-source it.

**Recommended:** Use `pyproject.toml` as source of truth. Have `__init__.py` read from
`importlib.metadata.version("imagen-mcp")` or set both to the same value.

**Files:** `pyproject.toml`, `src/__init__.py`

### 3.3 Pin FastMCP to Avoid Breakage

Currently `>=2.12.5` with no upper bound. FastMCP 3.0 is a **major breaking release** (moved
from `jlowin/fastmcp` to `PrefectHQ/fastmcp`, architectural rebuild). Without a pin, any
`uv lock --upgrade` will pull 3.0 and break the import.

**Action:** Change to `fastmcp>=2.12.5,<3.0` immediately. Migrate to 3.0 in
[Phase 8](#11-phase-8-fastmcp-30-migration-week-5).

**Files:** `pyproject.toml`

### 3.4 Wire Orphaned Model Fields or Remove Them

Three `ConversationalImageInput` fields are advertised in the MCP tool schema but **silently
ignored** -- a broken user contract:

| Field | Issue | Action |
|-------|-------|--------|
| `input_image_file_id` | OpenAI provider supports it, but `server.py` never passes it through | Wire through or remove |
| `input_image_path` | Completely phantom -- no downstream support whatsoever | Remove |
| `assistant_model` | Provider accepts it, server never bridges the gap | Wire through or remove |
| `dialogue_responses` | Vestigial -- zero references anywhere | Remove |

**Files:** `src/models/input_models.py`, `src/server.py`

---

## 4. Phase 1: Security Hardening (Week 1)

> **2 critical vulnerabilities, 6 high-severity issues.**
> Estimated effort: ~4 hours

### 4.1 CRITICAL: Gemini API Key in URL Query Parameter

**Location:** `server.py:546-551`

```python
# CURRENT (vulnerable)
response = await client.get(f"{api_base}/models?key={api_key}")
```

If `raise_for_status()` fires, the full URL (including key) lands in `str(e)` and is returned
to the user. Also visible to any HTTP proxy in the path and logged in tracebacks.

**Fix:** Use the `x-goog-api-key` HTTP header instead:
```python
# FIXED
response = await client.get(
    f"{api_base}/models",
    headers={"x-goog-api-key": api_key}
)
```

**Effort:** 5 minutes

### 4.2 CRITICAL: Arbitrary File Write via Path Traversal

**Location:** `config/paths.py:55-93`

Zero validation on `output_path`. A malicious prompt can write to arbitrary filesystem
locations. The code helpfully creates entire directory trees with `mkdir(parents=True)`.

**Fix:**
```python
resolved = Path(output_path).resolve()
allowed_base = Path(settings.output_dir).resolve()
if not str(resolved).startswith(str(allowed_base)):
    raise ValueError(f"Output path must be within {allowed_base}")
```

**Effort:** 30 minutes

### 4.3 HIGH: Sanitize Error Messages Returned to Users

**Location:** `server.py:322,457,523,588` -- all 4 error handlers return raw `str(e)`.

Combined with API keys in exception messages (local variables in tracebacks), this is a
direct credential leak path.

**Fix:** Create sanitized error messages:
```python
def _sanitize_error(e: Exception) -> str:
    """Return user-safe error message without credentials or internal details."""
    msg = str(e)
    # Strip potential API keys (32+ char alphanumeric strings)
    msg = re.sub(r'[A-Za-z0-9_-]{32,}', '[REDACTED]', msg)
    # Strip URLs that might contain keys
    msg = re.sub(r'key=[^&\s]+', 'key=[REDACTED]', msg)
    return msg
```

**Effort:** 30 minutes

### 4.4 HIGH: Add `repr=False` to API Key Pydantic Fields

**Location:** `input_models.py:137-145`

API key fields have no `exclude=True` or `repr=False` -- they'll appear in any Pydantic
model serialization or repr output.

**Fix:** `openai_api_key: str | None = Field(default=None, repr=False, exclude=True)`

**Effort:** 5 minutes

### 4.5 HIGH: Remove `os.path.expandvars()` from User-Supplied Paths

**Location:** `config/paths.py:18-21`

`expandvars("$HOME")` on user input = environment variable disclosure.

**Fix:** Only expand `~`, not arbitrary env vars. Use `Path.expanduser()` only.

**Effort:** 5 minutes

### 4.6 HIGH: Prevent Decompression Bombs in PIL

**Location:** `gemini_provider.py:253`

PIL processes user-supplied base64 with no format or dimension validation.

**Fix:**
```python
from PIL import Image
Image.MAX_IMAGE_PIXELS = 89_478_485  # Default, but explicit
# Validate dimensions after open, before any processing
```

**Effort:** 15 minutes

### 4.7 MEDIUM: Lock Down ConversationalImageInput

**Location:** `input_models.py:155`

`model_config = ConfigDict(extra="allow")` accepts arbitrary fields -- should be `"ignore"`
or `"forbid"`.

**Effort:** 2 minutes

### 4.8 MEDIUM: Set SQLite DB File Permissions

**Location:** `conversation_store.py:41`

Database stores prompts and images with default (world-readable) permissions.

**Fix:** `os.chmod(db_path, 0o600)` after creation.

**Effort:** 5 minutes

---

## 5. Phase 2: Performance Fixes (Week 1)

> **3 critical bottlenecks, 5 high-severity issues.**
> Estimated effort: ~3 hours

### 5.1 CRITICAL: Use Persistent HTTP Client for OpenAI

**Location:** `openai_provider.py:~120`

Currently creates a **new `httpx.AsyncClient`** per API call -- TLS handshake overhead of
~100-300ms per request.

**Fix:**
```python
class OpenAIProvider(ImageProvider):
    def __init__(self, settings):
        super().__init__(settings)
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(120.0),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
        )

    async def close(self):
        await self._client.aclose()
```

Wire `ProviderRegistry.close_all()` into server lifecycle (method exists but is never called).

**Effort:** 20 minutes

### 5.2 CRITICAL: Remove Unnecessary PIL Decode-Re-encode

**Location:** `gemini_provider.py:145-155`

Gemini returns valid PNG bytes. The code decodes to PIL, then re-encodes as PNG -- adding
~200-500ms per 4K image and doubling peak memory.

**Fix:** `base64.b64encode(image_bytes)` directly when no processing is needed.

**Effort:** 5 minutes

### 5.3 CRITICAL: Wrap File I/O in `asyncio.to_thread()`

**Location:** `base.py:271-302`

`base64.b64decode()` + `open().write()` runs synchronously inside `async` methods, blocking
the event loop for 15-55ms per save.

**Fix:**
```python
async def _save_image(self, ...):
    await asyncio.to_thread(self._save_image_sync, ...)
```

**Effort:** 15 minutes

### 5.4 HIGH: Add Timeout to Gemini API Calls

**Location:** `gemini_provider.py:~180`

No timeout -- a hung Gemini API call blocks the server indefinitely.

**Fix:** `asyncio.wait_for(self._generate(...), timeout=self.settings.request_timeout)`

**Effort:** 5 minutes

### 5.5 HIGH: Close PIL Images from Reference Processing

**Location:** `gemini_provider.py:160-175`

PIL images opened from base64 reference images are never closed -- slow memory/FD leak in
long-running sessions.

**Fix:** Use `with` context manager or explicit `.close()`.

**Effort:** 5 minutes

### 5.6 HIGH: Release `image_base64` After Disk Save

**Location:** Both providers

Multi-MB `image_base64` string retained on `ImageResult` after the image is saved to disk.

**Fix:** `result.image_base64 = None` after successful save, or make it an optional field
that's only populated when explicitly requested.

**Effort:** 10 minutes

### 5.7 HIGH: Persistent SQLite Connection

**Location:** `conversation_store.py`

Connection opened/closed on every method call (3-4x per generation). Also needs
`PRAGMA busy_timeout = 5000` for concurrent access.

**Fix:** Hold a persistent connection on the store instance. Ideally migrate to `aiosqlite`
for async compatibility.

**Effort:** 30 minutes

### 5.8 MEDIUM: Cache Compiled Regex Patterns

**Location:** `selector.py`, `dialogue.py`

~10 regex patterns recompiled on every prompt analysis.

**Fix:** Module-level `re.compile()` constants.

**Effort:** 15 minutes

### 5.9 MEDIUM: Remove Redundant `configure_logging()` Calls

**Location:** `server.py`

Called in all 5 tool handlers. Should be called once at server startup.

**Effort:** 5 minutes

---

## 6. Phase 3: Provider Modernization (Week 2)

> **Update to current API models and capabilities.**
> Estimated effort: ~8 hours

### 6.1 OpenAI Model Updates

#### Current State

The codebase hardcodes `"gpt-image-1"` in 4 places (ignoring the defined-but-unused
`DEFAULT_OPENAI_IMAGE_MODEL` constant).

#### Target Model Matrix

| Model | Released | Speed | Cost | Best For | Status |
|-------|:--------:|:-----:|:----:|----------|:------:|
| `gpt-image-1.5` | Dec 2025 | 4x faster than 1.0 | ~20% cheaper | **New default** -- text, editing, general | GA |
| `gpt-image-1` | Apr 2025 | Baseline | Baseline | Fallback, existing workflows | GA |

#### GPT-Image-1.5 New Capabilities to Support

| Feature | API Parameter | Current Support | Action |
|---------|:------------:|:---------------:|--------|
| Enhanced editing | Same endpoint, better results | Partial | Update docs |
| New sizes | `1536x1536` (square HD) | No | Add to size maps |
| Quality tiers | `quality: "low" / "medium" / "high" / "auto"` | No | Update quality handling |
| Output format | `output_format: "png" / "jpeg" / "webp"` | PNG only | Add format parameter |
| Compression | `output_compression: 0-100` (JPEG/WebP) | No | Add optional parameter |
| Background control | `background: "auto" / "transparent" / "opaque"` | No | Add parameter |

#### Implementation

```python
# src/config/constants.py
OPENAI_MODELS = {
    "gpt-image-1.5": {"speed": "fast", "cost": "low", "quality": "high"},
    "gpt-image-1": {"speed": "baseline", "cost": "baseline", "quality": "high"},
}
DEFAULT_OPENAI_IMAGE_MODEL = "gpt-image-1.5"

OPENAI_SUPPORTED_SIZES = {
    "1024x1024",   # Square
    "1536x1024",   # Landscape
    "1024x1536",   # Portrait
    "1536x1536",   # Square HD (new in 1.5)
    "auto",        # Let model decide (new in 1.5)
}

OPENAI_QUALITY_OPTIONS = ["low", "medium", "high", "auto"]
OPENAI_FORMAT_OPTIONS = ["png", "jpeg", "webp"]
```

**Files:** `src/config/constants.py`, `src/providers/openai_provider.py`, `src/models/input_models.py`

#### Pricing Reference (for selector scoring)

| Model | Quality | 1024x1024 | 1024x1536 | 1536x1536 |
|-------|---------|----------:|----------:|----------:|
| gpt-image-1.5 | low | $0.011 | $0.016 | $0.024 |
| gpt-image-1.5 | medium | $0.022 | $0.033 | $0.049 |
| gpt-image-1.5 | high | $0.044 | $0.066 | $0.099 |
| gpt-image-1 | low | $0.011 | $0.016 | N/A |
| gpt-image-1 | medium | $0.042 | $0.063 | N/A |
| gpt-image-1 | high | $0.167 | $0.250 | N/A |

### 6.2 Gemini Model Updates

#### Current State

The codebase references `gemini-3-pro-image-preview` (current), but also two **dead models**
and the overall model landscape has evolved significantly.

#### Target Model Matrix (Native Image Generation)

| Model ID | Codename | Speed | Quality | Max Res | Status |
|----------|----------|:-----:|:-------:|:-------:|:------:|
| `gemini-2.5-flash-preview-image-generation` | Nano Banana | Fast | Good | 1536x1536 | **GA** (Oct 2025) |
| `gemini-3-pro-image-preview` | Nano Banana Pro | Slower | Best | 1536x1536 | Preview |

#### Target Model Matrix (Imagen Family)

| Model ID | Speed | Quality | Max Res | Status |
|----------|:-----:|:-------:|:-------:|:------:|
| `imagen-4.0-generate-001` | Fast | Excellent | 2048x2048 | **GA** (Jan 2026) |
| `imagen-4.0-ultra-generate-001` | Slow | Best-in-class | 2048x2048 | GA |
| `imagen-4.0-fast-generate-001` | Fastest | Good | 2048x2048 | GA |

#### Models to REMOVE (Dead)

| Model | Shutdown Date |
|-------|:------------:|
| `gemini-2.0-flash-exp-image-generation` | Nov 14, 2025 |
| `imagen-3.0-generate-002` | Nov 10, 2025 |
| `imagen-3.0-fast-generate-001` | Nov 10, 2025 |

#### Gemini 2.5 Flash Image -- Key Facts

- **GA since October 2025** -- production-ready, stable
- Uses `google-genai` SDK `generateContent` with `response_modalities=["TEXT", "IMAGE"]`
- Supports **10 aspect ratios**: 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9
- **Native image generation** -- interleaves text + image in a single response
- Supports multi-turn conversational image refinement
- Supports up to 10 input images for editing/style reference
- Rate limit: 10 RPM (free), 30 RPM (paid)

#### Imagen 4 Family -- Key Facts

- **GA since January 2026** via Gemini API (`google-genai` SDK)
- Uses `client.models.generate_images()` method (different from Gemini native)
- Supports `number_of_images: 1-4` per request
- Supports **person generation** (configurable via `person_generation` param)
- Resolution: up to 2048x2048
- **No conversational refinement** -- single-shot only
- Three tiers: Fast (speed), Standard (balanced), Ultra (quality)

#### Implementation Plan

1. Update `GEMINI_MODELS` constant with full model matrix
2. Set `gemini-2.5-flash-preview-image-generation` as default (GA, fast, reliable)
3. Keep `gemini-3-pro-image-preview` as quality option
4. Add Imagen 4 family as new provider path (different API shape)
5. Update selector scoring for new model capabilities
6. Remove all references to dead models

**Files:** `src/config/constants.py`, `src/providers/gemini_provider.py`, `src/providers/selector.py`, `src/server.py`

### 6.3 Update Provider Auto-Selection Scoring

The selector needs updating for the new model landscape:

| Signal | Current Behavior | New Behavior |
|--------|-----------------|--------------|
| Text-heavy prompts | OpenAI (gpt-image-1) | OpenAI (gpt-image-1.5, even better at text) |
| Photorealistic | Gemini (3-pro) | Gemini (2.5-flash for speed, Imagen 4 Ultra for quality) |
| Product photography | Gemini | Imagen 4 (best photorealism) |
| Fast generation | No differentiation | gpt-image-1.5 (4x faster) or Imagen 4 Fast |
| High resolution | Gemini | Imagen 4 Ultra (2048x2048) |
| Conversational refinement | Either | Gemini native only (Imagen 4 doesn't support it) |

---

## 7. Phase 4: Error Handling & Robustness (Week 2)

> **Custom exception hierarchy, retry logic, structured error responses.**
> Estimated effort: ~5 hours

### 7.1 Custom Exception Hierarchy

Create `src/exceptions.py`:

```python
class ImagenError(Exception):
    """Base exception for imagen-mcp."""
    def __init__(self, message: str, *, user_message: str | None = None):
        super().__init__(message)
        self.user_message = user_message or "An unexpected error occurred"

class ConfigurationError(ImagenError):
    """Missing API key, invalid settings, etc."""
    pass

class ProviderError(ImagenError):
    """API call failures."""
    def __init__(self, message: str, *, provider: str, status_code: int | None = None, **kw):
        super().__init__(message, **kw)
        self.provider = provider
        self.status_code = status_code

class AuthenticationError(ProviderError):
    """Invalid or expired API key (401/403)."""
    pass

class RateLimitError(ProviderError):
    """Rate limit exceeded (429). Includes retry_after if available."""
    def __init__(self, message: str, *, retry_after: float | None = None, **kw):
        super().__init__(message, **kw)
        self.retry_after = retry_after

class GenerationError(ProviderError):
    """Model-level failure (content filter, invalid prompt, etc.)."""
    pass

class ValidationError(ImagenError):
    """Invalid prompt, bad parameters, unsupported size, etc."""
    pass
```

### 7.2 Map Exceptions to MCP Error Codes

In `server.py`, catch specific exceptions and return appropriate MCP errors:

| Exception | MCP Error Code | HTTP Analogy |
|-----------|:-------------:|:------------:|
| `ValidationError` | `INVALID_PARAMS` | 400 |
| `AuthenticationError` | `INVALID_PARAMS` (+ message) | 401 |
| `RateLimitError` | `INTERNAL_ERROR` (+ retry_after) | 429 |
| `GenerationError` | `INTERNAL_ERROR` | 500 |
| `ConfigurationError` | `INVALID_PARAMS` | 500 |

### 7.3 Wire Retry Logic into Both Providers

The base class has `_retry_with_backoff()` that **neither provider calls**. Either:

**Option A:** Wire the existing method:
```python
# In generate_image():
result = await self._retry_with_backoff(
    self._call_api,
    max_retries=3,
    retry_on=(RateLimitError, httpx.TransportError)
)
```

**Option B:** Replace with a decorator (cleaner):
```python
@retry(max_attempts=3, backoff_base=2.0, retry_on=(RateLimitError, TransportError))
async def _call_api(self, prompt: str, **kwargs) -> ImageResult:
    ...
```

OpenAI provider has manual 3-retry logic; Gemini has **zero retries**. Unify both.

### 7.4 Add Structured Error Responses

Currently errors return plain strings. Add structured error format:

```python
def _format_error(error: ImagenError, output_format: str) -> str:
    if output_format == "json":
        return json.dumps({
            "error": True,
            "error_type": type(error).__name__,
            "message": error.user_message,
            "provider": getattr(error, "provider", None),
            "retry_after": getattr(error, "retry_after", None),
        })
    return f"Error: {error.user_message}"
```

---

## 8. Phase 5: Architecture Modernization (Week 3)

> **Extract service layer, decompose god objects, make providers pluggable.**
> Estimated effort: ~15 hours

### 8.1 Extract Service Layer from `server.py`

`server.py` (610 lines) mixes tool registration, provider selection, conversation management,
error handling, and response formatting. Each handler repeats the same pattern.

#### Target Architecture

```
server.py (~150 lines)
    @mcp.tool() thin adapters: validate -> delegate -> format
        |
        v
GenerationService (~200 lines)
    generate() / converse() / list_providers() / list_conversations() / list_models()
    Error middleware, logging, metrics
        |
        v
ProviderRegistry -> OpenAI / Gemini / Future providers
ConversationStore -> SQLite persistence
DialogueService -> Pre-generation refinement
```

#### Benefits

- Each tool handler becomes ~15 lines (validate input, call service, format output)
- Error handling in one place (service layer middleware)
- Testable without MCP server scaffolding
- Provider-agnostic business logic

### 8.2 Decompose `ImageProvider` God Object

`ImageProvider` (base.py, 418 lines, 16 methods) mixes **6 concerns**:

| Concern | Methods | Extract To |
|---------|---------|-----------|
| Abstract interface | `generate_image`, `capabilities` | Keep in `ImageProvider` |
| Image I/O | `_save_image`, `_generate_filename` | `ImageSaver` utility |
| Retry logic | `_retry_with_backoff` | `retry_utils.py` decorator |
| Rate limiting | `_acquire_rate_limit` | Already in `rate_limiter.py` (wire it) |
| Conversation storage | `_store_conversation`, `_get_conversation` | Already in `conversation_store.py` (wire it) |
| Feature queries | `supports_feature`, `get_best_size_for_type` | Provider capabilities (data-driven) |

#### After Decomposition

```python
class ImageProvider(ABC):
    """Pure abstract interface -- nothing else."""

    @abstractmethod
    async def generate_image(self, prompt: str, **kwargs) -> ImageResult: ...

    @property
    @abstractmethod
    def capabilities(self) -> ProviderCapabilities: ...

    @property
    @abstractmethod
    def name(self) -> str: ...
```

### 8.3 Make Provider Registration Pluggable

Currently adding a 3rd provider requires touching **5+ files** and modifying `if/elif` chains.

#### Target: Self-Describing Providers

```python
class StabilityProvider(ImageProvider):
    name = "stability"
    env_key = "STABILITY_API_KEY"
    selector_hints = {
        "strengths": {"upscale", "inpaint", "realistic"},
        "speed_tier": "fast",
        "max_resolution": (2048, 2048),
    }

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supported_sizes=[...],
            supports_text=False,
            supports_editing=True,
            ...
        )
```

#### Registry Discovers Providers

```python
PROVIDER_CLASSES: list[type[ImageProvider]] = [
    OpenAIProvider,
    GeminiProvider,
    # StabilityProvider,  # Just add here
]

class ProviderRegistry:
    def __init__(self, settings: Settings):
        for cls in PROVIDER_CLASSES:
            key = getattr(settings, cls.env_key.lower(), None)
            if key:
                self._providers[cls.name] = cls(settings)
```

### 8.4 Separate Provider-Specific Fields from Shared Models

6 of 12 optional fields in `ImageGenerationInput` are Gemini-specific. This pollutes the
MCP tool schema that every client sees.

#### Target

```python
class OpenAIOptions(BaseModel):
    openai_model: Literal["gpt-image-1.5", "gpt-image-1"] | None = None
    quality: Literal["low", "medium", "high", "auto"] | None = None
    output_format: Literal["png", "jpeg", "webp"] | None = None

class GeminiOptions(BaseModel):
    gemini_model: str | None = None
    aspect_ratio: str | None = None
    reference_images: list[str] | None = None
    enable_google_search: bool = False

class ImageGenerationInput(BaseModel):
    prompt: str
    provider: Provider = "auto"
    size: str | None = None
    enhance_prompt: bool = True
    openai_options: OpenAIOptions | None = None
    gemini_options: GeminiOptions | None = None
```

### 8.5 Unify Duplicate Logic

| Duplication | Locations | Fix |
|-------------|-----------|-----|
| `_module_available()` | `cli.py:29`, `registry.py:23` | `src/utils.py` |
| API key dispatch `if/else` | `server.py:288-291`, `server.py:432-435` | Helper method |
| Logging event construction | `server.py:233-248`, `server.py:365-382` | Shared builder |
| `generate_image` skeleton | Both providers | Template Method pattern |
| Prompt length validation | `openai_provider.py:309`, `gemini_provider.py:144` | Base class |
| Size cross-conversion maps | Both providers | Centralize in constants |
| `_detect_image_type` | `selector.py:362`, `dialogue.py:210` | Single implementation |
| Conversation ID generation | `openai_provider.py:385`, `gemini_provider.py:224` | Base class method (already exists) |

---

## 9. Phase 6: Testing & CI/CD (Week 3-4)

> **From ~49% to >80% coverage with automated enforcement.**
> Estimated effort: ~12 hours

### 9.1 Create `conftest.py` with Shared Fixtures

Every test file independently does `os.environ.setdefault("OPENAI_API_KEY", "test-key")`.

```python
# tests/conftest.py
import pytest
from unittest.mock import AsyncMock

@pytest.fixture(autouse=True)
def fake_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("OUTPUT_DIR", "/tmp/imagen-test")

@pytest.fixture
def mock_openai_response():
    return AsyncMock(return_value=ImageResult(...))

@pytest.fixture
def mock_gemini_response():
    return AsyncMock(return_value=ImageResult(...))

@pytest.fixture
def conversation_store(tmp_path):
    return ConversationStore(db_path=tmp_path / "test.db")

@pytest.fixture
def settings(fake_env):
    get_settings.cache_clear()
    return get_settings()
```

### 9.2 Fill Test Coverage Gaps

| Module | Current | Target | Tests Needed |
|--------|:-------:|:------:|:-------------|
| `server.py` handlers | **0/5** | **5/5** | Full MCP tool tests for all 5 handlers |
| `gemini_provider.generate_image()` | **0** | **8+** | Success, error, timeout, retry, reference images |
| `cli.py` | **0/14** | **8/14** | Core REPL commands, one-shot mode |
| `dotenv.py` | **0/6** | **6/6** | Parse, write, edge cases |
| `rate_limiter.py` | **0/6** | **6/6** | Limits, burst detection, reset |
| `logging_config.py` | **0/4** | **3/4** | Setup, rotation, JSONL format |
| `conversation_store.py` | **9/11** | **11/11** | Cleanup, singleton |
| Error/retry paths | **~12/42** | **30+/42** | Fault injection for all error paths |

### 9.3 Add CI/CD Pipeline

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv sync --all-extras
      - run: uv run ruff format --check src/ tests/
      - run: uv run ruff check src/ tests/
      - run: uv run pyright src/
      - run: uv run pytest --cov=src --cov-report=xml --cov-fail-under=80
      - uses: codecov/codecov-action@v4
        with:
          file: coverage.xml
```

### 9.4 Add Pre-Commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.9.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: local
    hooks:
      - id: pyright
        name: pyright
        entry: uv run pyright
        language: system
        types: [python]
        pass_filenames: false
```

### 9.5 Add Coverage Configuration

```toml
# In pyproject.toml
[tool.coverage.run]
source = ["src"]
branch = true
omit = ["src/cli.py"]  # CLI tested separately

[tool.coverage.report]
fail_under = 80
show_missing = true
exclude_lines = [
    "pragma: no cover",
    "if TYPE_CHECKING:",
    "if __name__ ==",
]
```

---

## 10. Phase 7: Dependency & Packaging (Week 4)

> **Consolidate deps, update versions, modernize build.**
> Estimated effort: ~3 hours

### 10.1 Consolidate Dependencies into `pyproject.toml`

| Issue | Fix |
|-------|-----|
| Dev deps in `requirements.txt` + `dev-requirements.txt` | Move to `[project.optional-dependencies] dev = [...]` |
| `mcp` as direct dep (transitive via `fastmcp`) | Remove from direct deps |
| `types-requests` declared but `requests` not used | Remove |
| Conflicting pytest version floors | Single source in pyproject.toml |

```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "pytest-cov>=5.0",
    "pyright>=1.1.390",
    "ruff>=0.9.0",
    "pre-commit>=3.0",
]
```

### 10.2 Update Dependency Pins

| Package | Current Lock | Latest | Action |
|---------|:-----------:|:------:|--------|
| `fastmcp` | 2.13.3 | 3.0.2 | Pin `<3.0` now; migrate in Phase 8 |
| `google-genai` | 1.54.0 | 1.65.0+ | Upgrade (11+ minors of fixes) |
| `pillow` | 12.0.0 | 12.1.1 | Upgrade |
| `pydantic` | 2.12.3 | 2.14+ | Upgrade |
| `httpx` | 0.24.0 | 0.28+ | Upgrade (connection pool improvements) |
| `mcp` | 1.22.0 | 1.26.0 | Transitive -- remove from direct |

### 10.3 Add `py.typed` Marker

PEP 561 compliance -- allows downstream consumers to benefit from type annotations:

```bash
touch src/py.typed
```

Add to `pyproject.toml`:
```toml
[tool.setuptools.package-data]
src = ["py.typed"]
```

### 10.4 Consider Build Backend Upgrade

Current: `setuptools>=61.0`. Modern alternative: `hatchling>=1.26` with proper `[project]`
table (PEP 621). Lower priority -- setuptools works fine.

---

## 11. Phase 8: FastMCP 3.0 Migration (Week 5)

> **Upgrade from FastMCP 2.x to 3.0 for future-proofing.**
> Estimated effort: ~6 hours

### 11.1 What Changed in FastMCP 3.0

FastMCP 3.0 (GA October 2025, now at 3.0.2) is a major architectural rebuild by Prefect:

| Feature | FastMCP 2.x | FastMCP 3.0 |
|---------|:-----------:|:-----------:|
| Package | `jlowin/fastmcp` | `PrefectHQ/fastmcp` |
| Import | `from fastmcp import FastMCP` | Same (unchanged) |
| `@mcp.tool()` API | Yes | Yes (unchanged) |
| Transport | stdio only | stdio + **Streamable HTTP** |
| Background tasks | No | `@mcp.tool(task=True)` |
| Server composition | No | `mcp.import_server()` |
| OAuth 2.1 | No | Built-in |
| Elicitation | No | `ctx.elicit()` |
| Client SDK | Basic | Full `Client()` class |
| CLI | `fastmcp run` | Enhanced `fastmcp run/dev/install` |

### 11.2 Migration Steps

**Step 1: Update dependency**
```toml
# pyproject.toml
"fastmcp>=3.0.0,<4.0"
```

**Step 2: Check imports** -- The core import is unchanged:
```python
from fastmcp import FastMCP  # Works in both 2.x and 3.0
```

**Step 3: Update tool signatures** -- The `@mcp.tool()` decorator is backward-compatible.
No changes needed for basic tools.

**Step 4: Update `Context` usage** -- If using `Context` for logging:
```python
# 2.x
from fastmcp import Context

# 3.0 -- same, but Context gains new methods
from fastmcp import Context  # Now has ctx.elicit(), ctx.sample(), etc.
```

**Step 5: Test all 5 tools** -- Verify each tool works with the new version.

### 11.3 New Features to Leverage

After migration, enable:

| Feature | Benefit for imagen-mcp | Effort |
|---------|----------------------|:------:|
| `task=True` for image generation | Clients can poll progress instead of blocking | 1 hour |
| Streamable HTTP transport | Deploy as HTTP service, not just stdio | 2 hours |
| `ctx.elicit()` for conversational refinement | Replace custom dialogue system with native MCP | 4 hours |
| `fastmcp dev` for testing | Better DX during development | Free |

---

## 12. Phase 9: MCP Protocol Advancement (Week 6+)

> **Leverage latest MCP spec capabilities.**
> Estimated effort: ~12 hours

### 12.1 MCP Tasks for Async Image Generation

**MCP Spec:** 2025-11-25 introduced Tasks -- a "call-now, fetch-later" primitive.

Image generation is inherently long-running (5-30 seconds). Currently the tool blocks until
complete. With Tasks, the flow becomes:

```
Client -> tools/call -> Server returns task_id
Client -> tasks/get(task_id) -> Server returns progress/result
```

**Implementation with FastMCP 3.0:**
```python
@mcp.tool(task=True)
async def generate_image(params: ImageGenerationInput, ctx: Context) -> str:
    ctx.report_progress(0, 100, "Selecting provider...")
    recommendation = suggest_provider(params.prompt, ...)

    ctx.report_progress(20, 100, f"Generating with {recommendation.provider}...")
    result = await provider.generate_image(params.prompt, **kwargs)

    ctx.report_progress(90, 100, "Saving image...")
    # ... save and format
    return formatted_result
```

**Benefits:**
- No timeout issues for slow generations
- Client shows progress bar
- Multiple generations can run concurrently
- Client can cancel in-flight generations

### 12.2 MCP Elicitation for Conversational Refinement

**MCP Spec:** 2025-06-18 introduced Elicitation -- servers can request structured user input.

The current conversational image tool uses a **custom parameter-based dialogue system** with
a `dialogue_responses` field and multi-turn state tracking. This is complex and fragile.

MCP Elicitation provides this natively:

```python
@mcp.tool()
async def generate_image(params: ImageGenerationInput, ctx: Context) -> str:
    # Instead of custom dialogue system, use native elicitation
    if params.style is None:
        response = await ctx.elicit(
            message="What visual style would you like?",
            schema={
                "type": "object",
                "properties": {
                    "style": {
                        "type": "string",
                        "enum": ["photorealistic", "illustration", "cartoon", "oil-painting"]
                    },
                    "mood": {
                        "type": "string",
                        "description": "Desired mood (e.g., warm, dramatic, serene)"
                    }
                }
            }
        )
        if response.action == "accept":
            params.style = response.data["style"]
```

**Benefits:**
- First-class MCP interaction (not custom protocol workarounds)
- Client renders appropriate UI (dropdowns, text fields)
- JSON Schema validation built-in
- Simpler server code -- can potentially replace `services/dialogue.py` entirely

**Caveat:** Not all MCP clients support Elicitation yet. Maintain fallback.

### 12.3 Tool Annotations

**MCP Spec:** 2025-06-18 introduced Tool Annotations for metadata about tool behavior.

```python
@mcp.tool(
    annotations={
        "title": "Generate Image",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,  # Makes network calls
    }
)
async def generate_image(...):
```

### 12.4 Output Schemas

**MCP Spec:** 2025-06-18 introduced Output Schemas for typed tool responses.

Currently all tools return `str` (markdown or JSON text). With Output Schemas, clients know
the response structure at registration time:

```python
class ImageGenerationResult(BaseModel):
    image_path: str
    provider: str
    model: str
    size: str
    generation_time_ms: int
    prompt_used: str

@mcp.tool(output_schema=ImageGenerationResult)
async def generate_image(...) -> ImageGenerationResult:
    ...
```

### 12.5 Streamable HTTP Transport

**MCP Spec:** 2025-03-26 replaced SSE with Streamable HTTP.

Currently imagen-mcp only supports stdio transport (local process). With Streamable HTTP:
- Deploy as a web service
- Multiple clients can connect
- Stateless or stateful sessions
- Works behind load balancers and proxies

FastMCP 3.0 supports this out of the box:
```bash
# Deploy as HTTP
fastmcp run server:mcp --transport streamable-http --port 8000
```

---

## 13. Phase 10: New Features & Providers (Week 7+)

> **Expand capabilities based on modernized foundation.**
> Estimated effort: ~20+ hours

### 13.1 Add Imagen 4 as Distinct Provider Path

Imagen 4 uses a **different API shape** than Gemini native image generation. It's accessed
through the same `google-genai` SDK but via `client.models.generate_images()` instead of
`client.models.generate_content()`.

```python
class Imagen4Provider(ImageProvider):
    name = "imagen4"
    env_key = "GEMINI_API_KEY"  # Same key as Gemini

    async def generate_image(self, prompt, **kwargs):
        result = await self.client.models.generate_images(
            model=kwargs.get("model", "imagen-4.0-generate-001"),
            prompt=prompt,
            config=ImageGenerationConfig(
                number_of_images=1,
                output_mime_type="image/png",
                person_generation="ALLOW_ADULT",
            )
        )
        ...
```

**Decision:** Should Imagen 4 be a separate provider or a model option within the Gemini
provider? Recommendation: **Separate provider** because the API shape, capabilities, and
limitations (no conversation, no text interleaving) are fundamentally different.

### 13.2 Image Editing Support

Both OpenAI and Gemini support image editing (inpainting, outpainting, style transfer).

**OpenAI GPT-Image-1.5 editing:**
```python
result = client.images.edit(
    model="gpt-image-1.5",
    image=open("input.png", "rb"),
    mask=open("mask.png", "rb"),  # Optional
    prompt="Replace the sky with a sunset",
)
```

**Gemini native editing:**
- Send image + text prompt in `generateContent`
- Model understands "change X to Y" instructions natively

**New MCP tool:**
```python
@mcp.tool()
async def edit_image(
    image_path: str,
    prompt: str,
    mask_path: str | None = None,
    provider: Provider = "auto",
) -> str:
    ...
```

### 13.3 Batch Generation

Both providers support generating multiple images per prompt:
- OpenAI: `n=1-4` parameter
- Gemini: `number_of_images=1-4`
- Imagen 4: `number_of_images=1-4`

**New parameter:** `count: int = Field(default=1, ge=1, le=4)`

### 13.4 Output Format Options

GPT-Image-1.5 supports PNG, JPEG, and WebP output. Currently hardcoded to PNG.

**New parameters:**
```python
output_format: Literal["png", "jpeg", "webp"] = "png"
compression_quality: int = Field(default=90, ge=0, le=100)  # JPEG/WebP only
```

### 13.5 Cost Estimation Tool

Add a tool that estimates generation cost before committing:

```python
@mcp.tool()
async def estimate_cost(
    prompt: str,
    provider: Provider = "auto",
    quality: str = "medium",
    size: str = "1024x1024",
) -> str:
    """Estimate the cost of generating an image without actually generating it."""
    recommendation = suggest_provider(prompt, ...)
    cost = PRICING_TABLE[recommendation.provider][quality][size]
    return f"Estimated cost: ${cost:.4f} via {recommendation.provider}"
```

### 13.6 Prompt Enhancement Toggle Fix

`GeminiProvider.enhance_prompt()` exists but is **never called** (dead code).
The `enhance_prompt` parameter flows through but does nothing on the Gemini side.

**Fix:** Wire `enhance_prompt` into the Gemini generation flow, or remove the field if
enhancement should only be OpenAI-side.

---

## 14. Dead Code Removal Inventory

> **38 items to resolve: remove, wire up, or document as intentional.**

### 14.1 Dead Functions (1)

| Item | Location | Action |
|------|----------|--------|
| `create_app()` | `server.py:601` | Remove (not an entry point) |

### 14.2 Dead Methods (10)

| Item | Location | Action |
|------|----------|--------|
| `ImageProvider.get_best_size_for_type` | `base.py:195` | Remove (both providers override) |
| `OpenAIProvider.get_best_size_for_type` | `openai_provider.py:453` | Remove (zero callers) |
| `GeminiProvider.get_best_size_for_type` | `gemini_provider.py:418` | Remove (zero callers) |
| `GeminiProvider.get_best_aspect_ratio_for_type` | `gemini_provider.py:430` | Remove (never wired) |
| `GeminiProvider.enhance_prompt` | `gemini_provider.py:446` | **Wire up** or remove (see 13.6) |
| `ImageProvider.supports_feature` | `base.py:236` | Remove (use capabilities instead) |
| `ImageProvider._retry_with_backoff` | `base.py:302` | **Wire up** (see Phase 4.3) |
| `ConversationStore.cleanup_old_conversations` | `conversation_store.py:366` | **Wire up** on server startup/timer |
| `RateLimiter.get_status` | `rate_limiter.py:130` | Remove or expose via metrics |
| `RateLimiter.reset` | `rate_limiter.py:156` | Remove or expose via admin tool |

### 14.3 Dead Constants (5)

| Item | Location | Action |
|------|----------|--------|
| `OPENAI_MODELS` | `constants.py:21` | **Wire up** (providers should use it) |
| `DEFAULT_OPENAI_IMAGE_MODEL` | `constants.py:31` | **Wire up** (replace hardcoded strings) |
| `GEMINI_MAX_OBJECT_IMAGES` | `constants.py:75` | **Wire up** (enforce in validation) |
| `GEMINI_MAX_HUMAN_IMAGES` | `constants.py:76` | **Wire up** (enforce in validation) |
| `DEFAULT_TIMEOUT` | `constants.py:84` | **Wire up** (replace hardcoded timeouts) |

### 14.4 Dead/Buggy Model Fields (5)

| Item | Location | Action |
|------|----------|--------|
| `dialogue_responses` | `input_models.py:204` | Remove |
| `input_image_file_id` | `input_models.py:221` | **Wire through** server to provider (editing support) |
| `input_image_path` | `input_models.py:229` | Remove (no downstream support) |
| `assistant_model` | `input_models.py:257` | **Wire through** or remove |
| `ListConversationsInput.output_format` | `input_models.py:312` | **Fix**: handler should respect it |

### 14.5 Dead `__init__.py` Re-exports (15)

All re-exports in `src/config/__init__.py`, `src/providers/__init__.py`, and
`src/models/__init__.py` are unused -- every consumer imports directly from submodules.

**Action:** Remove dead re-exports or document them as public API surface.

### 14.6 Other Dead Code

| Item | Location | Action |
|------|----------|--------|
| No-op `.replace()` | `conversation_store.py:183` | Remove (replaces string with itself) |
| `src/tools/__init__.py` | Empty file, unused package | Remove |
| 5 Settings fields loaded but never used | `settings.py` | Wire into providers or remove |

---

## 15. API Research Reference

### 15.1 OpenAI Image API (Current as of Feb 2026)

**Available Models:**

| Model | Status | Endpoint | Key Capabilities |
|-------|:------:|----------|-----------------|
| `gpt-image-1.5` | GA | `POST /v1/images/generations` | 4x faster, cheaper, best text, editing, WebP |
| `gpt-image-1` | GA | Same | Original model, reliable baseline |

**GPT-Image-1.5 Full Parameter Reference:**

```python
response = client.images.generate(
    model="gpt-image-1.5",
    prompt="...",
    n=1,                           # 1-4 images
    size="1024x1024",              # 1024x1024, 1024x1536, 1536x1024, 1536x1536, auto
    quality="auto",                # low, medium, high, auto
    output_format="png",           # png, jpeg, webp
    output_compression=None,       # 0-100 (jpeg/webp only)
    background="auto",             # auto, transparent, opaque
    moderation="auto",             # auto, low
)
```

**Pricing (per image):**

| Model | Quality | 1024x1024 | 1024x1536 | 1536x1536 |
|-------|---------|----------:|----------:|----------:|
| gpt-image-1.5 | low | $0.011 | $0.016 | $0.024 |
| gpt-image-1.5 | medium | $0.022 | $0.033 | $0.049 |
| gpt-image-1.5 | high | $0.044 | $0.066 | $0.099 |
| gpt-image-1 | low | $0.011 | $0.016 | N/A |
| gpt-image-1 | medium | $0.042 | $0.063 | N/A |
| gpt-image-1 | high | $0.167 | $0.250 | N/A |

### 15.2 Google Gemini Image API (Current as of Feb 2026)

**Native Image Generation Models:**

| Model | Codename | Status | API Method | Conversational | Max Res |
|-------|----------|:------:|:----------:|:--------------:|:-------:|
| `gemini-2.5-flash-preview-image-generation` | Nano Banana | GA | `generateContent` | Yes | 1536x1536 |
| `gemini-3-pro-image-preview` | Nano Banana Pro | Preview | `generateContent` | Yes | 1536x1536 |

**Imagen Family Models:**

| Model | Status | API Method | Multi-Image | Max Res |
|-------|:------:|:----------:|:-----------:|:-------:|
| `imagen-4.0-generate-001` | GA | `generateImages` | Yes (1-4) | 2048x2048 |
| `imagen-4.0-ultra-generate-001` | GA | `generateImages` | Yes (1-4) | 2048x2048 |
| `imagen-4.0-fast-generate-001` | GA | `generateImages` | Yes (1-4) | 2048x2048 |

**Deprecated/Dead Models:**

| Model | Shutdown Date |
|-------|:------------:|
| `gemini-2.0-flash-exp-image-generation` | Nov 14, 2025 |
| `imagen-3.0-generate-002` | Nov 10, 2025 |
| `imagen-3.0-fast-generate-001` | Nov 10, 2025 |

**Gemini Native Image API:**
```python
from google import genai

client = genai.Client(api_key="...")
response = client.models.generate_content(
    model="gemini-2.5-flash-preview-image-generation",
    contents="A photorealistic portrait...",
    config=GenerateContentConfig(
        response_modalities=["TEXT", "IMAGE"],
    )
)
for part in response.candidates[0].content.parts:
    if part.inline_data:
        image_bytes = part.inline_data.data  # Raw bytes
```

**Imagen 4 API:**
```python
response = client.models.generate_images(
    model="imagen-4.0-generate-001",
    prompt="...",
    config=ImageGenerationConfig(
        number_of_images=1,
        output_mime_type="image/png",
        aspect_ratio="1:1",
        person_generation="ALLOW_ADULT",
        safety_filter_level="BLOCK_LOW_AND_ABOVE",
    )
)
for image in response.generated_images:
    image.image.save("output.png")
```

**Aspect Ratios (Gemini Native):** 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9

**Pricing (per image):**

| Model | Resolution | Cost |
|-------|-----------|-----:|
| Gemini 2.5 Flash Image | Standard | $0.020 |
| Gemini 3 Pro Image | Standard | ~$0.040 |
| Imagen 4 | Standard | $0.020 |
| Imagen 4 Ultra | Standard | $0.060 |
| Imagen 4 Fast | Standard | $0.010 |

### 15.3 FastMCP 3.0 Reference

**Key Migration Notes:**
- Import `from fastmcp import FastMCP` is unchanged
- `@mcp.tool()` decorator API is backward-compatible
- `Context` gains new methods: `ctx.elicit()`, `ctx.sample()`, `ctx.report_progress()`
- New: `@mcp.tool(task=True)` for background tasks
- New: Streamable HTTP transport
- New: Server composition via `mcp.import_server()`
- New: Built-in OAuth 2.1 support
- Breaking: Some internal APIs changed (session management, transport layer)

### 15.4 MCP Specification Versions

| Spec Version | Date | Key Additions |
|:------------:|:----:|---------------|
| 2024-11-05 | Nov 2024 | Initial release -- tools, resources, prompts |
| 2025-03-26 | Mar 2025 | Streamable HTTP transport (replaces SSE) |
| 2025-06-18 | Jun 2025 | Elicitation, tool annotations, output schemas, OAuth 2.1 |
| 2025-11-25 | Nov 2025 | Tasks (async workflows), structured content, enhanced auth |

---

## 16. Timeline & Effort Summary

### Priority-Ordered Execution Plan

| Phase | Scope | Effort | Priority | Dependencies |
|:-----:|-------|:------:|:--------:|:------------:|
| **0** | Critical fixes (dead models, version, pin) | ~3h | CRITICAL - Day 1 | None |
| **1** | Security hardening (8 fixes) | ~4h | CRITICAL - Week 1 | None |
| **2** | Performance fixes (9 fixes) | ~3h | CRITICAL - Week 1 | None |
| **3** | Provider modernization (models, scoring) | ~8h | HIGH - Week 2 | Phase 0 |
| **4** | Error handling (exceptions, retry, structured) | ~5h | HIGH - Week 2 | None |
| **5** | Architecture (service layer, decompose, pluggable) | ~15h | HIGH - Week 3 | Phases 1-4 |
| **6** | Testing & CI/CD (conftest, coverage, pipeline) | ~12h | HIGH - Week 3-4 | Phase 5 |
| **7** | Dependencies & packaging | ~3h | MEDIUM - Week 4 | Phase 6 |
| **8** | FastMCP 3.0 migration | ~6h | MEDIUM - Week 5 | Phase 7 |
| **9** | MCP protocol advancement (tasks, elicitation) | ~12h | LOW - Week 6+ | Phase 8 |
| **10** | New features (Imagen 4, editing, batch) | ~20h | LOW - Week 7+ | Phase 9 |

### Effort Totals

| Category | Hours |
|----------|------:|
| Critical fixes (Phases 0-2) | **10h** |
| Core modernization (Phases 3-5) | **28h** |
| Quality & packaging (Phases 6-7) | **15h** |
| Protocol advancement (Phases 8-9) | **18h** |
| New features (Phase 10) | **20h** |
| **Grand Total** | **~91h** |

### Quick Wins (Under 1 Hour, High Impact)

| # | Fix | Time | Impact |
|---|-----|:----:|--------|
| 1 | Move Gemini API key to header | 5 min | Eliminates critical security vuln |
| 2 | Add `repr=False` to API key fields | 5 min | Prevents credential in repr |
| 3 | Remove `os.path.expandvars()` from user paths | 5 min | Prevents env var disclosure |
| 4 | Remove PIL decode-re-encode round-trip | 5 min | -200-500ms per Gemini generation |
| 5 | Fix version mismatch | 2 min | Eliminates confusion |
| 6 | Pin FastMCP `<3.0` | 2 min | Prevents surprise breakage |
| 7 | Remove no-op `.replace()` | 2 min | Eliminates confusing dead code |
| 8 | Remove dead model references | 15 min | Eliminates user-facing errors |
| 9 | Compile regex patterns at module level | 15 min | Eliminates repeated compilation |
| 10 | Move `configure_logging()` to startup | 5 min | Eliminates 5x redundant calls |

**These 10 fixes take ~1 hour total and address 2 critical vulns, 1 critical perf issue, and 5 quality issues.**
