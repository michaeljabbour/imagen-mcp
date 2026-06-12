"""Approximate image-generation pricing for cost estimation.

These figures are **estimates** intended to help callers compare the
relative cost of providers/qualities before committing to a generation.
They are not billing-accurate: real cost depends on live provider pricing,
token usage (OpenAI gpt-image-2 bills image *output tokens*), and any
account-specific discounts. Always treat the output as a ballpark.

Sources (as of 2026-06): OpenAI gpt-image pricing tiers and Google
Gemini image pricing pages. Update ``PRICING`` when providers change rates.
"""

from __future__ import annotations

from dataclasses import dataclass

# OpenAI gpt-image-* — approximate USD per image by quality tier.
# A size multiplier scales the square baseline for larger canvases.
_OPENAI_QUALITY_USD: dict[str, float] = {
    "low": 0.011,
    "medium": 0.042,
    "high": 0.167,
    # "auto"/"standard"/"hd" map onto the medium tier for estimation.
    "auto": 0.042,
    "standard": 0.042,
    "hd": 0.167,
}

_OPENAI_SIZE_MULTIPLIER: dict[str, float] = {
    "256x256": 0.5,
    "512x512": 0.7,
    "1024x1024": 1.0,
    "1024x1536": 1.5,
    "1536x1024": 1.5,
    "1024x1792": 1.75,
    "1792x1024": 1.75,
    "auto": 1.0,
}

# Gemini (Nano Banana family) — approximate USD per image by resolution.
_GEMINI_SIZE_USD: dict[str, float] = {
    "1K": 0.020,
    "2K": 0.039,
    "4K": 0.120,
}

# Pro tier (Nano Banana Pro / gemini-3-pro-image-preview) costs more.
_GEMINI_PRO_MULTIPLIER = 1.8
_GEMINI_PRO_MODEL_MARKERS = ("pro",)


@dataclass
class CostEstimate:
    """Result of a cost estimation."""

    provider: str
    model: str | None
    quality: str | None
    size: str | None
    n: int
    per_image_usd: float | None
    total_usd: float | None
    approximate: bool = True
    note: str | None = None


def _estimate_openai(quality: str | None, size: str | None) -> tuple[float | None, str | None]:
    q = (quality or "auto").lower()
    s = (size or "1024x1024").lower().replace("X", "x")
    base = _OPENAI_QUALITY_USD.get(q)
    if base is None:
        return None, f"Unknown quality '{quality}' for OpenAI"
    mult = _OPENAI_SIZE_MULTIPLIER.get(s, 1.0)
    return round(base * mult, 4), None


def _estimate_gemini(model: str | None, size: str | None) -> tuple[float | None, str | None]:
    s = (size or "2K").upper()
    base = _GEMINI_SIZE_USD.get(s)
    if base is None:
        return None, f"Unknown size '{size}' for Gemini"
    if model and any(marker in model.lower() for marker in _GEMINI_PRO_MODEL_MARKERS):
        base *= _GEMINI_PRO_MULTIPLIER
    return round(base, 4), None


def estimate_generation_cost(
    provider: str,
    *,
    model: str | None = None,
    quality: str | None = None,
    size: str | None = None,
    n: int = 1,
) -> CostEstimate:
    """Estimate the cost of generating ``n`` images.

    Returns a :class:`CostEstimate`; ``per_image_usd``/``total_usd`` are
    ``None`` when the combination isn't in the pricing table (with the
    reason in ``note``).
    """
    provider = provider.lower()
    n = max(1, int(n))

    if provider == "openai":
        per_image, note = _estimate_openai(quality, size)
    elif provider == "gemini":
        per_image, note = _estimate_gemini(model, size)
    else:
        per_image, note = None, f"No pricing data for provider '{provider}'"

    total = round(per_image * n, 4) if per_image is not None else None
    return CostEstimate(
        provider=provider,
        model=model,
        quality=quality,
        size=size,
        n=n,
        per_image_usd=per_image,
        total_usd=total,
        note=note,
    )


def format_cost_estimate(est: CostEstimate) -> str:
    """Render a cost estimate as markdown."""
    lines = ["## 💵 Cost Estimate", ""]
    lines.append(f"**Provider:** {est.provider.title()}")
    if est.model:
        lines.append(f"**Model:** {est.model}")
    if est.quality:
        lines.append(f"**Quality:** {est.quality}")
    if est.size:
        lines.append(f"**Size:** {est.size}")
    lines.append(f"**Images:** {est.n}")
    lines.append("")

    if est.total_usd is not None:
        lines.append(f"**Estimated cost:** ~${est.total_usd:.4f} (${est.per_image_usd:.4f}/image)")
    else:
        lines.append(f"**Estimated cost:** unavailable — {est.note or 'no pricing data'}")

    lines.extend(
        [
            "",
            "> ⚠️ Approximate. Real cost depends on live provider pricing and "
            "(for OpenAI) actual image output tokens.",
        ]
    )
    return "\n".join(lines)
