# Provider Selection — Quick Reference

For users calling `imagen-mcp` directly (without `amplifier-bundle-imagen` or `amplifier-bundle-creative` composed). The bundles carry richer agent-level guidance; this is the focused decision card for raw-MCP callers.

Generation auto-selects a provider based on prompt content. Override via the `provider` parameter when you have a specific reason.

---

## Decision card

```
┌──────────────────────────────────────────────────────┐
│  Need TEXT readable in the image?                    │
│      → provider="openai", model="gpt-image-2"        │
│      Examples: menus, posters, UI mockups,           │
│                infographics, brand wordmarks         │
│                                                      │
│  Need REFERENCE images for continuity?               │
│      → provider="gemini", model="nano-banana-pro"    │
│        with reference_images=[base64_1, base64_2…]   │
│      Examples: same character across shots,          │
│                campaign continuity, brand identity   │
│                                                      │
│  Need PHOTOREAL hero, product, portrait, no text?    │
│      → provider="gemini", model="nano-banana-pro"    │
│      Examples: macro product beauty, editorial       │
│                portrait, lifestyle commercial        │
│                                                      │
│  Need TARGETED edit of a prior image?                │
│      → use edit_image (OpenAI), input_fidelity=high  │
│      Examples: "change the sky to sunset",           │
│                "remove the power lines"              │
│                                                      │
│  Need 4K resolution?                                 │
│      → provider="gemini", size="4K",                 │
│        model="nano-banana-pro"                       │
│                                                      │
│  Need transparent background (alpha)?                │
│      → provider="openai", background="transparent",  │
│        output_format="png"                           │
└──────────────────────────────────────────────────────┘
```

---

## Reference-image discipline (when continuity matters)

For multi-shot or character-driven work:

1. **First shot** — generate text-only OR with the operator's source-IP image as a reference.
2. **Subsequent shots** — pass the prior approved shot via `reference_images` (Gemini provider, base64-encoded).
3. **Persistent anchor** — keep the first approved shot in the reference set across the whole sequence; chain the immediate predecessor as the secondary ref.
4. **Setting-match ranking** — when multiple refs are available, prefer setting-match over face-clarity. Lighting > setting > composition > face quality.
5. **Downsize before sending** — Nano Banana Pro silently fails on refs > 1400 px on the longest edge. Resize first:
   ```bash
   convert SOURCE.png -resize 1400x1400\> -quality 95 RESIZED.png
   ```

This protocol is the difference between "three unrelated generations" and "a campaign that holds together." Skip it on multi-shot work and you'll burn iterations chasing identity drift.

---

## Quick provider notes

| Provider | Default model | Strengths | Limits |
|---|---|---|---|
| OpenAI | `gpt-image-2` | Text accuracy ~99%, sequential `edit_image`, transparent BG | Max 1792×1024; no reference-image grounding; no Google Search |
| Google | `nano-banana-2` (alias for `gemini-3.1-flash-image-preview`) | Fast Gemini Flash; full ref + grounding feature set | Slightly lower fidelity than Pro |
| Google | `nano-banana-pro` (alias for `gemini-3-pro-image-preview`) | Highest fidelity, Thinking mode, 4K, up to 14 ref images | Slower (15–25s); higher cost |

For deep capability comparison, deprecation watchlist, and cost/latency tables, see the companion `amplifier-bundle-imagen` repository's `docs/PROVIDER_COMPARISON.md`. That bundle wraps this MCP server with agent-level direction (image-director, image-prompt-engineer, image-editor, image-researcher) and is the recommended way to call `imagen-mcp` for non-trivial work.

---

## Cost awareness (rough, not binding)

| Operation | Approximate cost |
|---|---|
| gpt-image-2, 1024×1024, high quality | $0.03 – $0.08 |
| gpt-image-2, 1024×1536 or 1792×1024, high quality | $0.05 – $0.12 |
| Nano Banana Pro, 2K, no refs | $0.04 – $0.10 |
| Nano Banana Pro, 2K, 4 ref images | $0.05 – $0.15 |
| Nano Banana Pro, 4K | $0.10 – $0.25 |

For projects with > 30 generations, surface an estimate to the operator before kicking off.

---

## Override the auto-selector

The MCP's auto-selection is heuristic. To pin explicitly:

```python
# Force OpenAI (e.g., for text-heavy content the heuristic missed)
generate_image(prompt="…", provider="openai")

# Force Gemini Pro for highest-fidelity portrait
generate_image(prompt="…", provider="gemini", gemini_model="nano-banana-pro")

# Force a specific model
generate_image(prompt="…", provider="openai", openai_model="gpt-image-1.5")
```

When the operator has expressed a preference, treat it as authoritative — don't override even if your heuristic would choose differently.
