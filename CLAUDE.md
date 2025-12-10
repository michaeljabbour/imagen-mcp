# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**imagen-mcp** is a Python MCP (Model Context Protocol) server that provides intelligent multi-provider image generation to Claude Desktop. It automatically selects the best provider based on your prompt.

### Supported Providers

| Provider | Model | Best For |
|----------|-------|----------|
| **OpenAI** | GPT-Image-1 | Text rendering, infographics, comics, diagrams |
| **Gemini** | Nano Banana Pro (gemini-3-pro-image-preview) | Portraits, product photography, 4K output |

### Key Features

- **Auto Provider Selection** - Analyzes prompts to choose the best provider
- **Multi-turn Conversations** - Iterative refinement with context preservation
- **Pre-Generation Dialogue** - Guided questions refine vision before generating
- **Reference Images** - Up to 14 images for character/style consistency (Gemini)
- **Real-time Data** - Google Search grounding for current info (Gemini)
- **Full-quality PNGs** - High-resolution images saved to ~/Downloads/images/

## Development Commands

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set API keys (at least one required)
export OPENAI_API_KEY="your-openai-key"
export GEMINI_API_KEY="your-gemini-key"
```

### Testing
```bash
# Test new architecture loads
python3 -c "from src.server import mcp; print('✅ Server loads')"

# Test providers
python3 -c "from src.providers import get_provider_registry; print(get_provider_registry().list_providers())"

# Run tests
pytest

# Check logs
tail -f ~/Library/Logs/Claude/mcp-server-imagen.log
```

### Claude Desktop Configuration

**macOS** (`~/Library/Application Support/Claude/claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "imagen": {
      "command": "python3",
      "args": ["-m", "src.server"],
      "cwd": "/path/to/imagen-mcp",
      "env": {
        "OPENAI_API_KEY": "sk-...",
        "GEMINI_API_KEY": "AI..."
      }
    }
  }
}
```

## Architecture

### Multi-Provider Design

```
src/
├── server.py                 # MCP entry point with unified tools
├── config/
│   ├── constants.py          # Provider constants, keywords
│   └── settings.py           # Environment configuration
├── providers/
│   ├── base.py               # Abstract ImageProvider interface
│   ├── openai_provider.py    # OpenAI GPT-Image-1 implementation
│   ├── gemini_provider.py    # Gemini Nano Banana Pro implementation
│   ├── selector.py           # Auto-selection logic
│   └── registry.py           # Provider factory
├── models/
│   └── input_models.py       # Pydantic input models
└── services/                 # Dialogue, enhancement, storage
```

### Provider Selection Flow

1. User submits prompt to `generate_image` tool
2. `ProviderSelector` analyzes prompt for:
   - Text rendering keywords → OpenAI
   - Portrait/product keywords → Gemini
   - Reference images → Gemini (required)
   - Google Search grounding → Gemini (required)
   - 4K resolution → Gemini (required)
3. Selected provider generates image
4. Result returned with provider reasoning

### Provider Capabilities

| Feature | OpenAI | Gemini |
|---------|--------|--------|
| Text Rendering | ⭐⭐⭐ Excellent | ⭐⭐ Good |
| Photorealism | ⭐⭐ Good | ⭐⭐⭐ Excellent |
| Speed | ~60s | ~15s |
| Max Resolution | 1536x1024 | 4K |
| Sizes | 1024x1024, 1024x1536, 1536x1024 | 1K, 2K, 4K |
| Aspect Ratios | 3 | 10 |
| Reference Images | ❌ | ✅ (up to 14) |
| Real-time Data | ❌ | ✅ (Google Search) |
| Thinking Mode | ❌ | ✅ |

## MCP Tools

### `generate_image`
Main tool for image generation with auto provider selection.

```python
# Auto-selects OpenAI (text rendering)
generate_image(prompt="Menu card for Italian restaurant with prices")

# Auto-selects Gemini (portrait)
generate_image(prompt="Professional headshot with studio lighting")

# Force specific provider
generate_image(prompt="...", provider="gemini")

# Use reference images (auto-selects Gemini)
generate_image(prompt="...", reference_images=["base64..."])
```

### `conversational_image`
Multi-turn refinement with dialogue system.

### `list_providers`
Show available providers and their capabilities.

## Key Implementation Details

### OpenAI Provider (`src/providers/openai_provider.py`)

Uses Responses API with forced tool calling:
```python
payload = {
    "model": "gpt-4o",
    "tools": [{"function": {"name": "generate_image"}}],
    "tool_choice": {"function": {"name": "generate_image"}},
}
# → Extracts image from tool call → Calls /images/generations
```

### Gemini Provider (`src/providers/gemini_provider.py`)

Uses official Google GenAI SDK:
```python
from google import genai
from google.genai import types

client = genai.Client(api_key=key)
response = client.models.generate_content(
    model="gemini-3-pro-image-preview",
    contents=[*reference_images, prompt],
    config=types.GenerateContentConfig(
        response_modalities=["TEXT", "IMAGE"],
        image_config=types.ImageConfig(image_size="2K"),
    ),
)
```

### Auto-Selection Keywords

**OpenAI preferred:**
- text, label, menu, infographic, diagram, comic, dialogue, caption, title, headline, poster, certificate, badge

**Gemini preferred:**
- portrait, headshot, photo, photorealistic, product, studio, 4k, character consistency, weather, stock, current

**Gemini required:**
- current weather, today's, real-time, stock price, live (needs Google Search)
- Any use of reference images
- 4K resolution

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key | One of these |
| `GEMINI_API_KEY` | Google Gemini API key | required |
| `DEFAULT_PROVIDER` | Default: "auto" | No |
| `DEFAULT_OPENAI_SIZE` | Default: "1024x1024" | No |
| `DEFAULT_GEMINI_SIZE` | Default: "2K" | No |
| `ENABLE_GOOGLE_SEARCH` | Default: "false" | No |

## Troubleshooting

### "No providers available"
Set at least one API key: `OPENAI_API_KEY` or `GEMINI_API_KEY`

### "Gemini provider requires google-genai"
```bash
pip install google-genai pillow
```

### Wrong provider selected
Use explicit `provider` parameter to override auto-selection.

### Image not saved
Check ~/Downloads/images/ directory exists and is writable.

## Dependencies

```
mcp>=1.16.0           # MCP protocol
fastmcp>=2.12.5       # FastMCP framework
pydantic>=2.12.3      # Input validation
httpx>=0.24.0         # OpenAI HTTP client
google-genai>=1.52.0  # Gemini SDK
pillow>=10.4.0        # Image processing
```

## Version History

### v4.0.0 (Current)
- Multi-provider support (OpenAI + Gemini)
- Auto provider selection
- Gemini Nano Banana Pro with full features
- Restructured as `src/` package

### v3.0.0
- OpenAI GPT-Image-1 only
- Phase 1 dialogue system
- Simplified architecture
