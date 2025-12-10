# imagen-mcp

A Model Context Protocol (MCP) server for intelligent multi-provider image generation with Claude Desktop.

## Features

- **Auto Provider Selection** - Analyzes prompts to choose the best provider
- **Multi-Provider Support** - OpenAI GPT-Image-1 and Google Gemini
- **Reference Images** - Up to 14 images for character/style consistency (Gemini)
- **Real-time Data** - Google Search grounding for current info (Gemini)
- **High Resolution** - Up to 4K output (Gemini)
- **Full-quality PNGs** - Images saved to `~/Downloads/images/`

## Provider Comparison

| Feature | OpenAI GPT-Image-1 | Gemini Nano Banana Pro |
|---------|-------------------|------------------------|
| Text Rendering | Excellent | Good |
| Photorealism | Good | Excellent |
| Speed | ~60s | ~15s |
| Max Resolution | 1536x1024 | 4K |
| Sizes | 3 options | 1K, 2K, 4K |
| Aspect Ratios | 3 | 10 |
| Reference Images | No | Yes (up to 14) |
| Real-time Data | No | Yes (Google Search) |

**Use OpenAI for:** Text-heavy images, menus, infographics, comics, diagrams

**Use Gemini for:** Portraits, product photography, 4K output, reference images

## Installation

```bash
git clone https://github.com/yourusername/imagen-mcp.git
cd imagen-mcp
pip install -r requirements.txt
```

## Configuration

At least one API key is required. Both are recommended for auto-selection.

### Claude Desktop

**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

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

### Claude Code (CLI)

Add to `~/.claude/settings.json` or project `.mcp.json`:

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

Or use the CLI:

```bash
claude mcp add imagen -s user -- python3 -m src.server --cwd /path/to/imagen-mcp
```

### Gemini CLI

Add to `~/.gemini/settings.json`:

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

### OpenAI Codex CLI

Add to `~/.codex/config.json`:

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

### Generic MCP Client

For any MCP-compatible client, configure with:

- **Command:** `python3`
- **Args:** `["-m", "src.server"]`
- **Working Directory:** `/path/to/imagen-mcp`
- **Environment:** `OPENAI_API_KEY`, `GEMINI_API_KEY`

## Usage

### Auto Provider Selection

The server analyzes your prompt and selects the best provider:

```
"Create a menu card for an Italian restaurant"  → OpenAI (text rendering)
"Professional headshot with studio lighting"    → Gemini (photorealism)
"Infographic about climate change"              → OpenAI (diagram + text)
"Product shot of perfume on marble"             → Gemini (product photography)
```

### Manual Provider Selection

Override auto-selection with the `provider` parameter:

```
generate_image(prompt="...", provider="openai")
generate_image(prompt="...", provider="gemini")
```

### Gemini-Specific Features

```
# High resolution
generate_image(prompt="...", size="4K")

# Specific model
generate_image(prompt="...", gemini_model="gemini-2.0-flash-exp-image-generation")

# Reference images (base64 encoded)
generate_image(prompt="...", reference_images=["base64..."])

# Real-time data
generate_image(prompt="Current weather in NYC", enable_google_search=True)
```

## MCP Tools

### `generate_image`
Main tool for image generation with auto provider selection.

### `conversational_image`
Multi-turn refinement with dialogue system.

### `list_providers`
Show available providers and their capabilities.

### `list_gemini_models`
Query available Gemini image models from the API.

## Development

```bash
# Test server loads
python3 -c "from src.server import mcp; print('Server loads')"

# Test providers
python3 -c "from src.providers import get_provider_registry; print(get_provider_registry().list_providers())"

# Check logs
tail -f ~/Library/Logs/Claude/mcp-server-imagen.log
```

## Project Structure

```
imagen-mcp/
├── src/
│   ├── server.py              # MCP entry point
│   ├── config/
│   │   ├── constants.py       # Provider constants
│   │   └── settings.py        # Environment configuration
│   ├── providers/
│   │   ├── base.py            # Abstract provider interface
│   │   ├── openai_provider.py # OpenAI implementation
│   │   ├── gemini_provider.py # Gemini implementation
│   │   ├── selector.py        # Auto-selection logic
│   │   └── registry.py        # Provider factory
│   └── models/
│       └── input_models.py    # Pydantic input models
├── requirements.txt
├── CLAUDE.md
└── README.md
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key | One of these |
| `GEMINI_API_KEY` | Google Gemini API key | required |
| `GOOGLE_API_KEY` | Alias for GEMINI_API_KEY | |
| `DEFAULT_PROVIDER` | Default: "auto" | No |
| `DEFAULT_OPENAI_SIZE` | Default: "1024x1024" | No |
| `DEFAULT_GEMINI_SIZE` | Default: "2K" | No |
| `ENABLE_GOOGLE_SEARCH` | Default: "false" | No |

## Gemini Models

| Model ID | Description |
|----------|-------------|
| `gemini-3-pro-image-preview` | Default, highest quality |
| `gemini-2.0-flash-exp-image-generation` | Fast experimental |
| `imagen-3.0-generate-002` | Alternative model |

## Requirements

```
mcp>=1.16.0
fastmcp>=2.12.5
pydantic>=2.12.3
httpx>=0.24.0
google-genai>=1.52.0
pillow>=10.4.0
```

## License

MIT
