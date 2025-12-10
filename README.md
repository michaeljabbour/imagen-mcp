# imagen-mcp

A Model Context Protocol (MCP) server for intelligent multi-provider image generation.

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
chmod +x run.sh
```

## Configuration

At least one API key is required. Both are recommended for auto-selection.

### Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "imagen": {
      "command": "/path/to/imagen-mcp/run.sh",
      "args": [],
      "env": {
        "OPENAI_API_KEY": "sk-...",
        "GEMINI_API_KEY": "AI..."
      }
    }
  }
}
```

> **Note:** Claude Desktop doesn't support `cwd`, so use the `run.sh` wrapper script which handles the directory change.

Restart Claude Desktop (Cmd+Q, then reopen) after editing.

### Claude Code CLI

Use the CLI to add the server:

```bash
claude mcp add -s user imagen /path/to/imagen-mcp/run.sh
```

Then add environment variables by editing `~/.claude.json`:

```json
{
  "mcpServers": {
    "imagen": {
      "type": "stdio",
      "command": "/path/to/imagen-mcp/run.sh",
      "args": [],
      "env": {
        "OPENAI_API_KEY": "sk-...",
        "GEMINI_API_KEY": "AI..."
      }
    }
  }
}
```

Verify with:
```bash
claude mcp list
```

Reference: [Claude Code MCP Documentation](https://code.claude.com/docs/en/mcp)

### Gemini CLI

Edit `~/.gemini/settings.json`:

```json
{
  "mcpServers": {
    "imagen": {
      "command": "/path/to/imagen-mcp/run.sh",
      "args": [],
      "env": {
        "OPENAI_API_KEY": "sk-...",
        "GEMINI_API_KEY": "AI..."
      }
    }
  }
}
```

Reference: [Gemini CLI MCP Documentation](https://geminicli.com/docs/tools/mcp-server/)

### OpenAI Codex CLI

Edit `~/.codex/config.toml`:

```toml
[mcp_servers.imagen]
command = "/path/to/imagen-mcp/run.sh"
args = []

[mcp_servers.imagen.env]
OPENAI_API_KEY = "sk-..."
GEMINI_API_KEY = "AI..."
```

Or use the CLI:
```bash
codex mcp add imagen -- /path/to/imagen-mcp/run.sh
```

Reference: [Codex MCP Documentation](https://developers.openai.com/codex/mcp/)

### Generic MCP Client

For any MCP-compatible client:

| Setting | Value |
|---------|-------|
| Command | `/path/to/imagen-mcp/run.sh` |
| Args | `[]` |
| Environment | `OPENAI_API_KEY`, `GEMINI_API_KEY` |

## The Wrapper Script

The `run.sh` script handles the working directory requirement:

```bash
#!/bin/bash
cd /path/to/imagen-mcp
exec python3 -m src.server "$@"
```

This is necessary because the server runs as a Python module (`-m src.server`) which requires being in the project directory.

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

| Tool | Description |
|------|-------------|
| `generate_image` | Main tool with auto provider selection |
| `conversational_image` | Multi-turn refinement |
| `list_providers` | Show available providers and capabilities |
| `list_gemini_models` | Query available Gemini image models |

## Development

```bash
# Test server loads
python3 -c "from src.server import mcp; print('Server loads')"

# Test providers
python3 -c "from src.providers import get_provider_registry; print(get_provider_registry().list_providers())"

# Check logs (macOS)
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
├── run.sh                     # Wrapper script for MCP clients
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

## Sources

- [Claude Code MCP Documentation](https://code.claude.com/docs/en/mcp)
- [Gemini CLI MCP Documentation](https://geminicli.com/docs/tools/mcp-server/)
- [Codex MCP Documentation](https://developers.openai.com/codex/mcp/)
- [Model Context Protocol](https://modelcontextprotocol.io/)
