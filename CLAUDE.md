# CLAUDE.md

Instructions for Claude Code when working with this repository.

## Project Overview

**imagen-mcp** is a Python MCP server providing multi-provider image generation to Claude Desktop. It automatically selects the best provider based on prompt analysis.

## Supported Providers

| Provider | Model | Best For |
|----------|-------|----------|
| **OpenAI** | GPT-Image-1 | Text rendering, infographics, comics, diagrams |
| **Gemini** | Nano Banana Pro | Portraits, product photography, 4K output |

## Commands

```bash
# Test server loads
python3 -c "from src.server import mcp; print('OK')"

# Test providers
python3 -c "from src.providers import get_provider_registry; print(get_provider_registry().list_providers())"

# Check logs
tail -f ~/Library/Logs/Claude/mcp-server-imagen.log
```

## Architecture

```
src/
├── server.py                 # MCP entry point
├── config/
│   ├── constants.py          # Provider constants, keywords
│   └── settings.py           # Environment configuration
├── providers/
│   ├── base.py               # Abstract ImageProvider interface
│   ├── openai_provider.py    # OpenAI implementation
│   ├── gemini_provider.py    # Gemini implementation
│   ├── selector.py           # Auto-selection logic
│   └── registry.py           # Provider factory
└── models/
    └── input_models.py       # Pydantic input models
```

## Provider Selection

1. User submits prompt to `generate_image`
2. `ProviderSelector` analyzes prompt for keywords
3. Hard requirements force provider (reference images → Gemini, 4K → Gemini)
4. Soft preferences score each provider
5. Selected provider generates image

## MCP Tools

- `generate_image` - Main tool with auto provider selection
- `conversational_image` - Multi-turn refinement
- `list_providers` - Show available providers
- `list_gemini_models` - Query Gemini models

## Key Files

- `src/server.py` - MCP tools and entry point
- `src/providers/selector.py` - Provider selection logic
- `src/providers/openai_provider.py` - OpenAI GPT-Image-1
- `src/providers/gemini_provider.py` - Gemini Nano Banana Pro
- `src/config/constants.py` - Keywords for selection

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `GEMINI_API_KEY` | Gemini API key (or `GOOGLE_API_KEY`) |
| `DEFAULT_PROVIDER` | "auto", "openai", or "gemini" |

## Claude Desktop Config

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
