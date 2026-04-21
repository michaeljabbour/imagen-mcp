# Repository Guidelines

For contributors building and maintaining the MCP image generation server.

## Project Structure & Module Organization
- `src/server.py` is the MCP entry point exporting the registered tools.
- `src/providers/` holds provider implementations (`openai_provider.py`, `gemini_provider.py`), `selector.py` for auto-selection, and `registry.py` for factory wiring.
- `src/config/` contains constants and settings; `src/models/input_models.py` defines Pydantic request models.
- Tests live in `tests/` mirroring modules (`test_selector.py`, `test_providers.py`, `test_server.py`); `run.sh` is the wrapper used by MCP clients; dependencies sit in `requirements.txt` and `dev-requirements.txt`.

## Build, Test, and Development Commands
- Create a venv and install deps: `python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt -r dev-requirements.txt`.
- Run the server locally: `python -m src.server` (or `./run.sh` when invoked by clients); export `OPENAI_API_KEY`/`GEMINI_API_KEY` first.
- Format and lint: `ruff format . && ruff check . --fix`.
- Type check: `mypy src`.
- Tests: `pytest` (verbosity and discovery configured in `pytest.ini`).

## Coding Style & Naming Conventions
- Python 3.10+; Ruff line length 100; prefer explicit imports and typed signatures (mypy is strict: no implicit Optional, no untyped defs).
- Use snake_case for modules/functions, PascalCase for classes; keep provider IDs consistent with registry keys (`openai`, `gemini`).
- Keep side effects out of import time; guard script entry with `if __name__ == "__main__":` when needed.

## Testing Guidelines
- Add or extend tests in `tests/` near the related module using the `test_*.py`/`Test*`/`test_*` pattern.
- Mock external APIs—reuse dummy env vars as in `tests/test_selector.py`; avoid live requests in CI.
- Cover new branching in provider selection, config defaults, and tool metadata (reasoning, confidence, alternatives) when relevant.

## Commit & Pull Request Guidelines
- Follow Conventional Commits as in history (`feat:`, `fix:`, `docs:`, `chore:`); keep subjects under ~72 characters.
- PRs should summarize intent, list testing (`ruff`, `mypy`, `pytest`), and note impacted MCP tools or endpoints.
- Link issues when applicable and include before/after examples or log snippets for behavior changes.

## Security & Configuration Tips
- Never commit API keys; load via env (`OPENAI_API_KEY`, `GEMINI_API_KEY`, optional `GOOGLE_API_KEY` alias).
- Prefer `output_path` overrides for local testing to avoid cluttering `~/Downloads/images/`.
- Avoid logging sensitive prompts or keys; remove debug prints before merging.
