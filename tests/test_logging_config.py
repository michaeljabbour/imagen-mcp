"""Tests for logging configuration (src/services/logging_config.py)."""

from __future__ import annotations

import json
import logging

import pytest

from src.services import logging_config


@pytest.fixture(autouse=True)
def reset_logging_state(monkeypatch):
    """Reset the module-level configuration guard and root handlers.

    ``configure_logging`` is idempotent via a module global, and it attaches
    handlers to the root logger. Reset both so each test starts clean and
    writes into its own (tmp) log directory.
    """
    monkeypatch.setattr(logging_config, "_CONFIGURED", False)
    root = logging.getLogger()
    saved = root.handlers[:]
    events = logging.getLogger("imagen_mcp.events")
    saved_events = events.handlers[:]
    root.handlers.clear()
    events.handlers.clear()
    yield
    root.handlers[:] = saved
    events.handlers[:] = saved_events


def test_configure_logging_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    from src.config.settings import get_settings

    get_settings.cache_clear()

    logging_config.configure_logging()
    root = logging.getLogger()
    count_after_first = len(root.handlers)

    logging_config.configure_logging()  # second call should add nothing
    assert len(root.handlers) == count_after_first


def test_log_event_writes_jsonl(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    from src.config.settings import get_settings

    get_settings.cache_clear()

    logging_config.log_event("test.event", request_id="abc", count=3)

    # Flush handlers so the file is written.
    for handler in logging.getLogger("imagen_mcp.events").handlers:
        handler.flush()

    events_file = tmp_path / "logs" / "events.jsonl"
    assert events_file.is_file()

    lines = [line for line in events_file.read_text().splitlines() if line.strip()]
    payload = json.loads(lines[-1])
    assert payload["event"] == "test.event"
    assert payload["request_id"] == "abc"
    assert payload["count"] == 3
    assert "timestamp" in payload


def test_log_event_serializes_non_json_safe_values(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    from pathlib import Path

    from src.config.settings import get_settings

    get_settings.cache_clear()

    # Path is not JSON-serializable by default; default=str must handle it.
    logging_config.log_event("path.event", path=Path("/tmp/x.png"))
    for handler in logging.getLogger("imagen_mcp.events").handlers:
        handler.flush()

    events_file = tmp_path / "logs" / "events.jsonl"
    payload = json.loads(events_file.read_text().splitlines()[-1])
    assert payload["path"].endswith("x.png")
