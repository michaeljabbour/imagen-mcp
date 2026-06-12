"""Tests for the minimal .env loader/writer (src/config/dotenv.py)."""

from __future__ import annotations

import os

from src.config.dotenv import (
    _parse_line,
    get_dotenv_path,
    get_project_root,
    load_dotenv,
    upsert_dotenv,
)


class TestParseLine:
    def test_basic_pair(self):
        assert _parse_line("KEY=value") == ("KEY", "value")

    def test_blank_and_comment_ignored(self):
        assert _parse_line("") is None
        assert _parse_line("   ") is None
        assert _parse_line("# a comment") is None

    def test_export_prefix_stripped(self):
        assert _parse_line("export FOO=bar") == ("FOO", "bar")

    def test_no_equals_ignored(self):
        assert _parse_line("NOTAPAIR") is None

    def test_empty_key_ignored(self):
        assert _parse_line("=value") is None

    def test_single_quoted_value_literal(self):
        # Single quotes do not process escapes.
        assert _parse_line(r"K='a\nb'") == ("K", r"a\nb")

    def test_double_quoted_value_unescapes(self):
        assert _parse_line(r'K="a\nb\t\"c\""') == ("K", 'a\nb\t"c"')


class TestLoadDotenv:
    def test_missing_file_returns_empty(self, tmp_path):
        assert load_dotenv(tmp_path / "nope.env") == {}

    def test_loads_without_override_by_default(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text("FOO=fromfile\nBAR=baz\n")
        monkeypatch.setenv("FOO", "preexisting")

        loaded = load_dotenv(env_file)

        assert loaded == {"FOO": "fromfile", "BAR": "baz"}
        # FOO already set → not overridden; BAR newly set.
        assert os.environ["FOO"] == "preexisting"
        assert os.environ["BAR"] == "baz"

    def test_override_true_replaces_existing(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text("FOO=fromfile\n")
        monkeypatch.setenv("FOO", "preexisting")

        load_dotenv(env_file, override=True)
        assert os.environ["FOO"] == "fromfile"


class TestUpsertDotenv:
    def test_creates_file_and_writes_pairs(self, tmp_path):
        path = tmp_path / "sub" / ".env"
        upsert_dotenv(path, {"A": "1", "B": "two"})

        text = path.read_text()
        assert 'A="1"' in text
        assert 'B="two"' in text
        # Round-trips back through the parser.
        loaded = load_dotenv(path)
        assert loaded["A"] == "1"
        assert loaded["B"] == "two"

    def test_updates_existing_key_preserves_comments(self, tmp_path):
        path = tmp_path / ".env"
        path.write_text("# header comment\nA=old\nB=keep\n")

        upsert_dotenv(path, {"A": "new"})

        text = path.read_text()
        assert "# header comment" in text
        assert 'A="new"' in text
        assert "B=keep" in text
        assert "A=old" not in text

    def test_empty_updates_is_noop(self, tmp_path):
        path = tmp_path / ".env"
        upsert_dotenv(path, {})
        assert not path.exists()

    def test_permissions_locked_down(self, tmp_path):
        path = tmp_path / ".env"
        upsert_dotenv(path, {"SECRET": "x"})
        mode = path.stat().st_mode & 0o777
        assert mode == 0o600


class TestProjectRoot:
    def test_root_contains_pyproject(self):
        root = get_project_root()
        assert (root / "pyproject.toml").is_file()

    def test_dotenv_path_under_root(self):
        assert get_dotenv_path() == get_project_root() / ".env"
