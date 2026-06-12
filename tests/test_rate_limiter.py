"""Tests for the client-side rate limiter (src/services/rate_limiter.py)."""

from __future__ import annotations

import pytest

from src.services.rate_limiter import (
    DEFAULT_CONFIGS,
    RateLimitConfig,
    RateLimiter,
    get_rate_limiter,
)


@pytest.fixture
def fast_sleeps(monkeypatch):
    """Replace asyncio.sleep with a recorder so tests never actually wait."""
    waits: list[float] = []

    async def _fake_sleep(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr("src.services.rate_limiter.asyncio.sleep", _fake_sleep)
    return waits


class TestRateLimiter:
    @pytest.mark.asyncio
    async def test_first_request_does_not_wait(self, fast_sleeps):
        limiter = RateLimiter({"openai": RateLimitConfig(min_interval_seconds=2.0)})
        await limiter.acquire("openai")
        assert fast_sleeps == []

    @pytest.mark.asyncio
    async def test_min_interval_enforced(self, fast_sleeps):
        limiter = RateLimiter({"openai": RateLimitConfig(min_interval_seconds=5.0, burst_limit=99)})
        await limiter.acquire("openai")
        await limiter.acquire("openai")
        # Second call should have triggered a min-interval wait.
        assert any(w > 0 for w in fast_sleeps)

    @pytest.mark.asyncio
    async def test_requests_per_minute_limit(self, fast_sleeps):
        limiter = RateLimiter(
            {
                "openai": RateLimitConfig(
                    requests_per_minute=2, min_interval_seconds=0.0, burst_limit=99
                )
            }
        )
        await limiter.acquire("openai")
        await limiter.acquire("openai")
        await limiter.acquire("openai")  # exceeds 2/min → waits
        assert any(w > 0 for w in fast_sleeps)

    @pytest.mark.asyncio
    async def test_burst_limit_triggers_pause(self, fast_sleeps):
        limiter = RateLimiter(
            {
                "openai": RateLimitConfig(
                    requests_per_minute=99, min_interval_seconds=0.0, burst_limit=2
                )
            }
        )
        await limiter.acquire("openai")
        await limiter.acquire("openai")
        await limiter.acquire("openai")  # 3rd in burst window → pause
        assert 2.0 in fast_sleeps

    @pytest.mark.asyncio
    async def test_unknown_provider_uses_default_config(self, fast_sleeps):
        limiter = RateLimiter({})
        # Should not raise; falls back to RateLimitConfig() defaults.
        await limiter.acquire("mystery")
        # First request never waits, even on the default config.
        assert fast_sleeps == []

    def test_default_configs_present(self):
        assert "openai" in DEFAULT_CONFIGS
        assert "gemini" in DEFAULT_CONFIGS

    def test_singleton_is_stable(self):
        assert get_rate_limiter() is get_rate_limiter()
