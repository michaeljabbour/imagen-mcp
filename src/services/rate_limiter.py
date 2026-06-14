"""
Simple client-side rate limiter for API calls.

Prevents hitting rate limits by enforcing minimum intervals
between requests per provider.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting a provider."""

    requests_per_minute: int = 10
    min_interval_seconds: float = 1.0
    burst_limit: int = 3  # Max requests in quick succession


@dataclass
class ProviderState:
    """Tracks rate limit state for a provider."""

    last_request_time: float = 0.0
    request_times: list[float] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


# Default configs per provider (fallback when no settings-derived config is
# supplied). Tuned for parallelism — small interval, real burst allowance.
DEFAULT_CONFIGS: dict[str, RateLimitConfig] = {
    "openai": RateLimitConfig(
        requests_per_minute=10,
        min_interval_seconds=0.5,
        burst_limit=5,
    ),
    "gemini": RateLimitConfig(
        requests_per_minute=15,
        min_interval_seconds=0.5,
        burst_limit=5,
    ),
}


class RateLimiter:
    """
    Simple token bucket rate limiter.

    Enforces rate limits per provider to avoid 429 errors.
    """

    def __init__(self, configs: dict[str, RateLimitConfig] | None = None):
        """
        Initialize rate limiter.

        Args:
            configs: Provider-specific rate limit configs.
                     Defaults to DEFAULT_CONFIGS.
        """
        self.configs = configs or DEFAULT_CONFIGS
        self._states: dict[str, ProviderState] = {}

    def _get_state(self, provider: str) -> ProviderState:
        """Get or create state for a provider."""
        if provider not in self._states:
            self._states[provider] = ProviderState()
        return self._states[provider]

    def _get_config(self, provider: str) -> RateLimitConfig:
        """Get config for a provider."""
        return self.configs.get(provider, RateLimitConfig())

    async def acquire(self, provider: str) -> None:
        """
        Acquire permission to make a request.

        Blocks until the request is allowed under rate limits.

        Args:
            provider: Provider name (openai, gemini)
        """
        state = self._get_state(provider)
        config = self._get_config(provider)

        # Compute the required wait AND reserve the slot under the lock, then
        # release the lock BEFORE sleeping. Holding the lock across the sleep
        # (the previous behavior) turned the limiter into a global mutex that
        # serialized every request — the opposite of what we want for batch
        # parallelism. Reserving a future timestamp lets concurrent callers
        # stagger themselves without blocking each other.
        async with state.lock:
            now = time.time()

            # Clean old request times (keep last minute)
            cutoff = now - 60
            state.request_times = [t for t in state.request_times if t > cutoff]

            wait_time = 0.0

            # Requests-per-minute window
            if len(state.request_times) >= config.requests_per_minute:
                oldest = state.request_times[0]
                wait_time = max(wait_time, 60 - (now - oldest) + 0.1)

            # Minimum interval between requests
            time_since_last = now - state.last_request_time
            if time_since_last < config.min_interval_seconds:
                wait_time = max(wait_time, config.min_interval_seconds - time_since_last)

            # Burst limit (requests in the last 5 seconds)
            recent_cutoff = now - 5
            recent_requests = sum(1 for t in state.request_times if t > recent_cutoff)
            if recent_requests >= config.burst_limit:
                wait_time = max(wait_time, 2.0)

            # Reserve this request at its projected start time so the next
            # caller schedules itself after us (staggering, not serializing).
            reserved = now + wait_time
            state.last_request_time = reserved
            state.request_times.append(reserved)

        if wait_time > 0:
            logger.debug("Rate limit: %s waiting %.2fs before request", provider, wait_time)
            await asyncio.sleep(wait_time)


# Singleton instance
_limiter: RateLimiter | None = None


def get_rate_limiter() -> RateLimiter:
    """Get the singleton rate limiter instance, configured from settings."""
    global _limiter
    if _limiter is None:
        from ..config.settings import get_settings

        s = get_settings()
        configs = {
            "openai": RateLimitConfig(
                requests_per_minute=s.openai_rpm,
                min_interval_seconds=s.openai_min_interval_seconds,
                burst_limit=s.openai_burst_limit,
            ),
            "gemini": RateLimitConfig(
                requests_per_minute=s.gemini_rpm,
                min_interval_seconds=s.gemini_min_interval_seconds,
                burst_limit=s.gemini_burst_limit,
            ),
        }
        _limiter = RateLimiter(configs)
    return _limiter
