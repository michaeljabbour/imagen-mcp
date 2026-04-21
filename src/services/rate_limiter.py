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


# Default configs per provider
DEFAULT_CONFIGS: dict[str, RateLimitConfig] = {
    "openai": RateLimitConfig(
        requests_per_minute=10,
        min_interval_seconds=2.0,  # OpenAI is slower, space out more
        burst_limit=2,
    ),
    "gemini": RateLimitConfig(
        requests_per_minute=15,
        min_interval_seconds=1.0,
        burst_limit=3,
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

        async with state.lock:
            now = time.time()

            # Clean old request times (keep last minute)
            cutoff = now - 60
            state.request_times = [t for t in state.request_times if t > cutoff]

            # Check requests per minute
            if len(state.request_times) >= config.requests_per_minute:
                # Wait until oldest request falls out of window
                oldest = state.request_times[0]
                wait_time = 60 - (now - oldest) + 0.1
                if wait_time > 0:
                    logger.info(
                        f"Rate limit: waiting {wait_time:.1f}s for {provider} "
                        f"(hit {config.requests_per_minute}/min limit)"
                    )
                    await asyncio.sleep(wait_time)
                    now = time.time()

            # Check minimum interval
            time_since_last = now - state.last_request_time
            if time_since_last < config.min_interval_seconds:
                wait_time = config.min_interval_seconds - time_since_last
                logger.debug(f"Rate limit: waiting {wait_time:.2f}s for min interval")
                await asyncio.sleep(wait_time)
                now = time.time()

            # Check burst limit (requests in last 5 seconds)
            recent_cutoff = now - 5
            recent_requests = sum(1 for t in state.request_times if t > recent_cutoff)
            if recent_requests >= config.burst_limit:
                wait_time = 2.0  # Brief pause to avoid bursting
                logger.debug(f"Rate limit: burst pause {wait_time}s for {provider}")
                await asyncio.sleep(wait_time)
                now = time.time()

            # Record this request
            state.last_request_time = now
            state.request_times.append(now)


# Singleton instance
_limiter: RateLimiter | None = None


def get_rate_limiter() -> RateLimiter:
    """Get the singleton rate limiter instance."""
    global _limiter
    if _limiter is None:
        _limiter = RateLimiter()
    return _limiter
