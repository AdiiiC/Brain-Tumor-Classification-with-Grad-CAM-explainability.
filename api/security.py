"""
API key authentication and rate limiting.

Both are configured via environment variables so the same image can run
open (local dev) or locked down (production) without code changes.
"""

from __future__ import annotations

import hmac
import os
import time
from collections import defaultdict, deque
from threading import Lock

from fastapi import Depends, HTTPException, Request
from fastapi.security import APIKeyHeader

API_KEY_HEADER = "X-API-Key"

_api_key_scheme = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)


def _load_api_keys() -> set[str]:
    raw = os.getenv("API_KEYS", "")
    return {key.strip() for key in raw.split(",") if key.strip()}


def auth_enabled() -> bool:
    return bool(_load_api_keys())


async def require_api_key(api_key: str | None = Depends(_api_key_scheme)) -> str | None:
    """
    Validate the API key when any are configured.

    If API_KEYS is unset the API stays open, which keeps local development
    friction-free. Set API_KEYS in every deployed environment.
    """
    valid_keys = _load_api_keys()
    if not valid_keys:
        return None

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail=f"Missing API key. Supply it in the {API_KEY_HEADER} header.",
        )

    # compare_digest avoids leaking key material through timing differences
    if not any(hmac.compare_digest(api_key, valid) for valid in valid_keys):
        raise HTTPException(status_code=403, detail="Invalid API key.")

    return api_key


class SlidingWindowRateLimiter:
    """
    In-process sliding-window limiter.

    Adequate for a single-worker deployment. Multi-worker or multi-instance
    setups need a shared backend (Redis) instead.
    """

    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def check(self, identity: str) -> tuple[bool, int]:
        """Record a hit. Returns (allowed, seconds_until_reset)."""
        now = time.monotonic()
        cutoff = now - self.window_seconds

        with self._lock:
            hits = self._hits[identity]
            while hits and hits[0] < cutoff:
                hits.popleft()

            if len(hits) >= self.max_requests:
                return False, max(1, int(hits[0] + self.window_seconds - now))

            hits.append(now)
            return True, 0

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


def _client_identity(request: Request) -> str:
    """Prefer the API key, fall back to client IP."""
    api_key = request.headers.get(API_KEY_HEADER)
    if api_key:
        return f"key:{api_key}"
    client = request.client
    return f"ip:{client.host if client else 'unknown'}"


class RateLimit:
    """
    Dependency factory for per-endpoint limits.

    Heavy endpoints (comprehensive analysis, patch scanning) get tighter
    budgets than a plain health check.
    """

    def __init__(self, max_requests: int, window_seconds: int = 60, name: str = "default"):
        self.name = name
        self.limiter = SlidingWindowRateLimiter(max_requests, window_seconds)
        _REGISTRY[name] = self.limiter

    async def __call__(self, request: Request) -> None:
        if os.getenv("RATE_LIMIT_ENABLED", "true").lower() in ("false", "0", "no"):
            return

        identity = f"{self.name}:{_client_identity(request)}"
        allowed, retry_after = self.limiter.check(identity)
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please retry shortly.",
                headers={"Retry-After": str(retry_after)},
            )


_REGISTRY: dict[str, SlidingWindowRateLimiter] = {}


def reset_all_limiters() -> None:
    """Clear limiter state — used between tests."""
    for limiter in _REGISTRY.values():
        limiter.reset()


# Shared limit tiers.
standard_rate_limit = RateLimit(max_requests=60, window_seconds=60, name="standard")
heavy_rate_limit = RateLimit(max_requests=10, window_seconds=60, name="heavy")
batch_rate_limit = RateLimit(max_requests=5, window_seconds=60, name="batch")
