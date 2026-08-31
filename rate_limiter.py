"""rate_limiter.py — Token bucket rate limiting and burst throttling middleware.

This module provides thread-safe token bucket rate limiting per IP address,
along with ASGI/BaseHTTPMiddleware integration for FastAPI.
"""

from __future__ import annotations

import time
import threading
from typing import Dict, Tuple, Optional, Set
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

import config


class TokenBucket:
    """Thread-safe Token Bucket implementation.

    Attributes:
        capacity: Maximum number of tokens the bucket can hold (burst limit).
        refill_rate: Rate at which tokens are added per second.
        tokens: Current number of available tokens.
        last_update: Timestamp of the last token refill calculation.
    """

    def __init__(self, capacity: float, refill_rate: float) -> None:
        self.capacity = float(capacity)
        self.refill_rate = float(refill_rate)
        self.tokens = float(capacity)
        self.last_update = time.monotonic()
        self.lock = threading.Lock()

    def _refill(self, now: float) -> None:
        """Refill tokens based on elapsed time since last_update."""
        elapsed = now - self.last_update
        if elapsed > 0:
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self.last_update = now

    def consume(self, tokens: float = 1.0) -> Tuple[bool, int, float]:
        """Attempt to consume `tokens` from the bucket.

        Args:
            tokens: Number of tokens to consume.

        Returns:
            Tuple of (allowed: bool, remaining_tokens: int, retry_after: float).
        """
        with self.lock:
            now = time.monotonic()
            self._refill(now)

            if self.tokens >= tokens:
                self.tokens -= tokens
                remaining = int(self.tokens)
                return True, remaining, 0.0
            else:
                needed = tokens - self.tokens
                retry_after = (needed / self.refill_rate) if self.refill_rate > 0 else 60.0
                remaining = int(self.tokens)
                return False, remaining, retry_after


class RateLimiter:
    """Per-IP Rate Limiter managing TokenBucket instances with stale bucket eviction.

    Attributes:
        rpm: Requests per minute limit.
        burst: Maximum burst capacity (bucket size).
        ttl_seconds: Idle duration after which a stale bucket is evicted.
    """

    def __init__(
        self,
        rpm: Optional[int] = None,
        burst: Optional[int] = None,
        ttl_seconds: float = 3600.0,
    ) -> None:
        self.rpm = rpm if rpm is not None else config.RATE_LIMIT_RPM
        self.burst = burst if burst is not None else config.RATE_LIMIT_BURST
        self.refill_rate = self.rpm / 60.0
        self.ttl_seconds = ttl_seconds
        self.buckets: Dict[str, TokenBucket] = {}
        self.last_accessed: Dict[str, float] = {}
        self.lock = threading.Lock()

    def _cleanup_stale_buckets(self, now: float) -> None:
        """Evict buckets that have not been accessed within ttl_seconds."""
        stale_ips = [
            ip for ip, last_time in self.last_accessed.items()
            if now - last_time > self.ttl_seconds
        ]
        for ip in stale_ips:
            self.buckets.pop(ip, None)
            self.last_accessed.pop(ip, None)

    def get_bucket(self, client_ip: str) -> TokenBucket:
        """Retrieve or create a TokenBucket for the given client IP address."""
        with self.lock:
            now = time.monotonic()
            self._cleanup_stale_buckets(now)

            if client_ip not in self.buckets:
                self.buckets[client_ip] = TokenBucket(
                    capacity=self.burst,
                    refill_rate=self.refill_rate,
                )
            self.last_accessed[client_ip] = now
            return self.buckets[client_ip]

    def check_rate_limit(self, client_ip: str) -> Tuple[bool, int, float, int]:
        """Check rate limit for client IP.

        Returns:
            Tuple of (allowed: bool, remaining_tokens: int, retry_after: float, limit: int).
        """
        bucket = self.get_bucket(client_ip)
        allowed, remaining, retry_after = bucket.consume(1.0)
        return allowed, remaining, retry_after, int(self.burst)


# Global RateLimiter instance
default_rate_limiter = RateLimiter()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """ASGI Middleware enforcing token bucket rate limiting and burst throttling."""

    def __init__(
        self,
        app,
        rate_limiter: Optional[RateLimiter] = None,
        enabled: Optional[bool] = None,
        exempt_paths: Optional[Set[str]] = None,
    ) -> None:
        super().__init__(app)
        self.rate_limiter = rate_limiter or default_rate_limiter
        self.enabled = enabled
        self.exempt_paths = (
            exempt_paths if exempt_paths is not None else config.RATE_LIMIT_EXEMPT_PATHS
        )

    def is_exempt(self, path: str) -> bool:
        """Check if request path is exempt from rate limiting."""
        if path in self.exempt_paths:
            return True
        for exempt in self.exempt_paths:
            if exempt != "/" and path.startswith(exempt):
                return True
        return False

    def get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request headers or connection client."""
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        if request.client and request.client.host:
            return request.client.host
        return "127.0.0.1"

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        is_enabled = self.enabled if self.enabled is not None else getattr(config, "RATE_LIMIT_ENABLED", True)
        if not is_enabled:
            return await call_next(request)

        path = request.url.path
        if self.is_exempt(path):
            return await call_next(request)

        client_ip = self.get_client_ip(request)
        allowed, remaining, retry_after, limit = self.rate_limiter.check_rate_limit(client_ip)

        if not allowed:
            retry_seconds = max(1, int(retry_after) if retry_after > 0 else 1)
            headers = {
                "X-RateLimit-Limit": str(limit),
                "X-RateLimit-Remaining": str(max(0, remaining)),
                "Retry-After": str(retry_seconds),
            }
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Too many requests."},
                headers=headers,
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(max(0, remaining))
        return response
