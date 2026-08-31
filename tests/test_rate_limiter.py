"""tests/test_rate_limiter.py — Test suite for token bucket rate limiter and middleware.

Tests TokenBucket, RateLimiter, and FastAPI RateLimitMiddleware integration.
"""

import time
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
import config
from rate_limiter import TokenBucket, RateLimiter, RateLimitMiddleware


def test_token_bucket_initial_capacity():
    bucket = TokenBucket(capacity=5, refill_rate=1.0)
    assert bucket.tokens == 5.0
    allowed, remaining, retry_after = bucket.consume(1)
    assert allowed is True
    assert remaining == 4
    assert retry_after == 0.0


def test_token_bucket_exhaustion_and_retry_after():
    bucket = TokenBucket(capacity=2, refill_rate=1.0)
    assert bucket.consume(1)[0] is True
    assert bucket.consume(1)[0] is True
    allowed, remaining, retry_after = bucket.consume(1)
    assert allowed is False
    assert remaining == 0
    assert retry_after > 0.0


def test_token_bucket_replenishment():
    bucket = TokenBucket(capacity=2, refill_rate=10.0)
    bucket.consume(2)
    assert bucket.consume(1)[0] is False
    time.sleep(0.15)
    allowed, remaining, retry_after = bucket.consume(1)
    assert allowed is True


def test_rate_limiter_per_ip_isolation():
    limiter = RateLimiter(rpm=60, burst=1)
    allowed1, remaining1, _, _ = limiter.check_rate_limit("1.1.1.1")
    assert allowed1 is True

    allowed1_again, _, _, _ = limiter.check_rate_limit("1.1.1.1")
    assert allowed1_again is False

    allowed2, remaining2, _, _ = limiter.check_rate_limit("2.2.2.2")
    assert allowed2 is True


def test_rate_limiter_stale_bucket_eviction():
    limiter = RateLimiter(rpm=60, burst=5, ttl_seconds=0.05)
    limiter.get_bucket("1.1.1.1")
    assert "1.1.1.1" in limiter.buckets

    time.sleep(0.1)
    limiter.get_bucket("2.2.2.2")
    assert "1.1.1.1" not in limiter.buckets
    assert "2.2.2.2" in limiter.buckets


def test_middleware_fastapi_integration():
    test_app = FastAPI()
    limiter = RateLimiter(rpm=60, burst=2)
    test_app.add_middleware(
        RateLimitMiddleware,
        rate_limiter=limiter,
        enabled=True,
        exempt_paths={"/health"},
    )

    @test_app.get("/health")
    def health():
        return {"status": "ok"}

    @test_app.get("/api/test")
    def api_test():
        return {"data": "test"}

    client = TestClient(test_app)

    # Health check is exempt
    resp = client.get("/health")
    assert resp.status_code == 200
    assert "X-RateLimit-Limit" not in resp.headers

    # First API request
    resp1 = client.get("/api/test")
    assert resp1.status_code == 200
    assert resp1.headers["X-RateLimit-Limit"] == "2"
    assert resp1.headers["X-RateLimit-Remaining"] == "1"

    # Second API request (exhausts capacity 2)
    resp2 = client.get("/api/test")
    assert resp2.status_code == 200
    assert resp2.headers["X-RateLimit-Remaining"] == "0"

    # Third API request (exceeds capacity -> 429)
    resp3 = client.get("/api/test")
    assert resp3.status_code == 429
    assert resp3.headers["X-RateLimit-Limit"] == "2"
    assert "Retry-After" in resp3.headers
    assert resp3.json()["detail"] == "Rate limit exceeded. Too many requests."


def test_middleware_disabled_toggle():
    test_app = FastAPI()
    limiter = RateLimiter(rpm=60, burst=1)
    test_app.add_middleware(
        RateLimitMiddleware,
        rate_limiter=limiter,
        enabled=False,
    )

    @test_app.get("/api/test")
    def api_test():
        return {"data": "test"}

    client = TestClient(test_app)
    # Even multiple requests pass when disabled
    assert client.get("/api/test").status_code == 200
    assert client.get("/api/test").status_code == 200
