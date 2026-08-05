"""
Tests for Issue #225 — Request Correlation IDs.

Verifies that:
  1. Every response has an X-Request-ID header.
  2. The ID is a valid UUID when not supplied by the client.
  3. A client-supplied X-Request-ID header is echoed back.
  4. Different requests get different IDs.
  5. The logger includes request_id in JSON log output.
"""

import sys
import os
import uuid
import json
import logging
from unittest.mock import MagicMock

# Stub heavy deps so `import main` doesn't fail
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

for mod_name in [
    "qdrant_client", "qdrant_client.models", "sentence_transformers",
    "rank_bm25", "groq", "transformers", "torch", "openai", "pathspec",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

from fastapi.testclient import TestClient
from logger import JSONFormatter, ColorfulFormatter
import main


client = TestClient(main.app)


def test_response_has_request_id_header():
    """Every response must include an X-Request-ID header."""
    response = client.get("/health")
    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
    assert response.headers["X-Request-ID"] != ""


def test_generated_request_id_is_valid_uuid():
    """When no X-Request-ID header is supplied, the server must generate a valid UUID4."""
    response = client.get("/health")
    rid = response.headers["X-Request-ID"]
    # Should not raise ValueError if valid UUID
    parsed = uuid.UUID(rid)
    assert parsed.version == 4


def test_client_supplied_request_id_is_echoed():
    """A client-supplied X-Request-ID must be echoed back in the response."""
    custom_id = "my-custom-request-id-12345"
    response = client.get("/health", headers={"X-Request-ID": custom_id})
    assert response.headers["X-Request-ID"] == custom_id


def test_different_requests_get_different_ids():
    """Two consecutive requests without a supplied ID must get different IDs."""
    r1 = client.get("/health")
    r2 = client.get("/health")
    assert r1.headers["X-Request-ID"] != r2.headers["X-Request-ID"]


def test_json_logger_includes_request_id():
    """The JSONFormatter must include request_id in the log output when set."""
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test_file.py",
        lineno=42,
        msg="Test message with request ID",
        args=(),
        exc_info=None,
    )
    record.__dict__["request_id"] = "abc-123-def"

    formatted = formatter.format(record)
    data = json.loads(formatted)

    assert data["request_id"] == "abc-123-def"
    assert data["message"] == "Test message with request ID"


def test_json_logger_omits_request_id_when_not_set():
    """The JSONFormatter must NOT include request_id when it's not set."""
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test_file.py",
        lineno=42,
        msg="Test message without request ID",
        args=(),
        exc_info=None,
    )

    formatted = formatter.format(record)
    data = json.loads(formatted)

    assert "request_id" not in data


def test_colorful_formatter_includes_request_id():
    """The ColorfulFormatter must prepend [req:xxxx] when request_id is set."""
    formatter = ColorfulFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test_file.py",
        lineno=42,
        msg="Test message",
        args=(),
        exc_info=None,
    )
    record.__dict__["request_id"] = "abc-123"

    formatted = formatter.format(record)
    assert "[req:abc-123]" in formatted


def test_colorful_formatter_omits_request_id_when_not_set():
    """The ColorfulFormatter must NOT prepend [req:xxxx] when request_id is absent."""
    formatter = ColorfulFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test_file.py",
        lineno=42,
        msg="Test message",
        args=(),
        exc_info=None,
    )

    formatted = formatter.format(record)
    assert "[req:" not in formatted
