"""errors.py — Standardized error response utilities for DevWhisper.

This module provides helper functions and models to generate consistent JSON error
responses across all API endpoints. Using a centralized error format ensures
clients can reliably parse error details regardless of which endpoint fails.

Expected error shape:
    {
        "status": "error",
        "code": <int>,
        "message": <str>
    }

Usage:
    from errors import ErrorResponse, error_response
    return error_response(400, "Invalid query parameter")
"""

from pydantic import BaseModel
from fastapi.responses import JSONResponse


class ErrorResponse(BaseModel):
    """Pydantic model for a standardized API error response."""

    status: str = "error"
    code: int
    message: str


def error_response(status_code: int, detail: str) -> JSONResponse:
    """
    Create a standardized JSON error response.

    Returns a flat structure matching the ErrorResponse schema:
    { "status": "error", "code": <status_code>, "message": <detail> }

    Args:
        status_code: HTTP status code (e.g., 400, 401, 500).
        detail: Human-readable error description.

    Returns:
        A FastAPI JSONResponse with the error payload and matching status code.
    """
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "error",
            "code": status_code,
            "message": detail,
        },
    )
