"""
errors.py — Standardized error response utilities for DevWhisper.

This module provides helper functions to generate consistent JSON error
responses across all API endpoints. Using a centralized error format ensures
clients can reliably parse error details regardless of which endpoint fails.

Usage:
    from errors import error_response
    return error_response(400, "Invalid query parameter")
"""

from fastapi.responses import JSONResponse


def error_response(status_code: int, detail: str) -> JSONResponse:
    """
    Create a standardized JSON error response.

    Wraps the error detail in a consistent structure:
        { "error": { "status_code": <int>, "detail": <str> } }

    Args:
        status_code: HTTP status code (e.g., 400, 401, 500).
        detail: Human-readable error description.

    Returns:
        A FastAPI JSONResponse with the error payload and matching status code.
    """
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "status_code": status_code,
                "detail": detail,
            }
        },
    )
    
