"""errors.py — Standardized error response and control flow utilities for DevWhisper.

This module provides helper functions, Pydantic models, and decorators to generate
consistent JSON error responses and standardize backend execution flows across all API
endpoints and services.

Expected error shape:
    {
        "status": "error",
        "code": <int>,
        "message": <str>
    }

Usage:
    from errors import ErrorResponse, error_response, safe_execute
    return error_response(400, "Invalid query parameter")
"""

from pydantic import BaseModel
from fastapi.responses import JSONResponse
from functools import wraps
from logger import logger


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


def safe_execute(error_message: str = "An unexpected error occurred", default_return=None):
    """
    Standardized execution decorator to unify error handling and control flow 
    across backend processing modules.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error("Error in %s: %s", func.__name__, e, exc_info=True)
                return default_return
        return wrapper
    return decorator
