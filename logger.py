"""
logger.py — Structured and colored logging for DevWhisper.

This module configures the application-wide logger with two formatter modes:

    1. JSONFormatter — Structured JSON logs for production (set via LOG_FORMAT=json env var).
    2. ColorfulFormatter — ANSI-colored plain text logs for local development readability.

Both formatters include timestamp, log level, logger name, filename, line number,
function name, and the log message. JSONFormatter additionally captures custom
extra fields and exception tracebacks.

Usage:
    from logger import logger
    logger.info("Server started")
    logger.error("Something broke", exc_info=True)
"""

import os
import logging
import json
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """
    Custom formatter that outputs structured JSON logs.

    Includes standard LogRecord fields plus any custom extra fields passed
    via logger calls (e.g., logger.info("msg", extra={"user_id": 42})).
    Exception tracebacks are serialized as structured data.
    """

    # Standard attributes of LogRecord to exclude from custom extra fields
    STANDARD_ATTRIBUTES = {
        'args', 'asctime', 'created', 'exc_info', 'exc_text', 'filename',
        'funcName', 'levelname', 'levelno', 'lineno', 'module', 'msecs',
        'message', 'msg', 'name', 'pathname', 'process', 'processName',
        'relativeCreated', 'stack_info', 'thread', 'threadName'
    }

    def format(self, record: logging.LogRecord) -> str:
        """
        Convert a LogRecord into a JSON string.

        Args:
            record: The logging.LogRecord to format.

        Returns:
            JSON-encoded log line string.
        """
        log_record = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            "filename": record.filename,
            "lineno": record.lineno,
            "function": record.funcName,
        }

        # Serialize exception details if present
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        # Serialize custom extra fields passed in logger calls (e.g. logger.info("msg", extra={...}))
        for key, val in record.__dict__.items():
            if key not in self.STANDARD_ATTRIBUTES and not key.startswith('_'):
                log_record[key] = val

        return json.dumps(log_record)


class ColorfulFormatter(logging.Formatter):
    """
    Custom formatter that adds ANSI colors to console logs.

    Useful for local development readability. Each log level gets a distinct
    color: DEBUG (grey), INFO (green), WARNING (yellow), ERROR (red), CRITICAL (bold red).
    """

    GREY = "\x1b[38;20m"
    GREEN = "\x1b[32;20m"
    YELLOW = "\x1b[33;20m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    RESET = "\x1b[0m"

    FORMATS = {
        logging.DEBUG: GREY + "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s" + RESET,
        logging.INFO: GREEN + "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s" + RESET,
        logging.WARNING: YELLOW + "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s" + RESET,
        logging.ERROR: RED + "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s" + RESET,
        logging.CRITICAL: BOLD_RED + "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s" + RESET,
    }

    def format(self, record: logging.LogRecord) -> str:
        """
        Apply color formatting based on log level.

        Args:
            record: The logging.LogRecord to format.

        Returns:
            Colorized log line string.
        """
        log_fmt = self.FORMATS.get(record.levelno, self.FORMATS[logging.INFO])
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def setup_logging() -> None:
    """
    Configure the root logger with the chosen formatter.

    Reads LOG_FORMAT environment variable ("json" or "text"). Defaults to "text".
    Removes any existing handlers to avoid duplicate log entries.
    """
    log_format = os.getenv("LOG_FORMAT", "text").lower()
    root_logger = logging.getLogger()

    # Remove existing handlers to avoid duplicate log entries
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    handler = logging.StreamHandler()

    if log_format == "json":
        formatter = JSONFormatter()
    else:
        formatter = ColorfulFormatter()

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)


# Initialize logging when module is imported
setup_logging()
logger = logging.getLogger("devwhisper")
"""Application logger instance. Use this for all logging in DevWhisper."""
