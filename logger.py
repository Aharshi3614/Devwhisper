import os
import logging
import json
from datetime import datetime, timezone

class JSONFormatter(logging.Formatter):
    """Custom formatter to output structured JSON logs with support for custom extra attributes."""
    
    # Standard attributes of LogRecord to exclude from custom extra fields
    STANDARD_ATTRIBUTES = {
        'args', 'asctime', 'created', 'exc_info', 'exc_text', 'filename',
        'funcName', 'levelname', 'levelno', 'lineno', 'module', 'msecs',
        'message', 'msg', 'name', 'pathname', 'process', 'processName',
        'relativeCreated', 'stack_info', 'thread', 'threadName'
    }

    def format(self, record: logging.LogRecord) -> str:
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
    """Custom formatter to add ANSI colors to console logs for local development readability."""
    
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
        log_fmt = self.FORMATS.get(record.levelno, self.FORMATS[logging.INFO])
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def setup_logging():
    """Configure root logger to use JSON or colorful plain text logging."""
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
