import os
import logging
import json
from datetime import datetime, timezone

class JSONFormatter(logging.Formatter):
    """Custom formatter to output structured JSON logs."""
    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            "filename": record.filename,
            "lineno": record.lineno,
        }
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_record)

def setup_logging():
    """Configure root logger to use JSON or plain text logging."""
    log_format = os.getenv("LOG_FORMAT", "text").lower()
    root_logger = logging.getLogger()
    
    # Remove existing handlers to avoid duplicate log entries
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        
    handler = logging.StreamHandler()
    
    if log_format == "json":
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s"
        )
        
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

# Initialize logging when module is imported
setup_logging()
logger = logging.getLogger("devwhisper")
