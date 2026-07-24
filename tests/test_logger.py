import os
import json
import logging
from logger import JSONFormatter, setup_logging

def test_json_formatter_outputs_valid_json():
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test_file.py",
        lineno=42,
        msg="Hello %s",
        args=("world",),
        exc_info=None
    )
    
    formatted_str = formatter.format(record)
    data = json.loads(formatted_str)
    
    assert data["name"] == "test_logger"
    assert data["level"] == "INFO"
    assert data["message"] == "Hello world"
    assert data["filename"] == "test_file.py"
    assert data["lineno"] == 42
    assert "timestamp" in data
    assert "exception" not in data

def test_json_formatter_includes_exception():
    formatter = JSONFormatter()
    try:
        raise ValueError("Something went wrong")
    except ValueError:
        import sys
        exc_info = sys.exc_info()
        
    record = logging.LogRecord(
        name="test_logger",
        level=logging.ERROR,
        pathname="test_file.py",
        lineno=100,
        msg="An error occurred",
        args=(),
        exc_info=exc_info
    )
    
    formatted_str = formatter.format(record)
    data = json.loads(formatted_str)
    
    assert data["message"] == "An error occurred"
    assert "exception" in data
    assert "ValueError: Something went wrong" in data["exception"]

def test_setup_logging_configures_formatter(monkeypatch):
    # Test plain text formatter configuration (default)
    monkeypatch.setenv("LOG_FORMAT", "text")
    setup_logging()
    
    root_logger = logging.getLogger()
    assert len(root_logger.handlers) == 1
    handler = root_logger.handlers[0]
    # Check that it's NOT using JSONFormatter
    assert not isinstance(handler.formatter, JSONFormatter)
    
    # Test JSON formatter configuration
    monkeypatch.setenv("LOG_FORMAT", "json")
    setup_logging()
    
    root_logger = logging.getLogger()
    assert len(root_logger.handlers) == 1
    handler = root_logger.handlers[0]
    assert isinstance(handler.formatter, JSONFormatter)
