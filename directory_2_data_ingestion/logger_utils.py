"""
Logging utilities for the data ingestion pipeline.
"""

import logging
import os
from datetime import datetime
from typing import Optional

from config import config


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: str = "INFO"
) -> logging.Logger:
    """
    Set up logger with both file and console handlers.
    
    Args:
        name: Logger name
        log_file: Optional log file name
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(config.LOG_FORMAT)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = os.path.join(config.LOGS_DIR, log_file)
        file_handler = logging.FileHandler(log_path, mode='a')
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_ingestion_logger() -> logging.Logger:
    """Get logger for data ingestion with timestamp-based log file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"data_ingestion_{timestamp}.log"
    return setup_logger("data_ingestion", log_file, config.LOG_LEVEL)
