"""
log_config.py — Centralized logging configuration.

Provides a pre-configured logger instance that formats console output according
to the project's coding standards. Supports filtering info vs debug messages.
"""

import logging
import sys

def setup_logger(name: str = "omr_pipeline", level: int = logging.INFO) -> logging.Logger:
    """Initialize and return a configured logger.

    Args:
        name: Name of the logger (defaults to "omr_pipeline").
        level: Logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers if setup_logger is called multiple times
    if not logger.handlers:
        logger.setLevel(level)
        logger.propagate = False
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        formatter = logging.Formatter(
            fmt="[%(levelname)s] %(asctime)s - %(message)s",
            datefmt="%H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        
    return logger

# Default logger instance to be imported across the project
logger = setup_logger()
