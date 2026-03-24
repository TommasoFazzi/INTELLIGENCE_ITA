"""Logging configuration for the intelligence system"""

import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logger(name: str, log_level: str = None) -> logging.Logger:
    """
    Setup and configure logger for the application.

    Args:
        name: Logger name (usually __name__)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger instance

    Environment variables:
        LOG_LEVEL: Logging level (default: INFO)
        LOG_FILE:  Path to rotating log file. If set, logs are written both to
                   console and to the file (10 MB max, 5 backups). Disabled if
                   the variable is absent or empty.
    """
    if log_level is None:
        log_level = os.getenv("LOG_LEVEL", "INFO")

    level = getattr(logging, log_level.upper())
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler — added only once per logger instance
    if not any(isinstance(h, logging.StreamHandler) and
               not isinstance(h, RotatingFileHandler)
               for h in logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Optional rotating file handler — activated by LOG_FILE env var
    log_file = os.getenv("LOG_FILE", "").strip()
    if log_file and not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
        try:
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10 MB per file
                backupCount=5,
                encoding="utf-8",
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except OSError as exc:
            # Don't crash the application if the log file path is invalid
            logger.warning(f"Could not open log file '{log_file}': {exc}")

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger instance."""
    return setup_logger(name)
