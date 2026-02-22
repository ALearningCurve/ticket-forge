"""Centralized logging configuration for ticket-forge."""

import logging
import sys
from typing import Literal

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

# Default format for log messages
DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: LogLevel = "INFO") -> logging.Logger:
  """Get a configured logger for the given module name.

  Args:
      name: The name of the logger, typically __name__ from the calling module.
      level: The logging level, defaults to INFO.

  Returns:
      A configured Logger instance.
  """
  logger = logging.getLogger(name)

  # Only configure if not already configured
  if not logger.handlers:
    logger.setLevel(getattr(logging, level))

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level))

    formatter = logging.Formatter(DEFAULT_FORMAT, datefmt=DEFAULT_DATE_FORMAT)
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = False

  return logger


def configure_root_logger(level: LogLevel = "INFO") -> None:
  """Configure the root logger for the application.

  Call this once at application startup.

  Args:
      level: Logging level for root logger
  """
  logging.basicConfig(
    level=getattr(logging, level),
    format=DEFAULT_FORMAT,
    datefmt=DEFAULT_DATE_FORMAT,
    stream=sys.stdout,
  )
