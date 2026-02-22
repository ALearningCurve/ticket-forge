"""shared: Shared utilities for logging, configuration, and common helpers."""

from shared.logging import configure_root_logger, get_logger

configure_root_logger()

__version__ = "0.1.0"
__all__ = ["get_logger"]
