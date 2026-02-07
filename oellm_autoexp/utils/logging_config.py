"""Centralized logging configuration with environment variable support."""

from __future__ import annotations

import logging
import os
from typing import Literal


LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def get_log_level_from_env(
    default: int = logging.INFO,
    env_var: str = "OELLM_LOG_LEVEL",
) -> int:
    """Get log level from environment variable.

    Args:
        default: Default log level if environment variable is not set
        env_var: Name of environment variable to read (default: OELLM_LOG_LEVEL)

    Returns:
        Logging level constant (e.g., logging.DEBUG, logging.INFO)

    Environment Variable:
        OELLM_LOG_LEVEL: Can be set to DEBUG, INFO, WARNING, ERROR, or CRITICAL

    Examples:
        export OELLM_LOG_LEVEL=DEBUG
        export OELLM_LOG_LEVEL=INFO
        export OELLM_LOG_LEVEL=WARNING
    """
    level_str = os.getenv(env_var, "").upper()

    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    return level_map.get(level_str, default)


def configure_logging(
    verbose: bool = False,
    debug: bool = False,
    level: int | None = None,
    format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
    respect_env: bool = True,
) -> None:
    """Configure logging with optional environment variable support.

    Priority (highest to lowest):
        1. Explicit `level` parameter
        2. `debug` flag (if True)
        3. `verbose` flag (if True)
        4. Environment variable OELLM_LOG_LEVEL (if set and respect_env=True)
        5. Default (WARNING)

    Args:
        verbose: If True, set level to INFO (unless overridden)
        debug: If True, set level to DEBUG (unless overridden)
        level: Explicit log level (overrides all other settings)
        format: Log format string
        datefmt: Date format string
        respect_env: If True, check OELLM_LOG_LEVEL environment variable

    Examples:
        # Use environment variable (if set)
        configure_logging()  # Respects OELLM_LOG_LEVEL

        # Override with debug flag (takes precedence over environment)
        configure_logging(debug=True)

        # Explicit level (highest priority)
        configure_logging(level=logging.WARNING)

        # Ignore environment variable
        configure_logging(verbose=True, respect_env=False)
    """
    # Determine log level based on priority
    if level is not None:
        # Explicit level has highest priority
        final_level = level
    elif debug:
        # debug flag has second priority
        final_level = logging.DEBUG
    elif verbose:
        # verbose flag has third priority
        final_level = logging.INFO
    elif respect_env:
        # Environment variable has fourth priority
        final_level = get_log_level_from_env(default=logging.WARNING)
    else:
        # Default fallback
        final_level = logging.WARNING

    logging.basicConfig(
        level=final_level,
        format=format,
        datefmt=datefmt,
        force=True,  # Reconfigure if already configured
    )


__all__ = ["configure_logging", "get_log_level_from_env", "LogLevel"]
