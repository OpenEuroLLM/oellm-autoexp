"""Utilities for extracting Megatron arguments from the Megatron-LM parser."""

from __future__ import annotations

import argparse
import logging

from argparse_schema import (
    ActionSpec,
    ArgMetadata,
    build_cmdline_args,
    extract_default_args,
    get_action_specs,
    get_arg_metadata,
)

# Backwards compatibility aliases
MegatronActionSpec = ActionSpec
MegatronArgMetadata = ArgMetadata

LOGGER = logging.getLogger(__name__)

_MegatronParser: argparse.ArgumentParser | None = None


def get_megatron_parser() -> argparse.ArgumentParser:
    global _MegatronParser
    if _MegatronParser is None:
        try:
            from megatron.training.arguments import add_megatron_arguments
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Megatron-LM not available. Install the [megatron] extra or add submodules/Megatron-LM to PYTHONPATH"
            ) from exc
        parser = argparse.ArgumentParser(description="Megatron-LM Arguments", allow_abbrev=False)
        _MegatronParser = add_megatron_arguments(parser)
    return _MegatronParser


__all__ = [
    "MegatronActionSpec",
    "MegatronArgMetadata",
    "build_cmdline_args",
    "extract_default_args",
    "get_action_specs",
    "get_arg_metadata",
    "get_megatron_parser",
]
