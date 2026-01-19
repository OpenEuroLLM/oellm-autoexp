"""Utilities for extracting Megatron arguments from the Megatron-LM parser."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

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


def _maybe_add_submodule_to_path() -> None:
    """Ensure the bundled Megatron-LM submodule is importable."""

    root = Path(__file__).resolve().parents[2]
    candidate = root / "submodules" / "Megatron-LM"
    if candidate.exists():
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


def get_megatron_parser() -> argparse.ArgumentParser:
    global _MegatronParser
    if _MegatronParser is None:
        try:
            from megatron.training.arguments import add_megatron_arguments
        except ImportError:  # pragma: no cover - optional dependency
            _maybe_add_submodule_to_path()
            try:
                from megatron.training.arguments import add_megatron_arguments
            except ImportError as exc_inner:  # pragma: no cover - optional dependency
                raise ImportError(
                    "Megatron-LM not available. Install the [megatron] extra or add submodules/Megatron-LM"
                ) from exc_inner
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
