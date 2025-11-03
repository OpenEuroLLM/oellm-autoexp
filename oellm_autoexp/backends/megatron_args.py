"""Utilities for extracting Megatron arguments from the Megatron-LM parser."""

from __future__ import annotations

import argparse
import sys
from argparse import _StoreConstAction, _StoreFalseAction, _StoreTrueAction
from dataclasses import dataclass, field, MISSING
from enum import Enum
from pathlib import Path
from typing import Any
from collections.abc import Iterable, Mapping

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


@dataclass(kw_only=True)
class MegatronArgMetadata:
    arg_type: type | None = None
    default: Any = field(default_factory=MISSING)
    help: str | None = None
    choices: tuple[Any, ...] | None = None
    nargs: str | int | None = None
    element_type: type | None = None


def get_arg_metadata(parser: argparse.ArgumentParser) -> dict[str, MegatronArgMetadata]:
    metadata: dict[str, MegatronArgMetadata] = {}
    for action in parser._actions:  # noqa: SLF001
        if not action.dest:
            continue
        choices = None
        if action.choices is not None:
            normalized: list[Any] = []
            for option in action.choices:
                if isinstance(option, Enum):
                    normalized.append(getattr(option, "value", option.name))
                else:
                    normalized.append(option)
            choices = tuple(normalized)
        element_type: type | None = None
        if action.nargs in {"+", "*"}:
            element_type = action.type or str
        metadata[action.dest] = MegatronArgMetadata(
            arg_type=_extract_action_type(action),
            default=action.default,
            help=action.help or None,
            choices=choices,
            nargs=action.nargs,
            element_type=element_type,
        )
    return metadata


def extract_default_args(
    parser: argparse.ArgumentParser,
    exclude: Iterable[str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a mapping of Megatron argument defaults with optional
    overrides."""

    namespace = parser.parse_args(args=[])
    exclude_set = {item for item in (exclude or []) if item}
    overrides = dict(overrides or {})

    defaults: dict[str, Any] = {}
    for action in parser._actions:  # noqa: SLF001
        dest = getattr(action, "dest", None)
        if not dest or dest == "help" or dest in exclude_set:
            continue
        if dest in overrides:
            value = overrides[dest]
        else:
            value = getattr(namespace, dest, None)
        if isinstance(value, Enum):
            value = str(value).split(".")[-1]
        defaults[dest] = value
    return defaults


def build_cmdline_args(
    args: dict[str, Any],
    parser: argparse.ArgumentParser,
    metadata: dict[str, MegatronArgMetadata],
) -> list[str]:
    coerced = _coerce_arguments(args, metadata)
    cli_args: list[str] = []
    for key, value in coerced.items():
        cli_args.extend(_arg_to_cmdline(key, value, parser))
    return cli_args


def _coerce_arguments(
    args: dict[str, Any], metadata: dict[str, MegatronArgMetadata]
) -> dict[str, Any]:
    coerced: dict[str, Any] = {}
    for key, value in args.items():
        if key not in metadata:
            continue
        arg_meta = metadata[key]
        coerced[key] = _coerce_value(value, arg_meta.arg_type)
    return coerced


def _coerce_value(value: Any, target_type: type | None) -> Any:
    if target_type is None or value is None:
        return value
    origin = getattr(target_type, "__origin__", None)
    if origin is list:
        elem_type = target_type.__args__[0] if getattr(target_type, "__args__", None) else str
        if isinstance(value, str):
            value = value.split(",")
        if not isinstance(value, (list, tuple)):
            value = [value]
        return [elem_type(v) for v in value]
    if target_type is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "on"}
        return bool(value)
    try:
        return target_type(value)
    except Exception:  # pragma: no cover - rely on argparse to validate later
        return value


def _extract_action_type(action: argparse.Action) -> type | None:
    if isinstance(action, (_StoreTrueAction, _StoreFalseAction)):
        return bool
    if action.nargs in ("+", "*"):
        return list
    return action.type


def _arg_to_cmdline(arg: str, argval: Any, parser: argparse.ArgumentParser) -> list[str]:
    for action in parser._actions:  # noqa: SLF001
        if action.dest != arg:
            continue
        if isinstance(action, _StoreConstAction):
            if argval == action.const and argval != action.default:
                return [action.option_strings[0]]
            return []
        if action.nargs in ("+", "*"):
            if not argval:
                return []
            values = _ensure_iterable(argval)
            return [action.option_strings[0], *[str(v) for v in values]]
        if argval == action.default:
            return []
        return [action.option_strings[0], str(argval)]
    return []


def _ensure_iterable(value: Any) -> Iterable[Any]:
    if isinstance(value, (list, tuple)):
        return value
    return [value]


__all__ = [
    "MegatronArgMetadata",
    "build_cmdline_args",
    "extract_default_args",
    "get_arg_metadata",
    "get_megatron_parser",
]
