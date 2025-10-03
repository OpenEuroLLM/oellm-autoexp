"""Utilities for extracting Megatron arguments from the Megatron-LM parser."""

from __future__ import annotations

import argparse
from argparse import _StoreConstAction, _StoreFalseAction, _StoreTrueAction
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Type

_MegatronParser: Optional[argparse.ArgumentParser] = None


def get_megatron_parser() -> argparse.ArgumentParser:
    global _MegatronParser
    if _MegatronParser is None:
        try:
            from megatron.training.arguments import add_megatron_arguments
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Megatron-LM not available. Install the [megatron] extra or add submodules/megatron"
            ) from exc
        parser = argparse.ArgumentParser(description="Megatron-LM Arguments", allow_abbrev=False)
        _MegatronParser = add_megatron_arguments(parser)
    return _MegatronParser


@dataclass
class MegatronArgMetadata:
    arg_type: Type | None
    default: Any


def get_arg_metadata(parser: argparse.ArgumentParser) -> Dict[str, MegatronArgMetadata]:
    metadata: Dict[str, MegatronArgMetadata] = {}
    for action in parser._actions:  # noqa: SLF001
        if not action.dest:
            continue
        metadata[action.dest] = MegatronArgMetadata(_extract_action_type(action), action.default)
    return metadata


def build_cmdline_args(
    args: Dict[str, Any],
    parser: argparse.ArgumentParser,
    metadata: Dict[str, MegatronArgMetadata],
) -> list[str]:
    coerced = _coerce_arguments(args, metadata)
    cli_args: list[str] = []
    for key, value in coerced.items():
        cli_args.extend(_arg_to_cmdline(key, value, parser))
    return cli_args


def _coerce_arguments(args: Dict[str, Any], metadata: Dict[str, MegatronArgMetadata]) -> Dict[str, Any]:
    coerced: Dict[str, Any] = {}
    for key, value in args.items():
        if key not in metadata:
            continue
        arg_meta = metadata[key]
        coerced[key] = _coerce_value(value, arg_meta.arg_type)
    return coerced


def _coerce_value(value: Any, target_type: Type | None) -> Any:
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


def _extract_action_type(action: argparse.Action) -> Type | None:
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
    "get_arg_metadata",
    "get_megatron_parser",
]
