"""Utilities for extracting Megatron arguments from the Megatron-LM parser."""

from __future__ import annotations

import argparse
import sys
from argparse import _StoreFalseAction, _StoreTrueAction
from dataclasses import MISSING, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
from collections.abc import Iterable, Mapping

_MegatronParser: argparse.ArgumentParser | None = None


def _maybe_add_submodule_to_path() -> None:
    """Ensure the bundled Megatron-LM submodule is importable."""

    root = Path(__file__).resolve().parents[2]
    candidate = (
        root / "submodules" / "ROCm-Megatron-LM"
    )  # TODO Fix this hardcode to include both megatrons?
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


@dataclass(frozen=True)
class MegatronActionSpec:
    option_strings: tuple[str, ...]
    action_type: str
    nargs: int | str | None
    const: Any
    default: Any


def get_action_specs(parser: argparse.ArgumentParser) -> dict[str, MegatronActionSpec]:
    specs: dict[str, MegatronActionSpec] = {}
    for action in parser._actions:  # noqa: SLF001
        dest = getattr(action, "dest", None)
        if not dest or dest == "help":
            continue
        specs[dest] = MegatronActionSpec(
            option_strings=tuple(action.option_strings),
            action_type=_action_type_name(action),
            nargs=action.nargs,
            const=getattr(action, "const", None),
            default=action.default,
        )
    return specs


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
    metadata: Mapping[str, MegatronArgMetadata],
    action_specs: Mapping[str, MegatronActionSpec],
    *,
    skip_defaults: bool = True,
) -> list[str]:
    coerced = _coerce_arguments(args, metadata)
    cli_args: list[str] = []
    for key, value in coerced.items():
        spec = action_specs.get(key)
        if not spec:
            continue
        cli_args.extend(_spec_to_cmdline(spec, value, skip_defaults))
    return cli_args


def _coerce_arguments(
    args: dict[str, Any], metadata: Mapping[str, MegatronArgMetadata]
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


def _action_type_name(action: argparse.Action) -> str:
    name = action.__class__.__name__.lstrip("_").lower()
    if "storetrue" in name:
        return "store_true"
    if "storefalse" in name:
        return "store_false"
    if "storeconst" in name:
        return "store_const"
    if "appendconst" in name:
        return "append_const"
    if "append" in name:
        return "append"
    if "count" in name:
        return "count"
    return "store"


def _spec_to_cmdline(spec: MegatronActionSpec, argval: Any, skip_defaults: bool) -> list[str]:
    if not spec.option_strings:
        return []
    option = spec.option_strings[0]

    if spec.action_type in {"store_true", "store_false", "store_const"}:
        trigger = spec.const
        if argval == trigger and (not skip_defaults or argval != spec.default):
            return [option]
        return []

    if spec.action_type == "count":
        count = int(argval or 0)
        default_count = int(spec.default or 0)
        if skip_defaults and count == default_count:
            return []
        if count <= 0:
            return []
        return [option] * count

    if spec.nargs in ("+", "*"):
        if skip_defaults and argval == spec.default:
            return []
        if not argval:
            return []
        values = _ensure_iterable(argval)
        return [option, *[str(v) for v in values]]

    if skip_defaults and argval == spec.default:
        return []

    return [option, str(argval)]


def _ensure_iterable(value: Any) -> Iterable[Any]:
    if isinstance(value, (list, tuple)):
        return value
    return [value]


__all__ = [
    "MegatronActionSpec",
    "MegatronArgMetadata",
    "build_cmdline_args",
    "extract_default_args",
    "get_action_specs",
    "get_arg_metadata",
    "get_megatron_parser",
]
