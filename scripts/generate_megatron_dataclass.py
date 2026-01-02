#!/usr/bin/env python3
"""Generate a compoconf-compatible dataclass for Megatron-LM arguments."""

import argparse
import importlib.metadata as importlib_metadata
import keyword
from enum import Enum
from pathlib import Path
from textwrap import wrap
from typing import Any
import sys
from unittest.mock import MagicMock

# Mock all transformer_engine submodules before any imports
te_mock = MagicMock()
sys.modules["transformer_engine"] = te_mock
sys.modules["transformer_engine.pytorch"] = MagicMock()
sys.modules["transformer_engine.pytorch.router"] = MagicMock()
sys.modules["transformer_engine.pytorch.distributed"] = MagicMock()
sys.modules["transformer_engine.pytorch.tensor"] = MagicMock()
sys.modules["transformer_engine.pytorch.float8_tensor"] = MagicMock()
sys.modules["transformer_engine.pytorch.fp8"] = MagicMock()

_ORIG_IMPORTLIB_VERSION = importlib_metadata.version


def _mocked_version(name: str) -> str:
    if name == "transformer-engine":
        return "0.0.0"
    return _ORIG_IMPORTLIB_VERSION(name)


importlib_metadata.version = _mocked_version  # type: ignore[assignment]


from oellm_autoexp.backends.megatron_args import (  # noqa: E402
    MegatronArgMetadata,
    extract_default_args,
    get_arg_metadata,
    get_megatron_parser,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUTPUT = _REPO_ROOT / "oellm_autoexp" / "backends" / "megatron" / "config_schema.py"
_DEFAULT_CLI_OUTPUT = _REPO_ROOT / "oellm_autoexp" / "backends" / "megatron" / "cli_metadata.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help="Destination file for the generated dataclass.",
    )
    parser.add_argument(
        "--cli-output",
        type=Path,
        default=_DEFAULT_CLI_OUTPUT,
        help="Destination file for generated CLI metadata.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Comma-separated Megatron argument names to omit from the schema.",
    )
    parser.add_argument(
        "--no-add-aux",
        action="store_true",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Silence the success message when writing to disk.",
    )
    return parser.parse_args()


def _collect_exclusions(values: list[str]) -> set[str]:
    excluded: set[str] = set()
    for raw in values:
        for chunk in raw.split(","):
            token = chunk.strip()
            if token:
                excluded.add(token)
    return excluded


def _literal_token(value: Any) -> str:
    if isinstance(value, str):
        return repr(value)
    if value is None:
        return "None"
    return repr(value)


def _type_name(value: Any, *, default: str = "None") -> str:
    if value is None:
        return "None"
    if value is Any:
        return "Any"
    if isinstance(value, type):
        return value.__name__
    return default


def _type_repr(meta: MegatronArgMetadata, default: Any) -> tuple[str, set[str]]:
    imports: set[str] = set()
    base_type = meta.arg_type
    if meta.default is not None and default is None:
        default = meta.default
    if base_type is None and default is not None:
        base_type = type(default)

    if meta.choices:
        # If default is a string but choices are not (e.g., enum converted to string name),
        # use str type instead of Literal with incompatible types
        if (
            isinstance(default, str)
            and meta.choices
            and not all(isinstance(c, str) for c in meta.choices)
        ):
            type_repr = "str"
        else:
            literals = ", ".join(_literal_token(option) for option in meta.choices)
            imports.add("Literal")
            type_repr = f"Literal[{literals}]"
    elif base_type is bool:
        type_repr = "bool"
    elif base_type is int:
        type_repr = "int"
    elif base_type is float:
        type_repr = "float"
    elif base_type is str:
        type_repr = "str"
    elif base_type is list or meta.nargs in {"+", "*"}:
        elem_type = meta.element_type or (
            type(default[0]) if isinstance(default, list) and default else Any
        )
        if elem_type is Any:
            imports.add("Any")
            elem_repr = "Any"
        elif elem_type in {int, float, bool, str}:
            elem_repr = elem_type.__name__
        else:
            imports.add("Any")
            elem_repr = "Any"
        if default is not None and not isinstance(default, list):
            type_repr = f"list[{elem_repr}] | {elem_repr}"
        else:
            type_repr = f"list[{elem_repr}]"
    else:
        imports.add("Any")
        type_repr = "Any"

    if default is None:
        allow_none = False
        if meta.choices:
            allow_none = any(option is None for option in meta.choices)
        if not allow_none:
            type_repr = f"{type_repr} | None"
    return type_repr, imports


def _format_default(name: str, default: Any) -> tuple[str, set[str]]:
    imports: set[str] = set()
    if isinstance(default, list):
        imports.add("field")
        return f"field(default_factory=lambda: {repr(default)})", imports
    if isinstance(default, dict):
        imports.add("field")
        return f"field(default_factory=lambda: {repr(default)})", imports
    if isinstance(default, str):
        return repr(default), imports
    if isinstance(default, bool):
        return "True" if default else "False", imports
    if default is None:
        return "None", imports
    return repr(default), imports


def _summarise_help(meta: MegatronArgMetadata) -> str:
    help_text = meta.help or ""
    if meta.choices and not help_text:
        help_text = "Allowed values: " + ", ".join(str(choice) for choice in meta.choices)
    return help_text.strip()


def generate_dataclass(
    parser: argparse.ArgumentParser,
    metadata: dict[str, MegatronArgMetadata],
    defaults: dict[str, Any],
    output: Path,
    excluded: set[str],
    quiet: bool,
    added: dict[str, str] = {},
) -> None:
    fields: list[str] = []
    needs_field = False
    typing_imports: set[str] = set()

    for action in parser._actions:  # noqa: SLF001
        name = getattr(action, "dest", None)
        if not name or name == "help" or name in excluded:
            continue
        if name not in metadata or name not in defaults:
            continue
        if keyword.iskeyword(name):
            continue  # pragma: no cover - not expected currently
        meta = metadata[name]
        default = defaults[name]

        type_repr, type_imports = _type_repr(meta, default)
        typing_imports.update(type_imports)
        default_repr, default_imports = _format_default(name, default)
        if "field" in default_imports:
            needs_field = True
        field_lines: list[str] = []
        comment = _summarise_help(meta)
        if comment:
            wrapped = wrap(comment, width=88)
            for line in wrapped:
                field_lines.append(f"    # {line}")
        line = f"    {name}: {type_repr} = {default_repr}"
        field_lines.append(line)
        fields.append("\n".join(field_lines))
    for key, val in added.items():
        fields.append(f"    {key}: {str(val)}\n")
    header_lines = [
        '"""Megatron-LM configuration schema (auto-generated)."""',
        "",
    ]
    dataclasses_import = "from dataclasses import dataclass"
    if needs_field:
        dataclasses_import = "from dataclasses import dataclass, field"
    header_lines.append(dataclasses_import)
    if typing_imports:
        header_lines.append("from typing import " + ", ".join(sorted(typing_imports)))
    header_lines.append("from compoconf import ConfigInterface")
    header_lines.append("")
    header_lines.append("@dataclass")
    header_lines.append("class MegatronConfig(ConfigInterface):")
    header_lines.append('    """Typed projection of Megatron-LM CLI arguments."""')
    header_lines.append("    # Generated by scripts/generate_megatron_dataclass.py")
    body = "\n".join(header_lines) + "\n" + "\n\n".join(fields) + "\n"
    body += '\n__all__ = ["MegatronConfig"]\n'

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(body)
    if not quiet:
        print(f"Wrote {output}")


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


def _sanitize_default(value: Any) -> Any:
    if value is argparse.SUPPRESS:
        return None
    if isinstance(value, Enum):
        return getattr(value, "value", value.name)
    return value


def generate_cli_metadata(
    parser: argparse.ArgumentParser,
    metadata: dict[str, MegatronArgMetadata],
    defaults: dict[str, Any],
    output: Path,
    excluded: set[str],
    quiet: bool,
) -> None:
    specs_lines: list[str] = []
    metadata_lines: list[str] = []

    for dest in sorted(metadata.keys()):
        if dest in excluded:
            continue
        meta = metadata[dest]
        default_value = defaults.get(dest, meta.default)
        default_repr = (
            _literal_token(default_value) if default_value is not argparse.SUPPRESS else "None"
        )
        metadata_lines.append(
            "    "
            + repr(dest)
            + ": MegatronArgMetadata("
            + f"arg_type={_type_name(meta.arg_type)}, "
            + f"default={default_repr}, "
            + f"help={repr(meta.help)}, "
            + f"choices={repr(meta.choices)}, "
            + f"nargs={repr(meta.nargs)}, "
            + f"element_type={_type_name(meta.element_type, default='Any')}"
            + "),"
        )

    sorted_actions = sorted(
        (
            (getattr(action, "dest", None), action)
            for action in parser._actions  # noqa: SLF001
            if getattr(action, "dest", None)
        ),
        key=lambda pair: pair[0] or "",
    )
    for dest, action in sorted_actions:
        if not dest or dest == "help" or dest in excluded:
            continue
        option_strings = tuple(action.option_strings)
        default_value = defaults.get(dest, _sanitize_default(action.default))
        specs_lines.append(
            f'    "{dest}": MegatronActionSpec('
            f"option_strings={repr(option_strings)}, "
            f"action_type={repr(_action_type_name(action))}, "
            f"nargs={repr(action.nargs)}, "
            f"const={_literal_token(_sanitize_default(action.const))}, "
            f"default={_literal_token(default_value)}"
            "),"
        )

    body = "\n".join(
        [
            '"""Megatron CLI metadata (auto-generated)."""',
            "",
            "from dataclasses import dataclass",
            "from typing import Any, Mapping",
            "",
            "from oellm_autoexp.backends.megatron_args import MegatronArgMetadata",
            "",
            "@dataclass(frozen=True)",
            "class MegatronActionSpec:",
            "    option_strings: tuple[str, ...]",
            "    action_type: str",
            "    nargs: int | str | None",
            "    const: Any",
            "    default: Any",
            "",
            "MEGATRON_ARG_METADATA: Mapping[str, MegatronArgMetadata] = {",
            *metadata_lines,
            "}",
            "",
            "MEGATRON_ACTION_SPECS: Mapping[str, MegatronActionSpec] = {",
            *specs_lines,
            "}",
            "",
            '__all__ = ["MEGATRON_ARG_METADATA", "MEGATRON_ACTION_SPECS", "MegatronActionSpec"]',
            "",
        ]
    )
    body += "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(body)
    if not quiet:
        print(f"Wrote {output}")


def main() -> None:
    args = parse_args()
    excluded = _collect_exclusions(args.exclude)
    parser = get_megatron_parser()
    metadata = get_arg_metadata(parser)
    defaults = extract_default_args(parser)
    generate_dataclass(
        parser,
        metadata,
        defaults,
        args.output,
        excluded,
        args.quiet,
        added={} if args.no_add_aux else {"aux": "dict[str, Any] = field(default_factory=dict)"},
    )
    generate_cli_metadata(parser, metadata, defaults, args.cli_output, excluded, args.quiet)


if __name__ == "__main__":
    main()
