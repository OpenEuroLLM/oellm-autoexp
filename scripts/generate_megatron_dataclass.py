#!/usr/bin/env python3
"""Generate a compoconf-compatible dataclass for Megatron-LM arguments."""

from __future__ import annotations

import argparse
import keyword
from pathlib import Path
from textwrap import wrap
from typing import Any


import sys
from unittest.mock import MagicMock

# Mock all transformer_engine submodules before any imports
te_mock = MagicMock()
sys.modules["transformer_engine"] = te_mock
sys.modules["transformer_engine.pytorch"] = MagicMock()
sys.modules["transformer_engine.pytorch.distributed"] = MagicMock()
sys.modules["transformer_engine.pytorch.tensor"] = MagicMock()


from oellm_autoexp.backends.megatron_args import (  # noqa: E402
    MegatronArgMetadata,
    extract_default_args,
    get_arg_metadata,
    get_megatron_parser,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUTPUT = _REPO_ROOT / "oellm_autoexp" / "backends" / "megatron" / "config_schema.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help="Destination file for the generated dataclass.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Comma-separated Megatron argument names to omit from the schema.",
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


def _type_repr(meta: MegatronArgMetadata, default: Any) -> tuple[str, set[str]]:
    imports: set[str] = set()
    base_type = meta.arg_type
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


def generate_dataclass(output: str, excluded: set[str], quiet: bool) -> None:
    parser = get_megatron_parser()
    metadata = get_arg_metadata(parser)
    defaults = extract_default_args(parser)

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

    header_lines = [
        '"""Megatron-LM configuration schema (auto-generated)."""',
        "",
        "from __future__ import annotations",
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


def main() -> None:
    args = parse_args()
    excluded = _collect_exclusions(args.exclude)
    generate_dataclass(args.output, excluded, args.quiet)


if __name__ == "__main__":
    main()
