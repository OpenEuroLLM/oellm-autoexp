"""Generation of Python dataclasses and YAML defaults from argparse
metadata."""

from __future__ import annotations

import keyword
from textwrap import wrap
from typing import Any
from collections.abc import Mapping

from omegaconf import OmegaConf
from .converter import ArgMetadata, ActionSpec


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


def _type_repr(
    meta: ArgMetadata, default: Any, allow_scalar_lists: bool = True
) -> tuple[str, set[str]]:
    imports: set[str] = set()
    base_type = meta.arg_type

    # Use metadata default if provided default is None
    if meta.default is not None and default is None:
        default = meta.default
    if base_type is None and default is not None:
        base_type = type(default)

    if meta.choices:
        # Avoid Literal with mixed types if default is str but choices aren't
        if isinstance(default, str) and not all(isinstance(c, str) for c in meta.choices):
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

        if allow_scalar_lists and default is not None and not isinstance(default, list):
            type_repr = f"list[{elem_repr}] | {elem_repr}"
        else:
            type_repr = f"list[{elem_repr}]"
    else:
        imports.add("Any")
        type_repr = "Any"

    # Handle optional types
    if default is None:
        allow_none = False
        if meta.choices:
            allow_none = any(option is None for option in meta.choices)
        if not allow_none:
            type_repr = f"{type_repr} | None"

    return type_repr, imports


def _format_default(default: Any) -> tuple[str, set[str]]:
    imports: set[str] = set()
    if isinstance(default, (list, dict)):
        imports.add("field")
        return f"field(default_factory=lambda: {repr(default)})", imports
    if isinstance(default, str):
        return repr(default), imports
    if isinstance(default, bool):
        return "True" if default else "False", imports
    if default is None:
        return "None", imports
    return repr(default), imports


def generate_dataclass(
    metadata: Mapping[str, ArgMetadata],
    defaults: Mapping[str, Any],
    class_name: str = "GeneratedConfig",
    excluded: set[str] | None = None,
    allow_scalar_lists: bool = True,
    extra_fields: dict[str, str] | None = None,
) -> str:
    """Generate Python code for a compoconf-compatible dataclass."""
    excluded = excluded or set()
    extra_fields = extra_fields or {}

    fields: list[str] = []
    needs_field = False
    typing_imports: set[str] = set()

    for name in sorted(metadata.keys()):
        if name in excluded or keyword.iskeyword(name):
            continue
        if name not in defaults:
            continue

        meta = metadata[name]
        default = defaults[name]

        type_str, type_imports = _type_repr(meta, default, allow_scalar_lists=allow_scalar_lists)
        typing_imports.update(type_imports)

        default_str, default_imports = _format_default(default)
        if "field" in default_imports:
            needs_field = True

        field_lines: list[str] = []
        if meta.help:
            wrapped = wrap(meta.help.replace("\n", " ").strip(), width=84)
            for line in wrapped:
                field_lines.append(f"    # {line}")

        field_lines.append(f"    {name}: {type_str} = {default_str}")
        fields.append("\n".join(field_lines))

    for key, val in extra_fields.items():
        fields.append(f"    {key}: {val}")

    header = [
        '"""Auto-generated configuration schema."""',
        "",
        "from dataclasses import dataclass" + (", field" if needs_field else ""),
    ]
    if typing_imports:
        header.append("from typing import " + ", ".join(sorted(typing_imports)))
    header.append("from compoconf import ConfigInterface")
    header.append("")
    header.append("@dataclass")
    header.append(f"class {class_name}(ConfigInterface):")

    return "\n".join(header) + "\n" + "\n\n".join(fields) + "\n"


def generate_defaults_yaml(
    metadata: Mapping[str, ArgMetadata],
    defaults: Mapping[str, Any],
    excluded: set[str] | None = None,
    include_comments: bool = True,
) -> str:
    """Generate an annotated YAML file with default values."""
    excluded = excluded or set()
    filtered_defaults = {k: v for k, v in defaults.items() if k not in excluded}

    yaml_text = OmegaConf.to_yaml(OmegaConf.create(filtered_defaults), sort_keys=True)
    if not include_comments:
        return yaml_text

    lines = yaml_text.splitlines()
    for idx, line in enumerate(lines):
        if not line or line.startswith("#") or ":" not in line:
            continue
        key = line.split(":", 1)[0].strip()
        meta = metadata.get(key)
        if not meta or key in excluded:
            continue

        comment_parts = []
        if meta.choices:
            comment_parts.append(f"choices: {list(meta.choices)}")
        if meta.help:
            comment_parts.append(meta.help.replace("\n", " ").strip())

        if comment_parts:
            comment = " | ".join(comment_parts)
            lines[idx] = f"{line}  # {comment}"

    return "\n".join(lines) + "\n"


def generate_cli_metadata_code(
    metadata: Mapping[str, ArgMetadata],
    specs: Mapping[str, ActionSpec],
    excluded: set[str] | None = None,
) -> str:
    """Generate Python code for CLI metadata and action specs."""
    excluded = excluded or set()

    metadata_lines = []
    for dest in sorted(metadata.keys()):
        if dest in excluded:
            continue
        meta = metadata[dest]
        metadata_lines.append(
            f"    {dest!r}: ArgMetadata(\n"
            f"        arg_type={_type_name(meta.arg_type)},\n"
            f"        default={meta.default!r},\n"
            f"        help={meta.help!r},\n"
            f"        choices={meta.choices!r},\n"
            f"        nargs={meta.nargs!r},\n"
            f"        element_type={_type_name(meta.element_type, default='Any')}\n"
            "    ),"
        )

    specs_lines = []
    for dest in sorted(specs.keys()):
        if dest in excluded:
            continue
        spec = specs[dest]
        specs_lines.append(
            f"    {dest!r}: ActionSpec(\n"
            f"        option_strings={spec.option_strings!r},\n"
            f"        action_type={spec.action_type!r},\n"
            f"        nargs={spec.nargs!r},\n"
            f"        const={spec.const!r},\n"
            f"        default={spec.default!r}\n"
            "    ),"
        )

    lines = [
        '"""Auto-generated CLI metadata."""',
        "",
        "from typing import Any, Mapping",
        "from oellm_autoexp.argparse_schema import ArgMetadata, ActionSpec",
        "",
        "ARG_METADATA: Mapping[str, ArgMetadata] = {",
        *metadata_lines,
        "}",
        "",
        "ACTION_SPECS: Mapping[str, ActionSpec] = {",
        *specs_lines,
        "}",
    ]
    return "\n".join(lines) + "\n"
