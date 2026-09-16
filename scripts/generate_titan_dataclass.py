#!/usr/bin/env python3
"""Generate a compoconf-compatible dataclass snapshot for TorchTitan/Titan-
OELLM.

Use as:
PYTHONPATH=. python3 scripts/generate_titan_dataclass.py --custom-module oellm_autoexp.titan_custom_config:JobConfig
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import MISSING, Field
from pathlib import Path
from typing import Any, get_args, get_origin
import importlib
import inspect
import sys
from unittest.mock import MagicMock

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TITAN_ROOT = _REPO_ROOT / "submodules" / "titan-oellm"
_TORCHTITAN_ROOT = _TITAN_ROOT / "torchtitan"
_DEFAULT_OUTPUT = _REPO_ROOT / "oellm_autoexp" / "backends" / "titan" / "config_schema.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help="Destination file for the generated dataclass schema.",
    )
    parser.add_argument(
        "--base",
        default="titan_oellm.configs.oellm_job_config:JobConfig",
        help="Base JobConfig import path (module:attr).",
    )
    parser.add_argument(
        "--custom-module",
        default="",
        help="Optional custom config module to merge (module:JobConfig).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Silence success message.",
    )
    return parser.parse_args()


def _import_symbol(path: str):
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    if str(_TITAN_ROOT) not in sys.path:
        sys.path.insert(0, str(_TITAN_ROOT))
    if str(_TORCHTITAN_ROOT) not in sys.path:
        sys.path.insert(0, str(_TORCHTITAN_ROOT))
    if "tyro" not in sys.modules:
        sys.modules["tyro"] = MagicMock()
    module_path, _, symbol = path.partition(":")
    if not module_path or not symbol:
        raise ValueError(f"Invalid import path: {path}")
    mod = importlib.import_module(module_path)
    return getattr(mod, symbol)


def _merge_configs(base, custom):
    """Merge dataclass definitions (mirrors torchtitan
    ConfigManager._merge_configs)."""
    from dataclasses import field, fields, is_dataclass, make_dataclass

    result = []
    b_map = {f.name: f for f in fields(base)}
    c_map = {f.name: f for f in fields(custom)}

    for name, f in b_map.items():
        if name in c_map and is_dataclass(f.type) and is_dataclass(c_map[name].type):
            m_type = _merge_configs(f.type, c_map[name].type)
            result.append((name, m_type, field(default_factory=m_type)))
        elif name in c_map:
            result.append((name, c_map[name].type, c_map[name]))
        else:
            result.append((name, f.type, f))

    for name, f in c_map.items():
        if name not in b_map:
            result.append((name, f.type, f))

    return make_dataclass(f"Merged{base.__name__}", result, bases=(base,))


def _type_to_str(tp, imports: set[str]) -> str:
    if tp is None or tp is type(None):
        return "None"
    if tp is Any:
        imports.add("Any")
        return "Any"
    if isinstance(tp, str):
        return tp

    origin = get_origin(tp)
    args = get_args(tp)

    if origin is list:
        inner = _type_to_str(args[0] if args else Any, imports)
        return f"list[{inner}]"
    if origin is dict:
        k = _type_to_str(args[0] if args else Any, imports)
        v = _type_to_str(args[1] if len(args) > 1 else Any, imports)
        return f"dict[{k}, {v}]"
    if origin is tuple:
        if args:
            inner = ", ".join(_type_to_str(a, imports) for a in args)
            return f"tuple[{inner}]"
        return "tuple"
    if origin is set:
        inner = _type_to_str(args[0] if args else Any, imports)
        return f"set[{inner}]"
    if origin is type | None:
        inner = _type_to_str(args[0] if args else Any, imports)
        return f"type[{inner}] | None"
    if origin is None:
        pass

    if origin is list | dict | tuple | set:
        pass

    if origin is getattr(__import__("typing"), "Literal", None):
        imports.add("Literal")
        literals = ", ".join(repr(a) for a in args)
        return f"Literal[{literals}]"

    if origin is getattr(__import__("typing"), "Union", None):
        parts = [_type_to_str(a, imports) for a in args]
        return " | ".join(parts)

    if inspect.isclass(tp) and dataclasses.is_dataclass(tp):
        return tp.__name__

    if inspect.isclass(tp) and tp.__module__ == "builtins":
        return tp.__name__

    if inspect.isclass(tp):
        module = tp.__module__
        name = tp.__name__
        if module == "pathlib" and name == "Path":
            imports.add("Path")
            return "Path"
        return name

    imports.add("Any")
    return "Any"


def _default_to_str(field: Field, imports: set[str]) -> str | None:
    if field.default_factory is not MISSING:  # type: ignore[comparison-overlap]
        factory = field.default_factory  # type: ignore[assignment]
        if callable(factory) and getattr(factory, "__name__", "") == "<lambda>":
            try:
                value = factory()
            except Exception:
                value = MISSING
            if value is not MISSING:
                imports.add("field")
                return f"field(default_factory=lambda: {repr(value)})"
        if factory in {list, dict, set}:
            imports.add("field")
            return f"field(default_factory={factory.__name__})"
        if inspect.isclass(factory):
            imports.add("field")
            return f"field(default_factory={factory.__name__})"
        imports.add("field")
        return f"field(default_factory={factory})"

    if field.default is not MISSING:
        if isinstance(field.default, (list, dict, set)):
            imports.add("field")
            return f"field(default_factory=lambda: {repr(field.default)})"
        return repr(field.default)

    return None


def _collect_dataclasses(root) -> list[type]:
    seen: set[type] = set()
    ordered: list[type] = []

    def visit(cls):
        if cls in seen:
            return
        seen.add(cls)
        for f in dataclasses.fields(cls):
            tp = f.type
            origin = get_origin(tp)
            args = get_args(tp)
            if inspect.isclass(tp) and dataclasses.is_dataclass(tp):
                visit(tp)
            elif origin is not None:
                for a in args:
                    if inspect.isclass(a) and dataclasses.is_dataclass(a):
                        visit(a)
        ordered.append(cls)

    visit(root)
    return ordered


def generate_schema(output: Path, base_cls: type, custom_cls: type | None) -> None:
    if custom_cls is not None:
        root_cls = _merge_configs(base_cls, custom_cls)
    else:
        root_cls = base_cls

    classes = _collect_dataclasses(root_cls)

    imports: set[str] = set()
    lines: list[str] = []
    lines.append('"""Autogenerated TorchTitan/Titan-OELLM config schema."""')
    lines.append("from __future__ import annotations")
    lines.append("")
    lines.append("from dataclasses import dataclass, field")
    lines.append("from typing import Any, Literal")
    lines.append("")
    lines.append("# NOTE: This file is generated. Do not edit by hand.")
    lines.append("")

    for cls in classes:
        cls_imports: set[str] = set()
        lines.append("@dataclass")
        cls_name = "JobConfig" if cls is root_cls else cls.__name__
        lines.append(f"class {cls_name}:")
        if cls is root_cls:
            lines.append('    class_name: str = "TitanJobConfig"')
        for f in dataclasses.fields(cls):
            if cls is root_cls and f.name == "class_name":
                continue
            f_type = _type_to_str(f.type, cls_imports)
            default_src = _default_to_str(f, cls_imports)
            if default_src is None:
                lines.append(f"    {f.name}: {f_type}")
            else:
                lines.append(f"    {f.name}: {f_type} = {default_src}")
        if not dataclasses.fields(cls):
            lines.append("    pass")
        lines.append("")
        imports.update(cls_imports)

    # ensure required imports
    typing_imports = {"Any", "Literal"} | {i for i in imports if i in {"Any", "Literal"}}
    other_imports = {i for i in imports if i not in typing_imports and i != "field"}

    header: list[str] = []
    header.append('"""Autogenerated TorchTitan/Titan-OELLM config schema."""')
    header.append("from __future__ import annotations")
    header.append("")
    header.append("from dataclasses import dataclass, field")
    if typing_imports:
        header.append(f"from typing import {', '.join(sorted(typing_imports))}")
    if "Path" in other_imports:
        header.append("from pathlib import Path")
    header.append("")
    header.append("# NOTE: This file is generated. Do not edit by hand.")
    header.append("")

    # replace header
    body = lines[8:]  # after original header stub
    output.write_text("\n".join(header + body), encoding="utf-8")


def main() -> None:
    args = parse_args()
    base_cls = _import_symbol(args.base)
    custom_cls = _import_symbol(args.custom_module) if args.custom_module else None
    generate_schema(args.output, base_cls, custom_cls)
    if not args.quiet:
        print(f"Wrote Titan config schema to {args.output}")


if __name__ == "__main__":
    main()
