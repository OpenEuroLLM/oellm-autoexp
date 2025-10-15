#!/usr/bin/env python3
"""Generate annotated Megatron-LM configuration defaults."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from collections.abc import Iterable, Mapping

from omegaconf import OmegaConf

from oellm_autoexp.backends.megatron_args import (
    MegatronArgMetadata,
    extract_default_args,
    get_arg_metadata,
    get_megatron_parser,
)

import sys
from unittest.mock import MagicMock

# Mock all transformer_engine submodules before any imports
te_mock = MagicMock()
sys.modules["transformer_engine"] = te_mock
sys.modules["transformer_engine.pytorch"] = MagicMock()
sys.modules["transformer_engine.pytorch.distributed"] = MagicMock()
sys.modules["transformer_engine.pytorch.tensor"] = MagicMock()


_REPO_ROOT = Path(__file__).resolve().parent.parent

_DEFAULT_OUTPUT = _REPO_ROOT / "config" / "backend" / "megatron" / "base_defaults.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help="Destination file for the generated YAML (defaults to packaged base_defaults.yaml)",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Comma separated list of Megatron arguments to omit from the generated output.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="ARG=VALUE",
        help="Override a default value reported by Megatron (can be provided multiple times).",
    )
    parser.add_argument(
        "--no-comments",
        action="store_true",
        help="Skip appending choices/help text as YAML comments.",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print the generated YAML to stdout instead of writing to a file.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Silence the success message when writing to disk.",
    )
    return parser.parse_args()


def _collect_exclusions(values: Iterable[str]) -> set[str]:
    excluded: set[str] = set()
    for raw in values:
        for chunk in raw.split(","):
            token = chunk.strip()
            if token:
                excluded.add(token)
    return excluded


def _parse_override(value: str) -> tuple[str, object]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"Invalid override '{value}', expected KEY=VALUE")
    key, raw_value = value.split("=", 1)
    key = key.strip()
    parsed = _coerce_value(raw_value.strip())
    return key, parsed


def _coerce_value(raw: str) -> object:
    lowered = raw.lower()
    if lowered in {"none", "null"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw


def _build_comment(metadata: Mapping[str, MegatronArgMetadata], key: str) -> str:
    meta = metadata.get(key)
    if not meta:
        return ""
    parts: list[str] = []
    if meta.choices:
        parts.append(f"choices: {list(meta.choices)}")
    if meta.help:
        parts.append(meta.help.replace("\n", " ").strip())
    return " | ".join(part for part in parts if part)


def _annotate_yaml(
    yaml_text: str, metadata: Mapping[str, MegatronArgMetadata], excluded: set[str]
) -> str:
    lines = yaml_text.splitlines()
    for idx, line in enumerate(lines):
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key = stripped.split(":", 1)[0]
        if key in excluded or "#" in stripped:
            continue
        comment = _build_comment(metadata, key)
        if comment:
            lines[idx] = f"{line}  # {comment}"
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    excluded = _collect_exclusions(args.exclude)
    overrides = dict(_parse_override(entry) for entry in args.override)

    parser = get_megatron_parser()
    metadata = get_arg_metadata(parser)
    defaults = extract_default_args(parser, exclude=excluded, overrides=overrides)

    yaml_text = OmegaConf.to_yaml(OmegaConf.create(defaults), sort_keys=True)
    if not args.no_comments:
        yaml_text = _annotate_yaml(yaml_text, metadata, excluded)

    if args.stdout:
        print(yaml_text, end="")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml_text)
    if not args.quiet:
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
