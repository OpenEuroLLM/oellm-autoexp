#!/usr/bin/env python3
"""Render a Hydra configuration to stdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from compoconf import asdict
from oellm_autoexp.config.loader import load_config_reference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render autoexp config")
    parser.add_argument("config_ref", nargs="?", default="autoexp")
    parser.add_argument("-C", "--config-dir", default="config", type=Path)
    parser.add_argument("-o", "--override", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = load_config_reference(args.config_ref, args.config_dir, args.override)
    print(json.dumps(asdict(root), indent=2, default=str))


if __name__ == "__main__":
    main()
