#!/usr/bin/env python3
"""Render a Hydra configuration to stdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from compoconf import asdict
from oellm_autoexp.config.loader import load_config_reference
from oellm_autoexp.config.schema import ConfigSetup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render autoexp config")
    parser.add_argument("--config_name", default="autoexp")
    parser.add_argument("--config_path", default=None)
    parser.add_argument("-C", "--config-dir", default="config", type=Path)
    parser.add_argument("-o", "--overrides", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_setup = ConfigSetup(
        config_name=args.config_name,
        config_path=args.config_path,
        config_dir=args.config_dir,
        overrides=args.overrides,
    )
    root = load_config_reference(config_setup=config_setup)
    print(json.dumps(asdict(root), indent=2, default=str))


if __name__ == "__main__":
    main()
