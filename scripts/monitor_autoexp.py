#!/usr/bin/env python3
"""Resume monitoring for jobs described by a plan manifest."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from oellm_autoexp.workflow.host import (
    build_host_runtime,
    instantiate_controller,
    run_monitoring,
)
from oellm_autoexp.workflow.manifest import read_manifest
from omegaconf import OmegaConf


def _configure_logging(verbose: bool = False, debug: bool = False) -> None:
    level = logging.WARNING
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--use-fake-slurm", action="store_true", help="Use in-memory SLURM backend")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "override",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Hydra-style override applied to the monitoring config "
        "(e.g. debug_sync=true, log_signals[0].state.key=finished)",
    )
    return parser.parse_args(argv)


def _parse_override_value(raw: str) -> Any:
    try:
        wrapper = OmegaConf.create({"value": raw})
        return wrapper["value"]
    except Exception:
        return raw


def _apply_monitor_overrides(spec, overrides: list[str]) -> None:
    if not overrides:
        return
    cfg = OmegaConf.create(spec.config)
    for item in overrides:
        allow_new = item.startswith("+")
        expression = item[1:] if allow_new else item
        if "=" not in expression:
            raise SystemExit(f"Invalid monitor override '{item}'; expected KEY=VALUE syntax")
        key, value_str = expression.split("=", 1)
        key = key.strip()
        value = _parse_override_value(value_str.strip())
        OmegaConf.update(cfg, key, value, merge=True)
    spec.config = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _configure_logging(args.verbose, args.debug)

    manifest_path = Path(args.manifest).resolve()
    manifest = read_manifest(manifest_path)
    _apply_monitor_overrides(manifest.monitor, args.override)

    runtime = build_host_runtime(
        manifest,
        use_fake_slurm=args.use_fake_slurm,
        manifest_path=manifest_path,
    )

    controller = instantiate_controller(runtime)

    if not list(controller.jobs()):
        print("No jobs registered in monitoring session; nothing to monitor.")
        return

    print(f"Monitoring session: {runtime.state_store.session_id}")
    run_monitoring(runtime, controller)


if __name__ == "__main__":
    main()
