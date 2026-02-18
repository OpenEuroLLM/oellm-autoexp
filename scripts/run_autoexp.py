#!/usr/bin/env python3
"""Convenience wrapper to plan, submit, and optionally monitor in one shot."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Iterable
from uuid import uuid4

from compoconf import asdict

from oellm_autoexp.config.loader import load_config_reference
from oellm_autoexp.config.schema import ConfigSetup
from oellm_autoexp.orchestrator import (
    build_execution_plan,
    ExecutionPlan,
    submit_jobs,
    run_loop,
)
from oellm_autoexp.utils.logging_config import configure_logging


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="autoexp")
    parser.add_argument("--config-path", default=None)
    parser.add_argument("-C", "--config-dir", type=Path, default=Path("config"))
    parser.add_argument(
        "--dry-run", action="store_true", help="Plan and render without submitting jobs"
    )
    parser.add_argument("--no-monitor", action="store_true", help="Submit jobs but skip monitoring")
    parser.add_argument(
        "--monitor-state-dir",
        default="./monitor_state",
        type=Path,
        help="Monitoring state directory",
    )
    parser.add_argument("--no-verbose", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--array-subset",
        type=str,
        help="Comma-separated sweep indices or ranges (e.g., '0,3-5') to rerun.",
    )
    parser.add_argument(
        "overrides", nargs="*", default=[], help="Hydra-style overrides (`key=value`)."
    )
    return parser.parse_args(argv)


def _default_manifest_path(base_output_dir: str | Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    manifest_dir = Path(base_output_dir) / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    suffix = uuid4().hex[:6]
    return manifest_dir / f"plan_{timestamp}_{suffix}.json"


def _collect_git_metadata(repo_root: Path) -> dict[str, str | bool]:
    def _run(cmd: Iterable[str]) -> str:
        try:
            result = subprocess.run(
                list(cmd),
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            return ""
        return result.stdout.strip()

    commit = _run(["git", "rev-parse", "HEAD"]) or "unknown"
    status = _run(["git", "status", "--porcelain"])
    dirty = bool(status)
    diff = _run(["git", "diff"]) if dirty else ""
    return {
        "commit": commit,
        "dirty": dirty,
        "status": status,
        "diff": diff,
    }


def _sanitize_env() -> dict[str, str]:
    pattern = re.compile(r"(KEY|SECRET)", re.IGNORECASE)
    return {key: value for key, value in os.environ.items() if not pattern.search(key)}


def _write_job_provenance(
    plan: ExecutionPlan,
    *,
    args: argparse.Namespace | None = None,
    subset_indices: set[int] | None = None,
    overrides: list[str] = (),
) -> None:
    git_meta = _collect_git_metadata(REPO_ROOT)
    sanitized_env = _sanitize_env()
    base_payload = {
        "git": git_meta,
        "command": {key: str(val) for key, val in vars(args).items()} or list(sys.argv),
        "overrides": overrides,
        "subset_indices": sorted(subset_indices),
        "plan": asdict(plan),
        "environment": sanitized_env,
    }

    manifest_path = _default_manifest_path(plan.config_setup.monitor_state_dir)
    with open(manifest_path, "w") as fp:
        json.dump(base_payload, fp)


def _parse_subset(spec: str | None) -> set[int]:
    indices: set[int] = set()
    if not spec:
        return indices
    for token in spec.split(","):
        part = token.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                raise ValueError(f"invalid range '{part}'")
            indices.update(range(start, end + 1))
        else:
            indices.add(int(part))
    return indices


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging(not args.no_verbose, args.debug)

    config_dir = Path(args.config_dir)

    config_setup = ConfigSetup(
        pwd=os.path.abspath(os.curdir),
        config_name=args.config_name,
        config_dir=str(config_dir),
        overrides=args.overrides,
        monitor_state_dir=str(args.monitor_state_dir),
    )
    root = load_config_reference(config_setup=config_setup)

    try:
        subset_indices = _parse_subset(args.array_subset)
    except ValueError as exc:
        print(f"Invalid --array-subset argument: {exc}", file=sys.stderr)
        return

    plan = build_execution_plan(
        root,
        config_setup=config_setup,
        subset_indices=subset_indices or None,
    )

    if args.dry_run:
        exit(0)

    _write_job_provenance(
        plan,
        args=args,
        subset_indices=subset_indices,
        overrides=args.overrides,
    )

    res = submit_jobs(plan, no_error_catching=args.debug)

    if args.no_monitor:
        exit(0)

    run_loop(res.loop)


if __name__ == "__main__":
    main()
