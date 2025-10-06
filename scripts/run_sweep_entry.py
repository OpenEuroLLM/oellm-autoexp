#!/usr/bin/env python3
"""Execute a sweep entry described in sweep.json.

The JSON file is expected to have the structure produced by oellm_autoexp,
namely a top-level object with a "jobs" list where each entry contains:

{
  "name": str,
  "output_dir": str,
  "log_path": str,
  "parameters": {...},
  "launch": {"argv": [...], "environment": {...}}
}
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a sweep entry from sweep.json")
    parser.add_argument("--sweep", required=True, help="Path to sweep.json")
    parser.add_argument("--index", type=int, required=True, help="Entry index (typically SLURM array id)")
    parser.add_argument("--dry-run", action="store_true", help="Print command without executing")
    return parser.parse_args()


def load_entry(path: Path, index: int) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    jobs = payload.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError(f"Invalid sweep payload in {path}: missing 'jobs' list")
    try:
        entry = jobs[index]
    except IndexError as exc:  # pragma: no cover - indicates config error
        raise ValueError(f"Sweep index {index} out of range (jobs={len(jobs)})") from exc
    if not isinstance(entry, dict):
        raise ValueError(f"Sweep entry at index {index} is not a mapping")
    return entry


def prepare_environment(entry: dict) -> dict:
    launch_cfg = entry.get("launch", {})
    environment = launch_cfg.get("environment", {})
    if not isinstance(environment, dict):
        environment = {}
    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in environment.items()})
    env.setdefault("OELLM_AUTOEXP_JOB_NAME", str(entry.get("name", "")))
    return env


def main() -> None:
    args = parse_args()
    sweep_path = Path(args.sweep).expanduser().resolve()
    entry = load_entry(sweep_path, args.index)

    output_dir = Path(entry.get("output_dir", ""))
    if output_dir.as_posix():
        output_dir.mkdir(parents=True, exist_ok=True)

    launch_cfg = entry.get("launch", {})
    argv = launch_cfg.get("argv")
    if not isinstance(argv, list) or not argv:
        raise ValueError(f"Invalid launch configuration for sweep entry {args.index}")
    argv = [str(arg) for arg in argv]

    env = prepare_environment(entry)

    if args.dry_run:
        print("Command:", " ".join(argv))
        print("Environment overrides:", {k: env[k] for k in env if k not in os.environ or os.environ[k] != env[k]})
        return

    proc = subprocess.run(argv, env=env, check=False)
    if proc.returncode != 0:  # pragma: no cover - propagate failure upstream
        raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
