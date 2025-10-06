#!/usr/bin/env python3
"""Run autoexp inside a container similar to megatron-train utility."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch autoexp within a container")
    parser.add_argument("--image", required=True, help="Container image path")
    parser.add_argument("--apptainer-cmd", default="singularity")
    parser.add_argument("--config-ref", default="autoexp")
    parser.add_argument("-C", "--config-dir", default="config")
    parser.add_argument("-o", "--override", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true", help="Skip submission step")
    parser.add_argument("--fake-submit", action="store_true", help="Use fake SLURM submission inside the container")
    parser.add_argument("--show-command", action="store_true")
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Environment variable assignment passed to the container (VAR=VALUE).",
    )
    parser.add_argument(
        "--bind",
        action="append",
        default=[],
        help="Bind mount passed to the container (Singularity --bind syntax).",
    )
    parser.add_argument(
        "--no-submit",
        action="store_true",
        help="Only execute the dry-run phase inside the container.",
    )
    return parser.parse_args()


def _run_in_container(args: argparse.Namespace, cmd_parts: list[str]) -> int:
    apptainer_cmd = _build_container_command(args, cmd_parts)
    if args.show_command:
        print("Running:", " ".join(shlex.quote(c) for c in apptainer_cmd))
    result = subprocess.run(apptainer_cmd, check=False)
    return result.returncode


def _build_container_command(args: argparse.Namespace, cmd_parts: list[str]) -> list[str]:
    quoted_inner = " ".join(shlex.quote(part) for part in cmd_parts)
    command = [args.apptainer_cmd, "exec"]
    for bind in getattr(args, "bind", []) or []:
        command.extend(["--bind", bind])
    for env_assignment in getattr(args, "env", []) or []:
        command.extend(["--env", env_assignment])
    command.extend([args.image, "bash", "-lc", quoted_inner])
    return command


def main() -> None:
    args = parse_args()

    script = Path(__file__).resolve().parent / "run_autoexp.py"
    base_cmd = [
        "python",
        str(script),
        args.config_ref,
        "-C",
        str(args.config_dir),
    ]
    for override in args.override:
        base_cmd.extend(["-o", override])

    dry_cmd = base_cmd + ["--dry-run"]
    if _run_in_container(args, dry_cmd) != 0:
        raise SystemExit(1)

    if args.dry_run or args.no_submit:
        return

    submit_cmd = base_cmd.copy()
    if args.fake_submit:
        submit_cmd.append("--use-fake-slurm")
    else:
        raise SystemExit("Real SLURM submission not yet implemented; use --fake-submit")

    if _run_in_container(args, submit_cmd) != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
