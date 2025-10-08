#!/usr/bin/env python3
"""Run autoexp inside a container similar to megatron-train utility.

This script handles SLURM submission from outside the container:
1. Runs plan generation inside the container with --no-submit
2. Parses the sbatch command from container output
3. Executes sbatch on the host system where SLURM commands are available
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
from pathlib import Path

import oellm_autoexp.utils.run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch autoexp within a container")
    parser.add_argument("--image", required=True, help="Container image path")
    parser.add_argument("--apptainer-cmd", default="singularity")
    parser.add_argument("--config-name", default="autoexp")
    parser.add_argument("-C", "--config-dir", default="config")
    parser.add_argument("--no-run", action="store_true", help="Generate scripts but don't submit to SLURM")
    parser.add_argument("--fake-submit", action="store_true", help="Use fake SLURM submission inside the container")
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
        "--ihelp",
        action="store_true",
        help="Internal help",
    )
    parser.add_argument("override", nargs="*", default=[], help="Additional overrides.")
    return parser.parse_args()


def _build_container_command(args: argparse.Namespace, cmd_parts: list[str]) -> list[str]:
    """Build the apptainer/singularity exec command."""
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
        "--config-name",
        args.config_name,
        "-C",
        str(args.config_dir),
    ]

    # Run inside container with --no-submit to generate scripts and output sbatch command
    container_cmd = base_cmd + ["--no-submit"]
    if args.ihelp:
        container_cmd.append("--help")
    if args.fake_submit:
        container_cmd.append("--use-fake-slurm")

    container_cmd += args.override

    apptainer_cmd = _build_container_command(args, container_cmd)
    print("Running:", " ".join(shlex.quote(c) for c in apptainer_cmd))

    result = oellm_autoexp.utils.run.run_with_tee(apptainer_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise SystemExit(1)

    # Parse the sbatch command from container output
    sbatch_match = re.search(r"^Successful, to execute, run: (.*?)(sbatch .*)$", result.stdout, flags=re.MULTILINE)

    if not sbatch_match:
        # If no sbatch command found, this might be using fake SLURM or an error
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        if args.fake_submit:
            # Fake SLURM handles submission internally, so this is expected
            return
        print("Warning: No sbatch command found in output")
        return

    # Extract environment variables and sbatch command
    env_string = sbatch_match.group(1).strip()
    sbatch_cmd = sbatch_match.group(2).strip()

    print(result.stdout)
    print(f"\nSLURM Command: {sbatch_cmd}")

    if args.no_run:
        print("Skipping submission (--no-run specified)")
        return

    # Parse environment variables
    env = dict(os.environ)
    if env_string:
        for env_assignment in env_string.split():
            if "=" in env_assignment:
                key, value = env_assignment.split("=", 1)
                env[key] = value
                print(f"Setting env: {key}={value}")

    # Execute sbatch on the host
    print(f"Submitting: {sbatch_cmd}")
    sbatch_result = oellm_autoexp.utils.run.run_with_tee(
        shlex.split(sbatch_cmd), env=env, capture_output=True, text=True
    )

    print(sbatch_result.stdout)
    if sbatch_result.stderr:
        print("STDERR:", sbatch_result.stderr)

    if sbatch_result.returncode != 0:
        raise SystemExit(f"sbatch failed with return code {sbatch_result.returncode}")

    # Extract job ID and show log location hint
    job_match = re.search(r"Submitted batch job (\d+)", sbatch_result.stdout)
    if job_match:
        job_id = job_match.group(1)
        print(f"\nSuccessfully submitted job {job_id}")

        # Try to extract log paths from the container output
        log_matches = re.findall(r"log:\s*(\S+)", result.stdout)
        if log_matches:
            print("\nOutput logs:")
            for log_path in log_matches:
                print(f"  - {log_path}")
        else:
            print("\nCheck SLURM logs in your configured log directory")


if __name__ == "__main__":
    main()
