#!/usr/bin/env python3
"""Run autoexp with optional container execution.

This script intelligently handles execution based on container configuration:
- If --image is provided: Use that specific container image
- If --image is not provided: Parse config and use container.image if set
- If container config is null/empty: Execute run_autoexp.py directly (no container)

For containerized execution, this handles SLURM submission from outside the container:
1. Runs plan generation inside the container with --no-submit
2. Parses the sbatch command from container output
3. Executes sbatch on the host system where SLURM commands are available
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import sys
from pathlib import Path
from typing import Optional

from oellm_autoexp.utils.run import run_with_tee


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch autoexp within a container")
    parser.add_argument("--image", help="Container image path (not needed for --monitor-only)")
    parser.add_argument("--apptainer-cmd", default="singularity")
    parser.add_argument("--config-ref", default="autoexp")
    parser.add_argument("-C", "--config-dir", default="config")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Generate scripts but don't submit to SLURM")
    parser.add_argument("--no-monitor", action="store_true", help="Submit jobs but don't monitor them")
    parser.add_argument("--monitor-only", action="store_true", help="Skip submission, only monitor existing jobs")
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


def _load_container_config(config_ref: str, config_dir: Path, overrides: list[str]) -> Optional[dict]:
    """Load container configuration from config using OmegaConf (lazy import).

    Returns None if container is not configured, or a dict with container config.
    """
    try:
        # Lazy import to allow running with --image even without OmegaConf installed
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
    except ImportError:
        print("Warning: hydra/omegaconf not available, cannot auto-detect container config")
        return None

    config_dir = Path(config_dir).resolve()
    if not config_dir.exists():
        print(f"Error: Config directory not found: {config_dir}")
        return None

    try:
        with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            cfg = compose(config_name=config_ref, overrides=list(overrides))

        # Extract container config
        container_cfg = OmegaConf.to_container(cfg.get("container", None), resolve=True)
        return container_cfg if container_cfg else None
    except Exception as e:
        print(f"Warning: Failed to load container config: {e}")
        return None


def main() -> None:
    args = parse_args()

    script = Path(__file__).resolve().parent / "run_autoexp.py"
    base_cmd = [
        "python",
        str(script),
        "--config-ref",
        args.config_ref,
        "-C",
        str(args.config_dir),
    ]

    # Monitor-only mode: run monitoring from the host (no container needed)
    if args.monitor_only:
        monitor_cmd = base_cmd.copy()
        if args.debug:
            monitor_cmd.append("--debug")
        if args.verbose:
            monitor_cmd.append("--verbose")
        if args.fake_submit:
            monitor_cmd.append("--use-fake-slurm")
        monitor_cmd += args.override

        print("Running monitoring from host:", " ".join(shlex.quote(c) for c in monitor_cmd))
        result = run_with_tee(monitor_cmd)
        raise SystemExit(result.returncode)

    # Determine container image to use
    container_image = args.image
    container_runtime = getattr(args, "apptainer_cmd", "singularity")

    if not container_image:
        # Try to load from config
        container_config = _load_container_config(args.config_ref, Path(args.config_dir), args.override)

        if container_config:
            container_image = container_config.get("image")
            if container_config.get("runtime"):
                container_runtime = container_config["runtime"]

            if container_image:
                print(f"Using container from config: {container_image}")

        # If still no container, run directly without container
        if not container_image:
            print("No container configured, running directly on host...")
            direct_cmd = base_cmd.copy()
            if args.no_monitor:
                direct_cmd.append("--no-monitor")
            if args.debug:
                direct_cmd.append("--debug")
            if args.verbose:
                direct_cmd.append("--verbose")
            if args.fake_submit:
                direct_cmd.append("--use-fake-slurm")
            if args.ihelp:
                direct_cmd.append("--help")
            direct_cmd += args.override

            result = run_with_tee(direct_cmd)
            raise SystemExit(result.returncode)

    # At this point we have a container image to use
    print(f"Running in container: {container_image}")

    # Run inside container with --no-submit to generate scripts and output sbatch command
    container_cmd = base_cmd + ["--no-submit"]
    if args.ihelp:
        container_cmd.append("--help")
    if args.debug:
        container_cmd.append("--debug")
    if args.verbose:
        container_cmd.append("--verbose")
    if args.fake_submit:
        container_cmd.append("--use-fake-slurm")

    container_cmd += args.override

    # Override args.image for _build_container_command
    args.image = container_image
    args.apptainer_cmd = container_runtime

    apptainer_cmd = _build_container_command(args, container_cmd)
    print("Running:", " ".join(shlex.quote(c) for c in apptainer_cmd))

    result = run_with_tee(apptainer_cmd, capture_output=True, text=True)

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

    if args.dry_run:
        print("Skipping submission (--dry-run specified)")
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
    sbatch_result = run_with_tee(shlex.split(sbatch_cmd), env=env, capture_output=True, text=True)

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

    # If --no-monitor was specified, we're done
    if args.no_monitor:
        print("\nSkipping monitoring (--no-monitor specified)")
        print("To monitor later, run:")
        print(f"  python {script} --config-ref {args.config_ref} -C {args.config_dir} {' '.join(args.override)}")
        return

    # Otherwise, start monitoring on the host (outside container)
    print("\nStarting monitoring from host...")
    monitor_cmd = base_cmd.copy()
    if args.debug:
        monitor_cmd.append("--debug")
    if args.verbose:
        monitor_cmd.append("--verbose")
    if args.fake_submit:
        monitor_cmd.append("--use-fake-slurm")
    monitor_cmd += args.override

    monitor_result = run_with_tee(monitor_cmd)
    raise SystemExit(monitor_result.returncode)


if __name__ == "__main__":
    main()
