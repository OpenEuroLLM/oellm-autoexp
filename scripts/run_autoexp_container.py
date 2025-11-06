#!/usr/bin/env python3
"""Coordinate planning inside a container and submission on the host."""

from __future__ import annotations

import argparse
import re
import shlex
from pathlib import Path

from oellm_autoexp.utils.run import run_with_tee


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan in container, submit on host.")
    parser.add_argument("--image", help="Container image path")
    parser.add_argument("--apptainer-cmd", default="singularity")
    parser.add_argument("--container-python", default="python")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--config-ref", default="autoexp")
    parser.add_argument("-C", "--config-dir", type=Path, default=Path("config"))
    parser.add_argument("--plan-id", help="Explicit plan identifier")
    parser.add_argument("--no-submit", action="store_true", help="Only generate the plan manifest")
    parser.add_argument("--no-monitor", action="store_true", help="Submit jobs but skip monitoring")
    parser.add_argument("--monitor-only", action="store_true", help="Only run host monitoring")
    parser.add_argument(
        "--dry-run", action="store_true", help="Plan and describe submissions without sbatch"
    )
    parser.add_argument(
        "--use-fake-slurm", action="store_true", help="Use in-memory SLURM stub on host"
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Environment variable assignment passed to the container (VAR=VALUE)",
    )
    parser.add_argument(
        "--bind",
        action="append",
        default=[],
        help="Bind mount passed to the container (Singularity --bind syntax)",
    )
    parser.add_argument("override", nargs="*", default=[], help="Configuration overrides")
    return parser.parse_args(argv)


def _build_container_command(args: argparse.Namespace, inner: list[str]) -> list[str]:
    quoted = shlex.join(inner)
    command = [args.apptainer_cmd, "exec"]
    for bind in getattr(args, "bind", []) or []:
        command.extend(["--bind", bind])
    for env_assignment in getattr(args, "env", []) or []:
        command.extend(["--env", env_assignment])
    command.extend([args.image, "bash", "-lc", quoted])
    return command


def _load_container_config(config_ref: str, config_dir: Path, overrides: list[str]) -> dict | None:
    try:
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
    except ImportError:
        return None

    config_dir = Path(config_dir).resolve()
    if not config_dir.exists():
        return None

    try:
        hydra_overrides = [
            override.split("=")[0] + '="' + "=".join(override.split("=")[1:]) + '"'
            if "${" in override
            else override
            for override in overrides
        ]
        with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            cfg = compose(config_name=config_ref, overrides=list(hydra_overrides))
        container_cfg = OmegaConf.to_container(cfg.get("container", None), resolve=True)
        return container_cfg if container_cfg else None
    except Exception:
        return None


def _run_host_command(cmd: list[str], *, use_pty: bool = False) -> int:
    print("Running on host:", shlex.join(cmd))
    result = run_with_tee(cmd, use_pty=use_pty)
    return result.returncode


def _extract_manifest_path(output: str) -> Path | None:
    match = re.search(r"Plan manifest written to:\s*(.+)", output)
    if not match:
        return None
    candidate = match.group(1).strip()
    return Path(candidate).expanduser().resolve()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    scripts_dir = Path(__file__).resolve().parent
    plan_script = scripts_dir / "plan_autoexp.py"
    submit_script = scripts_dir / "submit_autoexp.py"
    monitor_script = scripts_dir / "monitor_autoexp.py"
    manifest_path: Path | None = args.manifest.resolve() if args.manifest else None

    if args.monitor_only:
        if manifest_path is None:
            raise SystemExit("monitor-only requires --manifest to be specified")
        host_cmd = [
            "python",
            str(monitor_script),
            "--manifest",
            str(manifest_path),
        ]
        if args.use_fake_slurm:
            host_cmd.append("--use-fake-slurm")
        if args.verbose:
            host_cmd.append("--verbose")
        if args.debug:
            host_cmd.append("--debug")
        raise SystemExit(_run_host_command(host_cmd))

    container_image = args.image
    container_runtime = args.apptainer_cmd

    if not container_image:
        cfg = _load_container_config(args.config_ref, args.config_dir, args.override)
        if cfg:
            container_image = cfg.get("image")
            if cfg.get("runtime"):
                container_runtime = cfg["runtime"]
            cfg_binds = cfg.get("bind") or []
            if cfg_binds:
                args.bind = [*cfg_binds, *getattr(args, "bind", [])]
            cfg_env = cfg.get("env") or {}
            if cfg_env:
                args.env = [*(f"{k}={v}" for k, v in cfg_env.items()), *getattr(args, "env", [])]
            if container_image:
                print(f"Using container from config: {container_image}")

    container_python = getattr(args, "container_python", "python")

    plan_cmd = [
        container_python,
        str(plan_script),
        "--config-ref",
        args.config_ref,
        "-C",
        str(args.config_dir),
    ]
    if manifest_path is not None:
        plan_cmd.extend(["--manifest", str(manifest_path)])
    if args.plan_id:
        plan_cmd.extend(["--plan-id", args.plan_id])
    if container_image:
        plan_cmd.extend(["--container-image", container_image])
        plan_cmd.extend(["--container-runtime", container_runtime])
    plan_cmd.extend(args.override)

    if container_image:
        args.image = container_image
        args.apptainer_cmd = container_runtime
        container_cmd = _build_container_command(args, plan_cmd)
        print("Running inside container:", shlex.join(container_cmd))
        plan_result = run_with_tee(container_cmd, text=True)
    else:
        print("No container configured; running planning step on host.")
        plan_result = run_with_tee(plan_cmd, text=True)

    if plan_result.returncode != 0:
        raise SystemExit(plan_result.returncode)

    if manifest_path is None:
        manifest_path = _extract_manifest_path(plan_result.stdout or "")
        if manifest_path is None:
            raise SystemExit(
                "Unable to determine manifest path from plan output; please pass --manifest explicitly."
            )

    if args.no_submit:
        print("Plan generated; skipping submission (--no-submit).")
        return

    submit_cmd = [
        "python",
        "-u",
        str(submit_script),
        "--manifest",
        str(manifest_path),
    ]
    if args.no_monitor:
        submit_cmd.append("--no-monitor")
    if args.dry_run:
        submit_cmd.append("--dry-run")
    if args.use_fake_slurm:
        submit_cmd.append("--use-fake-slurm")
    if args.verbose:
        submit_cmd.append("--verbose")
    if args.debug:
        submit_cmd.append("--debug")

    rc = _run_host_command(submit_cmd, use_pty=True)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
