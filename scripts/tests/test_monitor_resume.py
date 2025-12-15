#!/usr/bin/env python3
"""Exercise monitoring resume workflow on a SLURM cluster.

This script submits a lightweight plan, interrupts monitoring, and
verifies that a second monitoring session can pick up persisted state
without losing track of running jobs. It mirrors the manual workflow
operators typically follow when a login session drops or they need to
stop the monitor temporarily.

Usage (run from repository root, on a SLURM login node):

    python scripts/tests/test_monitor_resume.py \
        --config-ref autoexp \
        --plan-override project.name=debug_demo

The script will:
  1. Run :mod:`scripts.plan_autoexp` to generate a plan manifest.
  2. Submit the jobs with :mod:`scripts.submit_autoexp` using
     ``--no-monitor`` so the SLURM jobs keep running.
  3. Launch :mod:`scripts.monitor_autoexp` for a short window, then
     send a SIGINT to emulate an operator stopping the monitor.
  4. Launch the monitor a second time and confirm that the controller
     restores the persisted job and reattaches to SLURM state.
  5. Inspect the persisted state file to ensure resolved log paths are
     recorded without ``%j`` placeholders.

The script is intentionally verbose and prints the commands it runs so
you can re-run them manually if needed.  For end-to-end validation on a
real cluster you may want to increase ``--interrupt-after`` or
``--resume-duration`` to give SLURM time to start the job.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from collections.abc import Iterable

from oellm_autoexp.utils.run import run_with_tee


def _log(msg: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")


def _check_environment() -> None:
    """Basic sanity checks before kicking off SLURM jobs."""

    if shutil.which("sbatch") is None:
        raise SystemExit("sbatch not found in PATH; run from a SLURM login node")

    repo_root = Path(__file__).resolve().parents[2]
    marker = repo_root / "pyproject.toml"
    if not marker.exists():
        raise SystemExit(f"Expected to find {marker}; run from repository root")

    os.environ.setdefault("SLURM_ACCOUNT", os.environ.get("USER", "debug"))
    os.environ.setdefault("SLURM_PARTITION", "batch")
    os.environ.setdefault("SLURM_QOS", "normal")


def _plan_experiment(args: argparse.Namespace) -> Path:
    cmd = [
        sys.executable,
        "scripts/plan_autoexp.py",
        "--config-ref",
        args.config_ref,
        "--config-dir",
        str(args.config_dir),
    ]
    if args.manifest is not None:
        cmd.extend(["--manifest", str(args.manifest)])
    if args.plan_id:
        cmd.extend(["--plan-id", args.plan_id])
    if args.plan_override:
        cmd.extend(args.plan_override)

    result = run_with_tee(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit("plan_autoexp.py failed; see logs above")

    manifest_path: Path | None = None
    for line in result.stdout.splitlines():
        if line.startswith("Plan manifest written to:"):
            manifest_path = Path(line.split(":", 1)[1].strip())
            break
    if manifest_path is None:
        raise SystemExit("Unable to locate manifest path in plan output")
    _log(f"Manifest available at {manifest_path}")
    return manifest_path


def _read_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _submit_jobs(manifest_path: Path, *, monitor: bool = False) -> None:
    cmd = [
        sys.executable,
        "scripts/submit_autoexp.py",
        "--manifest",
        str(manifest_path),
    ]
    if not monitor:
        cmd.append("--no-monitor")
    result = run_with_tee(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit("submit_autoexp.py failed; see logs above")


def _resolve_state_file(manifest: dict) -> Path:
    state_dir = Path(manifest["monitoring_state_dir"])
    plan_id = manifest["plan_id"]
    state_file = state_dir / f"{plan_id}.json"
    return state_file


def _wait_for_state_file(path: Path, timeout: int) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if path.exists():
            return
        time.sleep(1)
    raise SystemExit(f"Timed out waiting for state file {path}")


def _stream_output(proc: subprocess.Popen[str], buffer: list[str]) -> None:
    assert proc.stdout is not None
    for line in proc.stdout:
        buffer.append(line.rstrip())
        print(line, end="", flush=True)


def _launch_monitor(
    session_id: str,
    state_dir: Path,
    overrides: Iterable[str],
) -> tuple[subprocess.Popen[str], list[str], threading.Thread]:
    cmd = [
        sys.executable,
        "scripts/monitor_autoexp.py",
        "--session",
        session_id,
        "--monitoring-state-dir",
        str(state_dir),
    ]
    cmd.extend(overrides)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    buffer: list[str] = []
    thread = threading.Thread(target=_stream_output, args=(proc, buffer), daemon=True)
    thread.start()
    return proc, buffer, thread


def _interrupt(
    proc: subprocess.Popen[str], thread: threading.Thread, *, wait: float = 10.0
) -> None:
    time.sleep(wait)
    _log(f"Sending SIGINT to monitor process {proc.pid}")
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        _log("Monitor did not exit after SIGINT; killing")
        proc.kill()
        proc.wait(timeout=5)
    thread.join(timeout=5)


def _ensure_resolved_paths(state_file: Path) -> None:
    payload = json.loads(state_file.read_text(encoding="utf-8"))
    jobs = payload.get("jobs", [])
    if not jobs:
        raise SystemExit(f"No jobs recorded in {state_file}")
    for job in jobs:
        resolved = job.get("resolved_log_path")
        if not resolved or "%" in resolved:
            raise SystemExit(
                "State file still uses unresolved log path templates; expected concrete paths"
            )
    _log(f"Resolved log paths confirmed in {state_file}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-ref", default="autoexp")
    parser.add_argument("--config-dir", type=Path, default=Path("config"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--plan-id", type=str, default="")
    parser.add_argument(
        "--plan-override",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Overrides forwarded to plan_autoexp.py",
    )
    parser.add_argument(
        "--monitor-override",
        nargs="*",
        default=["debug_sync=true"],
        metavar="KEY=VALUE",
        help="Hydra-style overrides forwarded to monitor_autoexp.py",
    )
    parser.add_argument("--interrupt-after", type=float, default=15.0)
    parser.add_argument("--resume-duration", type=float, default=20.0)
    parser.add_argument("--state-timeout", type=int, default=60)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    _check_environment()

    manifest_path = _plan_experiment(args)
    manifest = _read_manifest(manifest_path)

    _submit_jobs(manifest_path, monitor=False)

    state_file = _resolve_state_file(manifest)
    _log(f"Expecting monitor state at {state_file}")
    _wait_for_state_file(state_file, timeout=args.state_timeout)
    _ensure_resolved_paths(state_file)

    monitor_overrides = []
    for item in args.monitor_override:
        monitor_overrides.append(item)

    _log("Launching monitor for the first pass (will interrupt)...")
    session_id = manifest["plan_id"]
    state_dir = state_file.parent

    proc, buffer, thread = _launch_monitor(session_id, state_dir, monitor_overrides)
    _interrupt(proc, thread, wait=args.interrupt_after)

    if not any("Restored" in line for line in buffer):
        _log("Monitor output did not report restored jobs; check logs manually")

    _log("Launching monitor again to verify resume...")
    proc2, buffer2, thread2 = _launch_monitor(session_id, state_dir, monitor_overrides)
    _interrupt(proc2, thread2, wait=args.resume_duration)

    restored_lines = [line for line in buffer2 if "Restored" in line]
    if restored_lines:
        _log("Monitor reported restored jobs on resume:")
        for line in restored_lines:
            _log(f"  {line}")
    else:
        _log("Monitor did not print an explicit restore message; inspect session manually")

    _ensure_resolved_paths(state_file)
    _log("Monitoring resume test completed")


if __name__ == "__main__":
    main()
