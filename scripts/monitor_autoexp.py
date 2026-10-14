#!/usr/bin/env python3
"""Convenience wrapper to plan, submit, and optionally monitor in one shot."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Iterable
from uuid import uuid4


from oellm_autoexp.orchestrator import (
    DEFAULT_STALE_AFTER_S,
    ensure_no_live_monitor,
    run_loop,
    stop_path,
)
from oellm_autoexp.monitor.local_client import LocalCommandClient, LocalCommandClientConfig
from oellm_autoexp.monitor.slurm_client import SlurmClient, SlurmClientConfig
from oellm_autoexp.monitor.loop import JobFileStore, MonitorLoop
from oellm_autoexp.utils.logging_config import configure_logging


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monitor-state-dir", default="./monitor_state", type=Path)
    parser.add_argument("--no-verbose", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--session", default=None)
    parser.add_argument("--session-dir", default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Start even if another monitor's heartbeat is still fresh. Two "
        "monitors on one session race on every job record (double sbatch, lost "
        "log cursors), so only use this when you know the other one is gone.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=None,
        help="Seconds between polls (default 60). Mainly for tests; a shorter "
        "interval also shortens the heartbeat refresh, so keep --stale-after above it.",
    )
    parser.add_argument(
        "--stale-after",
        type=float,
        default=DEFAULT_STALE_AFTER_S,
        help="Seconds after which another monitor's heartbeat counts as dead.",
    )
    args = parser.parse_args(argv)
    if args.session is None and args.session_dir is None:
        print("Error either session or session-dir is required")
        parser.print_help()
        exit(1)
    return args


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

    slurm_client = SlurmClient(SlurmClientConfig())
    local_client = LocalCommandClient(LocalCommandClientConfig())
    # NB `Path(x) or fallback` does NOT work here: Path(None) raises TypeError
    # before the `or` is ever evaluated, so --session alone always crashed with
    #   TypeError: argument should be a str or an os.PathLike ... not 'NoneType'
    # and only --session-dir was usable.
    if args.session_dir:
        session_dir = Path(args.session_dir)
    else:
        session_dir = Path(args.monitor_state_dir) / args.session
    if not session_dir.exists():
        print(f"Session directory {session_dir} does not exist.")
        raise SystemExit(2)

    # Log to the session dir as well as the console: tmux scrollback dies with
    # the login node, so this is the only durable record of why a monitor
    # stopped.
    configure_logging(not args.no_verbose, args.debug, log_file=session_dir / "monitor.log")

    # Refuse rather than silently clearing: presence of the file IS the stop
    # request, and the supervisor reads the same file to decide whether to keep
    # its hands off. Removing it automatically here would let a supervised
    # relaunch quietly override a stop the user asked for.
    stop_file = stop_path(session_dir)
    if stop_file.exists():
        raise SystemExit(
            f"Stop requested for this session: {stop_file.read_text().strip()}\n"
            f"Remove it to resume:  rm {stop_file}"
        )

    ensure_no_live_monitor(session_dir, stale_after_s=args.stale_after, force=args.force)

    loop_kwargs = {}
    if args.poll_interval is not None:
        loop_kwargs["poll_interval_seconds"] = args.poll_interval
    loop = MonitorLoop(
        store=JobFileStore(str(session_dir)),
        slurm_client=slurm_client,
        local_client=local_client,
        **loop_kwargs,
    )
    # Attaching to a session we did not submit: the clients have no idea these
    # jobs exist, and SlurmClient.squeue() returns {} while it tracks nothing.
    # Without this the monitor runs forever without ever seeing a job status.
    loop.rehydrate()

    outcome = run_loop(loop)
    if outcome == "signal":
        print("\n[interrupted] Jobs were NOT cancelled -- they keep running in SLURM.")
        print(
            f"Resume with:  rm {stop_file} && python scripts/monitor_autoexp.py --session-dir {session_dir}"
        )
        raise SystemExit(130)
    if outcome == "failed":
        raise SystemExit(1)
    if outcome == "stopped":
        raise SystemExit(0)


if __name__ == "__main__":
    main()
