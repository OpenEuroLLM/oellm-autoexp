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


from oellm_autoexp.orchestrator import run_loop
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
        "--reset-jobs",
        nargs="+",
        metavar="PATTERN",
        default=None,
        help=(
            "Reset one or more jobs in the session so the monitor resubmits them. "
            "Each PATTERN is matched as a substring of the job filename. "
            "Example: --reset-jobs lr0.002_gbsz256_decay lr0.001_gbsz512_stable300BT"
        ),
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


def _reset_jobs(session_dir: Path, patterns: list[str]) -> None:
    """Reset matching jobs so the monitor will resubmit them on next poll."""
    import json, shutil

    matched = 0
    for path in sorted(session_dir.glob("*.job.json")):
        if not any(p in path.name for p in patterns):
            continue
        data = json.loads(path.read_text())
        rt = data.get("runtime", {})
        old = {k: rt.get(k) for k in ("final_state", "submitted", "runtime_job_id", "attempts")}
        rt["final_state"] = None
        rt["submitted"] = False
        rt["runtime_job_id"] = None
        rt["log_cursor"] = 0
        rt["last_status"] = None
        rt["action_state"] = {}
        new = {k: rt.get(k) for k in ("final_state", "submitted", "runtime_job_id", "attempts")}
        backup = path.with_suffix(".bak.json")
        shutil.copy2(path, backup)
        path.write_text(json.dumps(data, indent=2))
        print(f"  reset {path.name}")
        print(f"    before: {old}")
        print(f"    after:  {new}")
        matched += 1

    if matched == 0:
        print(f"No jobs matched patterns: {patterns}")
    else:
        print(f"\nReset {matched} job(s). Backups saved as *.bak.json")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging(not args.no_verbose, args.debug)

    session_dir = Path(args.session_dir) if args.session_dir else Path(args.monitor_state_dir) / args.session
    if not session_dir.exists():
        print(f"Session directory {session_dir} does not exist.")
        return

    if args.reset_jobs:
        _reset_jobs(session_dir, args.reset_jobs)
        return

    slurm_client = SlurmClient(SlurmClientConfig())
    local_client = LocalCommandClient(LocalCommandClientConfig())
    loop = MonitorLoop(
        store=JobFileStore(str(session_dir)),
        slurm_client=slurm_client,
        local_client=local_client,
    )

    run_loop(loop)


if __name__ == "__main__":
    main()
