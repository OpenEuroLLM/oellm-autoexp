#!/usr/bin/env python3
"""Reset job records in a monitor session so they can be re-submitted.

Usage:
    python tools/refresh_monitor.py --session-dir ./monitor_state/<session_id>
    python tools/refresh_monitor.py --session-dir ./monitor_state/<session_id> --dry-run
    python tools/refresh_monitor.py --session-dir ./monitor_state/<session_id> --only-finished

# Preview what will be reset
python tools/refresh_monitor.py --session-dir ./monitor_state/1776674681 --dry-run

# Apply and then restart monitoring
python tools/refresh_monitor.py --session-dir ./monitor_state/1776674681
bash run_monitoring.sh 1776674681
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _reset_runtime(runtime: dict) -> dict:
    """Return a copy of runtime with submission fields cleared."""
    return {
        **runtime,
        "submitted": False,
        "runtime_job_id": None,
        "start_ts": None,
        "end_ts": None,
        "log_cursor": 0,
        "last_status": None,
        "final_state": None,
        # attempts, condition_state, action_state are preserved intentionally
    }


def _needs_reset(runtime: dict, only_finished: bool) -> tuple[bool, str]:
    final = runtime.get("final_state")
    submitted = runtime.get("submitted", False)
    runtime_id = runtime.get("runtime_job_id")
    last_status = runtime.get("last_status")

    if final is not None:
        return True, f"final_state={final!r}"
    if not only_finished and submitted and runtime_id and last_status is None:
        return True, "limbo (submitted, no terminal status recorded)"
    return False, ""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", required=True, type=Path)
    parser.add_argument(
        "--only-finished",
        action="store_true",
        help="Only reset jobs with final_state set; skip limbo jobs",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print what would change, don't write")
    args = parser.parse_args(argv)

    session_dir = args.session_dir
    if not session_dir.exists():
        print(f"Session directory does not exist: {session_dir}")
        return

    paths = sorted(session_dir.glob("*.job.json"))
    if not paths:
        print("No job files found.")
        return

    reset_count = 0
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            print(f"  SKIP {path.name}: {e}")
            continue

        runtime = data.get("runtime", {})
        should_reset, reason = _needs_reset(runtime, args.only_finished)

        if not should_reset:
            print(f"  ok   {path.name}")
            continue

        print(f"  reset {path.name}  ({reason})")
        if not args.dry_run:
            data["runtime"] = _reset_runtime(runtime)
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        reset_count += 1

    if args.dry_run:
        print(f"\n[dry-run] would reset {reset_count}/{len(paths)} job(s)")
    else:
        print(f"\nReset {reset_count}/{len(paths)} job(s). Run the monitoring loop to re-submit.")


if __name__ == "__main__":
    main()

setfacl -m u:slaing00:rwx /leonardo_work/OELLM_prod2026/slaing00/multilingual_scaling

