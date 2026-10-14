#!/usr/bin/env python3
"""Close out monitor sessions whose jobs are long gone. Run ON the cluster.

A monitor that is killed -- login node reboot, Ctrl-C, a crash -- never gets to
write ``final_state`` on the jobs it was watching. The records stay "active"
forever. Measured on JUPITER 2026-08-28: **367 of 506** session directories
still held such a record, months after the jobs themselves ended.

That is not just untidy. Every tool that reads ``monitor_state`` has to guess
which sessions matter, and 35 of those sessions hold records with
``submitted: false`` -- so anything that naively "resumes what looks active"
would *submit old work*.

This script resolves them instead of flattening them. For every unfinished
record it asks SLURM what actually happened and writes the real answer:

    COMPLETED                                   -> final_state "finished"
    FAILED / CANCELLED / TIMEOUT / NODE_FAIL    -> final_state "cancelled"
    not in sacct any more, or never submitted   -> final_state "retired"

``finished`` and ``cancelled`` are the vocabulary the monitor itself uses, so
retired sessions read exactly like ones that ended under supervision. Verified
that sacct still answers for two-week-old job ids, so the real outcome is worth
recovering while it is still recoverable.

    python scripts/retire_sessions.py --state-dir monitor_state           # dry run
    python scripts/retire_sessions.py --state-dir monitor_state --apply
    python scripts/retire_sessions.py --session-dir monitor_state/1786722705 --apply

SAFETY
  * Dry run by default; nothing is written without ``--apply``.
  * A session with a job in the queue RIGHT NOW is skipped entirely, so a live
    run can never be retired out from under its monitor.
  * Provenance goes in ``runtime.action_state["retired"]``, never in a new
    top-level field: ``JobFileStore.load_all`` silently skips records it cannot
    parse, so an unknown key would make the job VANISH from monitoring rather
    than raise. (Verified.)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

# sacct state -> the final_state the monitor itself would have written.
_TERMINAL = {
    "COMPLETED": "finished",
    "CANCELLED": "cancelled",
    "FAILED": "cancelled",
    "TIMEOUT": "cancelled",
    "NODE_FAIL": "cancelled",
    "OUT_OF_MEMORY": "cancelled",
    "PREEMPTED": "cancelled",
    "BOOT_FAIL": "cancelled",
    "DEADLINE": "cancelled",
}
# States that mean the job has NOT ended; a session holding one is left alone.
_LIVE = {"RUNNING", "PENDING", "SUSPENDED", "REQUEUED", "RESIZING", "CONFIGURING"}

UNRESOLVED = "retired"


def _normalize(state: str) -> str:
    """`CANCELLED by 29685` -> `CANCELLED`, matching SlurmClient's parsing."""
    state = state.strip()
    for known in ("CANCELLED", "COMPLETED", "FAILED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY"):
        if known in state:
            return known
    return state.split()[0] if state else ""


def _run(argv: list[str]) -> str:
    if shutil.which(argv[0]) is None:
        return ""
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=120, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return proc.stdout if proc.returncode == 0 else ""


def queued_job_ids() -> set[str]:
    """Everything of ours currently in the queue.

    Empty if squeue is absent.
    """
    out = _run(["squeue", "-u", os.environ.get("USER") or "", "-h", "-o", "%i"])
    return {line.strip() for line in out.splitlines() if line.strip()}


def sacct_states(job_ids: list[str]) -> dict[str, tuple[str, str]]:
    """job_id -> (normalized state, end time).

    Batched; missing ids are absent.
    """
    states: dict[str, tuple[str, str]] = {}
    for start in range(0, len(job_ids), 200):  # keep the command line sane
        chunk = job_ids[start : start + 200]
        out = _run(
            [
                "sacct",
                "--jobs",
                ",".join(chunk),
                "-X",
                "--noheader",
                "--format",
                "JobID,State,End",
                "--parsable2",
            ]
        )
        for line in out.splitlines():
            parts = line.strip().split("|")
            if len(parts) < 2:
                continue
            states[parts[0].strip()] = (
                _normalize(parts[1]),
                parts[2].strip() if len(parts) > 2 else "",
            )
    return states


def _end_ts(end: str) -> float | None:
    try:
        return datetime.fromisoformat(end).timestamp()
    except ValueError:
        return None


def _load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _unfinished(payload: dict) -> bool:
    return (payload.get("runtime") or {}).get("final_state") is None


def session_dirs(state_dir: Path) -> list[Path]:
    return sorted(
        (d for d in state_dir.iterdir() if d.is_dir() and d.name != "manifests"),
        key=lambda d: d.name,
    )


def plan_session(session: Path, queued: set[str]) -> tuple[list[tuple], str | None]:
    """Return (per-record plan, skip reason).

    Records are (path, payload, job_id).
    """
    records: list[tuple] = []
    for path in sorted(session.glob("*.job.json")):
        payload = _load(path)
        if payload is None:
            records.append((path, None, None))  # unreadable; reported, never written
            continue
        if not _unfinished(payload):
            continue
        job_id = (payload.get("runtime") or {}).get("runtime_job_id")
        if job_id and str(job_id) in queued:
            return [], f"live job {job_id} is in the queue"
        records.append((path, payload, str(job_id) if job_id else None))
    return records, None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--state-dir", type=Path, help="monitor_state folder: every session in it")
    src.add_argument("--session-dir", type=Path, help="just this one session")
    parser.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    parser.add_argument("--quiet", action="store_true", help="summary only")
    args = parser.parse_args(argv)

    sessions = [args.session_dir] if args.session_dir else session_dirs(args.state_dir)
    missing = [s for s in sessions if not s.is_dir()]
    if missing:
        parser.error(f"no such session directory: {missing[0]}")

    queued = queued_job_ids()
    if not queued:
        print("NOTE: squeue returned nothing (no queued jobs, or squeue unavailable here).")
        print("      Sessions cannot be protected by the live-job check -- review the plan.\n")

    # Plan first, so all of sacct is asked in one batch rather than per session.
    plans: dict[Path, list[tuple]] = {}
    skipped: dict[Path, str] = {}
    for session in sessions:
        records, reason = plan_session(session, queued)
        if reason:
            skipped[session] = reason
        elif records:
            plans[session] = records

    job_ids = [jid for recs in plans.values() for (_p, _pl, jid) in recs if jid]
    states = sacct_states(sorted(set(job_ids)))

    counts: dict[str, int] = {}
    unreadable = 0
    changed_sessions = 0

    for session, records in plans.items():
        touched = False
        for path, payload, job_id in records:
            if payload is None:
                unreadable += 1
                if not args.quiet:
                    print(f"  {session.name}  {path.name}  <unreadable JSON> -> skipped")
                continue

            if job_id is None:
                slurm_state, end = "<never submitted>", ""
                final = UNRESOLVED
            elif job_id in states:
                slurm_state, end = states[job_id]
                if slurm_state in _LIVE:
                    # Gone from squeue between our two calls, or an odd state.
                    # Leave it: the next run will resolve it properly.
                    if not args.quiet:
                        print(f"  {session.name}  job {job_id}  {slurm_state} -> left alone")
                    continue
                final = _TERMINAL.get(slurm_state, UNRESOLVED)
            else:
                slurm_state, end = "<not in sacct>", ""
                final = UNRESOLVED

            counts[final] = counts.get(final, 0) + 1
            touched = True
            if not args.quiet:
                job_desc = f"job {job_id}" if job_id else "unsubmitted"
                print(f"  {session.name}  {job_desc}  {slurm_state} -> {final}")

            if args.apply:
                runtime = payload.setdefault("runtime", {})
                runtime["final_state"] = final
                if end and runtime.get("end_ts") is None:
                    runtime["end_ts"] = _end_ts(end)
                # action_state is a free-form dict, so this is schema-safe.
                # A new top-level key would make the record unparseable, and
                # load_all() skips unparseable files SILENTLY.
                runtime.setdefault("action_state", {})["retired"] = {
                    "at": time.time(),
                    "by": "scripts/retire_sessions.py",
                    "slurm_state": slurm_state,
                    "slurm_end": end or None,
                }
                _write_atomic(path, payload)
        if touched:
            changed_sessions += 1

    for session, reason in skipped.items():
        print(f"  {session.name}  SKIPPED ({reason})")

    total = sum(counts.values())
    detail = ", ".join(f"{n} {k}" for k, n in sorted(counts.items())) or "nothing"
    verb = "retired" if args.apply else "would be retired"
    print(
        f"\n{total} record(s) across {changed_sessions} session(s) {verb} [{detail}]"
        f"; {len(skipped)} session(s) skipped as live"
        + (f"; {unreadable} unreadable record(s)" if unreadable else "")
    )
    if not args.apply and total:
        print("Dry run -- pass --apply to write.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
