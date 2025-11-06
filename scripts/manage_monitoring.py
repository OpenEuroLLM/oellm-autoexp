#!/usr/bin/env python3
"""Utility to list and manage monitoring sessions."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from oellm_autoexp.persistence import MonitorStateStore


def list_sessions(monitoring_state_dir: str) -> None:
    """List all active monitoring sessions."""
    sessions = MonitorStateStore.list_sessions(monitoring_state_dir)

    if not sessions:
        print("No monitoring sessions found.")
        return

    print(f"Found {len(sessions)} monitoring session(s):\n")
    for session in sessions:
        created = datetime.fromtimestamp(session["created_at"]).strftime("%Y-%m-%d %H:%M:%S")
        print(f"Session ID: {session['session_id']}")
        print(f"  Project: {session['project_name']}")
        print(f"  Created: {created}")
        print(f"  Jobs: {session['job_count']}")
        print(f"  File: {session['session_path']}")
        print()


def show_session(monitoring_state_dir: str, session_id: str) -> None:
    """Show details of a specific monitoring session."""
    sessions = MonitorStateStore.list_sessions(monitoring_state_dir)
    session = next((s for s in sessions if s["session_id"] == session_id), None)

    if not session:
        print(f"Session {session_id} not found.")
        sys.exit(1)

    session_path = Path(session["session_path"])
    session_data = MonitorStateStore.load_session(session_path)

    if not session_data:
        print(f"Failed to load session {session_id}")
        sys.exit(1)

    print(f"Session: {session_id}")
    print(f"Project: {session_data['project_name']}")
    print(
        f"Created: {datetime.fromtimestamp(session_data['created_at']).strftime('%Y-%m-%d %H:%M:%S')}"
    )
    print(f"\nSession file: {session_path}")

    # Show job status
    jobs = session_data.get("jobs", [])
    if jobs:
        print(f"\nJobs: {len(jobs)}")
        for job in jobs:
            print(
                f"  - {job['name']} (job_id: {job['job_id']}, attempts: {job.get('attempts', 1)})"
            )
    else:
        print("\nNo jobs in session.")


def remove_session(monitoring_state_dir: str, session_id: str, force: bool = False) -> None:
    """Remove a monitoring session."""
    sessions = MonitorStateStore.list_sessions(monitoring_state_dir)
    session = next((s for s in sessions if s["session_id"] == session_id), None)

    if not session:
        print(f"Session {session_id} not found.")
        sys.exit(1)

    session_path = Path(session["session_path"])
    session_data = MonitorStateStore.load_session(session_path)

    if not session_data:
        print(f"Failed to load session {session_id}")
        sys.exit(1)

    # Check if there are active jobs
    if not force:
        jobs = session_data.get("jobs", [])
        if jobs:
            print(f"Warning: Session {session_id} has {len(jobs)} active job(s).")
            print("Use --force to remove anyway, or cancel the jobs first.")
            sys.exit(1)

    # Remove session file
    if session_path.exists():
        session_path.unlink()
        print(f"Removed session: {session_path}")

    print(f"Session {session_id} removed.")


def dump_config(monitoring_state_dir: str, session_id: str, output: str) -> None:
    """Dump the config from a monitoring session."""
    sessions = MonitorStateStore.list_sessions(monitoring_state_dir)
    session = next((s for s in sessions if s["session_id"] == session_id), None)

    if not session:
        print(f"Session {session_id} not found.")
        sys.exit(1)

    session_path = Path(session["session_path"])
    session_data = MonitorStateStore.load_session(session_path)

    if not session_data or "config" not in session_data:
        print(f"Failed to load config from session {session_id}")
        sys.exit(1)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(session_data["config"], indent=2), encoding="utf-8")
    print(f"Config dumped to: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage monitoring sessions")
    parser.add_argument(
        "--monitoring-state-dir",
        type=Path,
        default=Path("output/monitoring_state"),
        help="Monitoring state directory (default: output/monitoring_state)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # List command
    subparsers.add_parser("list", help="List all monitoring sessions")

    # Show command
    show_parser = subparsers.add_parser("show", help="Show details of a session")
    show_parser.add_argument("session_id", help="Session ID")

    # Remove command
    remove_parser = subparsers.add_parser("remove", help="Remove a monitoring session")
    remove_parser.add_argument("session_id", help="Session ID")
    remove_parser.add_argument(
        "--force",
        action="store_true",
        help="Force removal even if jobs are active",
    )

    # Dump config command
    dump_parser = subparsers.add_parser("dump-config", help="Dump config from a session")
    dump_parser.add_argument("session_id", help="Session ID")
    dump_parser.add_argument("--output", type=Path, required=True, help="Output file path")

    args = parser.parse_args()

    if args.command == "list":
        list_sessions(args.monitoring_state_dir)
    elif args.command == "show":
        show_session(args.monitoring_state_dir, args.session_id)
    elif args.command == "remove":
        remove_session(args.monitoring_state_dir, args.session_id, args.force)
    elif args.command == "dump-config":
        dump_config(args.monitoring_state_dir, args.session_id, args.output)


if __name__ == "__main__":
    main()
