#!/usr/bin/env python3
"""Resume monitoring for jobs recorded in monitoring state sessions."""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from compoconf import parse_config

from oellm_autoexp.workflow.host import (
    build_host_runtime,
    instantiate_controller,
    run_monitoring,
)
from oellm_autoexp.workflow.manifest import read_manifest
from oellm_autoexp.monitor.action_queue import ActionQueue
from oellm_autoexp.monitor.actions import ActionContext, MonitorActionInterface


def _configure_logging(verbose: bool = False, debug: bool = False) -> None:
    level = logging.WARNING
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@dataclass(kw_only=True)
class MonitorTarget:
    manifest_path: Path
    session_path: Path | None = None
    session_id: str | None = None
    job_count: int | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--manifest", type=Path, help="Path to a plan manifest")
    target_group.add_argument(
        "--session",
        type=str,
        help="Session ID or path to a monitoring session JSON (e.g. monitor/<id>.json)",
    )
    target_group.add_argument(
        "--all",
        action="store_true",
        help="Attach to every session in --monitoring-state-dir concurrently",
    )
    parser.add_argument(
        "--monitoring-state-dir",
        type=Path,
        default=Path("monitor"),
        help="Monitoring state directory containing <session_id>.json files (default: ./monitor)",
    )
    parser.add_argument("--use-fake-slurm", action="store_true", help="Use in-memory SLURM backend")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--cmd",
        choices=["resume", "tail", "events", "queue", "actions"],
        default="resume",
        help="Operation to perform.",
    )
    parser.add_argument(
        "--tail-lines",
        type=int,
        default=20,
        help="Number of lines to show per log when cmd=tail.",
    )
    return parser.parse_args(argv)


def _action_queue_path(state_store) -> Path:
    return state_store.session_path.with_suffix(".actions")


def _tail_logs(manifest, lines: int) -> None:
    for job in manifest.jobs:
        log_path = Path(job.log_path)
        print(f"=== {job.name} ({log_path}) ===")
        if not log_path.exists():
            print("log file not found\n")
            continue
        try:
            content = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError as exc:  # pragma: no cover - filesystem errors
            print(f"unable to read log: {exc}\n")
            continue
        tail = content[-lines:] if lines > 0 else content
        for line in tail:
            print(line)
        print()


def _render_timestamp(ts: float | None) -> str:
    if ts is None:
        return "n/a"
    return datetime.fromtimestamp(ts).isoformat(timespec="seconds")


def _print_events(state_store) -> None:
    events = state_store.load_events()
    if not events:
        print("No recorded monitor events.")
        return
    for record in sorted(events.values(), key=lambda r: r.last_seen_ts, reverse=True):
        print(
            f"{record.event_id} | {record.name} | status={record.status.value} "
            f"| count={record.count} | last={_render_timestamp(record.last_seen_ts)}"
        )


def _print_queue(state_store) -> None:
    queue = ActionQueue(_action_queue_path(state_store))
    records = queue.list()
    if not records:
        print("Action queue is empty.")
        return
    counts = Counter(record.status for record in records)
    print("Queue status:", ", ".join(f"{status}={counts[status]}" for status in sorted(counts)))
    for record in records:
        print(
            f"{record.queue_id} | {record.action_class} | status={record.status} "
            f"| event={record.event_id} | updated={_render_timestamp(record.updated_at)}"
        )


def _run_action_worker(runtime) -> None:
    queue = ActionQueue(_action_queue_path(runtime.state_store))
    processed = 0
    while True:
        record = queue.claim_next()
        if record is None:
            break
        events = runtime.state_store.load_events()
        event_record = events.get(record.event_id)
        if event_record is None:
            queue.mark_done(
                record.queue_id,
                status="failed",
                result={"error": "missing event record"},
            )
            continue
        cfg_dict = dict(record.config)
        cfg_dict.setdefault("class_name", record.action_class)
        action_cfg = parse_config(MonitorActionInterface.cfgtype, cfg_dict)
        action = action_cfg.instantiate(MonitorActionInterface)
        context = ActionContext(
            event=event_record,
            job_metadata=record.metadata.get("job", {}),
            workspace=None,
            env={},
        )
        try:
            result = action.execute(context)
        except Exception as exc:  # pragma: no cover - runtime failures
            queue.mark_done(
                record.queue_id,
                status="failed",
                result={"error": str(exc)},
            )
            continue
        action.update_event(event_record, result)
        runtime.state_store.upsert_event(event_record)
        queue.mark_done(
            record.queue_id,
            status="done" if result.status == "success" else "failed",
            result={"message": result.message, "metadata": result.metadata},
        )
        processed += 1
    print(f"Processed {processed} queued action(s).")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _configure_logging(args.verbose, args.debug)

def _resolve_targets(args: argparse.Namespace) -> list[MonitorTarget]:
    if args.manifest:
        manifest_path = Path(args.manifest).resolve()
        if not manifest_path.exists():
            raise SystemExit(f"Manifest not found: {manifest_path}")
        return [MonitorTarget(manifest_path=manifest_path)]

    state_dir = _select_state_dir(args.monitoring_state_dir)
    if args.session:
        session_path = _resolve_session_path(args.session, state_dir)
        target = _load_session_target(session_path)
        if target.job_count == 0 and not args.include_completed:
            print(
                f"Session {target.session_id} has no active jobs; use --include-completed to monitor anyway."
            )
            return []
        return [target]

    if args.all:
        sessions = MonitorStateStore.list_sessions(state_dir)
        if not sessions:
            print(f"No monitoring sessions found under {state_dir}")
            return []
        targets: list[MonitorTarget] = []
        for entry in sessions:
            if entry.get("job_count", 0) == 0 and not args.include_completed:
                continue
            session_path = Path(entry["session_path"]).resolve()
            try:
                target = _load_session_target(session_path)
            except SystemExit as exc:
                logging.warning(str(exc))
                continue
            if target.job_count == 0 and not args.include_completed:
                continue
            targets.append(target)
        return targets

    raise SystemExit("No monitoring target provided")


def _monitor_single(target: MonitorTarget, args: argparse.Namespace) -> None:
    manifest = read_manifest(target.manifest_path)
    _apply_monitor_overrides(manifest.monitor, args.override)

    runtime = build_host_runtime(
        manifest,
        use_fake_slurm=args.use_fake_slurm,
        manifest_path=target.manifest_path,
    )

    if args.cmd == "resume":
        controller = instantiate_controller(runtime)
        if not list(controller.jobs()):
            print("No jobs registered in monitoring session; nothing to monitor.")
            return
        print(f"Monitoring session: {runtime.state_store.session_id}")
        run_monitoring(runtime, controller)
    elif args.cmd == "tail":
        _tail_logs(manifest, max(1, args.tail_lines))
    elif args.cmd == "events":
        _print_events(runtime.state_store)
    elif args.cmd == "queue":
        _print_queue(runtime.state_store)
    elif args.cmd == "actions":
        _run_action_worker(runtime)


def _monitor_all(targets: list[MonitorTarget], args: argparse.Namespace) -> None:
    threads: list[threading.Thread] = []

    def run_target(target: MonitorTarget) -> None:
        try:
            _monitor_single(target, args)
        except SystemExit as exc:
            logging.error(str(exc))

    for target in targets:
        thread = threading.Thread(target=run_target, args=(target,), daemon=False)
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()


def _select_state_dir(requested: Path) -> Path:
    requested = requested.resolve()
    if requested.exists():
        return requested
    # Fall back between common defaults to smooth over older configurations
    fallbacks: list[Path] = []
    if requested == Path("monitor").resolve():
        fallbacks.append(Path("output/monitoring_state"))
    elif requested == Path("output/monitoring_state").resolve():
        fallbacks.append(Path("monitor"))
    for candidate in fallbacks:
        if candidate.exists():
            return candidate.resolve()
    return requested


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _configure_logging(args.verbose, args.debug)

    targets = _resolve_targets(args)
    if not targets:
        return

    if args.all and len(targets) > 1:
        _monitor_all(targets, args)
    else:
        _monitor_single(targets[0], args)


if __name__ == "__main__":
    main()
