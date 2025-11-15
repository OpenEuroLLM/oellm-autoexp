#!/usr/bin/env python3
"""Resume monitoring for jobs described by a plan manifest."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from dataclasses import dataclass
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="Plan manifest to monitor.")
    parser.add_argument(
        "--session",
        help="Existing monitoring session ID or path (alternative to --manifest).",
    )
    parser.add_argument(
        "--monitoring-state-dir",
        type=Path,
        help="Location of monitoring_state/ (required with --session).",
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
    parser.add_argument(
        "--monitor-override",
        action="append",
        default=[],
        metavar="key=value",
        help="Override monitor config entries before instantiating the controller.",
    )
    return parser.parse_args(argv)


@dataclass
class SessionTarget:
    session_id: str
    session_path: Path
    manifest_path: Path


def _resolve_session_path(value: str, state_dir: Path | None) -> Path:
    candidate = Path(value)
    if candidate.exists():
        return candidate.resolve()
    if state_dir is None:
        raise SystemExit("--monitoring-state-dir is required when --session is not a path")
    path = Path(state_dir) / f"{value}.json"
    if not path.exists():
        raise SystemExit(f"Session file not found: {path}")
    return path.resolve()


def _load_session_target(path: Path) -> SessionTarget:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"Unable to read session file {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Session file {path} is not valid JSON: {exc}") from exc

    manifest = payload.get("manifest_path")
    if not manifest:
        raise SystemExit(f"Session file {path} missing 'manifest_path'")
    manifest_path = Path(manifest).expanduser().resolve()
    if not manifest_path.exists():
        raise SystemExit(f"Manifest referenced by session not found: {manifest_path}")

    session_id = payload.get("session_id") or path.stem
    return SessionTarget(session_id=session_id, session_path=path, manifest_path=manifest_path)


def _apply_monitor_overrides(spec, overrides: list[str]) -> None:
    if not overrides:
        return
    config: dict[str, Any] = dict(spec.config or {})

    for override in overrides:
        if "=" not in override:
            raise SystemExit(f"Invalid override (expected key=value): {override}")
        key, value = override.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"Invalid override key in: {override}")
        parsed: Any = value.strip()
        lower = parsed.lower()
        if lower in {"true", "false"}:
            parsed = lower == "true"
        else:
            try:
                parsed = int(parsed)
            except ValueError:
                try:
                    parsed = float(parsed)
                except ValueError:
                    pass

        target = config
        parts = key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
            if not isinstance(target, dict):
                raise SystemExit(f"Cannot assign nested key '{key}' (non-dict encountered)")
        target[parts[-1]] = parsed

    spec.config = config


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

    manifest_path: Path
    if args.session:
        session_path = _resolve_session_path(args.session, args.monitoring_state_dir)
        target = _load_session_target(session_path)
        manifest_path = target.manifest_path
        print(f"Resolved session '{target.session_id}' -> manifest {manifest_path}")
    elif args.manifest:
        manifest_path = Path(args.manifest).resolve()
    else:
        raise SystemExit("Provide either --manifest or --session")

    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")
    manifest = read_manifest(manifest_path)
    _apply_monitor_overrides(manifest.monitor, args.monitor_override)

    runtime = build_host_runtime(
        manifest,
        use_fake_slurm=args.use_fake_slurm,
        manifest_path=manifest_path,
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


if __name__ == "__main__":
    main()
