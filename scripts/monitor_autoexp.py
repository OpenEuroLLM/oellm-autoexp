#!/usr/bin/env python3
"""Resume monitoring for jobs recorded in monitoring state sessions."""

from __future__ import annotations

import argparse
import logging
import threading
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from oellm_autoexp.workflow.host import (
    build_host_runtime,
    instantiate_controller,
    run_monitoring,
)
from oellm_autoexp.workflow.manifest import read_manifest
from oellm_autoexp.persistence.state_store import MonitorStateStore
from omegaconf import OmegaConf


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
        "override",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Hydra-style override applied to the monitoring config "
        "(e.g. debug_sync=true, log_signals[0].state.key=finished)",
    )
    parser.add_argument(
        "--include-completed",
        action="store_true",
        help="Include sessions without active jobs (default: skip completed ones)",
    )
    return parser.parse_args(argv)


def _parse_override_value(raw: str) -> Any:
    try:
        wrapper = OmegaConf.create({"value": raw})
        return wrapper["value"]
    except Exception:
        return raw


def _apply_monitor_overrides(spec, overrides: list[str]) -> None:
    if not overrides:
        return
    cfg = OmegaConf.create(spec.config)
    for item in overrides:
        allow_new = item.startswith("+")
        expression = item[1:] if allow_new else item
        if "=" not in expression:
            raise SystemExit(f"Invalid monitor override '{item}'; expected KEY=VALUE syntax")
        key, value_str = expression.split("=", 1)
        key = key.strip()
        value = _parse_override_value(value_str.strip())
        OmegaConf.update(cfg, key, value, merge=True)
    spec.config = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]


def _resolve_session_path(candidate: str, state_dir: Path) -> Path:
    path = Path(candidate)
    if path.suffix == ".json" and path.exists():
        return path.resolve()
    derived = state_dir / f"{candidate}.json"
    if derived.exists():
        return derived.resolve()
    raise SystemExit(
        f"Session file not found for '{candidate}'. Provide a path to the JSON file or"
        f" ensure it exists under {state_dir}."
    )


def _load_session_target(session_path: Path) -> MonitorTarget:
    payload = MonitorStateStore.load_session(session_path)
    if not payload:
        raise SystemExit(f"Unable to load monitoring session from {session_path}")
    manifest_path = payload.get("manifest_path")
    if not manifest_path:
        raise SystemExit(
            f"Monitoring session {session_path} does not record 'manifest_path'. "
            "Please rerun with --manifest pointing to the original plan."
        )
    session_id = payload.get("session_id") or session_path.stem
    job_count = len(payload.get("jobs", []))
    return MonitorTarget(
        manifest_path=Path(manifest_path).resolve(),
        session_path=session_path.resolve(),
        session_id=session_id,
        job_count=job_count,
    )


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

    controller = instantiate_controller(runtime)

    if not list(controller.jobs()):
        print(
            f"No jobs registered in monitoring session {runtime.state_store.session_id}; nothing to monitor."
        )
        return

    session_file = runtime.state_store.session_path
    if target.session_path and os.path.abspath(session_file) != os.path.abspath(
        target.session_path
    ):
        logging.warning(
            "Session file mismatch: runtime points to %s but CLI provided %s",
            session_file,
            target.session_path,
        )
    print(f"Monitoring session: {runtime.state_store.session_id} ({session_file})")
    run_monitoring(runtime, controller)


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
