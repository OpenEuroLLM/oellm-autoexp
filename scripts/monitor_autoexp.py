#!/usr/bin/env python3
"""Resume monitoring for jobs described by a plan manifest."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from oellm_autoexp.utils.logging_config import configure_logging
from oellm_autoexp.workflow.host import (
    build_host_runtime,
    instantiate_controller,
    run_monitoring,
)
from scripts.utils_plan import read_manifest

LOGGER = logging.getLogger(__name__)


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
        choices=["resume"],
        default="resume",
        help="Operation to perform.",
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
    path = Path(state_dir) / value
    if not path.exists():
        path_file = Path(state_dir) / f"{value}.json"
        if path_file.exists():
            return path_file.resolve()
        raise SystemExit(f"Session directory not found: {path}")
    return path.resolve()


def _load_session_target(path: Path) -> SessionTarget:
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise SystemExit(f"Unable to read session file {path}: {exc}") from exc

        manifest = payload.get("manifest_path")
        if not manifest:
            raise SystemExit(f"Session file {path} missing 'manifest_path'")
        manifest_path = Path(manifest).expanduser().resolve()
        return SessionTarget(
            session_id=payload.get("session_id") or path.stem,
            session_path=path,
            manifest_path=manifest_path,
        )

    raise NotImplementedError(
        "Resuming from directory session not fully implemented yet without metadata file"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(debug=args.debug)

    if args.manifest:
        manifest_path = args.manifest.resolve()
        if not manifest_path.exists():
            LOGGER.error(f"Manifest not found: {manifest_path}")
            return 1
    elif args.session:
        try:
            session_path = _resolve_session_path(args.session, args.monitoring_state_dir)
            target = _load_session_target(session_path)
            manifest_path = target.manifest_path
        except NotImplementedError as e:
            LOGGER.error(str(e))
            return 1
    else:
        LOGGER.error("Must provide --manifest or --session")
        return 1

    try:
        manifest = read_manifest(manifest_path)
    except Exception as exc:
        LOGGER.error(f"Failed to load manifest: {exc}")
        return 1

    runtime = build_host_runtime(
        manifest,
        use_fake_slurm=args.use_fake_slurm,
        manifest_path=manifest_path,
    )

    if args.cmd == "resume":
        controller = instantiate_controller(runtime)
        run_monitoring(runtime, controller)
        return 0

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
