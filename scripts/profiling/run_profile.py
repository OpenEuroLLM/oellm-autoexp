#!/usr/bin/env python3
"""Rank-aware in-container launcher for rocprofv3 and Nsight Systems."""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import socket
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from oellm_autoexp.profiling.models import CaptureManifest


def _options(encoded: str) -> dict[str, Any]:
    if not encoded:
        return {}
    return json.loads(base64.urlsafe_b64decode(encoded.encode()).decode())


def _version(provider: str, executable: str) -> str | None:
    if provider == "rocprofv3":
        return os.environ.get("ROCM_VERSION")
    try:
        result = subprocess.run(
            [executable, "--version"], capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return (result.stdout or result.stderr).strip().splitlines()[0] or None


def _rocprof_command(
    output_dir: Path,
    rank: int,
    capture: set[str],
    options: dict[str, Any],
    command: list[str],
) -> tuple[list[str], list[str]]:
    result = ["rocprofv3"]
    flags = {
        "kernel": "--kernel-trace",
        "runtime": "--hip-trace",
        "communication": "--rccl-trace",
        "memory": "--memory-copy-trace",
        "markers": "--marker-trace",
    }
    result.extend(flag for domain, flag in flags.items() if domain in capture)
    if options.get("stats", True):
        result.append("--stats")
    formats = options.get("output_formats", ["csv"])
    result.extend(["--output-format", *map(str, formats)])
    result.extend(
        [
            "-d",
            str(output_dir),
            "-o",
            f"rank{rank}_pid%pid%",
            "--",
            *command,
        ]
    )
    expected = [f"rank{rank}_pid*_kernel_trace.csv"]
    if "json" in formats:
        expected.append(f"rank{rank}_pid*_results.json")
    return result, expected


def _nsys_command(
    output_dir: Path,
    rank: int,
    capture: set[str],
    options: dict[str, Any],
    command: list[str],
) -> tuple[list[str], list[str]]:
    traces: list[str] = []
    if {"kernel", "runtime", "memory", "communication"} & capture:
        traces.append("cuda")
    if "markers" in capture:
        traces.append("nvtx")
    if "runtime" in capture:
        traces.append("osrt")
    output = output_dir / f"rank{rank}"
    result = [
        "nsys",
        "profile",
        f"--trace={','.join(dict.fromkeys(traces))}",
        f"--output={output}",
        f"--force-overwrite={'true' if options.get('force_overwrite', True) else 'false'}",
        f"--sample={options.get('sample', 'none')}",
    ]
    export = options.get("export", "sqlite")
    if export:
        result.append(f"--export={export}")
    result.extend(["--", *command])
    return result, [f"rank{rank}.nsys-rep", f"rank{rank}.sqlite"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", required=True, choices=["rocprofv3", "nsys"])
    parser.add_argument("--mode", default="timeline", choices=["timeline"])
    parser.add_argument("--ranks", default="0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--capture", default="kernel,runtime,communication,memory")
    parser.add_argument("--provider-options", default="")
    parser.add_argument("--analysis-options", default="")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        print("error: missing application command after --", file=sys.stderr)
        return 2

    rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", "0")))
    selected = {int(value) for value in args.ranks.split(",") if value}
    if selected and rank not in selected:
        os.execvp(command[0], command)

    executable = shutil.which(args.provider)
    if executable is None:
        print(
            f"warning: {args.provider} not found; launching application without profiling",
            file=sys.stderr,
        )
        os.execvp(command[0], command)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    options = _options(args.provider_options)
    analysis_options = _options(args.analysis_options)
    capture = {value for value in args.capture.split(",") if value}
    if args.provider == "rocprofv3":
        profile_command, expected = _rocprof_command(
            args.output_dir, rank, capture, options, command
        )
    else:
        profile_command, expected = _nsys_command(
            args.output_dir, rank, capture, options, command
        )
    manifest = CaptureManifest(
        schema_version=1,
        provider=args.provider,
        mode=args.mode,
        rank=rank,
        hostname=socket.gethostname(),
        command=command,
        tool_path=executable,
        tool_version=_version(args.provider, executable),
        output_dir=str(args.output_dir),
        expected_artifacts=expected,
        provider_options=options,
        analysis_options=analysis_options,
    )
    payload = json.dumps(asdict(manifest), indent=2) + "\n"
    (args.output_dir / f"profile_manifest_rank{rank}.json").write_text(
        payload, encoding="utf-8"
    )
    (args.output_dir / "profile_manifest.json").write_text(payload, encoding="utf-8")
    print(
        f"run_profile: provider={args.provider} rank={rank} output={args.output_dir}",
        file=sys.stderr,
    )
    os.execvp(profile_command[0], profile_command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
