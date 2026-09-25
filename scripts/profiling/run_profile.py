#!/usr/bin/env python3
"""Rank-aware in-container launcher for rocprofv3 and Nsight Systems."""

from __future__ import annotations

import argparse
import base64
import json
import os
import shlex
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


def _analysis_command(
    output_dir: Path,
    options: dict[str, Any],
    *,
    python_prefix: list[str],
) -> list[str]:
    base_dir = output_dir.parent
    job_id = os.environ.get("SLURM_JOB_ID")
    command = [
        *python_prefix,
        "scripts/profiling/analyze_profile.py",
        str(output_dir),
        "--output-dir",
        str(output_dir / "analysis"),
        "--steady-start",
        str(options.get("steady_start_iteration", 5)),
    ]
    steady_end = options.get("steady_end_iteration")
    if steady_end is not None:
        command.extend(["--steady-end", str(steady_end)])
    if job_id:
        stdout = base_dir / "logs" / f"stdout-{job_id}.log"
        config = base_dir / f"config-{job_id}.yaml"
        command.extend(["--stdout", str(stdout), "--config", str(config)])
    if options.get("canvas"):
        command.extend(
            ["--canvas-out", str(output_dir / "analysis" / "profile-analysis.canvas.tsx")]
        )
    return command


def _render_analysis_sbatch(
    output_dir: Path,
    options: dict[str, Any],
    command: list[str],
) -> Path:
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    job_id = os.environ.get("SLURM_JOB_ID", "profile")
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=profile-analysis-{job_id}",
        f"#SBATCH --cpus-per-task={int(options.get('cpus', 8))}",
        f"#SBATCH --mem={options.get('memory', '32G')}",
        f"#SBATCH --time={options.get('time', '00:20:00')}",
        f"#SBATCH --output={analysis_dir}/stdout-%j.log",
        f"#SBATCH --error={analysis_dir}/stderr-%j.log",
    ]
    if options.get("account"):
        lines.append(f"#SBATCH --account={options['account']}")
    if options.get("partition"):
        lines.append(f"#SBATCH --partition={options['partition']}")
    lines.extend(["", f"cd {shlex.quote(str(Path.cwd()))}", shlex.join(command), ""])
    path = analysis_dir / f"analyze-{job_id}.sbatch"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _run_or_submit_analysis(output_dir: Path, options: dict[str, Any]) -> None:
    if not options.get("enabled") or not options.get("auto_run"):
        return
    execution = options.get("execution", "dependent_slurm")
    if execution == "same_job":
        command = _analysis_command(output_dir, options, python_prefix=[sys.executable])
        result = subprocess.run(command, check=False)
        if result.returncode:
            print(
                f"warning: profiling analysis exited with code {result.returncode}",
                file=sys.stderr,
            )
        return

    job_id = os.environ.get("SLURM_JOB_ID")
    sbatch = shutil.which("sbatch")
    if not job_id or not sbatch:
        print(
            "warning: dependent analysis requested outside Slurm; running in current job",
            file=sys.stderr,
        )
        command = _analysis_command(output_dir, options, python_prefix=[sys.executable])
        subprocess.run(command, check=False)
        return

    command = _analysis_command(
        output_dir, options, python_prefix=["uv", "run", "--python", "3.12"]
    )
    script = _render_analysis_sbatch(output_dir, options, command)
    result = subprocess.run(
        [sbatch, f"--dependency=afterok:{job_id}", str(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        print(f"warning: failed to submit profiling analysis: {result.stderr}", file=sys.stderr)
    else:
        print(f"run_profile: submitted dependent analysis: {result.stdout.strip()}", file=sys.stderr)


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
    result = subprocess.run(profile_command, check=False)
    if result.returncode == 0:
        _run_or_submit_analysis(args.output_dir, analysis_options)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
