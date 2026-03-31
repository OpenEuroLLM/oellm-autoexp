"""Find stable sweep jobs that did not finish (e.g. SLURM time limit) and print resume commands.

Scans each stable job's output directory for ``current.log`` (the active log; ignore older
``slurm-*.log`` files). A job counts as **finished** when Megatron exits cleanly, detected by
tail-of-log markers such as ``exiting program at iteration`` or ``after training is done``.
Jobs that end with ``DUE TO TIME LIMIT``, OOM, or similar are treated as **incomplete**.

Does not modify ``resume_timed_out.py``; builds the same ``run_autoexp.py`` commands.

Usage:
  python scripts/automated_resume_timed_out.py \\
    --config-name experiments/swagatam/test_moe_130M_300BT_bsz_256.yaml \\
    --config-dir config

  # Show per-job status (complete / incomplete / unknown) and exit_interval
  python scripts/automated_resume_timed_out.py ... --verbose

  # Only consider these stable indices (comma-separated / ranges)
  python scripts/automated_resume_timed_out.py ... --indices 0,12,24-30

  # Run resume commands one after another (blocks until each job finishes — rarely what you want)
  python scripts/automated_resume_timed_out.py ... --sequential

  # Start each resume in its own detached tmux session (returns immediately; use attach to watch)
  python scripts/automated_resume_timed_out.py ... --tmux

  # Print commands only (default); --dry-run with --sequential or --tmux skips execution
"""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Literal

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = Path("/leonardo_work/OELLM_prod2026/users/shaldar0/.conda/llm/bin/python")

LogStatus = Literal["complete", "incomplete", "unknown"]


def parse_indices(spec: str) -> list[int]:
    indices: list[int] = []
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
            indices.extend(range(start, end + 1))
        else:
            indices.append(int(part))
    return sorted(set(indices))


def _maybe_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def analyze_log_tail(lines: list[str]) -> tuple[LogStatus, str]:
    """Scan from the last line upward for definitive training outcome."""
    for line in reversed(lines):
        if "after training is done" in line:
            return "complete", "found: after training is done"
        if "exiting program at iteration" in line:
            return "complete", "found: exiting program at iteration"
        if "DUE TO TIME LIMIT" in line:
            return "incomplete", "SLURM time limit"
        if "CANCELLED AT" in line and "TIME LIMIT" in line:
            return "incomplete", "SLURM time limit (cancelled)"
        if "OutOfMemoryError" in line or "CUDA out of memory" in line:
            return "incomplete", "CUDA / OOM"
        if "FATAL ERROR" in line:
            return "incomplete", "FATAL ERROR"
    return "unknown", "no completion or failure markers in tail"


def last_iteration_from_log(lines: list[str]) -> int | None:
    """Best-effort last iteration from Megatron progress lines."""
    pat = re.compile(r"iteration\s+(\d+)/\s*(\d+)")
    last = None
    for line in lines:
        m = pat.search(line)
        if m:
            last = int(m.group(1))
    return last


def read_current_log(log_path: Path, max_bytes: int = 8_000_000) -> list[str]:
    """Read log file; for large files, only keep the tail (completion is always at the end)."""
    if not log_path.is_file():
        return []
    size = log_path.stat().st_size
    if size <= max_bytes:
        return log_path.read_text(errors="replace").splitlines()
    with log_path.open("rb") as f:
        f.seek(max(0, size - max_bytes))
        f.readline()  # drop potential partial line
        data = f.read().decode(errors="replace")
    return data.splitlines()


def resolve_output_dir(base_output_dir: str, repo_root: Path) -> Path:
    p = Path(base_output_dir)
    if p.is_absolute():
        return p
    return repo_root / p


def tmux_session_name(prefix: str, index: int) -> str:
    """tmux session names: alphanumeric, _, - (keep short and unique per index)."""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in prefix)
    return f"{safe}-{index}"


def build_tmux_command(repo_root: Path, session_name: str, cmd: list[str]) -> list[str]:
    """tmux new-session -d runs the shell command and returns immediately (detached).

    Always ``cd`` to the repo root (oellm-autoexp) first — ``run_autoexp`` expects that cwd
    for relative paths such as ``config/`` and ``results/``.
    """
    root = repo_root.resolve()
    inner = f"cd {shlex.quote(str(root))} && exec " + " ".join(shlex.quote(str(x)) for x in cmd)
    return ["tmux", "new-session", "-d", "-s", session_name, "bash", "-lc", inner]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-name", required=True, help="Config name (Hydra)")
    parser.add_argument("--config-dir", type=Path, default=Path("config"))
    parser.add_argument(
        "--python",
        type=Path,
        default=DEFAULT_PYTHON,
        help=f"Python for run_autoexp (default: {DEFAULT_PYTHON})",
    )
    parser.add_argument(
        "--indices",
        default="",
        help="Optional subset of stable indices (e.g. 0,3,12-15). Default: all stable.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print status for every stable job and exit_interval / last iter",
    )
    exec_grp = parser.add_mutually_exclusive_group()
    exec_grp.add_argument(
        "--sequential",
        action="store_true",
        help="Run each resume in the foreground, one after another (waits for each to finish)",
    )
    exec_grp.add_argument(
        "--tmux",
        action="store_true",
        help="Start each resume in a detached tmux session (non-blocking; one session per job)",
    )
    parser.add_argument(
        "--tmux-prefix",
        default="aexpresume",
        help="Session name prefix for --tmux (sessions are named {prefix}-{index})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --sequential or --tmux: only print what would run, do not execute",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        default=[],
        help="Extra Hydra overrides passed through to run_autoexp (e.g. ++slurm.sbatch.time=48:00:00)",
    )
    args = parser.parse_args()

    idx_parts = [args.indices] if args.indices else []
    clean_overrides = []
    for token in args.overrides:
        stripped = token.strip().rstrip(",")
        if stripped and all(c in "0123456789-" for c in stripped):
            idx_parts.append(stripped)
        else:
            clean_overrides.append(token)
    if idx_parts:
        args.indices = ",".join(filter(None, idx_parts))
        args.overrides = clean_overrides

    subset: set[int] | None = None
    if args.indices:
        try:
            subset = set(parse_indices(args.indices))
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    sys.path.insert(0, str(REPO_ROOT))
    from oellm_autoexp.config.loader import load_config_reference
    from oellm_autoexp.config.schema import ConfigSetup
    from oellm_autoexp.orchestrator import build_execution_plan

    config_setup = ConfigSetup(
        pwd=str(REPO_ROOT),
        config_name=args.config_name,
        config_dir=str(args.config_dir),
        overrides=args.overrides,
        monitor_state_dir="./monitor_state",
    )
    root = load_config_reference(config_setup=config_setup)
    plan = build_execution_plan(root, config_setup=config_setup)

    stable_jobs: list[tuple[int, object, Path]] = []
    for job in plan.jobs:
        stage = getattr(job.config, "stage", "")
        if stage != "stable":
            continue
        idx = getattr(job.config, "index", None)
        if idx is None:
            continue
        if subset is not None and idx not in subset:
            continue
        base = getattr(job.config.job, "base_output_dir", None)
        if not base:
            continue
        out = resolve_output_dir(str(base), REPO_ROOT)
        stable_jobs.append((idx, job.config, out))

    stable_jobs.sort(key=lambda x: x[0])

    if not stable_jobs:
        print("No stable jobs match (check --indices or config).", file=sys.stderr)
        sys.exit(1)

    to_resume: list[tuple[int, str, LogStatus, str]] = []

    print(f"Checking {len(stable_jobs)} stable job(s) (current.log in each output dir)\n")

    for idx, cfg, out_dir in stable_jobs:
        meg = getattr(cfg.backend, "megatron", None)
        exit_interval = _maybe_int(getattr(meg, "exit_interval", None)) if meg else None
        train_iters = _maybe_int(getattr(meg, "train_iters", None)) if meg else None

        log_path = out_dir / "current.log"
        lines = read_current_log(log_path)
        if not lines:
            status: LogStatus = "incomplete"
            reason = f"missing or empty log: {log_path}"
            last_it = None
        else:
            status, reason = analyze_log_tail(lines)
            last_it = last_iteration_from_log(lines)

        base_output = str(getattr(cfg.job, "base_output_dir", "") or "").rstrip("/")
        load_path = f"{base_output}/checkpoints" if base_output else f"{out_dir}/checkpoints"

        if args.verbose:
            ei = exit_interval if exit_interval is not None else "null"
            ti = train_iters if train_iters is not None else "null"
            li = last_it if last_it is not None else "?"
            short = Path(base_output).name if base_output else out_dir.name
            print(
                f"  [{idx}] {short}: status={status} | "
                f"exit_interval={ei} train_iters={ti} last_iter≈{li} | {reason}"
            )

        if status != "complete":
            to_resume.append((idx, load_path, status, reason))

    need = len(to_resume)
    done = len(stable_jobs) - need
    print(
        f"Summary: {done} complete, {need} need resume (incomplete or unknown)\n"
    )

    if not to_resume:
        print("Nothing to resume.")
        return

    commands: list[tuple[int, list[str]]] = []
    for idx, load_path, _, _ in to_resume:
        # Each resume must expand to exactly one sweep point; otherwise one tmux would
        # submit multiple SLURM jobs (unexpected).
        one = build_execution_plan(root, config_setup=config_setup, subset_indices={idx})
        if len(one.jobs) != 1:
            print(
                f"Error: --array-subset {idx} expands to {len(one.jobs)} job(s), expected 1.",
                file=sys.stderr,
            )
            for j in one.jobs:
                print(
                    f"  index={getattr(j.config, 'index', None)} "
                    f"stage={getattr(j.config, 'stage', None)}",
                    file=sys.stderr,
                )
            sys.exit(1)

        cmd = [
            str(args.python),
            str(REPO_ROOT / "scripts" / "run_autoexp.py"),
            "--config-name",
            args.config_name,
            "--config-dir",
            str(args.config_dir),
            "--array-subset",
            str(idx),
            f"++backend.megatron.load={load_path}",
            *args.overrides,
        ]
        commands.append((idx, cmd))

    print(f"Resume {len(commands)} job(s): indices {[c[0] for c in commands]}")
    print(
        "(Each command is one run_autoexp → one SLURM job ID; multi-node `nodes: N` is one allocation.)\n"
    )
    for idx, cmd in commands:
        print(f"  [{idx}] load={cmd[-1].split('=', 1)[1]}")

    if args.tmux and args.dry_run:
        print("\n--tmux --dry-run: would start these detached tmux sessions:\n")

    if args.tmux:
        sessions: list[tuple[int, str]] = []
        for idx, cmd in commands:
            name = tmux_session_name(args.tmux_prefix, idx)
            tmux_cmd = build_tmux_command(REPO_ROOT, name, cmd)
            sessions.append((idx, name))
            if args.dry_run:
                print(f"# Index {idx} -> session {name}")
                print(" ".join(shlex.quote(x) for x in tmux_cmd))
                print()
            else:
                rc = subprocess.run(tmux_cmd)
                if rc.returncode != 0:
                    print(
                        f"tmux failed for index {idx} (session {name}). "
                        "If the session already exists, run: "
                        f"tmux kill-session -t {shlex.quote(name)}",
                        file=sys.stderr,
                    )
                    sys.exit(rc.returncode)
        if not args.dry_run:
            print(f"Started {len(sessions)} detached tmux session(s).\n")
            for idx, name in sessions:
                print(f"  [{idx}] tmux attach -t {shlex.quote(name)}")
            print("\nList: tmux ls")
        return

    if args.sequential and args.dry_run:
        print("\n--sequential --dry-run: commands below would be executed.\n")

    if args.sequential and not args.dry_run:
        for idx, cmd in commands:
            print(f"\n>>> Resuming index {idx}...")
            rc = subprocess.run(cmd, cwd=REPO_ROOT)
            if rc.returncode != 0:
                print(f"Index {idx} failed with exit code {rc.returncode}", file=sys.stderr)
                sys.exit(rc.returncode)
        print("\nAll resume jobs finished.")
        return

    print("\nRun each command in a separate terminal/screen, or use --tmux to spawn detached sessions.\n")
    for idx, cmd in commands:
        print(f"# Index {idx}:")
        print(" ".join(cmd))
        print()


if __name__ == "__main__":
    main()
