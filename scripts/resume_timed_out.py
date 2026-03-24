"""Resume timed-out or failed jobs from their last checkpoint.

When SLURM jobs hit the time limit (e.g. 24h), they exit with "DUE TO TIME LIMIT".
Megatron saves checkpoints periodically, so you can resume from the last one.

Supports both stable and decay jobs:
  - Stable jobs: load from their own checkpoint directory.
  - Decay jobs: load from the corresponding stable sibling's checkpoint directory.

Usage:
  # Resume specific indices (comma-separated or ranges, spaces OK)
  python scripts/resume_timed_out.py \\
    --config-name experiments/swagatam/test_moe_130M_300BT_filtering \\
    --indices 0,3,6,48

  # Resume decay jobs
  python scripts/resume_timed_out.py \\
    --config-name experiments/swagatam/test_moe_130M_300BT_filtering \\
    --indices 1,2,4,5

  # Dry-run: print commands without executing
  python scripts/resume_timed_out.py ... --dry-run

  # Resume one job at a time
  python scripts/resume_timed_out.py ... --sequential

Each job is resumed with:
  - --array-subset <index>  (run only that sweep point)
  - ++backend.megatron.load=<checkpoint_dir>
    (own checkpoints for stable, stable sibling's checkpoints for decay)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = Path("/leonardo_work/OELLM_prod2026/users/shaldar0/.conda/llm/bin/python")


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-name", required=True, help="Config name (e.g. experiments/swagatam/test_moe_130M_300BT_filtering)")
    parser.add_argument("--config-dir", type=Path, default=Path("config"))
    parser.add_argument(
        "--python",
        type=Path,
        default=DEFAULT_PYTHON,
        help=f"Python executable for run_autoexp (default: {DEFAULT_PYTHON})",
    )
    parser.add_argument(
        "--indices",
        default="",
        help="Comma-separated indices or ranges (e.g. 0,1,5-10,48). Accepts both stable and decay indices. Use --list-stable / --list-decay to see available indices.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run jobs one at a time (wait for each to finish before starting next)",
    )
    parser.add_argument(
        "--list-stable",
        action="store_true",
        help="List all stable indices and exit (useful to see which indices to pass)",
    )
    parser.add_argument(
        "--list-decay",
        action="store_true",
        help="List decay indices with job names (to find index for a failed decay job)",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        default=[],
        help="Extra Hydra overrides (e.g. ++slurm.sbatch.time=48:00:00)",
    )
    args = parser.parse_args()

    # Handle spaces in --indices (e.g. "--indices 0, 3, 6" gets shell-split
    # into args.indices="0," with overrides=["3,", "6"]). Reclaim stray
    # numeric fragments from overrides and merge them back into indices.
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

    if args.list_stable:
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
        stable_jobs = [
            (job.config.index, getattr(job.config.job, "base_output_dir", ""))
            for job in plan.jobs
            if getattr(job.config, "stage", "") == "stable"
        ]
        print(f"Stable indices ({len(stable_jobs)}) (index -> output_dir):")
        for idx, out_dir in sorted(stable_jobs, key=lambda x: x[0]):
            name = Path(out_dir).name if out_dir else ""
            print(f"  {idx}: {name}")
        return

    if args.list_decay:
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
        decay_jobs = [
            (job.config.index, getattr(job.config.job, "base_output_dir", ""))
            for job in plan.jobs
            if getattr(job.config, "stage", "").startswith("decay")
        ]
        print(f"Decay indices ({len(decay_jobs)}) (index -> output_dir):")
        for idx, out_dir in sorted(decay_jobs, key=lambda x: x[0]):
            name = Path(out_dir).name if out_dir else ""
            print(f"  {idx}: {name}")
        return

    try:
        indices = parse_indices(args.indices)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if not indices:
        print("No indices specified. Use --indices with --list-stable / --list-decay to see available indices.", file=sys.stderr)
        sys.exit(1)

    sys.path.insert(0, str(REPO_ROOT))
    from oellm_autoexp.config.loader import load_config_reference
    from oellm_autoexp.config.schema import ConfigSetup
    from oellm_autoexp.hydra_staged_sweep import expand_sweep
    from oellm_autoexp.orchestrator import build_execution_plan

    config_setup = ConfigSetup(
        pwd=str(REPO_ROOT),
        config_name=args.config_name,
        config_dir=str(args.config_dir),
        overrides=args.overrides,
        monitor_state_dir="./monitor_state",
    )
    root = load_config_reference(config_setup=config_setup)

    points = expand_sweep(root.sweep)
    all_indices = {p.index for p in points}
    invalid = sorted(set(indices) - all_indices)
    if invalid:
        print(f"Warning: indices {invalid} not found in sweep, skipping", file=sys.stderr)
    resume_indices = sorted(set(indices) & all_indices)
    if not resume_indices:
        print("No valid indices found. Use --list-stable / --list-decay to see available indices.", file=sys.stderr)
        sys.exit(1)

    plan = build_execution_plan(
        root,
        config_setup=config_setup,
        subset_indices=set(resume_indices),
    )

    # Build index -> load_path from plan.
    # Stable jobs: load from their own checkpoint directory.
    # Decay jobs: load from the stable sibling's checkpoint (already resolved
    # via ${sibling.stable.job.base_output_dir}/checkpoints in the config).
    index_to_load: dict[int, str] = {}
    for job in plan.jobs:
        cfg = job.config
        idx = getattr(cfg, "index", None)
        if idx is None:
            continue
        stage = getattr(cfg, "stage", "")
        base_output = getattr(cfg.job, "base_output_dir", None)
        if not base_output:
            print(f"Warning: no base_output_dir for index {idx}, skipping", file=sys.stderr)
            continue
        if stage == "stable":
            load_path = f"{base_output}/checkpoints"
        else:
            megatron_cfg = getattr(cfg.backend, "megatron", None)
            resolved_load = getattr(megatron_cfg, "load", None) if megatron_cfg else None
            if resolved_load:
                load_path = str(resolved_load)
            else:
                print(f"Warning: no resolved load path for decay index {idx}, falling back to own checkpoints", file=sys.stderr)
                load_path = f"{base_output}/checkpoints"
        index_to_load[idx] = load_path

    missing = set(resume_indices) - set(index_to_load)
    if missing:
        print(f"Warning: indices {sorted(missing)} not found in plan (filtered out?)", file=sys.stderr)

    commands: list[tuple[int, list[str]]] = []
    for idx in resume_indices:
        if idx not in index_to_load:
            continue
        load_path = index_to_load[idx]
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

    if not commands:
        print("No valid resume commands to run.", file=sys.stderr)
        sys.exit(1)

    print(f"Resuming {len(commands)} job(s): indices {[c[0] for c in commands]}")
    for idx, cmd in commands:
        print(f"  [{idx}] load={index_to_load[idx]}")

    if args.dry_run:
        for idx, cmd in commands:
            print(f"\n# Index {idx}:")
            print(" ".join(cmd))
        return

    if args.sequential:
        for idx, cmd in commands:
            print(f"\n>>> Resuming index {idx}...")
            rc = subprocess.run(cmd, cwd=REPO_ROOT)
            if rc.returncode != 0:
                print(f"Job {idx} failed with exit code {rc.returncode}", file=sys.stderr)
                sys.exit(rc.returncode)
        print("\nAll jobs completed.")
        return

    # Parallel: run each in background (each has its own monitor)
    print("\nSubmitting jobs (each runs in foreground with its own monitor)...")
    print("Run each command in a separate terminal/screen, or use --sequential to run one by one.\n")
    for idx, cmd in commands:
        print(f"# Index {idx}:")
        print(" ".join(cmd))
        print()
    print("Copy-paste the commands above into separate terminals, or run with --sequential.")


if __name__ == "__main__":
    main()
