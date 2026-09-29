#!/usr/bin/env python3
"""Plan and submit weighted checkpoint merges (WSM / "model soup") over a run's
torch_dist checkpoints, producing merged torch_dist checkpoints that feed
straight into scripts/mass_convert_checkpoints.py.

The merge itself is done by NVIDIA's tools/checkpoint/weighted_merge.py
(Megatron-LM PR #5114). That tool is metadata-driven: it reads each source
checkpoint's DCP metadata, never builds a model, and imports no model code, so
it runs CPU-only and works against a pinned Megatron whose training source we
do not want to touch. This wrapper only decides *which* windows to merge and
submits one Slurm job per merge.

Why merging happens before HF conversion: the tool reads and writes torch_dist
and nothing else, TE FP8 _extra_state is copied (not averaged) from one chosen
source and has no equivalent in safetensors, and a merged torch_dist checkpoint
stays loadable for evaluation or resume. Averaging after conversion would throw
all of that away.

Each --window produces its own output run directory, so several merge durations
can be planned in one invocation and compared:

    <output-dir>/<label>_wsm<W>_<style>/checkpoints/iter_<END>

which is shaped like an ordinary run, so conversion is the usual call with the
*source* run's training config (a merged checkpoint has no logs/ of its own):

    uv run python scripts/mass_convert_checkpoints.py \\
        --checkpoints-dir <output-dir>/<label>_wsm16_minus-sqrt/checkpoints \\
        --training-config <source run>/logs/current.yaml \\
        --run-label <label>wsm16 ...

Example (prelude on Leonardo, three merge durations at once):

    uv run python scripts/merge_checkpoints.py \\
        --checkpoints-dir /leonardo_scratch/.../baby_9b_dense/checkpoints \\
        --output-dir /leonardo_scratch/large/userexternal/$USER/prelude-merged \\
        --run-label prelude --min-iteration-interval 2400 \\
        --window 8 --window 16 --window 32 --stride 10 \\
        --merge-tool /path/to/Megatron-LM/tools/checkpoint/weighted_merge.py \\
        --megatron-root /path/to/Megatron-LM \\
        --container-image /path/to/image.sif --singularity-bind /leonardo_scratch \\
        --account OELLM_prod2026 --partition boost_usr_prod --qos boost_qos_dbg \\
        --nodes 1 --ntasks-per-node 8 --time-limit 01:00:00 --dry-run

Start with --dry-run: it prints the planned windows and passes --dry-run to the
merge tool, which validates every source's layout and the resolved input list
without writing anything.

Note on --extra-state-source-index: the merge copies Transformer Engine
_extra_state (FP8 amax history and scales) from one input rather than averaging
it, defaulting to the *earliest* input in the window. For a checkpoint published
as iter_<END> the endpoint is usually the more sensible source. The resolved
input count is printed by a --dry-run, so set the index explicitly once that
count is known rather than guessing here.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ITER_RE = re.compile(r"^iter_(\d+)$")


def discover_iterations(ckpt_dir: Path, pattern: str) -> list[int]:
    """Iterations of complete torch_dist checkpoints under ckpt_dir."""
    iters = []
    for d in sorted(ckpt_dir.glob(pattern)):
        m = ITER_RE.match(d.name)
        if not m:
            continue
        # A checkpoint dir can be a symlink into another project's space and be
        # dangling or unreadable for us; skip it rather than failing discovery.
        try:
            if not d.is_dir() or not (d / "metadata.json").exists():
                continue
        except OSError as exc:
            print(f"  skipping {d.name}: {exc.strerror or exc}")
            continue
        iters.append(int(m.group(1)))
    return sorted(iters)


def plan_windows(iters: list[int], interval: int, window: int, stride: int, ends: list[int] | None):
    """(start, end) iteration pairs for each merge.

    Windows are laid out on the regular grid (iterations that are multiples of
    `interval`), which is what the merge tool's own --min-iteration-interval
    filtering selects. Off-grid checkpoints -- segment-boundary saves such as
    iter_0084238 -- are deliberately not used as endpoints: including one would
    place two inputs a few hundred iterations apart and over-weight that point.
    """
    grid = [i for i in iters if i % interval == 0]
    span = (window - 1) * interval
    if ends:
        endpoints = [e for e in ends if e in grid]
        missing = sorted(set(ends) - set(endpoints))
        if missing:
            print(f"  requested endpoints not on the grid, ignored: {missing}")
    else:
        eligible = [e for e in grid if (e - span) in grid]
        # Anchor the series at the newest checkpoint and walk back, so the most
        # recent merge point is always present regardless of how the stride divides.
        endpoints = sorted(eligible[::-1][::stride])
    return [(e - span, e) for e in endpoints]


def free_slots(qos: str, max_concurrent: int) -> int:
    out = subprocess.run(
        ["squeue", "-u", os.environ["USER"], "--noheader", "--format", "%q"],
        capture_output=True,
    ).stdout.decode()
    return max_concurrent - out.count(qos)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--checkpoints-dir", required=True, type=Path)
    ap.add_argument("--checkpoint-pattern", default="iter_*")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--run-label", required=True, help="Source run label, e.g. b1 or prelude")
    ap.add_argument(
        "--window",
        type=int,
        action="append",
        required=True,
        help="Checkpoints per merge (merge duration). Repeatable to plan several durations.",
    )
    ap.add_argument(
        "--stride", type=int, default=1, help="Checkpoints between consecutive window endpoints"
    )
    ap.add_argument(
        "--end-iteration",
        type=int,
        action="append",
        default=None,
        help="Merge only these endpoints (repeatable); overrides --stride",
    )
    ap.add_argument(
        "--min-iteration-interval",
        type=int,
        required=True,
        help="Checkpoint grid spacing, e.g. 2000 on the 32B runs, 2400 on prelude",
    )
    ap.add_argument(
        "--merge-style",
        default="minus-sqrt",
        choices=["linear", "minus-sqrt", "linear__reverse", "minus-sqrt__reverse"],
    )
    ap.add_argument("--merge-save-dtype", default="same")
    ap.add_argument(
        "--extra-state-source-index",
        type=int,
        default=None,
        help="Which resolved input supplies TE _extra_state (default: the tool's, the earliest)",
    )
    ap.add_argument(
        "--balance-rank-work",
        action="store_true",
        help="Bin-pack source DCP chunks across ranks (uneven source layouts only)",
    )
    ap.add_argument("--merge-tool", required=True, type=Path, help="Path to weighted_merge.py")
    ap.add_argument(
        "--megatron-root",
        default=None,
        type=Path,
        help="Tree providing megatron.core; omit to use whatever the image ships",
    )
    ap.add_argument("--container-image", required=True)
    ap.add_argument("--singularity-bind", action="append", default=[])
    ap.add_argument("--account", required=True)
    ap.add_argument("--partition", required=True)
    ap.add_argument("--qos", default=None)
    ap.add_argument("--reservation", default=None)
    ap.add_argument("--time-limit", default="02:00:00")
    ap.add_argument("--nodes", type=int, default=1)
    ap.add_argument(
        "--ntasks-per-node", type=int, default=8, help="Reader ranks per node; merging is I/O-bound"
    )
    ap.add_argument("--cpus-per-task", type=int, default=8)
    ap.add_argument(
        "--gres",
        default=None,
        help="Passed through as --gres=... . The merge never uses a GPU, but every "
        "cluster we have only offers GPU partitions, so an allocation may require it.",
    )
    ap.add_argument("--max-concurrent-jobs", type=int, default=4)
    ap.add_argument("--force", action="store_true", help="Re-merge windows already present")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and run the merge tool with --dry-run (validates layouts, writes nothing)",
    )
    args = ap.parse_args()

    iters = discover_iterations(args.checkpoints_dir, args.checkpoint_pattern)
    grid = [i for i in iters if i % args.min_iteration_interval == 0]
    print(
        f"discovered {len(iters)} checkpoints, {len(grid)} on the "
        f"{args.min_iteration_interval}-iteration grid"
    )
    if not grid:
        sys.exit("no checkpoints on the requested grid; check --min-iteration-interval")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "logs").mkdir(exist_ok=True)

    jobs = []
    for window in args.window:
        series = f"{args.run_label}_wsm{window}_{args.merge_style}"
        out_root = args.output_dir / series / "checkpoints"
        pairs = plan_windows(
            grid, args.min_iteration_interval, window, args.stride, args.end_iteration
        )
        planned, skipped = [], 0
        for start, end in pairs:
            if not args.force and (out_root / f"iter_{end:07d}").exists():
                skipped += 1
                continue
            planned.append((start, end))
        tokens = window * args.min_iteration_interval
        print(
            f"[{series}] window={window} ckpts ({tokens} iterations): "
            f"{len(planned)} to merge, {skipped} already present"
        )
        for start, end in planned:
            jobs.append({"series": series, "out_root": out_root, "start": start, "end": end})

    if not jobs:
        print("nothing to do")
        return 0

    binds = " ".join(f"--bind {b}" for b in args.singularity_bind)
    pythonpath = ":".join(str(p) for p in (args.megatron_root,) if p)
    env = f"--env PYTHONNOUSERSITE=1{f' --env PYTHONPATH={pythonpath}' if pythonpath else ''}"

    submitted_path = args.output_dir / "submitted_merges.json"
    submitted = json.loads(submitted_path.read_text()) if submitted_path.exists() else []

    for job in jobs:
        start, end = job["start"], job["end"]
        window = (end - start) // args.min_iteration_interval + 1
        merge_cmd = (
            f"python {args.merge_tool}"
            f" --merge-inputs {args.checkpoints_dir}"
            f" --start-checkpoint {start} --end-checkpoint {end}"
            f" --min-iteration-interval {args.min_iteration_interval}"
            f" --min-checkpoints {window}"
            f" --merge-style {args.merge_style}"
            f" --merge-save-dtype {args.merge_save_dtype}"
            # Our checkpoints carry distributed optimizer state, which the tool
            # refuses to merge or silently copy; this is the explicit opt-out.
            f" --merge-ignore-non-model-state"
            f" --merge-output {job['out_root']}"
            f" --output-iteration {end}"
        )
        if args.extra_state_source_index is not None:
            merge_cmd += f" --extra-state-source-index {args.extra_state_source_index}"
        if args.balance_rank_work:
            merge_cmd += " --merge-balance-rank-work"
        if args.dry_run:
            merge_cmd += " --dry-run"

        jobname = f"wsm_{job['series']}_{end:07d}"
        task = (
            "export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=$SLURM_LOCALID; "
            f"singularity exec {binds} {env} {args.container_image} {merge_cmd}"
        )
        ntasks = args.nodes * args.ntasks_per_node
        cmd = [
            "sbatch",
            f"--account={args.account}",
            f"--partition={args.partition}",
            *([f"--qos={args.qos}"] if args.qos else []),
            *([f"--reservation={args.reservation}"] if args.reservation else []),
            *([f"--gres={args.gres}"] if args.gres else []),
            f"--time={args.time_limit}",
            f"--nodes={args.nodes}",
            f"--ntasks-per-node={args.ntasks_per_node}",
            f"--cpus-per-task={args.cpus_per_task}",
            f"--job-name={jobname}",
            f"--output={args.output_dir}/logs/{jobname}-%j.log",
            f"--error={args.output_dir}/logs/{jobname}-%j.err",
            "--parsable",
            (
                "--wrap=export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1); "
                f"export MASTER_PORT=29500; srun --ntasks={ntasks} bash -c '{task}'"
            ),
        ]

        if args.dry_run:
            print(f"  [dry-run] {jobname}: merge {start}..{end} ({window} ckpts)")
            print(f"      {shlex.join(cmd)}")
            continue

        if args.qos:
            while free_slots(args.qos, args.max_concurrent_jobs) <= 0:
                time.sleep(10)
        proc = subprocess.run(cmd, capture_output=True)
        jid = proc.stdout.decode().strip()
        if proc.returncode != 0 or not jid:
            sys.exit(
                f"sbatch failed for {jobname} (exit {proc.returncode}): "
                f"{proc.stderr.decode().strip() or 'no stderr'}"
            )
        submitted.append(
            {"jobname": jobname, "jobid": jid, "series": job["series"], "start": start, "end": end}
        )
        print(f"submitted {jobname} ({window} ckpts, {start}..{end}) -> {jid}")
        submitted_path.write_text(json.dumps(submitted, indent=2))

    if args.dry_run:
        print(f"--dry-run: {len(jobs)} merge(s) planned, nothing submitted")
    else:
        print(f"all {len(jobs)} merge job(s) submitted (tracked in {submitted_path})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
