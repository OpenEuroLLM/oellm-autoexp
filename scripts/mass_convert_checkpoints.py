#!/usr/bin/env python3
"""Batch-convert Megatron checkpoints to sharded HuggingFace format, with a
canonical-prompt validation pass, submitted to Slurm as parallel-task jobs that
respect a QOS's per-user job cap.

Re-running is safe/idempotent: any checkpoint whose --output-dir/<iter>
already exists is skipped unless --force is passed, so a partial or killed
run can just be re-invoked as-is.

Example (Leonardo, debug QOS, 2-job cap -> group-size 16 gives 32 parallel
conversions at a time):

    uv run python scripts/mass_convert_checkpoints.py \\
        --checkpoints-dir /leonardo_scratch/fast/OELLM_prod2026/production_training/baby_9b_dense/checkpoints \\
        --training-config /leonardo_scratch/fast/OELLM_prod2026/production_training/baby_9b_dense/logs/current.yaml \\
        --checkpoints-dir /leonardo_work/OELLM_prod2026/production_training/baby_9b_dense/checkpoints \\
        --training-config /leonardo_work/OELLM_prod2026/production_training/baby_9b_dense/logs/current.yaml \\
        --output-dir /leonardo_scratch/large/userexternal/$USER/prelude-ckpts \\
        --container-image /leonardo_work/OELLM_prod2026/container_images/OELLM_autoexp_MegatronTrainingNoRoot_base_2510_x86_64_202603060943.sif \\
        --singularity-bind /leonardo_scratch --singularity-bind /leonardo --singularity-bind /leonardo_work/ \\
        --account OELLM_prod2026 --qos boost_qos_dbg --max-concurrent-jobs 2 --group-size 16

When multiple --checkpoints-dir/--training-config pairs are given (matched
by order), earlier pairs take priority for a given iteration if the same
iter exists in more than one dir (useful when a run's checkpoint location
moved mid-training but you want a single output tree).

Use --dry-run to see the discovered checkpoints and job plan without
submitting anything.

For incremental/ongoing use (e.g. a periodic watcher that re-invokes this once
new checkpoints appear), pass --group-size 1 so each checkpoint gets its own
Slurm job -- this lets a caller chain a dependent upload job
(sbatch --dependency=afterok:<jobid>) onto each individual conversion job.

Pass --hf-repo-id when multiple independent operators each convert into their
own --output-dir against the same training run (e.g. each running their own
copy of this tool from their own scratch space, since a scratch dir generally
can't be shared across HPC accounts). A checkpoint already published as a
complete branch on that HF Hub repo is skipped even if it isn't present in
this --output-dir, so operators don't redundantly reconvert what someone else
already finished and uploaded.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def discover_checkpoints(dirs_and_configs, pattern: str):
    seen = {}
    for ckpt_dir, config in dirs_and_configs:
        ckpt_dir = Path(ckpt_dir)
        for d in sorted(ckpt_dir.glob(pattern)):
            if not d.is_dir() or d.name in seen:
                continue
            try:
                has_metadata = (d / "metadata.json").exists()
            except PermissionError:
                print(
                    f"  skipping {d.name} in {ckpt_dir}: permission denied, cannot check metadata.json"
                )
                continue
            if not has_metadata:
                print(
                    f"  skipping {d.name} in {ckpt_dir}: no metadata.json (broken/incomplete checkpoint)"
                )
                continue
            seen[d.name] = {"iter": d.name, "megatron_path": str(d), "megatron_config": str(config)}
    return [seen[k] for k in sorted(seen)]


def free_slots(qos: str, max_concurrent: int) -> int:
    out = subprocess.run(
        ["squeue", "-u", os.environ["USER"], "--noheader", "--format", "%q"],
        capture_output=True,
    ).stdout.decode()
    return max_concurrent - out.count(qos)


def hf_complete_iters(repo_id: str) -> set[str]:
    """Iter names already published as a complete branch on repo_id.

    "Complete" mirrors upload_to_hf.py's own check: the branch must
    actually have the weight files, not just exist as a ref.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    try:
        branches = {b.name for b in api.list_repo_refs(repo_id).branches}
    except Exception as exc:  # noqa: BLE001
        print(
            f"warning: could not query HF repo {repo_id} ({exc}); "
            "not skipping any checkpoints based on Hub state",
            file=sys.stderr,
        )
        return set()

    complete = set()
    for branch in branches:
        try:
            files = set(api.list_repo_files(repo_id, revision=branch))
        except Exception:  # noqa: BLE001
            continue
        has_weights = any(f in ("model.safetensors", "model.safetensors.index.json") for f in files)
        if has_weights and "config.json" in files and "validation.json" in files:
            complete.add(branch)
    return complete


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--checkpoints-dir",
        action="append",
        required=True,
        help="Megatron checkpoints dir (iter_* subdirs). Repeatable; earlier wins on iter conflicts.",
    )
    ap.add_argument(
        "--training-config",
        action="append",
        required=True,
        help="Training config YAML, one per --checkpoints-dir, same order.",
    )
    ap.add_argument("--checkpoint-pattern", default="iter_*")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--repo-root", default=REPO_ROOT, type=Path)
    ap.add_argument("--bridge-root", default=None, type=Path)
    ap.add_argument("--resources", default=None, type=Path)
    ap.add_argument("--container-image", required=True)
    ap.add_argument("--hf-model", default="openeurollm/Qwen3-0.9B-ne")
    ap.add_argument("--tokenizer", default="openeurollm/Qwen3-0.9B-ne")
    ap.add_argument("--derive-hf-arch", default="qwen3")
    ap.add_argument("--max-shard-size", default="5GB")
    ap.add_argument(
        "--singularity-bind",
        action="append",
        required=True,
        help="Path to bind into the conversion job's container (repeatable). "
        "No default: bind paths are cluster-specific.",
    )
    ap.add_argument(
        "--hf-repo-id",
        default=None,
        help="If set, also skip checkpoints already published as a complete branch on this "
        "HF Hub repo, even if absent from --output-dir (see module docstring)",
    )
    ap.add_argument("--account", required=True)
    ap.add_argument("--partition", default="boost_usr_prod")
    ap.add_argument("--qos", default="boost_qos_dbg")
    ap.add_argument("--time-limit", default="00:30:00")
    ap.add_argument("--group-size", type=int, default=16, help="Checkpoints (Slurm tasks) per job")
    ap.add_argument(
        "--max-concurrent-jobs", type=int, default=2, help="Match the QOS's MaxJobsPerUser"
    )
    ap.add_argument("--cpus-per-task", type=int, default=4)
    ap.add_argument(
        "--gpus-per-node",
        type=int,
        default=4,
        help="GPUs per node on the target partition; pins --ntasks-per-node so job size "
        "in nodes is deterministic regardless of cluster fragmentation",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only convert the first N discovered checkpoints (testing)",
    )
    ap.add_argument(
        "--force", action="store_true", help="Reconvert even if the output already exists"
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan only; discover + write manifests but don't submit",
    )
    args = ap.parse_args()

    if len(args.checkpoints_dir) != len(args.training_config):
        ap.error(
            "--checkpoints-dir and --training-config must be repeated the same number of times, in matching order"
        )

    bridge_root = args.bridge_root or (args.repo_root / "submodules" / "Megatron-Bridge")
    resources = args.resources or (
        args.repo_root / "oellm_autoexp" / "postprocess" / "resources" / "megatron_bridge"
    )

    print("discovering checkpoints...")
    checkpoints = discover_checkpoints(
        list(zip(args.checkpoints_dir, args.training_config)), args.checkpoint_pattern
    )
    print(f"discovered {len(checkpoints)} valid checkpoints")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "logs").mkdir(exist_ok=True)
    (args.output_dir / "validation").mkdir(exist_ok=True)
    (args.output_dir / "manifests").mkdir(exist_ok=True)

    already_on_hf = set()
    if args.hf_repo_id:
        already_on_hf = hf_complete_iters(args.hf_repo_id)
        print(f"{len(already_on_hf)} checkpoint(s) already complete on {args.hf_repo_id}")

    tasks = []
    skipped = 0
    for c in checkpoints:
        hf_path = args.output_dir / c["iter"]
        if not args.force and (hf_path.exists() or c["iter"] in already_on_hf):
            skipped += 1
            continue
        tasks.append(
            {
                **c,
                "hf_path": str(hf_path),
                "hf_model": args.hf_model,
                "tokenizer": args.tokenizer,
                "derive_hf_arch": args.derive_hf_arch,
                "bridge_root": str(bridge_root),
                "resources": str(resources),
                "validation_json": str(hf_path / "validation.json"),
            }
        )
    print(f"{len(tasks)} to convert, {skipped} already present (skipped; use --force to redo)")

    if args.limit:
        tasks = tasks[: args.limit]
        print(f"--limit applied: converting {len(tasks)}")

    if not tasks:
        print("nothing to do")
        return 0

    groups = [tasks[i : i + args.group_size] for i in range(0, len(tasks), args.group_size)]
    print(
        f"{len(groups)} job(s) of up to {args.group_size} tasks each "
        f"(max {args.max_concurrent_jobs} concurrent under qos={args.qos})"
    )

    manifest_paths = []
    for gi, group in enumerate(groups):
        p = args.output_dir / "manifests" / f"group_{gi:03d}.json"
        p.write_text(json.dumps(group, indent=2))
        manifest_paths.append(p)

    if args.dry_run:
        print("--dry-run: manifests written, nothing submitted:")
        for gi, group in enumerate(groups):
            print(f"  group {gi:03d}: {len(group)} tasks -> {manifest_paths[gi]}")
        return 0

    submitted_path = args.output_dir / "manifests" / "submitted_jobs.json"
    submitted = json.loads(submitted_path.read_text()) if submitted_path.exists() else []
    for gi, (group, manifest_path) in enumerate(zip(groups, manifest_paths)):
        n = len(group)
        jobname = f"ckpt_conv_g{gi:03d}"
        while free_slots(args.qos, args.max_concurrent_jobs) <= 0:
            time.sleep(10)
        binds = " ".join(f"--bind {b}" for b in args.singularity_bind)
        task_cmd = (
            f"singularity exec --nv {binds} "
            f"--env PYTHONPATH={args.repo_root}:{bridge_root}/src --env PYTHONNOUSERSITE=1 "
            f"--env HF_HOME=$HOME/.cache/huggingface --env CUDA_DEVICE_MAX_CONNECTIONS=1 "
            f"{args.container_image} python -m oellm_autoexp.backends.megatron_bridge.convert_and_validate_task "
            f"--manifest {manifest_path} --max-shard-size {args.max_shard_size}"
        )
        ntasks_per_node = min(n, args.gpus_per_node)
        cmd = [
            "sbatch",
            f"--account={args.account}",
            f"--partition={args.partition}",
            f"--qos={args.qos}",
            f"--time={args.time_limit}",
            f"--ntasks={n}",
            "--gpus-per-task=1",
            f"--cpus-per-task={args.cpus_per_task}",
            f"--ntasks-per-node={ntasks_per_node}",
            f"--job-name={jobname}",
            f"--output={args.output_dir}/logs/{jobname}-%j.log",
            f"--error={args.output_dir}/logs/{jobname}-%j.err",
            "--parsable",
            f"--wrap=srun --ntasks={n} bash -c '{task_cmd}'",
        ]
        jid = subprocess.run(cmd, capture_output=True).stdout.decode().strip()
        submitted.append({"jobname": jobname, "jobid": jid, "n": n, "manifest": str(manifest_path)})
        print(f"submitted {jobname} ({n} tasks) -> {jid}")
        submitted_path.write_text(json.dumps(submitted, indent=2))

    print(
        f"all {len(groups)} job(s) submitted ({len(submitted)} total tracked in {submitted_path})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
