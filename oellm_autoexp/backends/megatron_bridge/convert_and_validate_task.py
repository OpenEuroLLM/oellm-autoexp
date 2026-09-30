"""Single-checkpoint task entry point for mass conversion.

Reads one entry from a JSON manifest (a list of per-checkpoint task specs),
selected by --task-index (or $SLURM_PROCID when running under `srun
--ntasks=N`), and runs convert (run_export) + validate (validate_export)
in-process for that checkpoint. Meant to be invoked once per Slurm task by
scripts/mass_convert_checkpoints.py; see that script's docstring for the
end-to-end workflow.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from oellm_autoexp.backends.megatron_bridge.validate_export import validate


def _parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument(
        "--task-index",
        type=int,
        default=None,
        help="Index into the manifest. Defaults to $SLURM_PROCID.",
    )
    ap.add_argument("--max-shard-size", default="5GB")
    return ap.parse_args()


def _pin_local_gpu() -> None:
    """Give each task its own GPU.

    Sites that reject ``--gpus-per-task`` need a job-level ``--gres=gpu:N``
    instead, which makes every task on the node see all N devices; each then
    defaults to ``cuda:0`` and they fight over one card. Conversion hides this
    because it initialises on CPU, but validation loads the whole model onto the
    GPU and the second task OOMs. Narrow the visible set to this task's own
    device, before anything initialises CUDA.
    """
    local_id = os.environ.get("SLURM_LOCALID")
    if local_id is None:
        return
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = [d for d in visible.split(",") if d] if visible else None
    if devices and len(devices) > 1:
        chosen = devices[int(local_id) % len(devices)]
    elif devices:
        return  # already narrowed to one device
    else:
        chosen = local_id
    os.environ["CUDA_VISIBLE_DEVICES"] = chosen
    print(f"  task {local_id}: CUDA_VISIBLE_DEVICES={chosen}")


def _free_gpu() -> None:
    """Drop cached and unreferenced GPU allocations from the conversion
    stage."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            before = torch.cuda.memory_allocated() / 2**30
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            after = torch.cuda.memory_allocated() / 2**30
            print(f"  gpu memory after conversion: {before:.1f} GiB -> {after:.1f} GiB allocated")
    except ImportError:
        pass


def main() -> int:
    args = _parse()
    task_index = args.task_index if args.task_index is not None else int(os.environ["SLURM_PROCID"])

    tasks = json.loads(args.manifest.read_text())
    if task_index >= len(tasks):
        print(
            f"[task {task_index}] no checkpoint assigned (manifest has {len(tasks)} entries), exiting"
        )
        return 0

    task = tasks[task_index]
    it = task["iter"]
    hf_path = Path(task["hf_path"])
    _pin_local_gpu()
    print(f"[task {task_index}] iter={it} -> {hf_path}")

    if hf_path.exists():
        print(f"[task {task_index}] {hf_path} already exists, skipping conversion")
    else:
        # Conversion runs as a child process, not in-process: Megatron-Bridge keeps
        # live references to the loaded model, so its GPU memory (63 GiB for a 32B)
        # survives `del`/`empty_cache()` and is still held when validation loads the
        # export onto the same device. Letting the process exit is the only reliable
        # way to give it back.
        cmd = [
            sys.executable,
            "-m",
            "oellm_autoexp.backends.megatron_bridge.run_export",
            "--megatron-path",
            task["megatron_path"],
            "--hf-path",
            str(hf_path),
            "--hf-model",
            task["hf_model"],
            "--tokenizer",
            task["tokenizer"],
            "--bridge-root",
            task["bridge_root"],
            "--resources",
            task["resources"],
            "--megatron-config",
            task["megatron_config"],
            "--max-shard-size",
            args.max_shard_size,
        ]
        if task.get("derive_hf_arch"):
            cmd += ["--derive-hf-arch", task["derive_hf_arch"]]
        if task.get("vocab_size") is not None:
            cmd += ["--vocab-size", str(task["vocab_size"])]
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            return proc.returncode
        print(f"[task {task_index}] convert done: {it}")

    # Conversion loads the Megatron model onto this GPU, and validation then
    # loads the HF export onto the same one. A 32B in bf16 is ~64 GB, so with
    # both resident the second load hits OOM on a 95 GB device. Release the
    # first before starting the second.
    _free_gpu()

    validate(hf_path, Path(task["validation_json"]))
    print(f"[task {task_index}] validate done: {it}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
