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
import sys
from pathlib import Path

from oellm_autoexp.backends.megatron_bridge.run_export import run_export
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


def main() -> int:
    args = _parse()
    task_index = args.task_index if args.task_index is not None else int(os.environ["SLURM_PROCID"])

    tasks = json.loads(args.manifest.read_text())
    if task_index >= len(tasks):
        print(f"[task {task_index}] no checkpoint assigned (manifest has {len(tasks)} entries), exiting")
        return 0

    task = tasks[task_index]
    it = task["iter"]
    hf_path = Path(task["hf_path"])
    print(f"[task {task_index}] iter={it} -> {hf_path}")

    if hf_path.exists():
        print(f"[task {task_index}] {hf_path} already exists, skipping conversion")
    else:
        run_export(
            megatron_path=Path(task["megatron_path"]),
            hf_path=hf_path,
            hf_model=task["hf_model"],
            tokenizer=task["tokenizer"],
            bridge_root=Path(task["bridge_root"]),
            resources=Path(task["resources"]),
            derive_hf_arch=task.get("derive_hf_arch"),
            megatron_config=Path(task["megatron_config"]),
            max_shard_size=args.max_shard_size,
        )
        print(f"[task {task_index}] convert done: {it}")

    validate(hf_path, Path(task["validation_json"]))
    print(f"[task {task_index}] validate done: {it}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
