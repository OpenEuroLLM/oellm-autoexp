#!/usr/bin/env python3
"""Build a Megatron checkpoint VIEW that Megatron-Bridge can actually load.

Our training checkpoints contain only the sharded tensors -- ``*.distcp``, ``.metadata``,
``metadata.json`` -- and no ``common.pt``. Megatron-Bridge needs that file: it holds the
Megatron ``args`` the exporter builds its model from. Without it the bridge infers a model
from the derived HF config, the inferred model's ``_extra_state`` covers a handful of layers
instead of all 64, and the load dies in validation with

    Invalid access pattern: 59 ShardedObject are missing

That is what killed conversion jobs 1736104, 1740243 and 1740301 -- always at the
dist-checkpoint load, ~12 minutes in, on a checkpoint that is itself complete.

This creates a directory that SYMLINKS every file of the source checkpoint and adds a real
``common.pt``. A 32B checkpoint is ~442 GB over 2049 files, so linking rather than copying is
the difference between seconds and hours.

    python3 scripts/korbi/make_megatron_view.py SRC_ITER_DIR DST_ITER_DIR --common-pt REF

The reference ``common.pt`` may be borrowed from another checkpoint OF THE SAME
ARCHITECTURE -- its `args` describe the model, not the weights. The relevant fields for the
32B flagship are

    num_layers 64  hidden_size 5120  ffn_hidden_size 25600  num_attention_heads 64
    padded_vocab_size 262144  tensor_model_parallel_size 1  pipeline_model_parallel_size 1
    fp8 None

TP1/PP1 and fp8 off are deliberate: the exporter builds a plain unsharded bf16 model and
dist-checkpointing reshards the saved TP4/PP4/VPP4 layout into it on load.

Borrowing is only safe if the architectures match, so this VERIFIES the reference against
the checkpoint's own ``.metadata`` before writing anything. A mismatch would otherwise
produce an export that loads without complaint and is silently wrong.
"""

import argparse
import shutil
import sys
from pathlib import Path

# Tensors whose global_shape pins the architecture, and which arg each axis must equal.
_SHAPE_CHECKS = [
    ("decoder.layers.self_attention.linear_proj.weight", {0: "num_layers", 1: "hidden_size"}),
    ("output_layer.weight", {0: "padded_vocab_size", 1: "hidden_size"}),
]


def verify_architecture(src: Path, common_pt: Path) -> list[str]:
    """Compare the reference args against the checkpoint's own metadata.

    Returns a list of problems; empty means consistent. Returns a single
    note (not a problem) if torch is unavailable, so the view can still
    be built on a login node.
    """
    try:
        import torch
        from torch.distributed.checkpoint import FileSystemReader
    except Exception as exc:  # pragma: no cover - environment dependent
        return [f"NOTE: torch unavailable ({exc.__class__.__name__}); architecture NOT verified"]

    args = getattr(
        torch.load(common_pt, map_location="cpu", weights_only=False).get("args"), "__dict__", {}
    )
    if not args:
        return ["reference common.pt has no `args`"]

    meta = FileSystemReader(str(src)).read_metadata().state_dict_metadata
    problems = []
    checked = 0
    for key, axes in _SHAPE_CHECKS:
        entry = meta.get(key)
        shape = getattr(entry, "size", None)
        if shape is None:
            continue
        for axis, arg_name in axes.items():
            if axis >= len(shape) or arg_name not in args:
                continue
            checked += 1
            if int(shape[axis]) != int(args[arg_name]):
                problems.append(
                    f"{key} axis {axis} is {int(shape[axis])} but reference {arg_name}="
                    f"{args[arg_name]}"
                )
    if not checked:
        problems.append("could not check any tensor shape against the reference args")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", type=Path, help="source iter_XXXXXXX directory")
    ap.add_argument("dst", type=Path, help="view directory to create")
    ap.add_argument("--common-pt", type=Path, required=True, help="reference common.pt")
    ap.add_argument(
        "--skip-verify",
        action="store_true",
        help="do not check the reference architecture against the checkpoint metadata",
    )
    args = ap.parse_args()

    if not (args.src / ".metadata").is_file():
        print(f"FATAL: {args.src} has no .metadata -- not a dist checkpoint", file=sys.stderr)
        return 1
    if not args.common_pt.is_file():
        print(f"FATAL: reference not found: {args.common_pt}", file=sys.stderr)
        return 1
    if (args.src / "common.pt").is_file():
        print(f"NOTE: {args.src} already has common.pt; a view may be unnecessary")

    if not args.skip_verify:
        problems = verify_architecture(args.src, args.common_pt)
        fatal = [p for p in problems if not p.startswith("NOTE:")]
        for p in problems:
            print(f"[view]   {p}")
        if fatal:
            print("FATAL: reference common.pt does not match this checkpoint", file=sys.stderr)
            return 1

    args.dst.mkdir(parents=True, exist_ok=True)
    linked = 0
    for entry in sorted(args.src.iterdir()) + [args.src / ".metadata"]:
        if entry.name == "common.pt" or not entry.exists():
            continue
        target = args.dst / entry.name
        if target.is_symlink() or target.exists():
            target.unlink()
        target.symlink_to(entry.resolve())
        linked += 1

    shutil.copyfile(args.common_pt, args.dst / "common.pt")

    print(f"[view] {args.dst}")
    print(f"[view]   symlinked {linked} entries from {args.src}")
    print(f"[view]   common.pt copied from {args.common_pt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
