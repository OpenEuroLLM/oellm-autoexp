#!/usr/bin/env python3
"""Batch-convert the multilingual architecture-scaling checkpoints to HF dirs.

Walks a training output group, picks the run directories matching a pattern,
and runs :mod:`oellm_autoexp.hf_export.convert_megatron_to_hf` on each. Existing
outputs are skipped, so it is safe to re-run as more cooldowns finish.

The default pattern selects the *cooldown finals* (``*_firstcd_decay<N>BT``) —
the 124 checkpoints the downstream-eval sweep scores.

Examples::

    # everything, into <group>_hf/ next to the training tree
    python scripts/convert_arch_scaling_to_hf.py

    # one variant only (what the sweep's per-variant convert stage runs)
    python scripts/convert_arch_scaling_to_hf.py --filter 'qwen3_gdn7_nope_*_firstcd_*'

    # the stable branch checkpoints instead of the cooldown finals
    python scripts/convert_arch_scaling_to_hf.py --filter '*_stable' --iter 38147
"""

from __future__ import annotations

import argparse
import fnmatch
import logging
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oellm_autoexp.hf_export.convert_megatron_to_hf import convert  # noqa: E402

LOGGER = logging.getLogger("convert_arch_scaling")

DEFAULT_GROUP = "architecture_scaling_variants_multilingual_main_7to1"
DEFAULT_TOKENIZER = "/e/data1/datasets/products/openeurollm/tokenizers/tokenizer-256k"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--train-group",
        type=Path,
        default=None,
        help="training output group dir (default: $OUTPUT_DIR/%s)" % DEFAULT_GROUP,
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="destination root (default: <train-group>_hf)",
    )
    p.add_argument("--filter", default="*_firstcd_decay*BT", help="glob over run dir names")
    p.add_argument("--tokenizer", type=Path, default=Path(DEFAULT_TOKENIZER))
    p.add_argument("--iter", type=int, default=None, help="force one iteration for every run")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    train_group = args.train_group or Path(
        os.environ.get("OUTPUT_DIR", "./output")
    ) / DEFAULT_GROUP
    out_root = args.out_root or train_group.parent / f"{train_group.name}_hf"

    runs = sorted(
        d
        for d in train_group.iterdir()
        if d.is_dir() and fnmatch.fnmatch(d.name, args.filter)
    )
    LOGGER.info("%d run dirs match %r under %s", len(runs), args.filter, train_group)

    converted = skipped = failed = 0
    for run in runs:
        out_dir = out_root / run.name
        if (out_dir / "model.safetensors").exists() and not args.overwrite:
            skipped += 1
            continue
        if not (run / "latest_checkpointed_iteration.txt").exists() and args.iter is None:
            LOGGER.warning("%s: no checkpoint yet, skipping", run.name)
            skipped += 1
            continue
        if args.dry_run:
            LOGGER.info("would convert %s -> %s", run.name, out_dir)
            converted += 1
            continue
        try:
            convert(run, out_dir, args.tokenizer, args.iter)
            converted += 1
        except Exception:  # keep going; one bad run must not stall the batch
            failed += 1
            LOGGER.error("FAILED %s\n%s", run.name, traceback.format_exc())

    LOGGER.info("converted=%d skipped=%d failed=%d -> %s", converted, skipped, failed, out_root)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
