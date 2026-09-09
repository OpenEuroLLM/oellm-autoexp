#!/usr/bin/env python3
"""Build a softcapped VIEW of an exported HF checkpoint, without copying the
weights.

Megatron-Bridge cannot carry `final_logit_softcapping` into a Qwen3 export (it supports it
only for Gemma 2 / Gemma-VL, and the Qwen3 HF config has no such field), so a model trained
with the cap exports to a checkpoint that runs uncapped. That is the wrong model for
likelihood-scored evaluation: the cap changes the softmax normalizer, so per-token
log-probabilities differ and multiple-choice options can reorder.

This creates a sibling directory that SYMLINKS every file of the export except config.json,
writes a patched config.json pointing at `modeling_qwen3_softcap.Qwen3SoftcapForCausalLM`,
and copies that module in beside it. A 32B export is ~64 GB, so symlinking rather than
copying is the difference between seconds and an hour.

    python3 scripts/korbi/make_capped_hf_view.py SRC DST --cap 30.0 --template MODELING.py

Idempotent: an existing DST is refreshed rather than failing, so a re-run after a partial
conversion is safe.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ARCH = "Qwen3SoftcapForCausalLM"
AUTO_MAP = {"AutoModelForCausalLM": "modeling_qwen3_softcap.Qwen3SoftcapForCausalLM"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", type=Path, help="exported HF checkpoint directory")
    ap.add_argument("dst", type=Path, help="capped view to create")
    ap.add_argument("--cap", type=float, required=True, help="final_logit_softcapping value")
    ap.add_argument("--template", type=Path, required=True, help="modeling_qwen3_softcap.py")
    args = ap.parse_args()

    src_cfg = args.src / "config.json"
    if not src_cfg.is_file():
        print(f"FATAL: no config.json in {args.src} -- conversion incomplete?", file=sys.stderr)
        return 1
    if not args.template.is_file():
        print(f"FATAL: template not found: {args.template}", file=sys.stderr)
        return 1

    args.dst.mkdir(parents=True, exist_ok=True)

    # Symlink everything except config.json, which is replaced, and the module we drop in.
    linked = 0
    for entry in sorted(args.src.iterdir()):
        if entry.name in ("config.json", "modeling_qwen3_softcap.py"):
            continue
        target = args.dst / entry.name
        if target.is_symlink() or target.exists():
            target.unlink()
        os.symlink(entry.resolve(), target)
        linked += 1

    cfg = json.loads(src_cfg.read_text())
    original_arch = cfg.get("architectures")
    cfg["architectures"] = [ARCH]
    cfg["auto_map"] = AUTO_MAP
    cfg["final_logit_softcapping"] = args.cap
    (args.dst / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    shutil.copyfile(args.template, args.dst / "modeling_qwen3_softcap.py")

    print(f"[capped-view] {args.dst}")
    print(f"[capped-view]   symlinked {linked} file(s) from {args.src}")
    print(f"[capped-view]   architectures {original_arch} -> [{ARCH!r}]")
    print(f"[capped-view]   final_logit_softcapping = {args.cap}")
    print("[capped-view]   NOTE: loading this requires trust_remote_code=True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
