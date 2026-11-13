#!/usr/bin/env python3
"""Table of the layer-swap test (docs/recall_surgery_ambitious.md §11; fixedvalid_eval_layer_swap_mn5): every arm's
mean eval loss minus its base's (b<B>_base) on identical tokens, per validation set (seq_length of the dump), and
for swaps the fraction of the zero-ablation's cost of the same layers (swap / zero: 0 = the donor layer fits like
the base's own, 1 = as bad as removing the layer's output).

    python scripts/korbi/production/layer_swap_table.py [--root /gpfs/scratch/ehpc533/hybrid_nucleus/fixedvalid_eval/layer_swap]

Needs torch (the training image: singularity exec <sif> python3 ...). Arms without a complete dump are listed as
pending.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import os
import re

import torch


def load_dump(d: str) -> dict:
    """{token hash: (seq_length, losses[mask])} of one MEGATRON_EVAL_TOKEN_DUMP directory."""
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "r*_*.pt"))):
        x = torch.load(f)
        key = hashlib.sha256(x["tokens"].numpy().tobytes()).hexdigest()
        if key in out:  # the ranks of one data-parallel group write the same micro-batch
            continue
        m = x["loss_mask"].reshape(x["losses"].shape).bool()
        out[key] = (x.get("seq_length"), x["losses"][m].double())
    return out


def diff(base: dict, arm: dict) -> dict:
    """{seq_length: (tokens, base mean, arm mean - base mean)} over the micro-batches both have."""
    acc = {}
    for key, (seq, lb) in base.items():
        if key not in arm:
            continue
        la = arm[key][1]
        s = acc.setdefault(seq, [0, 0.0, 0.0])
        s[0] += lb.numel()
        s[1] += lb.sum().item()
        s[2] += (la - lb).sum().item()
    return {seq: (n, sb / n, sd / n) for seq, (n, sb, sd) in acc.items() if n}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="/gpfs/scratch/ehpc533/hybrid_nucleus/fixedvalid_eval/layer_swap")
    ap.add_argument("--match", default="", help="only arms matching this regex (and their bases); the login node "
                    "kills a full table at its CPU limit")
    a = ap.parse_args(argv)
    arms = sorted(os.path.basename(p) for p in glob.glob(os.path.join(a.root, "b*")))
    if a.match:
        keep = {x for x in arms if re.search(a.match, x)}
        arms = [x for x in arms if x in keep or (x.endswith("_base") and any(k.split("_")[0] == x.split("_")[0] for k in keep))]
    bases = {}
    for arm in arms:
        if arm.endswith("_base") and glob.glob(os.path.join(a.root, arm, "token_dump", "r*_*.pt")):
            bases[arm.split("_")[0]] = load_dump(os.path.join(a.root, arm, "token_dump"))
    rows, pending = [], []
    for arm in arms:
        m = re.match(r"b(\d+)_(\w+?)_(zero|d\d+)$", arm)
        if not m:
            continue
        b, layers, what = m.groups()
        dump = os.path.join(a.root, arm, "token_dump")
        if f"b{b}" not in bases or not glob.glob(os.path.join(dump, "r*_*.pt")):
            pending.append(arm)
            continue
        rows.append((int(b), layers, what, diff(bases[f"b{b}"], load_dump(dump))))
    seqs = sorted({s for r in rows for s in r[3]}, key=str)
    zero = {(b, layers): d for b, layers, what, d in rows if what == "zero"}
    head = f"{'base':>6} {'layers':>7} {'donor':>6} {'dist':>6} | " + " | ".join(
        f"dloss@{s} (swap/zero)" for s in seqs)
    print(head)
    print("-" * len(head))
    order = {"zero": -1}
    for b, layers, what, d in sorted(rows, key=lambda r: (r[0], r[1], order.get(r[2], 0), -int(r[2][1:]) if r[2] != "zero" else 0)):
        donor = what if what == "zero" else what[1:]
        dist = "" if what == "zero" else str(b - int(what[1:]))
        cells = []
        for s in seqs:
            if s not in d:
                cells.append(f"{'-':>21}")
                continue
            z = zero.get((b, layers), {}).get(s)
            frac = f"{d[s][2] / z[2]:.2f}" if what != "zero" and z and abs(z[2]) > 1e-9 else ""
            cells.append(f"{d[s][2]:+.4f} ({frac:>5})" if frac else f"{d[s][2]:+.4f}        ")
        print(f"{b:>6} {layers:>7} {donor:>6} {dist:>6} | " + " | ".join(f"{c:>21}" for c in cells))
    for name, d in bases.items():
        print(f"{name} base mean loss: " + ", ".join(
            f"seq {s} {sum(v[1].sum().item() for v in d.values() if v[0] == s) / max(1, sum(v[1].numel() for v in d.values() if v[0] == s)):.4f}"
            for s in sorted({v[0] for v in d.values()}, key=str)))
    if pending:
        print(f"pending ({len(pending)}): {' '.join(pending)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
