#!/usr/bin/env python3
"""How many RMSNorm channels are actually switched off, not just the smallest
one.

Item 12 in docs/64k-debug/DEBUG.md rests on `min` -- the single smallest gain out
of ~5,120 channels x 64 layers, tracked over 22 checkpoints. That is an extreme
order statistic: the minimum of a large sample drifts downward as the
distribution widens, whether or not the layer has lost any capacity. So "the
minimum fell from 0.9375 to 0.0066" may describe one unlucky channel rather than
a collapse.

This records the distribution instead: what FRACTION of channels sit below
10% / 25% / 50% of their layer's median |gain|. If that fraction grows, channels
really are switching off. If it stays flat while `min` wanders, item 12 is an
artefact of the statistic and closes.

Magnitudes throughout: a gain of -0.5 is a sign flip, not an off channel, so the
thresholds are on |gain| and the sign is reported separately.

CPU only, no GPU, seconds per checkpoint -- it reads just the five 1-D gain
tensors out of each ~450 GB checkpoint, reusing the loader in
scan_norm_gains.py.

  apptainer exec /e/project1/e-sta-openeurollm/container/\
MegatronTraining-JUPITER-te218-fa3_aarch64_202608280932.sif \
    python3 scripts/scan_gain_distribution.py \
      /e/scratch/e-sta-openeurollm/production_training/\
oellm_32b_dense_prod_dataopt5_gbs4096_lr3e-4/checkpoints \
      --csv docs/64k-debug/data/gain_distribution.csv
"""

import argparse
import csv
import re
from pathlib import Path

import torch

from scan_norm_gains import SHORT, _fmt, load_gains

FRACS = (0.50, 0.25, 0.10, 0.01)


def rows_for(it, gains):
    """One row per (tensor, layer); layer -1 for the unstacked final norm."""
    for key, t in gains.items():
        t = t.unsqueeze(0) if t.ndim == 1 else t
        layers = range(t.shape[0]) if t.shape[0] > 1 else [-1]
        for i, layer in enumerate(layers):
            g = t[i].float()
            a = g.abs()
            med = a.median().item()
            row = {
                "iter": it,
                "tensor": SHORT[key],
                "layer": layer,
                "n": a.numel(),
                "median_abs": med,
                "min_abs": a.min().item(),
                "frac_negative": (g < 0).float().mean().item(),
            }
            for f in FRACS:
                # NB fraction of channels below f*median, not a count past a
                # fixed cutoff: the median moves over training, and a fixed
                # cutoff would conflate "channels switched off" with "the whole
                # layer got smaller".
                key_name = f"frac_below_{int(f * 100):02d}pct_med"
                row[key_name] = (a < f * med).float().mean().item() if med > 0 else float("nan")
            row["p01_over_med"] = (
                (torch.quantile(a, 0.01).item() / med) if med > 0 else float("nan")
            )
            row["p10_over_med"] = (
                (torch.quantile(a, 0.10).item() / med) if med > 0 else float("nan")
            )
            yield row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_root", type=Path)
    ap.add_argument("--csv", type=Path)
    args = ap.parse_args()

    ckpts = sorted(
        args.ckpt_root.glob("iter_*"), key=lambda p: int(re.search(r"\d+", p.name).group())
    )
    if not ckpts:
        raise SystemExit(f"no iter_* checkpoints under {args.ckpt_root}")

    all_rows, header_done = [], False
    for ckpt in ckpts:
        it = int(re.search(r"\d+", ckpt.name).group())
        try:
            gains = load_gains(ckpt)
        except Exception as e:
            print(f"{it:<8} skipped: {type(e).__name__}")
            continue
        rows = list(rows_for(it, gains))
        all_rows += rows

        # Summary: worst layer's fraction below 10% of median, per tensor --
        # the number item 12 actually needs.
        by_tensor = {}
        for r in rows:
            by_tensor.setdefault(r["tensor"], []).append(r["frac_below_10pct_med"])
        by_tensor = {k: max(v) for k, v in by_tensor.items()}
        if not header_done:
            print("worst-layer fraction of channels below 10% of layer median |gain|\n")
            print("iter    " + "  ".join(f"{n:>16}" for n in by_tensor))
            header_done = True
        print(f"{it:<8}" + "  ".join(f"{by_tensor[n]:>16.5f}" for n in by_tensor))

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
            w.writeheader()
            w.writerows(_fmt(r) for r in all_rows)
        print(f"\nwrote {len(all_rows)} rows -> {args.csv}")


if __name__ == "__main__":
    main()
