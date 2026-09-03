#!/usr/bin/env python3
"""Adam moments and the implied effective step size, straight from a
checkpoint.

Tests the one mechanism on the list with a real threshold in it. The Adam step is

    lr * m / (sqrt(v) + eps)

As the gradient signal shrinks, sqrt(v) shrinks with it, so the step does NOT
shrink proportionally; once sqrt(v) falls towards adam_eps the denominator stops
tracking and the effective step GROWS. That is a threshold, not a drift, and the
moments are fp32 in both FP8 and bf16 runs -- so the bf16 control cannot rule it
out, which is exactly the profile the trigger must have.

The distributed optimizer stores flat per-bucket buffers, not per-parameter
tensors, so there are no parameter names here. Bucket layout is deterministic
for a fixed parallel config, so bucket_idx_N is the same parameters in every
checkpoint and a fixed subset is comparable across them.

  apptainer exec <sif> python3 scripts/scan_optimizer_state.py <iter_dir> \\
      --csv docs/64k-debug/data/optimizer_state.csv
"""

import argparse
import csv
import re
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp

ADAM_EPS = 1e-8
SAMPLE = 2_000_000  # elements kept for percentiles; full tensor is ~150M


def bucket_keys(meta, n_buckets, spread=False):
    """N shards with all three buffers present.

    A shard is identified by its FULL prefix, not by bucket_idx. The keys look
    like `...group_idx_G.gbuf_idx_B.dtype_(...).bucket_idx_N`, and bucket_idx is
    only the last component: at this parallel config bucket_idx 0 alone covers
    64 distinct (group, gbuf) shards holding 30.7B of the model's 33.9B
    parameters. An earlier version of this function keyed on bucket_idx alone,
    so all 64 collided in a dict and one arbitrary survivor won -- it reported
    "5 buckets" while actually covering 1.49B parameters, 4.4% of the model,
    chosen by dict iteration order. Any figure produced before this fix is a
    sample of that arbitrary slice, not a statement about the model.

    `spread` picks shards evenly across the sorted list rather than the first n.
    """
    pat = re.compile(r"(.*bucket_idx_(\d+))\.(exp_avg|exp_avg_sq|param)$")
    found = {}
    for k in meta:
        m = pat.match(k)
        if m:
            found.setdefault(m.group(1), {})[m.group(3)] = k
    ok = sorted(b for b, v in found.items() if len(v) == 3)
    if n_buckets < len(ok):
        if spread:
            idx = (
                [round(i * (len(ok) - 1) / (n_buckets - 1)) for i in range(n_buckets)]
                if n_buckets > 1
                else [0]
            )
            ok = [ok[i] for i in sorted(set(idx))]
        else:
            ok = ok[:n_buckets]
    return [(b, found[b]) for b in ok]


def stats(m, v, p, gen):
    """Reductions on the full buffers; percentiles on a fixed random subset."""
    sqrt_v = v.clamp_min(0).sqrt()
    step = m.abs() / (sqrt_v + ADAM_EPS)  # the Adam step, before lr

    idx = torch.randint(0, m.numel(), (min(SAMPLE, m.numel()),), generator=gen)
    sv_s = sqrt_v[idx]
    st_s = step[idx]
    q = torch.tensor([0.01, 0.5, 0.99])

    sv_q = torch.quantile(sv_s, q)
    st_q = torch.quantile(st_s, q)
    return {
        "m_rms": m.pow(2).mean().sqrt().item(),
        "sqrt_v_rms": sqrt_v.pow(2).mean().sqrt().item(),
        "param_rms": p.pow(2).mean().sqrt().item(),
        "sqrt_v_p01": sv_q[0].item(),
        "sqrt_v_p50": sv_q[1].item(),
        "sqrt_v_p99": sv_q[2].item(),
        # How much of the buffer has sqrt(v) small enough that adam_eps starts
        # to matter in the denominator. This is the threshold, if there is one.
        "frac_sqrtv_below_1e6eps": (sqrt_v < ADAM_EPS * 1e6).float().mean().item(),
        "frac_sqrtv_below_1e3eps": (sqrt_v < ADAM_EPS * 1e3).float().mean().item(),
        "frac_sqrtv_below_10eps": (sqrt_v < ADAM_EPS * 10).float().mean().item(),
        "step_p50": st_q[1].item(),
        "step_p99": st_q[2].item(),
        # Step relative to the weight it is applied to: the scale-free version.
        "rel_step_p50": st_q[1].item() / max(p.abs().mean().item(), 1e-12),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--buckets", type=int, default=6)
    ap.add_argument(
        "--spread",
        action="store_true",
        help="sample buckets evenly across the model, not the first n",
    )
    args = ap.parse_args()

    reader = dcp.FileSystemReader(str(args.ckpt))
    meta = reader.read_metadata().state_dict_metadata
    buckets = bucket_keys(meta, args.buckets, spread=args.spread)
    if not buckets:
        raise SystemExit(f"no optimizer buckets in {args.ckpt}")

    it = int(re.search(r"\d+", args.ckpt.name).group())
    gen = torch.Generator().manual_seed(0)  # same sample every checkpoint
    rows = []
    for b, keys in buckets:
        buf = {}
        for name, k in keys.items():
            md = meta[k]
            buf[name] = torch.empty(md.size, dtype=md.properties.dtype)
        dcp.load({keys[n]: t for n, t in buf.items()}, storage_reader=reader)
        rows.append(
            {
                "iter": it,
                "bucket": b,
                "numel": buf["exp_avg"].numel(),
                **stats(buf["exp_avg"], buf["exp_avg_sq"], buf["param"], gen),
            }
        )
        del buf

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    new = not args.csv.exists()
    with open(args.csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    med = sorted(r["step_p50"] for r in rows)[len(rows) // 2]
    sv = sorted(r["sqrt_v_p50"] for r in rows)[len(rows) // 2]
    print(
        f"{it}: {len(rows)} buckets, median sqrt(v) {sv:.3e}, median |step| {med:.4f} -> {args.csv}"
    )


if __name__ == "__main__":
    main()
