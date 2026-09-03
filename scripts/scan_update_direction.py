#!/usr/bin/env python3
"""Is the model moving faster, or the same speed in worse directions?

WHY THIS EXISTS
---------------
Under Adam the update is `LR * m / (sqrt(v) + eps)`, so the gradient MAGNITUDE is
divided out. That is why the flagship's gradient norm can fall from 0.512 to
0.457 while the loss rises: the two are not mechanically coupled, and looking for
a correlation between them is looking at the wrong variable. What is not
normalised away is where the weights actually go.

This measures two things over the checkpoint series, exactly and model-wide:

  step   ||dW|| per interval -- is the model displacing further per 4,000 steps?
  cos    cos(dW_t, dW_t-1)   -- do consecutive displacements still agree?

A model descending a consistent basin keeps a positive cosine between successive
displacements. A model that has entered a regime where it overshoots and
reverses loses that agreement, and the cosine falls toward zero or below. If the
turn is an optimization-regime change, this is where it should be visible.

WHY THE MASTER WEIGHTS, NOT THE bf16 WEIGHTS
--------------------------------------------
`optimizer.distributed.*.param` holds the fp32 master weights the optimizer
actually updates. Differencing the bf16 copies instead would subtract two rounded
numbers, and one ulp of bf16 near a weight of 0.01 is comparable to the update
being measured -- the difference would be dominated by rounding noise, biasing
every cosine toward zero. The master weights have no such problem.

These flat per-bucket buffers are laid out by DP rank, so their layout is only
comparable across checkpoints of the SAME run at the SAME DP size. That holds
here (all 80 bucket shapes are identical across every checkpoint read), and it is
asserted at load time rather than assumed.

WHAT THIS CANNOT SEE
--------------------
Checkpoints are 4,000 steps apart, so each dW is the sum of 4,000 updates.
Per-step oscillation -- the edge-of-stability signature -- averages out inside a
single dW and is invisible here. What survives is reversal at the 4,000-step
scale. A null result therefore does not exclude edge-of-stability; it excludes
large-scale directional reversal only. Resolving the former needs the 500-step
checkpoint cadence that item 19 also asks for.

No GPU is used: this is CPU and I/O only.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp


def bucket_keys(reader):
    """The fp32 master-weight buffers, largest first so peak RAM shows up
    early."""
    meta = reader.read_metadata().state_dict_metadata
    ks = {k: v for k, v in meta.items() if k.endswith(".param")}
    return sorted(ks, key=lambda k: -ks[k].size[0]), ks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_root", type=Path)
    ap.add_argument("--iters", required=True, help="comma-separated, evenly spaced")
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--limit-buckets", type=int, default=0, help="0 = all; else first N")
    ap.add_argument("--stride", type=int, default=1, help="take every Nth bucket")
    ap.add_argument("--offset", type=int, default=0, help="starting bucket for --stride")
    ap.add_argument(
        "--partial",
        type=Path,
        help="write raw accumulators here instead of a report; combine with --combine",
    )
    ap.add_argument(
        "--combine",
        nargs="*",
        help="sum these partial .json files into the final report; no checkpoints read",
    )
    args = ap.parse_args()

    # The accumulators are plain sums over parameters, so splitting the buckets
    # across jobs and adding the partials afterwards is exactly equal to one
    # sequential pass. The work is I/O bound, so this is the only way the full
    # 1.1 TB read finishes in reasonable wall time.
    if args.combine:
        acc = None
        for f in args.combine:
            d = json.loads(Path(f).read_text())
            if acc is None:
                acc = d
            else:
                assert d["iters"] == acc["iters"], "partials span different checkpoints"
                for fld in ("sq_d", "dots", "sq_w"):
                    acc[fld] = [a + b for a, b in zip(acc[fld], d[fld])]
                acc["ntot"] += d["ntot"]
                acc["nbuckets"] += d["nbuckets"]
        report(acc["iters"], acc["sq_d"], acc["dots"], acc["sq_w"], acc["ntot"], args.csv)
        print(f"combined {len(args.combine)} partials, {acc['nbuckets']} buckets")
        return

    iters = [s.strip() for s in args.iters.split(",")]
    readers = {it: dcp.FileSystemReader(str(args.ckpt_root / f"iter_{it}")) for it in iters}
    keys, meta = bucket_keys(readers[iters[0]])
    for it in iters[1:]:
        k2, m2 = bucket_keys(readers[it])
        assert k2 == keys and all(m2[k].size == meta[k].size for k in keys), (
            f"bucket layout differs at iter_{it}; these checkpoints are not comparable"
        )
    if args.limit_buckets:
        keys = keys[: args.limit_buckets]
    keys = keys[args.offset :: args.stride]
    n_int = len(iters) - 1
    print(f"{len(keys)} buckets, {len(iters)} checkpoints, {n_int} intervals", flush=True)

    # Global accumulators, summed exactly over every parameter in every bucket.
    sq_d = [0.0] * n_int  # ||dW_i||^2
    dots = [0.0] * (n_int - 1)  # dW_i . dW_i+1
    sq_w = [0.0] * len(iters)  # ||W_t||^2
    ntot = 0

    for bi, k in enumerate(keys):
        ws = []
        for it in iters:
            buf = {k: torch.empty(meta[k].size, dtype=meta[k].properties.dtype)}
            dcp.load(buf, storage_reader=readers[it])
            ws.append(buf[k])
        ntot += ws[0].numel()
        for t, w in enumerate(ws):
            sq_w[t] += float(w.pow(2).sum(dtype=torch.float64))
        ds = [ws[i + 1] - ws[i] for i in range(n_int)]
        del ws
        for i, d in enumerate(ds):
            sq_d[i] += float(d.pow(2).sum(dtype=torch.float64))
        for i in range(n_int - 1):
            dots[i] += float((ds[i] * ds[i + 1]).sum(dtype=torch.float64))
        del ds
        print(f"  bucket {bi + 1}/{len(keys)} {k.split('.')[-3]} done", flush=True)

    if args.partial:
        args.partial.parent.mkdir(parents=True, exist_ok=True)
        args.partial.write_text(
            json.dumps(
                {
                    "iters": iters,
                    "sq_d": sq_d,
                    "dots": dots,
                    "sq_w": sq_w,
                    "ntot": ntot,
                    "nbuckets": len(keys),
                }
            )
        )
        print(f"partial ({len(keys)} buckets, {ntot / 1e9:.3f}B params) -> {args.partial}")
        return

    report(iters, sq_d, dots, sq_w, ntot, args.csv)


def report(iters, sq_d, dots, sq_w, ntot, csv_path):
    n_int = len(iters) - 1
    print(f"\n{ntot / 1e9:.3f}B parameters, exact global sums\n")
    rows = []
    print(f"{'interval':<22}{'||dW||':>12}{'||dW||/||W||':>14}{'cos(prev,this)':>16}")
    for i in range(n_int):
        nd = math.sqrt(sq_d[i])
        rel = nd / math.sqrt(sq_w[i])
        c = ""
        if i > 0:
            c = dots[i - 1] / (math.sqrt(sq_d[i - 1]) * nd)
        lab = f"{iters[i]}->{iters[i + 1]}"
        print(f"{lab:<22}{nd:>12.4f}{rel:>14.6f}{(f'{c:+.4f}' if c != '' else '-'):>16}")
        rows.append(
            {
                "from": iters[i],
                "to": iters[i + 1],
                "norm_dW": f"{nd:.6f}",
                "rel_dW": f"{rel:.8f}",
                "cos_with_prev": (f"{c:.6f}" if c != "" else ""),
                "n_params": ntot,
            }
        )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n-> {csv_path}")


if __name__ == "__main__":
    main()
