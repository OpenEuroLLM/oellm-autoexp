#!/usr/bin/env python3
"""Attention cost per training sequence on the real document layout (doclen_sample.py's packed phase-2 sequences):
forward + backward of one attention layer at v4b's shapes for every variant on N sampled sequences, then the per-step
estimate -- the mean, and the expected slowest of R data-parallel ranks (each rank one sequence: the step waits for
it). Variants:
  dense_masked  -- varlen flash attention over the documents (the production setting, document masking)
  dense_full    -- causal over the whole sequence (no document masking)
  block / block_rowmask -- page_maxpool_pg, one selection per 64 queries (flex / rowmask kernel), per document
  tok40         -- per-token selection inside 40 block candidates (rowmask), per document
  *_full        -- the sparse variants without document masking (one span)

    attn_packed_bench.py --seqs seqs_64k.json [--n 24] [--ranks 384 512] [--out json]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seqs", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--ranks", type=int, nargs="+", default=[384, 512])
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--kv", type=int, default=2)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--variants", nargs="+",
                    default=["dense_masked", "dense_full", "block", "block_rowmask", "tok40", "block_full",
                             "tok40_full"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from megatron.core.transformer.experimental_attention_variant import page_sparse as ps

    d = json.load(open(a.seqs))
    L, seqs = d["seq"], d["seqs"]
    rng = random.Random(0)
    # half random, half from the longest-document tail (the stragglers), weights kept for the mean
    order = sorted(range(len(seqs)), key=lambda i: max(seqs[i]))
    tail = order[int(0.9 * len(order)):]
    pick = rng.sample(range(len(seqs)), a.n // 2) + rng.sample(tail, a.n - a.n // 2)
    dev, dt = "cuda", torch.bfloat16
    scale = a.dim**-0.5
    base = dict(selector="page_maxpool", page=64, topk=32, window=512, sink_pages=1, per_group=True)
    cfgs = {"block": (ps.PageSparseConfig(**base, select_block=64), "flex"),
            "block_rowmask": (ps.PageSparseConfig(**base, select_block=64), "rowmask"),
            "tok40": (ps.PageSparseConfig(**base, select_block=1, block_candidates=40, candidate_block=64),
                      "rowmask")}

    def timed(fn):
        fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(a.steps):
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
        return min(ts)

    rows = []
    for si in pick:
        segs = seqs[si]
        torch.manual_seed(si)
        q = torch.randn(1, a.heads, L, a.dim, device=dev, dtype=dt, requires_grad=True)
        k = torch.randn(1, a.kv, L, a.dim, device=dev, dtype=dt, requires_grad=True)
        v = torch.randn(1, a.kv, L, a.dim, device=dev, dtype=dt, requires_grad=True)
        bounds = [0]
        for s in segs:
            bounds.append(bounds[-1] + s)
        row = {"seq": si, "docs": len(segs), "longest": max(segs)}
        for var in a.variants:
            full = var.endswith("_full")
            if var == "dense_full":
                fn = lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True).sum().backward()
            elif var == "dense_masked":
                try:  # the production kernel class: one varlen flash call for every document
                    from flash_attn import flash_attn_varlen_func

                    cu = torch.tensor(bounds, dtype=torch.int32, device=dev)
                    mx = max(segs)
                    fn = lambda: flash_attn_varlen_func(q[0].transpose(0, 1), k[0].transpose(0, 1),
                                                        v[0].transpose(0, 1), cu, cu, mx, mx, causal=True,
                                                        softmax_scale=scale).sum().backward()
                except ImportError:  # one SDPA call per document (overstates many-document sequences)
                    def fn():
                        tot = 0
                        for x, y in zip(bounds[:-1], bounds[1:]):
                            tot = tot + F.scaled_dot_product_attention(q[:, :, x:y], k[:, :, x:y], v[:, :, x:y],
                                                                       is_causal=True, enable_gqa=True).sum()
                        tot.backward()
            else:
                cfg, kern = cfgs[var.removesuffix("_full")]
                spans = ([(q, k, v, 0, None)] if full else
                         [(q[:, :, x:y], k[:, :, x:y], v[:, :, x:y], 0, None) for x, y in zip(bounds[:-1], bounds[1:])])
                fn = lambda: sum(o.sum() for o in ps.flex_packed_page_attention(spans, cfg, scale, 64,
                                                                                kernel=kern)).backward()
            try:
                row[var] = timed(fn)
            except torch.OutOfMemoryError:
                row[var] = None
        rows.append(row)
        print(json.dumps(row), flush=True)
    # estimate: random half -> the mean; all -> the tail; the slowest of R ranks by resampling the empirical layout
    # costs (the tail picks stand for the top 10 % of layouts)
    summary = {}
    rnd, tl_ = rows[: a.n // 2], rows[a.n // 2:]
    for var in a.variants:
        r = [x[var] for x in rnd if x.get(var) is not None]
        t = [x[var] for x in tl_ if x.get(var) is not None]
        if not r:
            continue
        pool = r + t * max(1, round(len(r) / max(1, len(t)) / 9))  # tail at ~10 % weight
        est = {}
        for R in a.ranks:
            # P(no rank draws from the top 10 %) = 0.9^R ~ 0 for R >= 100: the step ~ the tail's typical maximum
            reps = [max(rng.choice(t if rng.random() < 0.1 else r) for _ in range(R)) for _ in range(200)] if t else []
            est[f"slowest_of_{R}"] = statistics.mean(reps) if reps else None
        summary[var] = {"mean": statistics.mean(r), "tail_mean": statistics.mean(t) if t else None, **est}
    print(json.dumps(summary, indent=1), flush=True)
    if a.out:
        json.dump({"args": vars(a), "gpu": torch.cuda.get_device_name(), "rows": rows, "summary": summary},
                  open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
