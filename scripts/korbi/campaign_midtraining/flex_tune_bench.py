#!/usr/bin/env python3
"""FlexAttention kernel options for the page-sparse training path (page_sparse.flex_packed_page_attention, per KV
group, block-wise selection): forward + backward time per candidate kernel_options at v4b's attention shapes, each
candidate checked against the default options' output and q / k / v gradients first (a wrong tile choice must not
count). One GPU; documents packed as in training.

    flex_tune_bench.py [--docs 65536] [--selector page_maxpool] [--dim 256] [--out json]
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import time


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs", type=int, nargs="+", default=[65536])
    ap.add_argument("--selector", default="page_maxpool")
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--kv", type=int, default=2)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import torch
    from megatron.core.transformer.experimental_attention_variant import page_sparse as ps

    dev, dt = "cuda", torch.bfloat16
    torch.manual_seed(0)
    spans = []
    for L in a.docs:
        q = torch.randn(1, a.heads, L, a.dim, device=dev, dtype=dt, requires_grad=True)
        k = torch.randn(1, a.kv, L, a.dim, device=dev, dtype=dt, requires_grad=True)
        v = torch.randn(1, a.kv, L, a.dim, device=dev, dtype=dt, requires_grad=True)
        spans.append((q, k, v, 0, None))
    leaves = [t for sp in spans for t in sp[:3]]
    cfg = ps.PageSparseConfig(selector=a.selector, page=64, topk=32, window=512, sink_pages=1, select_block=64,
                              per_group=True)
    scale = a.dim**-0.5
    g_out = [torch.randn(1, a.heads, L, a.dim, device=dev, dtype=dt) for L in a.docs]
    default = ps.flex_kernel_options

    def run(opts):
        ps.flex_kernel_options = (lambda q, qb, kb: dict(opts)) if opts is not None else default
        for t in leaves:
            t.grad = None
        outs = ps.flex_packed_page_attention(spans, cfg, scale, 64)
        sum((o * g).sum() for o, g in zip(outs, g_out)).backward()
        return [o.detach().float() for o in outs], [t.grad.float() for t in leaves]

    def timed(opts):
        run(opts)
        torch.cuda.synchronize()
        ts = []
        for _ in range(a.steps):
            t0 = time.perf_counter()
            run(opts)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
        return statistics.median(ts)

    ref_out, ref_grad = run(None)
    base = default(spans[0][0], 64, 64)
    rows = [{"opts": "default " + json.dumps(base), "ms": timed(None)}]
    print(json.dumps(rows[-1]), flush=True)
    cands = []
    for bm, bn, w, s in itertools.product((32, 64), (32, 64), (4, 8), (1, 2, 3)):
        cands.append({"BLOCK_M": bm, "BLOCK_N": bn, "num_warps": w, "num_stages": s})
    for m1, n1, m2, n2 in [(32, 64, 64, 32), (64, 64, 64, 64), (32, 32, 32, 32), (64, 32, 32, 64)]:
        for w in (4, 8):
            cands.append({**base, "BLOCK_M1": m1, "BLOCK_N1": n1, "BLOCK_M2": m2, "BLOCK_N2": n2, "num_warps": w})
    for opts in cands:
        row = {"opts": json.dumps(opts)}
        try:
            out, grad = run(opts)
            err = max(float((x - y).abs().max() / y.abs().max().clamp(min=1e-6))
                      for x, y in zip(out + grad, ref_out + ref_grad))
            row["max_rel_err"] = err
            if err > 2e-2:
                row["rejected"] = "differs from the default options"
            else:
                row["ms"] = timed(opts)
        except Exception as e:  # noqa: BLE001 -- an unsupported tile / resource limit: report, continue
            row["error"] = f"{type(e).__name__}: {str(e)[:120]}"
        rows.append(row)
        print(json.dumps(row), flush=True)
    ok = sorted((r for r in rows if "ms" in r), key=lambda r: r["ms"])
    print("best:", json.dumps(ok[:5], indent=1), flush=True)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"args": vars(a), "gpu": torch.cuda.get_device_name(), "rows": rows}, fh, indent=2)


if __name__ == "__main__":
    main()
