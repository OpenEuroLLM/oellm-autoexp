#!/usr/bin/env python3
"""Per-token page selection in training: what would it cost? One attention layer, forward + backward, one GPU:
  dense      -- SDPA causal (flash)
  block      -- the current training path: page_maxpool_pg, one selection per 64 queries, packed flex
  token_fla  -- every query token its own pages (1 sink + the window's pages + top-32 page_mean pages, per KV group):
                the per-token page scores + top-k (chunked), then FLA's NSA selection kernel (parallel_nsa) with the
                query heads per KV group padded to 16 with zero heads (the kernel's minimum) -- an upper bound
Reports ms per fwd + bwd and the selection's share.

    token_sparse_bench.py [--lengths 65536] [--heads 16] [--kv 2] [--dim 256]
"""

from __future__ import annotations

import argparse
import statistics
import time


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lengths", type=int, nargs="+", default=[65536])
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--kv", type=int, default=2)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--steps", type=int, default=5)
    a = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from fla.ops.nsa.parallel import parallel_nsa
    from megatron.core.transformer.experimental_attention_variant import page_sparse as ps

    dev, dt = "cuda", torch.bfloat16
    page, topk, wp, sinks = 64, 32, 512 // 64, 1
    G = a.heads // a.kv
    GP = max(16, 1 << (G - 1).bit_length())

    def timed(fn):
        fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(a.steps):
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
        return statistics.median(ts)

    for L in a.lengths:
        torch.manual_seed(0)
        q = torch.randn(1, a.heads, L, a.dim, device=dev, dtype=dt, requires_grad=True)
        k = torch.randn(1, a.kv, L, a.dim, device=dev, dtype=dt, requires_grad=True)
        v = torch.randn(1, a.kv, L, a.dim, device=dev, dtype=dt, requires_grad=True)
        res = {}
        res["dense"] = timed(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
                             .sum().backward())
        cfg = ps.PageSparseConfig(selector="page_maxpool", page=page, topk=topk, window=512, sink_pages=sinks,
                                  select_block=64, per_group=True)
        res["block"] = timed(lambda: ps.flex_packed_page_attention([(q, k, v, 0, None)], cfg, a.dim**-0.5, 64)[0]
                             .sum().backward())

        P = -(-L // page)

        def select():  # [1, L, kv, S] page indices per token and KV group (-1: none)
            with torch.no_grad():
                kp = ps.page_pool(k, page, "mean")  # [1, g, P, d]
                pos = torch.arange(L, device=dev)
                cur = pos // page
                out = []
                for gi in range(a.kv):
                    rows = []
                    for c0 in range(0, L, 8192):
                        qc = q[0, gi * G:(gi + 1) * G, c0:c0 + 8192]  # [G, c, d]
                        sc = torch.einsum("gcd,pd->cp", qc, kp[0, gi].to(qc.dtype)).float()
                        pc = torch.arange(P, device=dev)
                        cand = (pc[None] >= sinks) & (pc[None] < (cur[c0:c0 + 8192, None] - wp))
                        sc = sc.masked_fill(~cand, float("-inf"))
                        val, sel = sc.topk(min(topk, P), dim=-1)
                        rows.append(torch.where(torch.isfinite(val), sel, -1))
                    selp = torch.cat(rows, 0)  # [L, topk]
                    win = cur[:, None] - torch.arange(wp, -1, -1, device=dev)[None]  # [L, wp + 1]
                    win = torch.where(win >= sinks, win, -1)
                    sink = torch.zeros(L, sinks, dtype=torch.long, device=dev)
                    row = torch.cat([sink, win, selp], 1)
                    S2 = 1 << (row.shape[1] - 1).bit_length()  # the kernel needs a power-of-2 S: pad with -1 (skipped)
                    out.append(F.pad(row, (0, S2 - row.shape[1]), value=-1))
                return torch.stack(out, 1)[None]  # [1, L, g, S]

        res["token_select_only"] = timed(select)
        idx = select()
        S = idx.shape[-1]
        qt = q.detach().transpose(1, 2).reshape(1, L, a.kv, G, a.dim)
        qp = torch.zeros(1, L, a.kv, GP, a.dim, device=dev, dtype=dt)
        qp[..., :G, :] = qt
        qp = qp.reshape(1, L, a.kv * GP, a.dim).requires_grad_()
        kt, vt = (t.detach().transpose(1, 2).contiguous().requires_grad_() for t in (k, v))

        def token():
            o = parallel_nsa(qp, kt, vt, g_slc=torch.ones(1, L, a.kv * GP, device=dev, dtype=dt), block_indices=idx,
                             block_counts=S, block_size=page, scale=a.dim**-0.5)
            o.sum().backward()

        res["token_fla_kernel"] = timed(token)
        res["token_total"] = res["token_select_only"] + res["token_fla_kernel"]
        print(f"L={L} heads {a.heads}/{a.kv} d{a.dim} G={G} (padded {GP}) S={S}: "
              + "  ".join(f"{n} {t:.1f} ms" for n, t in res.items()), flush=True)


if __name__ == "__main__":
    main()
