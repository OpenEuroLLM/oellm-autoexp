#!/usr/bin/env python3
"""Pure per-token page selection in training, timed on REAL selections: one attention layer's forward + backward at
v4b's shapes (16 / 2 heads, head dim 256) on one GPU, with the page masks a real model's queries choose
(token_union_probe.py --lists: page_maxpool top 32, window 512, 1 sink, per KV group, on held-out long documents).
The kernel's time depends only on the masks, so random q / k / v with replayed masks time the real selections.

Per lists file (one document) and attention layer:
  rowmask64   -- rowmask_block_attention, 64-query tiles (each loops over its 64 queries' page union)
  hp<T>       -- rowmask_head_packed_attention, tiles of T queries x the group's 8 heads (union of T queries)
  b<B>        -- one selection per B queries (--blocks): each query its anchor's (the B-block's first query's) real
                 pages plus its own window and sinks, in head-packed tiles of token_tile queries (ub<B>: the union)
  u64 / u<T>  -- mean page union per tile
and once per length: dense (SDPA causal), block (page_maxpool, one selection per 64 queries, packed flex and rowmask),
tok40 (per-token inside 40 block candidates, rowmask), select (the pure per-token selection + head-packed lists on
random q / k: token_span_lists, the training path's overhead) and token_path (the whole training path,
flex_packed_page_attention with kernel rowmask, on random q / k -- random queries choose near-disjoint pages: an
upper bound for the union).

    token_replay_bench.py --lists <dir>/lists_64k_doc0.pt [...] [--tiles 8 16] [--out json]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lists", nargs="+", required=True)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--tiles", type=int, nargs="+", default=[8, 16])
    ap.add_argument("--blocks", type=int, nargs="*", default=[4, 8, 16])
    ap.add_argument("--layers", type=int, default=0, help="layers per file (0: all)")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from megatron.core.transformer.experimental_attention_variant import page_sparse as ps
    from megatron.core.transformer.experimental_attention_variant import rowmask_attention as ra

    dev, dt = "cuda", torch.bfloat16
    scale = a.dim**-0.5

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

    def allowed_from_lists(idx, bits, cnt, P):
        """64-query tiles' lists [G, nq, NB] -> the page mask [G, nq * 64, P]."""
        G, nq, NB = idx.shape
        valid = torch.arange(NB, device=dev)[None, None] < cnt[..., None]
        ii = torch.where(valid, idx.long(), P)  # padding slots -> a dropped column
        rows = ((bits[..., None] >> torch.arange(64, device=dev)) & 1).bool()  # [G, nq, NB, 64]
        out = torch.zeros(G, nq, 64, P + 1, dtype=torch.bool, device=dev)
        out.scatter_(3, ii[:, :, None, :].expand(G, nq, 64, NB), rows.permute(0, 1, 3, 2))
        return out[..., :P].reshape(G, nq * 64, P)

    res = {"gpu": torch.cuda.get_device_name(), "files": {}, "per_length": {}}
    done_lengths = set()
    for path in a.lists:
        layers = torch.load(path)
        names = list(layers)[: a.layers or None]
        frow = {}
        for name in names:
            idx, bits, cnt = (t.to(dev) for t in layers[name])
            G, nq, _ = idx.shape
            L = nq * 64
            P = L // 64
            torch.manual_seed(0)
            q = torch.randn(a.heads, L, a.dim, device=dev, dtype=dt, requires_grad=True)
            k = torch.randn(G, L, a.dim, device=dev, dtype=dt, requires_grad=True)
            v = torch.randn(G, L, a.dim, device=dev, dtype=dt, requires_grad=True)
            pos = torch.arange(L, device=dev, dtype=torch.int32)
            row = {"L": L, "u64": float(cnt.float().mean())}
            row["rowmask64"] = timed(lambda: ra.rowmask_block_attention(q, k, v, idx, bits, cnt, pos, pos, scale, True)
                                     .sum().backward())
            allowed = allowed_from_lists(idx, bits, cnt, P)
            for T in a.tiles:
                lt = ra.lists_from_allowed(allowed, tq=T)
                row[f"u{T}"] = float(lt[2].float().mean())
                try:
                    row[f"hp{T}"] = timed(lambda: ra.rowmask_head_packed_attention(q, k, v, *lt, pos, pos, scale, T)
                                          .sum().backward())
                except Exception as e:  # noqa: BLE001  (shared memory on small GPUs)
                    row[f"hp{T}"] = f"{type(e).__name__}"
                del lt
            if a.blocks:  # one selection per B queries: the anchor's real pages, the query's own window / sinks
                tqb = ps.token_tile(a.heads // G, dev)
                p = torch.arange(P, device=dev)
                cur = torch.arange(L, device=dev) // 64
                fixed = (p[None, :] < 1) | ((p[None, :] >= (cur - 8)[:, None]) & (p[None, :] <= cur[:, None]))
                for B in a.blocks:
                    anc = (torch.arange(L, device=dev) // B) * B
                    ab = (allowed[:, anc] | fixed[None]) & (p[None, None, :] <= cur[None, :, None])
                    lt = ra.lists_from_allowed(ab, tq=tqb)
                    del ab
                    row[f"ub{B}"] = float(lt[2].float().mean())
                    row[f"b{B}"] = timed(lambda: ra.rowmask_head_packed_attention(q, k, v, *lt, pos, pos, scale, tqb)
                                         .sum().backward())
                    del lt
            del allowed
            frow[name] = row
            print(f"{path} {name}: " + "  ".join(f"{x} {y:.1f}" if isinstance(y, float) else f"{x} {y}"
                                                 for x, y in row.items()), flush=True)
            if L not in done_lengths:  # the reference variants and the training path, once per length
                done_lengths.add(L)
                ref = {}
                qq, kk, vv = (t.detach()[None].requires_grad_() for t in (q, k, v))
                ref["dense"] = timed(lambda: F.scaled_dot_product_attention(qq, kk, vv, is_causal=True,
                                                                            enable_gqa=True).sum().backward())
                base = dict(selector="page_maxpool", page=64, topk=32, window=512, sink_pages=1, per_group=True)
                blk = ps.PageSparseConfig(**base, select_block=64)
                tok40 = ps.PageSparseConfig(**base, select_block=1, block_candidates=40, candidate_block=64)
                pure = ps.PageSparseConfig(**base, select_block=1)
                variants = [("block", blk, "flex"), ("block_rowmask", blk, "rowmask"), ("tok40", tok40, "rowmask"),
                            ("token_path", pure, "rowmask")]
                for B in a.blocks:  # the whole B-block training path (anchor scoring + lists + kernel), random q / k
                    variants.append((f"b{B}_path", ps.PageSparseConfig(**base, select_block=B), "rowmask"))
                for nm, cfg, kern in variants:
                    try:
                        ref[nm] = timed(lambda: ps.flex_packed_page_attention([(qq, kk, vv, 0, None)], cfg, scale, 64,
                                                                              kernel=kern)[0].sum().backward())
                    except torch.OutOfMemoryError:
                        ref[nm] = None
                tq = ps.token_tile(a.heads // G, dev)
                with torch.no_grad():
                    ref["select"] = timed(lambda: ps.token_span_lists(qq, kk, pure, 0, scale, None, tq))
                res["per_length"][str(L)] = ref
                print(f"L={L}: " + "  ".join(f"{x} {y:.1f}" if y is not None else f"{x} OOM"
                                             for x, y in ref.items()), flush=True)
            del q, k, v
            torch.cuda.empty_cache()
        res["files"][path] = frow
        if a.out:
            json.dump(res, open(a.out, "w"), indent=1)
    # summary per length: the layers' mean kernel time per variant
    by_len = {}
    for frow in res["files"].values():
        for row in frow.values():
            by_len.setdefault(row["L"], []).append(row)
    for L, rows in sorted(by_len.items()):
        keys = [x for x in rows[0] if x != "L" and all(isinstance(r.get(x), float) for r in rows)]
        mean = {x: statistics.mean(r[x] for r in rows) for x in keys}
        res.setdefault("summary", {})[str(L)] = {"layers": len(rows), **mean, **res["per_length"].get(str(L), {})}
        print(f"[summary] L={L} ({len(rows)} layers): " + "  ".join(f"{x} {y:.1f}" for x, y in mean.items()),
              flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
