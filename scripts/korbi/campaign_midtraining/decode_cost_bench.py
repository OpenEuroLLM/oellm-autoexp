#!/usr/bin/env python3
"""Decode cost of one attention layer at 128k-1M context on one GPU (batch 1, one new token per step): dense
attention against the sparse variants with decode-time caches, as a serving implementation would keep them (page
means / compressed entries updated incrementally, not recomputed per step):

  dense            the full KV on the GPU, SDPA over all keys
  page_mean        KV on the GPU + cached page means; per KV group the top-k pages by q . mean key, + sink + window
  page_mean_host   page means + the window on the GPU, the full KV in pinned host memory: the selected pages are
                   gathered on the host and copied over PCIe every step (the campaign's 1M plan, §3.2)
  page_mean_hier   page_mean scoring groups of G pages first (top g groups), then the pages inside them
  csa2             only the compressed entries (m keys -> 1) + the raw window are kept; a lightning indexer over the
                   entries (index heads x dim) picks the top entries; csa2_small: a DeepSeek-sized indexer (4 x 64)
  page_maxpool     InfLLM-v2 scoring: cached means of half-page windows at quarter-page stride, a page scored by the
                   max over the 5 windows overlapping it; page_maxpool_host: the same with the KV in host memory
  lsa              bucket centroids (8 buckets per closed 512-token block = one per 64 tokens on average), the KV kept
                   bucket-contiguous once a block closes, so selected buckets gather like pages; lsa_host: host KV

Reported per method and context: ms per step (median), GPU bytes the layer keeps resident, host -> GPU bytes per
step. Selection itself is timed with the attention; the indexer's / scoring's share is printed separately.

    decode_cost_bench.py [--lengths 131072 262144 524288 1048576] [--heads 16 --kv 2 --dim 256] [--out out.json]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
import torch.nn.functional as F


def timed(fn, steps=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(steps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def attend(q, k, v):
    """q [1, nh, 1, d], k / v [1, nkv, n, d] -> [1, nh, 1, d] (GQA inside SDPA)."""
    return F.scaled_dot_product_attention(q, k, v, enable_gqa=q.shape[1] != k.shape[1])


def page_tokens(top, page, window_start, L, dev):
    """Token indices of the selected pages [nkv, k] + the sink page + the window -> [nkv, n]."""
    nkv = top.shape[0]
    tok = (top[..., None] * page + torch.arange(page, device=dev)).reshape(nkv, -1)
    sink = torch.arange(page, device=dev).expand(nkv, -1)
    win = torch.arange(window_start, L, device=dev).expand(nkv, -1)
    return torch.cat([sink, tok, win], 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lengths", type=int, nargs="+", default=[131072, 262144, 524288, 1048576])
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--kv", type=int, default=2)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--page", type=int, default=64)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--hier-group", type=int, default=16)
    ap.add_argument("--hier-topk", type=int, default=8)
    ap.add_argument("--compress", type=int, default=4)
    ap.add_argument("--topk-entries", type=int, default=512)
    ap.add_argument("--index-heads", type=int, default=32)
    ap.add_argument("--index-dim", type=int, default=128)
    ap.add_argument("--methods", nargs="+",
                    default=["dense", "page_mean", "page_mean_host", "page_maxpool", "page_maxpool_host", "lsa",
                             "lsa_host", "csa2", "csa2_small"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev, dt = "cuda", torch.bfloat16
    nh, nkv, d, page = a.heads, a.kv, a.dim, a.page
    rep = nh // nkv
    g = torch.Generator(device=dev).manual_seed(0)
    res = []
    for L in a.lengths:
        P = L // page
        q = torch.randn(1, nh, 1, d, device=dev, dtype=dt, generator=g)
        qg = q[0, :, 0].view(nkv, rep, d).float().sum(1)  # [nkv, d]: the group's heads' summed query (per-group score)
        win0 = L - a.window
        cand_hi = win0 // page  # pages fully before the window
        st, sub = page // 4, page // 2
        J = 4 * P + 1

        def window_means(keys):  # [nkv, L, d] -> [nkv, J, d]: means of [j st, j st + sub)
            cs = F.pad(F.pad(keys.float(), (0, 0, 0, sub)).cumsum(1), (0, 0, 1, 0))
            starts = torch.arange(J, device=keys.device) * st
            return ((cs[:, starts + sub] - cs[:, starts]) / sub).to(dt)

        def scorer(kind, keys):
            """(select() -> top pages [nkv, k], aux tensors kept resident) for page_mean / page_maxpool / lsa."""
            if kind == "page_maxpool":
                wm = window_means(keys)

                def select():
                    sc = torch.einsum("gd,gjd->gj", qg, wm.float())
                    pooled = F.max_pool1d(sc[:, None], kernel_size=5, stride=4, padding=1)[:, 0, :P]
                    return pooled[:, 1:cand_hi].topk(a.topk, dim=-1).indices + 1
                return select, [wm]
            # page_mean: page means; lsa: bucket centroids (one per 64 tokens on average, bucket-contiguous KV)
            pm = keys.view(nkv, P, page, d).float().mean(2).to(dt)

            def select():
                sc = torch.einsum("gd,gpd->gp", qg, pm.float())[:, 1:cand_hi]
                return sc.topk(a.topk, dim=-1).indices + 1
            return select, [pm]

        for m in a.methods:
            torch.cuda.empty_cache()
            row = {"length": L, "method": m}
            try:
                if m in ("page_maxpool", "lsa"):
                    k = torch.randn(1, nkv, L, d, device=dev, dtype=dt, generator=g)
                    v = torch.randn_like(k)
                    select, aux = scorer(m, k[0])

                    def step():
                        idx = page_tokens(select(), page, win0, L, dev)
                        kk = torch.gather(k[0], 1, idx[..., None].expand(-1, -1, d))[None]
                        vv = torch.gather(v[0], 1, idx[..., None].expand(-1, -1, d))[None]
                        return attend(q, kk, vv)

                    row["ms"] = timed(step)
                    row["select_ms"] = timed(select)
                    row["gpu_mb"] = (2 * k.numel() + sum(t.numel() for t in aux)) * 2 / 2**20
                    row["h2d_mb"] = 0.0
                    del k, v, aux
                elif m in ("page_maxpool_host", "lsa_host"):
                    kh = torch.randn(nkv, P, page, d, dtype=dt).pin_memory()
                    vh = torch.randn(nkv, P, page, d, dtype=dt).pin_memory()
                    select, aux = scorer(m.replace("_host", ""), kh.view(nkv, L, d).to(dev))
                    torch.cuda.empty_cache()
                    kw = kh.view(nkv, L, d)[:, win0:].to(dev)
                    vw = vh.view(nkv, L, d)[:, win0:].to(dev)
                    ks = kh[:, :1].reshape(nkv, page, d).to(dev)
                    vs = vh[:, :1].reshape(nkv, page, d).to(dev)

                    kbuf = torch.empty(nkv, a.topk, page, d, dtype=dt).pin_memory()  # reused every step
                    vbuf = torch.empty_like(kbuf).pin_memory()

                    def step():
                        top = select().cpu()
                        for i in range(nkv):
                            torch.index_select(kh[i], 0, top[i], out=kbuf[i])
                            torch.index_select(vh[i], 0, top[i], out=vbuf[i])
                        kk = torch.cat([ks, kbuf.to(dev, non_blocking=True).reshape(nkv, -1, d), kw], 1)[None]
                        vv = torch.cat([vs, vbuf.to(dev, non_blocking=True).reshape(nkv, -1, d), vw], 1)[None]
                        return attend(q, kk, vv)

                    row["ms"] = timed(step, steps=30, warmup=5)
                    row["gpu_mb"] = (sum(t.numel() for t in aux) + 2 * kw.numel() + 2 * ks.numel()) * 2 / 2**20
                    row["h2d_mb"] = 2 * nkv * a.topk * page * d * 2 / 2**20
                    row["host_mb"] = 2 * kh.numel() * 2 / 2**20
                    del kh, vh, aux
                elif m in ("dense", "page_mean", "page_mean_hier"):
                    k = torch.randn(1, nkv, L, d, device=dev, dtype=dt, generator=g)
                    v = torch.randn_like(k)
                    resident = 2 * k.numel() * 2
                    if m == "dense":
                        row["ms"] = timed(lambda: attend(q, k, v))
                    else:
                        pm = k[0].view(nkv, P, page, d).float().mean(2).to(dt)  # cached page means [nkv, P, d]
                        resident += pm.numel() * 2
                        if m == "page_mean":
                            def select():
                                sc = torch.einsum("gd,gpd->gp", qg, pm.float())[:, 1:cand_hi]
                                return sc.topk(a.topk, dim=-1).indices + 1
                        else:
                            G = a.hier_group
                            gm = pm[:, : (P // G) * G].view(nkv, P // G, G, d).float().mean(2).to(dt)
                            resident += gm.numel() * 2

                            def select():
                                gs = torch.einsum("gd,gnd->gn", qg, gm.float())[:, : cand_hi // G]
                                gtop = gs.topk(a.hier_topk, dim=-1).indices  # [nkv, hk]
                                pages = (gtop[..., None] * G + torch.arange(G, device=dev)).reshape(nkv, -1)
                                pages = pages.clamp(1, cand_hi - 1)
                                ps_ = torch.einsum("gd,gnd->gn", qg, torch.gather(pm, 1, pages[..., None].expand(-1, -1, d)).float())
                                return torch.gather(pages, 1, ps_.topk(a.topk, dim=-1).indices)

                        def step():
                            idx = page_tokens(select(), page, win0, L, dev)  # [nkv, n]
                            kk = torch.gather(k[0], 1, idx[..., None].expand(-1, -1, d))[None]
                            vv = torch.gather(v[0], 1, idx[..., None].expand(-1, -1, d))[None]
                            return attend(q, kk, vv)

                        row["ms"] = timed(step)
                        row["select_ms"] = timed(select)
                    row["gpu_mb"] = resident / 2**20
                    row["h2d_mb"] = 0.0
                    del k, v
                elif m == "page_mean_host":
                    kh = torch.randn(nkv, P, page, d, dtype=dt).pin_memory()  # host KV, page-major
                    vh = torch.randn(nkv, P, page, d, dtype=dt).pin_memory()
                    pm = kh.float().mean(2).to(dt).to(dev)
                    kw = kh.view(nkv, L, d)[:, win0:].to(dev)
                    vw = vh.view(nkv, L, d)[:, win0:].to(dev)
                    ks = kh[:, :1].reshape(nkv, page, d).to(dev)  # the sink page stays on the GPU too
                    vs = vh[:, :1].reshape(nkv, page, d).to(dev)

                    kbuf = torch.empty(nkv, a.topk, page, d, dtype=dt).pin_memory()  # reused every step
                    vbuf = torch.empty_like(kbuf).pin_memory()

                    def step():
                        sc = torch.einsum("gd,gpd->gp", qg, pm.float())[:, 1:cand_hi]
                        top = (sc.topk(a.topk, dim=-1).indices + 1).cpu()  # [nkv, k] (a sync: the host gathers)
                        for i in range(nkv):
                            torch.index_select(kh[i], 0, top[i], out=kbuf[i])
                            torch.index_select(vh[i], 0, top[i], out=vbuf[i])
                        kk = torch.cat([ks, kbuf.to(dev, non_blocking=True).reshape(nkv, -1, d), kw], 1)[None]
                        vv = torch.cat([vs, vbuf.to(dev, non_blocking=True).reshape(nkv, -1, d), vw], 1)[None]
                        return attend(q, kk, vv)

                    row["ms"] = timed(step, steps=30, warmup=5)
                    row["gpu_mb"] = (pm.numel() + 2 * kw.numel() + 2 * ks.numel()) * 2 / 2**20
                    row["h2d_mb"] = 2 * nkv * a.topk * page * d * 2 / 2**20
                    row["host_mb"] = 2 * kh.numel() * 2 / 2**20
                    del kh, vh
                elif m in ("csa2", "csa2_small"):
                    ih, idim = (a.index_heads, a.index_dim) if m == "csa2" else (4, 64)
                    Lc = L // a.compress
                    kc = torch.randn(1, nkv, Lc, d, device=dev, dtype=dt, generator=g)  # compressed entries only
                    vc = torch.randn_like(kc)
                    kw = torch.randn(1, nkv, a.window, d, device=dev, dtype=dt, generator=g)
                    vw = torch.randn_like(kw)
                    ki = torch.randn(Lc, idim, device=dev, dtype=dt, generator=g)  # index keys
                    qi = torch.randn(ih, idim, device=dev, dtype=dt, generator=g)
                    wi = torch.randn(ih, device=dev, dtype=torch.float32, generator=g)

                    def select():
                        s = torch.relu(qi @ ki.T).float()  # [H, Lc]
                        return (wi[:, None] * s).sum(0)[: Lc - a.window // a.compress].topk(a.topk_entries).indices

                    def step():
                        top = select()
                        kk = torch.cat([kc[:, :, top], kw], 2)
                        vv = torch.cat([vc[:, :, top], vw], 2)
                        return attend(q, kk, vv)

                    row["ms"] = timed(step)
                    row["select_ms"] = timed(select)
                    row["gpu_mb"] = (2 * kc.numel() + 2 * kw.numel() + ki.numel()) * 2 / 2**20
                    row["h2d_mb"] = 0.0
                    del kc, vc, ki
            except torch.OutOfMemoryError:
                row["oom"] = True
            res.append(row)
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"args": vars(a), "gpu": torch.cuda.get_device_name(), "rows": res}, fh, indent=2)


if __name__ == "__main__":
    main()
