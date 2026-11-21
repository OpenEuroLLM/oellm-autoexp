#!/usr/bin/env python3
"""Attention cost per training sequence on the real document layout (doclen_sample.py's packed phase-2 sequences):
forward + backward of one attention layer at v4b's shapes for every variant on N sampled sequences, then the per-step
estimate -- the mean, and the expected slowest of R data-parallel ranks (each rank one sequence: the step waits for
it). Variants:
  dense_masked  -- varlen flash attention over the documents (the production setting, document masking)
  dense_full    -- causal over the whole sequence (no document masking)
  block / block_rowmask -- page_maxpool_pg, one selection per 64 queries (flex / rowmask kernel), per document
  tok40         -- per-token selection inside 40 block candidates (rowmask), per document
  token         -- pure per-token selection (each query its own top 32, head-packed rowmask tiles), per document, on
                   the random q / k (random queries pick near-disjoint pages: an upper bound)
  tok_b4 / tok_b8 / tok_b16 -- one selection per 4 / 8 / 16 queries on the head-packed per-token path
  token_real    -- the same kernel with REAL page masks (--masks: token_union_probe.py's lists of one long held-out
                   document; a document of n tokens takes the first n queries' masks -- a query's selection depends
                   only on the keys before it) plus the selection's own cost (timed on the random q / k)
  *_full        -- the sparse variants without document masking (one span)

    attn_packed_bench.py --seqs seqs_64k.json [--n 24] [--ranks 384 512] [--masks lists_256k_doc0.pt] [--out json]
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
    ap.add_argument("--masks", default=None, help="token_union_probe lists (.pt) for token_real")
    ap.add_argument("--mask-layer", type=int, default=0, help="which attention layer's masks")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from megatron.core.transformer.experimental_attention_variant import page_sparse as ps

    d = json.load(open(a.seqs))
    L, seqs = d["seq"], d["seqs"]
    rng = random.Random(0)
    # a third random, a third from the longest-document 10 %, a third from the most-documents 10 % (both kinds of
    # straggler), mixed back at their frequencies for the estimate
    n3 = a.n // 3
    by_long = sorted(range(len(seqs)), key=lambda i: max(seqs[i]))[int(0.9 * len(seqs)):]
    by_docs = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))[int(0.9 * len(seqs)):]
    pick = rng.sample(range(len(seqs)), a.n - 2 * n3) + rng.sample(by_long, n3) + rng.sample(by_docs, n3)
    kind = ["random"] * (a.n - 2 * n3) + ["long"] * n3 + ["docs"] * n3
    dev, dt = "cuda", torch.bfloat16
    scale = a.dim**-0.5
    base = dict(selector="page_maxpool", page=64, topk=32, window=512, sink_pages=1, per_group=True)
    cfgs = {"block": (ps.PageSparseConfig(**base, select_block=64), "flex"),
            "block_rowmask": (ps.PageSparseConfig(**base, select_block=64), "rowmask"),
            "tok40": (ps.PageSparseConfig(**base, select_block=1, block_candidates=40, candidate_block=64),
                      "rowmask"),
            "token": (ps.PageSparseConfig(**base, select_block=1), "rowmask"),
            # one selection per B queries on the head-packed per-token path (anchor scoring; per tile at most
            # 16 / B + 1 selections: random q / k bound the union)
            "tok_b4": (ps.PageSparseConfig(**base, select_block=4), "rowmask"),
            "tok_b8": (ps.PageSparseConfig(**base, select_block=8), "rowmask"),
            "tok_b16": (ps.PageSparseConfig(**base, select_block=16), "rowmask")}
    real = None
    if a.masks:  # the real page masks [G, Lm, Pm] of one long document
        from megatron.core.transformer.experimental_attention_variant import rowmask_attention as ra

        layers = torch.load(a.masks)
        idx_m, bits_m, cnt_m = (t.to(dev) for t in layers[list(layers)[a.mask_layer]])
        G_, nq_, NB_ = idx_m.shape
        Pm = nq_
        valid = torch.arange(NB_, device=dev)[None, None] < cnt_m[..., None]
        ii = torch.where(valid, idx_m.long(), Pm)
        rows_ = ((bits_m[..., None] >> torch.arange(64, device=dev)) & 1).bool()
        real = torch.zeros(G_, nq_, 64, Pm + 1, dtype=torch.bool, device=dev)
        real.scatter_(3, ii[:, :, None, :].expand(G_, nq_, 64, NB_), rows_.permute(0, 1, 3, 2))
        real = real[..., :Pm].reshape(G_, nq_ * 64, Pm)
        del idx_m, bits_m, cnt_m, rows_, ii, valid
        tq_real = ps.token_tile(a.heads // a.kv, dev)
        span_all = base["sink_pages"] + base["window"] // base["page"] + 1 + base["topk"]  # pages selected anyway

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
        row = {"seq": si, "kind": kind[len(rows)], "docs": len(segs), "longest": max(segs)}
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
            elif var == "token_real":
                if real is None or max(segs) > real.shape[1]:
                    continue
                cfg, kern = cfgs["token"]
                lens = list(segs)
                long_ = [i for i, n in enumerate(lens) if -(-n // 64) > span_all]
                short_ = [i for i in range(len(lens)) if i not in long_]
                qs_, ks_, vs_ = q.split(lens, 2), k.split(lens, 2), v.split(lens, 2)
                sels = [ra.lists_from_allowed(real[:, :lens[i], :-(-lens[i] // 64)].contiguous(), tq=tq_real)
                        for i in long_]

                def fn(cfg=cfg, long_=long_, short_=short_, qs_=qs_, ks_=ks_, vs_=vs_, sels=sels):
                    outs = []
                    if short_:  # documents that select every page: the dense fast path, as in training
                        outs += ps.flex_packed_page_attention([(qs_[i], ks_[i], vs_[i], 0, None) for i in short_],
                                                              cfg, scale, 64, kernel="rowmask")
                    if long_:
                        outs += ps.flex_packed_page_attention([(qs_[i], ks_[i], vs_[i], 0, None) for i in long_],
                                                              cfg, scale, 64, selections=sels, kernel="rowmask")
                    sum(o.sum() for o in outs).backward()

                def sel_fn(long_=long_, qs_=qs_, ks_=ks_):
                    with torch.no_grad():
                        for i in long_:
                            ps.token_span_lists(qs_[i], ks_[i], cfgs["token"][0], 0, scale, None, tq_real)

                try:
                    row["token_real_select"] = timed(sel_fn) if long_ else 0.0
                except torch.OutOfMemoryError:
                    row["token_real_select"] = None
            else:
                cfg, kern = cfgs[var.removesuffix("_full")]
                def fn(cfg=cfg, kern=kern, full=full):
                    # documents as ONE split, as PageSparseCoreAttention does (slices backpropagate a full-size zero
                    # tensor each)
                    if full:
                        spans = [(q, k, v, 0, None)]
                    else:
                        spans = [(a_, b_, c_, 0, None) for a_, b_, c_ in zip(q.split(segs, 2), k.split(segs, 2),
                                                                             v.split(segs, 2))]
                    sum(o.sum() for o in ps.flex_packed_page_attention(spans, cfg, scale, 64, kernel=kern)).backward()
            try:
                row[var] = timed(fn)
            except torch.OutOfMemoryError:
                row[var] = None
        rows.append(row)
        print(json.dumps(row), flush=True)
    # estimate: random third -> the mean; a rank draws a long-tail / many-documents layout with 10 % each
    for x in rows:  # the per-token training path's cost: its kernel plus its selection
        if x.get("token_real") is not None and x.get("token_real_select") is not None:
            x["token_real_total"] = x["token_real"] + x["token_real_select"]
    summary = {}
    for var in a.variants + (["token_real_total"] if "token_real" in a.variants else []):
        grp = {k: [x[var] for x in rows if x["kind"] == k and x.get(var) is not None] for k in ("random", "long", "docs")}
        if not grp["random"]:
            continue
        est = {}
        for R in a.ranks:
            def draw():
                u = rng.random()
                g = grp["long"] if u < 0.1 and grp["long"] else grp["docs"] if u < 0.2 and grp["docs"] else grp["random"]
                return rng.choice(g)
            est[f"slowest_of_{R}"] = statistics.mean(max(draw() for _ in range(R)) for _ in range(200))
        summary[var] = {"mean": statistics.mean(grp["random"]),
                        "long_tail_mean": statistics.mean(grp["long"]) if grp["long"] else None,
                        "docs_tail_mean": statistics.mean(grp["docs"]) if grp["docs"] else None, **est}
    print(json.dumps(summary, indent=1), flush=True)
    if a.out:
        json.dump({"args": vars(a), "gpu": torch.cuda.get_device_name(), "rows": rows, "summary": summary},
                  open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
