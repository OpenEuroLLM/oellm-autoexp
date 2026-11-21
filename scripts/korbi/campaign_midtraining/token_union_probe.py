#!/usr/bin/env python3
"""Pure per-token page selection on REAL activations: how many distinct pages does a tile of T consecutive queries
need? A block-sparse kernel that loops over a tile's page union (rowmask_attention.py: one bit word per (tile, page))
costs about union x tile, so the union size decides whether per-token selection can be cheaper than dense.

An HF export (long_attention_probe.py's loader and chunked prefill) runs dense attention over long held-out documents;
every attention layer (incl. kvshare / ckda-branch attention) also selects per token and per KV group with
page_maxpool (page 64, top 32, window 512, 1 sink -- the pmp_tok arm), exactly as page_sparse.select_pages does, and
records per tile of T in --tiles queries the union of pages its queries attend (sinks + window + selection), by
position bin. With --lists the 64-query tiles' (idx, bits, cnt) lists of every layer (rowmask kernel inputs) are saved
per document length: the kernel benchmark replays them (token_sparse_bench.py --replay).

    token_union_probe.py --model <export> --docs <prefix> --lengths 65536,262144 --out <dir> [--lists]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import long_attention_probe as lap  # noqa: E402

ps = lap.ps
BINS = [0, 8192, 32768, 65536, 131072, 262144, 1 << 30]


def bin_name(lo, hi):
    return f"{lo // 1024}k-{hi // 1024}k" if hi < 1 << 30 else f"{lo // 1024}k+"


class Capture(torch.nn.Module):
    """Stands in for an attention mixer's ``sparse`` module: dense attention (as the mixer's own dense path) plus the
    per-token selection statistics of the call's queries over all keys so far."""

    uses_hidden = False  # the selector needs no hidden input (the model then keeps its fused paths)

    def __init__(self, name, probe, past_mask):
        super().__init__()
        self.name, self.probe, self.past_mask = name, probe, past_mask

    def forward(self, q, k, v, x, past, state=None):
        nh, nkv, d = q.shape[1], k.shape[1], q.shape[-1]
        if past == 0:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=d**-0.5, enable_gqa=nh != nkv)
        else:
            rep = nh // nkv
            kk = k.repeat_interleave(rep, dim=1) if rep > 1 else k
            vv = v.repeat_interleave(rep, dim=1) if rep > 1 else v
            o = F.scaled_dot_product_attention(q, kk, vv, attn_mask=self.past_mask(q.shape[2], k.shape[2]),
                                               scale=d**-0.5)
        if self.probe.on:
            self.probe.record(self.name, q, k, past)
        return o, None


class UnionProbe:
    def __init__(self, tiles, chunk, want_lists):
        self.cfg = ps.PageSparseConfig(selector="page_maxpool", page=64, topk=32, window=512, sink_pages=1,
                                       select_block=1)
        self.tiles, self.chunk, self.want_lists = tiles, chunk, want_lists
        self.on = False
        self.hist = defaultdict(lambda: defaultdict(list))  # (layer, T, bin) -> per-tile union sizes
        self.lists = defaultdict(list)  # layer -> [(idx, bits, cnt) per group, per chunk]
        self.seconds = 0.0

    @torch.no_grad()
    def record(self, name, q, k, past):
        t0 = time.time()
        nh, s, nkv = q.shape[1], q.shape[2], k.shape[1]
        rep = nh // nkv
        scale = q.shape[-1] ** -0.5
        if past % 64 or s % 64:
            raise ValueError("chunks must be 64-aligned")
        per_chunk = []
        for c0 in range(0, s, self.chunk):
            c1 = min(s, c0 + self.chunk)
            groups = []
            for gi in range(nkv):
                allowed = ps.select_pages(q[:, gi * rep:(gi + 1) * rep, c0:c1], k[:, gi:gi + 1], self.cfg,
                                          past + c0, scale)[0]  # [c, P]
                pos0 = past + c0
                for T in self.tiles:
                    n = (c1 - c0) // T
                    u = allowed[: n * T].view(n, T, -1).any(1).sum(-1)  # [n] union per tile
                    tpos = pos0 + torch.arange(n, device=u.device) * T
                    for lo, hi in zip(BINS[:-1], BINS[1:]):
                        m = (tpos >= lo) & (tpos < hi)
                        if bool(m.any()):
                            self.hist[(name, T, bin_name(lo, hi))][gi].append(u[m].cpu())
                groups.append(allowed)
            if self.want_lists:
                per_chunk.append(lists_from_allowed(torch.stack(groups)))
        if self.want_lists:
            self.lists[name].extend(per_chunk)
        torch.cuda.synchronize(q.device)
        self.seconds += time.time() - t0

    def take_lists(self):
        """layer -> (idx [G, nq, NB], bits, cnt) over the whole document (the chunks' lists concatenated)."""
        out = {}
        for name, parts in self.lists.items():
            NB = max(p[0].shape[-1] for p in parts)
            idx = torch.cat([F.pad(p[0], (0, NB - p[0].shape[-1])) for p in parts], 1).cpu()
            bits = torch.cat([F.pad(p[1], (0, NB - p[1].shape[-1])) for p in parts], 1).cpu()
            cnt = torch.cat([p[2] for p in parts], 1).cpu()
            out[name] = (idx, bits, cnt)
        self.lists.clear()
        return out

    def summary(self):
        res = {}
        for (name, T, b), groups in sorted(self.hist.items()):
            u = torch.cat([torch.cat(v) for v in groups.values()]).float()
            res.setdefault(name, {}).setdefault(str(T), {})[b] = {
                "tiles": int(u.numel()), "mean": float(u.mean()), "p50": float(u.quantile(0.5)),
                "p90": float(u.quantile(0.9)), "max": float(u.max())}
        return res


def lists_from_allowed(allowed):
    """rowmask_attention.lists_from_allowed, the module loaded by path like page_sparse.py (no megatron import)."""
    if "rowmask" not in _MOD:
        import importlib.util

        path = lap.REPO / "submodules/Megatron-LM/megatron/core/transformer/experimental_attention_variant/rowmask_attention.py"
        spec = importlib.util.spec_from_file_location("rowmask_attention_by_path", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _MOD["rowmask"] = mod
    return _MOD["rowmask"].lists_from_allowed(allowed)


_MOD = {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--docs", default="")
    ap.add_argument("--lengths", default="65536,262144")
    ap.add_argument("--n-docs", default="2,1", help="documents per length")
    ap.add_argument("--chunk", type=int, default=16384, help="prefill chunk")
    ap.add_argument("--select-chunk", type=int, default=2048, help="queries per selection block")
    ap.add_argument("--tiles", default="8,16,32,64")
    ap.add_argument("--bos", type=int, default=1)
    ap.add_argument("--lists", action="store_true", help="save the 64-query tiles' lists per layer and length")
    ap.add_argument("--replace-sparse", action="store_true",
                    help="a sparse-trained export: attend densely and record the per-token selection anyway")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    lengths = [int(x) for x in a.lengths.split(",")]
    ndocs = [int(x) for x in a.n_docs.split(",")]
    t0 = time.time()
    model = lap.rp.load_model(a.model, a.tiny)
    probe = UnionProbe([int(x) for x in a.tiles.split(",")], a.select_chunk, a.lists)
    names = []
    for mname, m in model.named_modules():
        if type(m).__name__ in ("OellmXSAAttention", "OellmKVShareAttention") and hasattr(m, "sparse"):
            if m.sparse is not None and not a.replace_sparse:
                raise SystemExit(f"{a.model}: {mname} already sparse (--replace-sparse: attend densely anyway)")
            m.sparse = Capture(mname, probe, sys.modules[type(m).__module__]._past_mask)
            names.append(mname)
    print(f"[union] model loaded in {time.time() - t0:.0f}s; attention modules: {names}", flush=True)
    meta = {"model": a.model, "layers": names, "cfg": vars(probe.cfg), "docs": {}}
    for L, n in zip(lengths, ndocs):
        docs = lap.long_docs(a.docs, L, n, a.bos)
        probe.on = True
        t1 = time.time()
        for di, doc in enumerate(docs):
            lap.prefill(model, doc, a.chunk)
            if a.lists:
                torch.save(probe.take_lists(), a.out / f"lists_{L // 1024}k_doc{di}.pt")
        probe.on = False
        meta["docs"][str(L)] = len(docs)
        print(f"[union] L={L}: {len(docs)} docs in {time.time() - t1:.0f}s (selection {probe.seconds:.0f}s)",
              flush=True)
        summ = probe.summary()
        (a.out / "union.json").write_text(json.dumps({"meta": meta, "union": summ}, indent=1, default=str))
    for name, per_t in probe.summary().items():
        for T, bins in per_t.items():
            print(f"[union] {name} T={T}: " + "  ".join(f"{b} mean {v['mean']:.1f} p90 {v['p90']:.0f} max "
                                                       f"{v['max']:.0f}" for b, v in bins.items()), flush=True)
    print(f"[union] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
