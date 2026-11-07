"""Step 4 of the sparse-attention variant plan (docs/campaign_3b_hybrid_midtraining.md): is each variant's current
implementation affordable for the s3 training comparison? One attention layer of v4 (16 query heads, 2 KV heads,
head dim 256 since the hd256 restart), one document of L tokens, bf16: forward and forward + backward time and
peak memory of Megatron's page_sparse / sparse_variants code (the exact references; flex for the page selectors)
against dense attention (SDPA, flash / cuDNN).

    python sparse_attention_speed.py --out <dir> --lengths 16384,65536
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
VAR = REPO / "submodules/Megatron-LM/megatron/core/transformer/experimental_attention_variant"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, VAR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ps = _load("page_sparse")
sv = _load("sparse_variants")

# matched budget: ~2.7k keys per query (page 64, window 512, 1 sink page, 32 selected pages / buckets / entries)
BASE = dict(page=64, topk=32, window=512, sink_pages=1)
ARMS = {
    "dense": None,
    "window": ("flex", dict(BASE, selector="window")),
    "page_mean_flex": ("flex", dict(BASE, selector="page_mean")),
    "quest_flex": ("flex", dict(BASE, selector="quest")),
    "page_mean_hier_flex": ("flex", dict(BASE, selector="page_mean", hier_group=16, hier_topk=8)),
    # one selection per block of 64 queries (the block's first query selects): the page union is the budget
    "page_mean_b64_flex": ("flex", dict(BASE, selector="page_mean", select_block=64)),
    "quest_b64_flex": ("flex", dict(BASE, selector="quest", select_block=64)),
    "page_mean_hier_b64_flex": ("flex", dict(BASE, selector="page_mean", hier_group=16, hier_topk=8,
                                             select_block=64)),
    "page_mean_gathered": ("gathered", dict(BASE, selector="page_mean")),
    "nsa": ("nsa", dict(block=32, stride=16, page=64, topk=32, sink_pages=1, select_window=512, window=512,
                        gate_init=(0.0, 0.0, 0.0))),
    "nsa_flex": ("nsa_flex", dict(block=32, stride=16, page=64, topk=32, sink_pages=1, select_window=512,
                                  window=512, gate_init=(0.0, 0.0, 0.0))),
    "nsa_b64_flex": ("nsa_flex", dict(block=32, stride=16, page=64, topk=32, sink_pages=1, select_window=512,
                                      window=512, gate_init=(0.0, 0.0, 0.0), select_block=64)),
    "csa2": ("csa2", dict(compress=4, topk_entries=512, window=512, page=64, sink_pages=1)),
    "csa2_b64_flex": ("csa2_flex", dict(compress=4, topk_entries=512, window=512, page=64, sink_pages=1,
                                        select_block=64)),
    "csa2_b64_kv16_flex": ("csa2_flex", dict(compress=4, topk_entries=512, window=512, page=64, sink_pages=1,
                                             select_block=64, kv_block=16)),
    "lsa_b64_kv16_flex": ("lsa_flex", dict(pool=4, block=512, buckets=32, topk=32, window=512, page=64,
                                           sink_pages=1, select_block=64, kv_block=16)),
    "lsa_b64_flex": ("lsa_flex", dict(pool=4, block=512, buckets=32, topk=32, window=512, page=64, sink_pages=1,
                                      select_block=64)),
    "lsa": ("lsa", dict(pool=4, block=512, buckets=32, topk=32, window=512, page=64, sink_pages=1)),
}


def build(name, hidden, hq, g, d, dev):
    spec = ARMS[name]
    if spec is None:
        return lambda q, k, v, x: ps.dense_attention(q, k, v, 0, d**-0.5), []
    kind, kw = spec
    if kind in ("flex", "gathered"):
        cfg = ps.PageSparseConfig(**kw)
        if kind == "flex":
            return lambda q, k, v, x: ps.flex_page_sparse_attention(q, k, v, cfg, 0, d**-0.5, None, 64), []
        return lambda q, k, v, x: ps.page_sparse_attention(q, k, v, cfg, 0, d**-0.5), []
    if kind in ("nsa", "nsa_flex"):
        m = sv.NSA(hidden, hq, d, sv.NSAConfig(**kw)).to(dev, torch.bfloat16)
        m.flex = kind == "nsa_flex"
        return lambda q, k, v, x: m(q, k, v, x, 0, d**-0.5), list(m.parameters())
    if kind in ("csa2", "csa2_flex"):
        m = sv.CSA2(hidden, g, d, sv.CSA2Config(**kw)).to(dev, torch.bfloat16)
        m.flex = kind == "csa2_flex"

        def f(q, k, v, x):
            z = m.token_features(x)
            return m(q, k, v, x, z, 0, d**-0.5) + 0 * m.indexer_loss(q, k, x, z, 0, d**-0.5)
        return f, list(m.parameters())
    m = sv.LSA(sv.LSAConfig(**kw))
    m.flex = kind == "lsa_flex"
    return lambda q, k, v, x: m(q, k, v, 0, d**-0.5), []


@torch.no_grad()
def union_share(L, hq, g, d, cfg, dev, q_block=64):
    """Flex's attended (query block, page) pairs over the causal dense ones, on the bench's random inputs."""
    gen = torch.Generator(device=dev).manual_seed(0)
    q = torch.randn(1, hq, L, d, device=dev, dtype=torch.bfloat16, generator=gen)
    k = torch.randn(1, g, L, d, device=dev, dtype=torch.bfloat16, generator=gen)
    allowed = ps.select_pages(q, k, cfg, 0, d**-0.5)[0]
    nq = -(-L // q_block)
    blk = torch.nn.functional.pad(allowed, (0, 0, 0, nq * q_block - L)).view(nq, q_block, -1).any(1)
    causal = sum(min(-(-((i + 1) * q_block) // cfg.page), blk.shape[1]) for i in range(nq))
    return float(blk.sum()) / causal


def timed(fn, iters):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--lengths", default="16384,65536")
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--groups", type=int, default=2)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--hidden", type=int, default=2048)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    hq, g, d, h = args.heads, args.groups, args.dim, args.hidden
    rows = []
    for L in [int(x) for x in args.lengths.split(",")]:
        for name in args.arms.split(","):
            row = {"arm": name, "L": L}
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                gen = torch.Generator(device=dev).manual_seed(0)
                q = torch.randn(1, hq, L, d, device=dev, dtype=torch.bfloat16, generator=gen).requires_grad_()
                k = torch.randn(1, g, L, d, device=dev, dtype=torch.bfloat16, generator=gen).requires_grad_()
                v = torch.randn(1, g, L, d, device=dev, dtype=torch.bfloat16, generator=gen).requires_grad_()
                x = torch.randn(1, L, h, device=dev, dtype=torch.bfloat16, generator=gen).requires_grad_()
                fn, params = build(name, h, hq, g, d, dev)
                base = torch.cuda.memory_allocated()

                def fwd():
                    with torch.no_grad():
                        fn(q, k, v, x)

                def fwd_bwd():
                    fn(q, k, v, x).float().square().mean().backward()
                    for t in (q, k, v, x, *params):
                        t.grad = None

                fwd_bwd()  # warm-up (flex compile, autotune), both paths
                fwd()
                row["fwd_ms"] = timed(fwd, args.iters)
                torch.cuda.reset_peak_memory_stats()
                row["fwd_bwd_ms"] = timed(fwd_bwd, args.iters)
                row["peak_gb"] = (torch.cuda.max_memory_allocated() - base) / 2**30
            except torch.cuda.OutOfMemoryError:
                row["error"] = "OOM"
            except Exception as exc:  # noqa: BLE001 -- one arm failing does not stop the table
                row["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            finally:
                q = k = v = x = fn = params = None
            rows.append(row)
            print(json.dumps(row), flush=True)
            (args.out / "speed.json").write_text(json.dumps(rows, indent=1))
    # the attended page union per block of 64 queries (flex's work), as a share of the causal dense blocks
    for r in rows:
        spec = ARMS.get(r["arm"])
        if spec and spec[0] == "flex" and "error" not in r:
            r["union_share"] = union_share(r["L"], hq, g, d, ps.PageSparseConfig(**spec[1]), dev)
    (args.out / "speed.json").write_text(json.dumps(rows, indent=1))
    lines = ["# Sparse-attention speed (one layer, one document, bf16, "
             f"{hq}x{d} q / {g} KV heads; {torch.cuda.get_device_name()})", "",
             "| L | arm | fwd ms | fwd+bwd ms | vs dense | extra peak GB | page union / dense |",
             "|---|---|---:|---:|---:|---:|---:|"]
    dense = {r["L"]: r.get("fwd_bwd_ms") for r in rows if r["arm"] == "dense"}
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['L'] // 1024}k | {r['arm']} | {r['error']} | | | | |")
            continue
        rel = r["fwd_bwd_ms"] / dense[r["L"]] if dense.get(r["L"]) else float("nan")
        lines.append(f"| {r['L'] // 1024}k | {r['arm']} | {r['fwd_ms']:.1f} | {r['fwd_bwd_ms']:.1f} | {rel:.2f}x | "
                     f"{r['peak_gb']:.1f} | {r.get('union_share', float('nan')):.3f} |")
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
