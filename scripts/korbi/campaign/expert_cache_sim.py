"""Expert-offloading study from MoE routing traces (Megatron ROUTER_TRACE_OUT, trace5 suites).

Decode at batch 1 with the routed experts in host memory and a GPU cache holding a fraction f of
all expert bytes: what share of the per-token expert reads hit the cache? Compared between MoE
shapes at equal cache bytes (f of each model's experts), e.g. 144 x w top-3 vs 288 x w/2 top-6.

    python scripts/korbi/campaign/expert_cache_sim.py base=<trace.pt> g2=<trace.pt> [--fracs ...]

Per layer and router call (one call = one 8k sequence = one decode session, cache reset):
- lru: least-recently-used cache of round(f E) experts, filled on demand;
- static: the round(f E) experts most used in the other half of the sequences, pinned;
- reuse: share of a token's experts that the previous token also used;
- window: distinct experts touched by 16 / 256 consecutive tokens, as a share of E.
Hit rates are per (token, expert) read; all experts of a model have the same size, so this is
also the byte hit rate.
"""

import argparse
from collections import OrderedDict

import numpy as np
import torch


def lru_hits(ids: np.ndarray, cap: int) -> float:
    if cap <= 0:
        return 0.0
    cache, hits, total = OrderedDict(), 0, 0
    for row in ids:
        for e in row:
            e = int(e)
            total += 1
            if e in cache:
                hits += 1
                cache.move_to_end(e)
            else:
                cache[e] = None
                if len(cache) > cap:
                    cache.popitem(last=False)
    return hits / max(total, 1)


def static_hits(ids: np.ndarray, pinned: np.ndarray) -> float:
    return float(np.isin(ids, pinned).mean())


def reuse(ids: np.ndarray) -> float:
    prev, cur = ids[:-1], ids[1:]
    same = (cur[:, :, None] == prev[:, None, :]).any(-1)
    return float(same.mean())


def window_share(ids: np.ndarray, w: int, n_exp: int) -> float:
    n = (len(ids) // w) * w
    if n == 0:
        return float("nan")
    blocks = ids[:n].reshape(-1, w * ids.shape[1])
    return float(np.mean([len(np.unique(b)) for b in blocks]) / n_exp)


def study(path: str, fracs, max_tokens: int, max_calls: int):
    d = torch.load(path, map_location="cpu")
    n_exp, k = d["num_experts"], d["topk"]
    rows = {f"lru@{f}": [] for f in fracs} | {f"static@{f}": [] for f in fracs}
    rows |= {"reuse": [], "win16": [], "win256": []}
    for layer, calls in sorted(d["trace"].items()):
        calls = [c.numpy().astype(np.int64) for c in calls[:max_calls]]
        half = max(len(calls) // 2, 1)
        for part, other in ((calls[:half], calls[half:] or calls[:half]),
                            (calls[half:] or calls[:half], calls[:half])):
            counts = np.bincount(np.concatenate([c.ravel() for c in other]), minlength=n_exp)
            order = np.argsort(-counts)
            for c in part:
                c = c[:max_tokens]
                for f in fracs:
                    cap = int(round(f * n_exp))
                    rows[f"lru@{f}"].append(lru_hits(c, cap))
                    rows[f"static@{f}"].append(static_hits(c, order[:cap]))
                rows["reuse"].append(reuse(c))
                rows["win16"].append(window_share(c, 16, n_exp))
                rows["win256"].append(window_share(c, 256, n_exp))
    return n_exp, k, len(d["trace"]), {key: float(np.nanmean(v)) for key, v in rows.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("traces", nargs="+", help="name=path")
    p.add_argument("--fracs", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3, 0.5])
    p.add_argument("--max-tokens", type=int, default=2048, help="tokens per sequence (LRU is O(n k))")
    p.add_argument("--max-calls", type=int, default=16)
    a = p.parse_args()
    res = {}
    for spec in a.traces:
        name, path = spec.split("=", 1)
        res[name] = study(path, a.fracs, a.max_tokens, a.max_calls)
    keys = [f"lru@{f}" for f in a.fracs] + [f"static@{f}" for f in a.fracs] + ["reuse", "win16", "win256"]
    print("| model | experts x top-k | layers | " + " | ".join(keys) + " |")
    print("|---|---|---:|" + "---:|" * len(keys))
    for name, (n_exp, k, layers, r) in res.items():
        print(f"| {name} | {n_exp} x {k} | {layers} | " + " | ".join(f"{r[x]:.3f}" for x in keys) + " |")


if __name__ == "__main__":
    main()
