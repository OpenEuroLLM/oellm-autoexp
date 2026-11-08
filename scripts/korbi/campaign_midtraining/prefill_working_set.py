"""Prefill-informed expert working set (docs/campaign_3b_hybrid_midtraining.md §7, 2026-11-05) on R0 traces
(router_probe.py --trace-batches): rank experts by their use in the prompt (first half of a row: 1k of 2k, or 32k
of 64k), pin the top-C per layer for the continuation (second half), and compare with global popularity (other
domains) and an LRU cache of C experts warmed on the prompt. prompt16: one pinned set for 16 streams.

    python prefill_working_set.py <traces dir>
"""

import collections
import glob
import sys
from collections import OrderedDict

import numpy as np

E, LAYERS = 360, 48
GROUPS = {"L0-2": range(0, 3), "L3-15": range(3, 16), "L16-47": range(16, 48)}


def counts(idx):
    return np.bincount(idx.ravel(), minlength=E)


def lru_hit_rate(prompt, cont, cap):
    cache = OrderedDict()

    def touch(e):
        cache.pop(e, None)
        cache[e] = 1
        if len(cache) > cap:
            cache.popitem(last=False)

    for tok in prompt[-4096:]:
        for e in tok:
            touch(e)
    hits = 0
    for tok in cont:
        hits += sum(e in cache for e in tok)
        for e in tok:
            touch(e)
    return hits / cont.size


def per_stream(name, idx, data, half, caps, lru_rows=4):
    res = collections.defaultdict(list)
    dom = name.rsplit("_", 1)[0]
    for layer in range(LAYERS):
        other = sum(counts(v[:, : v.shape[1] // 2, layer]) for k, v in data.items() if k.rsplit("_", 1)[0] != dom)
        for r in range(idx.shape[0]):
            prompt, cont = idx[r, :half, layer], idx[r, half : 2 * half, layer]
            order = np.argsort(-(counts(prompt) + 1e-3 * other))
            for cap in caps:
                pinned = np.zeros(E, bool)
                pinned[order[:cap]] = True
                popular = np.zeros(E, bool)
                popular[np.argsort(-other)[:cap]] = True
                res[("prompt", cap, layer)].append(pinned[cont].mean())
                res[("global", cap, layer)].append(popular[cont].mean())
                if r < lru_rows and cap in (32, 64):
                    res[("lru", cap, layer)].append(lru_hit_rate(prompt, cont, cap))
    return res


def batch16(idx, half, caps):
    res = collections.defaultdict(list)
    for layer in range(LAYERS):
        order = np.argsort(-counts(idx[:, :half, layer]))
        cont = idx[:, half : 2 * half, layer]
        for cap in caps:
            pinned = np.zeros(E, bool)
            pinned[order[:cap]] = True
            res[("prompt16", cap, layer)].append(pinned[cont].mean())
    return res


def main():
    data = {f.split("/")[-1][:-4]: np.load(f)["idx"].astype(np.int64) for f in sorted(glob.glob(sys.argv[1] + "/*.npz"))}
    by_dom = collections.defaultdict(lambda: collections.defaultdict(list))
    for name, idx in data.items():
        long = name.startswith("long")
        half = 32768 if long else 1024
        res = per_stream(name, idx, data, half, (16, 32, 64, 128))
        if not long:
            res.update(batch16(idx, half, (64, 128, 192)))
        for key, v in res.items():
            by_dom[name.rsplit("_", 1)[0]][key] += v
    for dom, res in by_dom.items():
        print(f"== {dom}")
        for pol in ("prompt", "global", "lru", "prompt16"):
            for cap in sorted({c for p, c, _ in res if p == pol}):
                cells = [f"{g} {np.mean([x for layer in ls for x in res[(pol, cap, layer)]]):.3f}" for g, ls in GROUPS.items()]
                total = np.mean([x for layer in range(LAYERS) for x in res[(pol, cap, layer)]])
                print(f"  {pol:8s} C={cap:3d}  all {total:.3f}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
