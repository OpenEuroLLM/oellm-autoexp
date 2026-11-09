"""Expert-cache eviction with knowledge of future routing (P2 follow-up, docs/campaign_3b_hybrid_midtraining.md): how
many expert transfers could a predictor of the coming tokens' routing save over LRU, for v4 decoding with offloaded
experts on a 24 GB GPU?

The R0 routing traces (top-5 expert ids of all 48 MoE layers per token; the same traces and step streams as
spec_gate.py) drive one cache over all (layer, expert) pairs. A decode step needs every expert of its token(s) -- the
union over 1 token (batch 1) or over 16 sessions' tokens (batch 16) -- on the GPU; a step's misses are loaded and,
when the cache is full, a resident expert not needed by this step is evicted by the policy:

  lru        the least recently used (baseline; reproduces spec_gate's misses)
  opt        Belady: the one used farthest in the future (the bound for ANY predictor of future routing)
  la<n>      lookahead n (1 .. 256): the routing of the next n steps is known; experts needed within it are kept
             (the one needed latest goes first if all are), the others are evicted LRU -- what a predictor of the
             next n tokens' routing could at best do
  la1_r<r>   lookahead 1 with a recall-r predictor: each expert of the next step is known with probability r
             (R0: token-ahead recall@5 <= 0.42) -- the realistic middle

Misses per step after a 25 % warm-up, and the decode rate of spec_gate's cost model (a step takes the larger of
the GPU time and the PCIe time of its misses times the one-layer-ahead prefetch's over-fetch).

    python cache_policy_sim.py --traces <dir with *.npz> --out <dir> [--caps 3000,6000]
"""

from __future__ import annotations

import argparse
import ctypes
import glob
import json
import os
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

LAYERS, EXPERTS = 48, 360
NKEYS = LAYERS * EXPERTS
EXPERT_MB = 3 * 2048 * 768 * (1 + 4 / 128) / 2**20  # FP8 + 1x128 block scales, 3 GLU matrices (as spec_gate)

_C = r"""
#include <stdint.h>
#include <stdlib.h>
typedef struct { int64_t p; int32_t k; } E;
typedef struct { E *a; int64_t n, cap; int max; } H;  /* binary heap: min on p (max if max = 1) */
static int before(H *h, E x, E y) { return h->max ? (x.p > y.p || (x.p == y.p && x.k < y.k))
                                                 : (x.p < y.p || (x.p == y.p && x.k < y.k)); }
static void push(H *h, int64_t p, int32_t k) {
  if (h->n == h->cap) { h->cap = h->cap ? 2 * h->cap : 1024; h->a = realloc(h->a, h->cap * sizeof(E)); }
  int64_t i = h->n++; E x = {p, k};
  while (i > 0) { int64_t q = (i - 1) / 2; if (!before(h, x, h->a[q])) break; h->a[i] = h->a[q]; i = q; }
  h->a[i] = x;
}
static E pop(H *h) {
  E top = h->a[0], x = h->a[--h->n]; int64_t i = 0;
  for (;;) { int64_t c = 2 * i + 1; if (c >= h->n) break;
    if (c + 1 < h->n && before(h, h->a[c + 1], h->a[c])) c++;
    if (!before(h, h->a[c], x)) break; h->a[i] = h->a[c]; i = c; }
  if (h->n) h->a[i] = x;
  return top;
}
static uint32_t hsh(uint32_t a, uint32_t b) {
  uint32_t x = a * 2654435761u ^ (b + 0x9e3779b9u + (a << 6) + (a >> 2)); x ^= x >> 16; x *= 0x7feb352du;
  x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x;
}
/* keys[n] deduped per step, starts[nsteps + 1], nextstep[n] = step of the key's next access (INT32_MAX: none).
   la < 0: OPT (everything known); la = 0: LRU; la > 0: the next la steps known, each key with probability recall.
   Returns misses counted from step `warm` on. */
int64_t simulate(const int32_t *keys, const int64_t *starts, int32_t nsteps, const int32_t *nextstep,
                 int32_t nkeys, int32_t cap, int32_t la, double recall, int32_t warm) {
  char *cached = calloc(nkeys, 1), *known = calloc(nkeys, 1);
  int32_t *last = calloc(nkeys, sizeof(int32_t)), *nxt = calloc(nkeys, sizeof(int32_t));
  int32_t *instep = malloc(nkeys * sizeof(int32_t));
  for (int32_t k = 0; k < nkeys; k++) instep[k] = -1;
  H A = {0}, B = {0}; B.max = 1;  /* A: unknown keys by last use (LRU); B: known keys by next use (farthest first) */
  E *hold = malloc(sizeof(E) * 65536);
  uint32_t thr = (uint32_t)(recall * 4294967295.0);
  int32_t size = 0; int64_t miss = 0;
  for (int32_t t = 0; t < nsteps; t++) {
    if (la > 0 && t + la < nsteps) {  /* the window's new last step: its keys become known (with prob recall) */
      int32_t u = t + la;
      for (int64_t i = starts[u]; i < starts[u + 1]; i++) {
        int32_t k = keys[i];
        if (cached[k] && !known[k] && nxt[k] == u && hsh(k, u) <= thr) { known[k] = 1; push(&B, u, k); }
      }
    }
    for (int64_t i = starts[t]; i < starts[t + 1]; i++) instep[keys[i]] = t;
    for (int64_t i = starts[t]; i < starts[t + 1]; i++) {
      int32_t k = keys[i];
      if (cached[k]) continue;
      if (t >= warm) miss++;
      if (size < cap) { cached[k] = 1; size++; continue; }
      int32_t victim = -1, nh = 0;
      while (A.n) {  /* the least recently used unknown key, not needed by this step */
        E e = pop(&A);
        if (!cached[e.k] || known[e.k] || last[e.k] != e.p) continue;
        if (instep[e.k] == t) { hold[nh++] = e; if (nh == 65536) break; continue; }
        victim = e.k; break;
      }
      for (int32_t j = 0; j < nh; j++) push(&A, hold[j].p, hold[j].k);
      nh = 0;
      if (victim < 0) {
        while (B.n) {  /* else the known key needed farthest ahead */
          E e = pop(&B);
          if (!cached[e.k] || !known[e.k] || nxt[e.k] != e.p) continue;
          if (instep[e.k] == t) { hold[nh++] = e; if (nh == 65536) break; continue; }
          victim = e.k; break;
        }
        for (int32_t j = 0; j < nh; j++) push(&B, hold[j].p, hold[j].k);
      }
      if (victim < 0) continue;  /* the step alone exceeds the cache: streamed, not kept */
      cached[victim] = 0; known[victim] = 0;
      cached[k] = 1;
    }
    for (int64_t i = starts[t]; i < starts[t + 1]; i++) {  /* this step's keys: new last / next use */
      int32_t k = keys[i];
      if (!cached[k]) continue;
      last[k] = t; nxt[k] = nextstep[i];
      int kn = la < 0 ? nextstep[i] != INT32_MAX
             : (la > 0 && nextstep[i] <= t + la && hsh(k, nextstep[i]) <= thr);
      known[k] = (char)kn;
      if (kn) push(&B, nextstep[i], k); else push(&A, t, k);
    }
  }
  free(cached); free(known); free(last); free(nxt); free(instep); free(A.a); free(B.a); free(hold);
  return miss;
}
"""


def _lib():
    d = Path(tempfile.gettempdir()) / "cache_policy_sim_c"
    d.mkdir(exist_ok=True)
    so = d / "sim.so"
    if not so.exists():
        src = d / "sim.c"
        src.write_text(_C)
        subprocess.check_call([os.environ.get("CC", "gcc"), "-O2", "-shared", "-fPIC", str(src), "-o", str(so)])
    lib = ctypes.CDLL(str(so))
    lib.simulate.restype = ctypes.c_int64
    lib.simulate.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p, ctypes.c_int32,
                             ctypes.c_int32, ctypes.c_int32, ctypes.c_double, ctypes.c_int32]
    return lib


def step_stream(idx: np.ndarray):
    """idx [B, T, L, k] -> per decode step (position t of every sequence): the distinct (layer, expert) keys,
    concatenated, and step boundaries (spec_gate.step_stream with w = adv = 1)."""
    B, T, L, K = idx.shape
    keys_all = np.arange(L)[None, None, :, None] * EXPERTS + idx.astype(np.int64)
    chunks = [np.unique(keys_all[:, t].reshape(-1)) for t in range(T)]
    starts = np.zeros(T + 1, dtype=np.int64)
    starts[1:] = np.cumsum([len(c) for c in chunks])
    return np.concatenate(chunks).astype(np.int32), starts


def next_steps(keys: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """For every access: the step of the same key's next access (INT32_MAX: none)."""
    step = np.repeat(np.arange(len(starts) - 1, dtype=np.int64), np.diff(starts))
    order = np.lexsort((step, keys))
    nxt = np.full(len(keys), np.iinfo(np.int32).max, dtype=np.int64)
    same = keys[order][1:] == keys[order][:-1]
    nxt[order[:-1][same]] = step[order[1:][same]]
    return nxt.astype(np.int32)


POLICIES = {"lru": (0, 1.0), "opt": (-1, 1.0), "la1": (1, 1.0), "la2": (2, 1.0), "la4": (4, 1.0),
            "la8": (8, 1.0), "la16": (16, 1.0), "la64": (64, 1.0), "la256": (256, 1.0), "la1_r0.4": (1, 0.4),
            "la1_r0.6": (1, 0.6)}


def run_scenario(job):
    name, path_rows, bs, caps = job
    arrs = [np.load(p)["idx"] for p in path_rows["files"]]
    idx = np.concatenate(arrs, 0)
    if bs == 1 and path_rows["session"]:
        idx = idx.reshape(1, -1, LAYERS, 5)  # one session through all the domain's rows
    elif bs == 16:
        idx = idx[:16]
    keys, starts = step_stream(idx)
    nxt = next_steps(keys, starts)
    nsteps = len(starts) - 1
    warm = int(nsteps * 0.25)
    lib = _lib()
    out = {"steps": nsteps - warm, "union": float(np.diff(starts)[warm:].mean())}
    for c in caps:
        for pol, (la, rec) in POLICIES.items():
            m = lib.simulate(keys.ctypes.data, starts.ctypes.data, nsteps, nxt.ctypes.data, NKEYS, c, la, rec, warm)
            out[f"{pol}@{c}"] = m / (nsteps - warm)
    print(f"[sim] {name}: " + ", ".join(f"{k} {v:.1f}" for k, v in out.items() if "@" in k), flush=True)
    return name, bs, out


def step_ms(union, misses, bs, expert_mb, a):
    gpu = (a.resident_gb * 1024 + union * expert_mb) / (a.dev_gbs * 1024) * 1e3 + 0.02 * bs
    pcie = misses * a.prefetch_factor * expert_mb / (a.pcie_gbs * 1024) * 1e3
    return max(gpu, pcie), gpu, pcie


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--traces", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--caps", default="3000,6000")
    p.add_argument("--pcie-gbs", type=float, default=23.7)  # measured (offload_bench)
    p.add_argument("--dev-gbs", type=float, default=883.0)  # measured
    p.add_argument("--resident-gb", type=float, default=2.6)  # [est.] as spec_gate
    p.add_argument("--prefetch-factor", type=float, default=2.0)  # [est.] as spec_gate
    p.add_argument("--workers", type=int, default=12)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    caps = [int(c) for c in a.caps.split(",")]
    files = sorted(glob.glob(os.path.join(a.traces, "*.npz")))
    doms = {}
    for f in files:
        doms.setdefault(os.path.basename(f).rsplit("_", 1)[0], []).append(f)
    jobs = []
    for dom, fs in doms.items():
        if dom.startswith("long"):
            jobs += [(f"{dom}#{i}", {"files": [f], "session": False}, 1, caps) for i, f in enumerate(fs)]
        else:
            jobs.append((dom, {"files": fs, "session": True}, 1, caps))
            jobs.append((f"{dom}_b16", {"files": fs, "session": False}, 16, caps))
    with ProcessPoolExecutor(a.workers) as ex:
        results = list(ex.map(run_scenario, jobs))
    res = {"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}, "expert_mb": EXPERT_MB,
           "policies": {k: {"lookahead": v[0], "recall": v[1]} for k, v in POLICIES.items()}, "sims": {}}
    for name, bs, out in results:
        res["sims"][name] = out | {"bs": bs}
    # decode rate: FP8 at each cap; int4 (half the bytes) at the doubled cap (same memory)
    rates = {}
    for name, r in res["sims"].items():
        bs = r["bs"]
        for c in caps:
            for pol in POLICIES:
                rates[f"{name}|fp8@{c}|{pol}"] = bs * 1e3 / step_ms(r["union"], r[f"{pol}@{c}"], bs, EXPERT_MB, a)[0]
                if 2 * c in caps or c == caps[0]:
                    c2 = 2 * c
                    if f"{pol}@{c2}" in r:
                        rates[f"{name}|int4@{c2}|{pol}"] = bs * 1e3 / step_ms(r["union"], r[f"{pol}@{c2}"], bs,
                                                                             EXPERT_MB / 2, a)[0]
    res["tok_s"] = rates
    (a.out / "cache_policy.json").write_text(json.dumps(res, indent=1))
    write_md(res, a.out, caps)


def write_md(res, out, caps):
    pols = list(POLICIES)
    L = ["# Expert-cache eviction with future routing knowledge (v4 R0 traces)", "",
         "Misses per decode step (experts; after a 25 % warm-up). lru = baseline, opt = Belady (bound for any "
         "predictor), la<n> = next n steps' routing known, la1_r<r> = next step known with recall r. Simulated "
         "from measured routing; tok/s from spec_gate's cost model (measured PCIe 23.7 GB/s / device 883 GB/s, "
         "estimated 2.6 GB resident read and 2x prefetch over-fetch).", ""]
    for c in caps:
        L += [f"## Cache {c} experts ({c * EXPERT_MB / 1024:.1f} GB FP8)", "",
              "| scenario | union | " + " | ".join(pols) + " |", "|---|---:|" + "---:|" * len(pols)]
        for name, r in res["sims"].items():
            L.append(f"| {name} | {r['union']:.0f} | " + " | ".join(f"{r[f'{p}@{c}']:.1f}" for p in pols) + " |")
        L.append("")
    b1 = [n for n, r in res["sims"].items() if r["bs"] == 1]
    b16 = [n for n, r in res["sims"].items() if r["bs"] == 16]
    L += ["## Decode tok/s (mean over scenarios)", "", "| setting | " + " | ".join(pols) + " |",
          "|---|" + "---:|" * len(pols)]
    for label, names in (("batch 1", b1), ("batch 16 (total)", b16)):
        for fmt in [f"fp8@{c}" for c in caps] + [f"int4@{caps[-1]}"]:
            vals = []
            for pol in pols:
                xs = [res["tok_s"].get(f"{n}|{fmt}|{pol}") for n in names]
                xs = [x for x in xs if x is not None]
                vals.append(f"{np.mean(xs):.1f}" if xs else "-")
            L.append(f"| {label} {fmt} | " + " | ".join(vals) + " |")
    (out / "cache_policy.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()
