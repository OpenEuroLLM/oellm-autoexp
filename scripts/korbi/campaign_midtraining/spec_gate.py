"""P3 gate of the midtraining campaign (docs/campaign_3b_hybrid_midtraining.md): does speculative decoding (MTP or a
block drafter such as DFlash) pay off when the experts are offloaded to host memory on a 24 GB consumer GPU?

From the R0 routing traces of v4 (scripts/korbi/campaign_midtraining/router_probe.py: top-5 expert ids of all 48 MoE
layers for every token of held-out text; causal, so prefill routing = decode routing for the same tokens):

1. expert unions: distinct experts per layer over w consecutive tokens (batch 1) and over 16 sequences x w (batch
   16) -- what one verify step of w positions has to have on the GPU;
2. an LRU expert cache over all (layer, expert) pairs (exact LRU stack distances, so every cache size from one
   pass): misses per decode step of w positions advancing by the expected accepted length;
3. the speculative-decoding cost model: a step's time is the larger of the GPU time (resident weights + the
   step's cached experts read at device bandwidth, + per-position compute) and the PCIe time of its misses (the
   one-layer-ahead prefetch overlaps fetches with compute, substitute-on-miss removes stalls: R0/R2). A verify step
   of w = k + 1 positions advances by j = 1 .. k + 1 tokens with the truncated-geometric probabilities of
   acceptance alpha (P(j) = alpha^(j-1) (1 - alpha), P(k+1) = alpha^k); each j is simulated as a fixed advance
   and the expected time weighted by P(j). Throughput = E[tokens] / E[time] relative to plain decoding.
   Optimistic: the rejected positions route the true tokens (their experts are reused later, as in the traces).
   Pessimistic: they route wrong tokens, every one of them costing a plain step's misses on top.

    python spec_gate.py --traces <dir with *.npz> --out <dir> [--pcie-gbs 24 --dev-gbs 900]
"""

from __future__ import annotations

import argparse
import ctypes
import glob
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

LAYERS, EXPERTS = 48, 360
EXPERT_MB = 3 * 2048 * 768 * (1 + 4 / 128) / 2**20  # FP8 weights + fp32 1x128 block scales, 3 GLU matrices

_C = r"""
#include <stdint.h>
#include <stdlib.h>
/* LRU stack distances of a stream of steps: keys[n] (dedupe within a step is the caller's), step boundaries
   starts[nsteps+1]. For each access: the number of distinct keys used since this key's previous use (its stack
   depth; -1 = first use). A Fenwick tree over access times marks every key's latest use. */
static void add(int32_t *t, int64_t n, int64_t i, int32_t v) { for (i++; i <= n; i += i & -i) t[i] += v; }
static int64_t sum(int32_t *t, int64_t i) { int64_t s = 0; for (i++; i > 0; i -= i & -i) s += t[i]; return s; }
void stack_dist(const int32_t *keys, int64_t n, int32_t nkeys, int32_t *out) {
  int32_t *tree = calloc(n + 1, sizeof(int32_t));
  int64_t *last = malloc(sizeof(int64_t) * nkeys);
  for (int32_t k = 0; k < nkeys; k++) last[k] = -1;
  for (int64_t i = 0; i < n; i++) {
    int32_t k = keys[i];
    if (last[k] < 0) out[i] = -1;
    else { out[i] = (int32_t)(sum(tree, i - 1) - sum(tree, last[k])); add(tree, n, last[k], -1); }
    add(tree, n, i, 1); last[k] = i;
  }
  free(tree); free(last);
}
"""


def _lib():
    d = Path(tempfile.gettempdir()) / "spec_gate_c"
    d.mkdir(exist_ok=True)
    so = d / "sd.so"
    if not so.exists():
        src = d / "sd.c"
        src.write_text(_C)
        subprocess.check_call([os.environ.get("CC", "gcc"), "-O2", "-shared", "-fPIC", str(src), "-o", str(so)])
    lib = ctypes.CDLL(str(so))
    lib.stack_dist.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int32, ctypes.c_void_p]
    return lib


LIB = None


def stack_distances(keys: np.ndarray) -> np.ndarray:
    global LIB
    LIB = LIB or _lib()
    keys = np.ascontiguousarray(keys, dtype=np.int32)
    out = np.empty(len(keys), dtype=np.int32)
    LIB.stack_dist(keys.ctypes.data, len(keys), LAYERS * EXPERTS, out.ctypes.data)
    return out


def step_stream(idx: np.ndarray, w: int, adv: int):
    """idx [B, T, L, k] -> per decode step (positions t .. t + w - 1 of every sequence, then advance by adv): the
    distinct (layer, expert) keys, concatenated, and the per-step key counts."""
    B, T, L, K = idx.shape
    keys_all = (np.arange(L)[None, None, :, None] * EXPERTS + idx.astype(np.int64))  # [B, T, L, K]
    chunks, counts = [], []
    for t in range(0, T - w + 1, adv):
        u = np.unique(keys_all[:, t:t + w].reshape(-1))
        chunks.append(u)
        counts.append(len(u))
    return np.concatenate(chunks), np.array(counts)


def misses_per_step(idx, w, adv, caps, warm_frac=0.25):
    """Mean distinct experts and LRU misses per step for each cache capacity (experts), after a warm-up share."""
    keys, counts = step_stream(idx, w, adv)
    sd = stack_distances(keys)
    step_of = np.repeat(np.arange(len(counts)), counts)
    start = int(len(counts) * warm_frac)
    m = step_of >= start
    n_steps = len(counts) - start
    res = {"union": float(counts[start:].mean()), "steps": int(n_steps)}
    for c in caps:
        miss = (sd[m] < 0) | (sd[m] >= c)
        res[f"miss@{c}"] = float(miss.sum() / n_steps)
    return res


def load(trace_dir):
    out = {}
    for f in sorted(glob.glob(os.path.join(trace_dir, "*.npz"))):
        dom = os.path.basename(f).rsplit("_", 1)[0]
        out.setdefault(dom, []).append(np.load(f)["idx"])
    return out


def expected_tokens(alpha, k):
    return (1 - alpha ** (k + 1)) / (1 - alpha) if alpha < 1 else k + 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--traces", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--caps", default="1000,2000,3000,4000", help="expert-cache capacities (experts)")
    p.add_argument("--pcie-gbs", type=float, default=24.0, help="effective host->GPU GB/s (measure: offload_bench)")
    p.add_argument("--dev-gbs", type=float, default=900.0, help="effective device-memory GB/s")
    p.add_argument("--resident-gb", type=float, default=2.6,
                   help="non-expert weights read per step (FP8 trunk + shared experts + bf16 head) [est.]")
    p.add_argument("--prefetch-factor", type=float, default=2.0,
                   help="fetched / true misses with the one-layer-ahead top-10 prefetch (false positives) [est.]")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    caps = [int(c) for c in args.caps.split(",")]
    data = load(args.traces)
    res = {"expert_mb": EXPERT_MB, "caps": caps, "args": vars(args) | {"out": str(args.out)}}

    # 1. unions over w consecutive tokens (batch 1 = each row alone; batch 16 = 16 rows together)
    unions = {}
    for dom, arrs in data.items():
        idx = np.concatenate(arrs, 0) if arrs[0].shape[0] > 1 else arrs[0]
        for w in (1, 2, 3, 4, 5, 6, 8, 9):
            T = idx.shape[1]
            b1 = []
            for t in range(0, T - w + 1, max(1, T // 512)):
                win = idx[:, t:t + w]  # [B, w, L, K]
                b1.append(np.mean([len(np.unique(win[r, :, layer])) for r in range(0, idx.shape[0], 4)
                                   for layer in range(0, LAYERS, 6)]))
            u = {"b1": float(np.mean(b1))}
            if idx.shape[0] >= 16:
                b16 = []
                for t in range(0, T - w + 1, max(1, T // 128)):
                    win = idx[:16, t:t + w]
                    b16.append(np.mean([len(np.unique(win[:, :, layer])) for layer in range(0, LAYERS, 6)]))
                u["b16"] = float(np.mean(b16))
            unions[f"{dom}|{w}"] = u
            print(f"[union] {dom} w={w}: {u}", flush=True)
    res["unions"] = unions

    # 2./3. LRU misses per step and the speculative-decoding cost model, batch 1 on the 64k documents and on the
    # domain rows (each row one session, concatenated per domain), batch 16 on the domain rows
    def step_time(union, misses, w, bs):
        gpu = (args.resident_gb * 1024 + union * EXPERT_MB) / (args.dev_gbs * 1024) * 1e3  # ms, weight reads
        gpu += 0.02 * w * bs  # ms, per-position compute (attention / CKDA / Engram) [est.]
        pcie = misses * args.prefetch_factor * EXPERT_MB / (args.pcie_gbs * 1024) * 1e3
        return max(gpu, pcie), gpu, pcie

    sims = {}
    scen = []
    for dom, arrs in data.items():
        if dom.startswith("long"):
            scen += [(f"{dom}#{i}", a, 1) for i, a in enumerate(arrs)]
        else:
            rows = np.concatenate(arrs, 0)  # [32, 2048, L, K]
            scen.append((f"{dom}", rows.reshape(1, -1, LAYERS, 5), 1))  # one session through all its rows
            scen.append((f"{dom}_b16", rows[:16], 16))
    alphas = (0.5, 0.6, 0.7, 0.8, 0.9)
    ks = (1, 2, 4, 8)
    for name, idx, bs in scen:
        base = misses_per_step(idx, 1, 1, caps)
        sims[f"{name}|w1"] = base
        for k in ks:
            for j in range(1, k + 2):
                sims[f"{name}|k{k}|adv{j}"] = misses_per_step(idx, k + 1, j, caps)
        print(f"[sim] {name}: base {base}", flush=True)
        (args.out / "spec_gate.json").write_text(json.dumps(res | {"sims": sims}, indent=1))

    # throughput relative to plain decoding, per cache size: E[tokens] / E[time] over the advance distribution
    table = {}
    for name, idx, bs in scen:
        base = sims[f"{name}|w1"]
        for c in caps:
            t0, g0, p0 = step_time(base["union"], base[f"miss@{c}"], 1, bs)
            for k in ks:
                for a in alphas:
                    probs = [a ** (j - 1) * (1 - a) for j in range(1, k + 1)] + [a ** k]
                    e_tok = sum(pj * j for pj, j in zip(probs, range(1, k + 2)))
                    t_opt = t_pes = 0.0
                    for pj, j in zip(probs, range(1, k + 2)):
                        r = sims[f"{name}|k{k}|adv{j}"]
                        t_opt += pj * step_time(r["union"], r[f"miss@{c}"], k + 1, bs)[0]
                        extra = (k + 1 - j) * base[f"miss@{c}"]  # rejected positions: wrong tokens, fresh misses
                        t_pes += pj * step_time(r["union"] + (k + 1 - j) * base["union"] / max(bs, 1),
                                                r[f"miss@{c}"] + extra, k + 1, bs)[0]
                    table[f"{name}|cap{c}|k{k}|a{a}"] = {
                        "speedup_opt": e_tok * t0 / t_opt, "speedup_pes": e_tok * t0 / t_pes, "e_tokens": e_tok,
                        "t_plain_ms": t0, "gpu_plain_ms": g0, "pcie_plain_ms": p0}
    res["sims"] = sims
    res["speedup"] = table
    (args.out / "spec_gate.json").write_text(json.dumps(res, indent=1))
    write_summary(res, args.out, caps, alphas)


def write_summary(res, out, caps, alphas):
    L = ["# P3 gate: speculative decoding with offloaded experts (v4 routing traces)", "",
         f"Expert {res['expert_mb']:.2f} MB (FP8 + scales). Estimated GPU/PCIe constants: "
         f"see spec_gate.json args. Measured: routing traces only.", "",
         "## Distinct experts per layer over w consecutive tokens", "",
         "| domain | " + " | ".join(f"w={w}" for w in (1, 2, 3, 4, 5, 6, 8, 9)) + " |",
         "|---|" + "---:|" * 8]
    doms = sorted({k.split("|")[0] for k in res["unions"]})
    for d in doms:
        L.append(f"| {d} b1 | " + " | ".join(f"{res['unions'][f'{d}|{w}']['b1']:.1f}" for w in (1, 2, 3, 4, 5, 6, 8, 9))
                 + " |")
        if "b16" in res["unions"][f"{d}|1"]:
            L.append(f"| {d} b16 | " + " | ".join(f"{res['unions'][f'{d}|{w}']['b16']:.1f}"
                                                 for w in (1, 2, 3, 4, 5, 6, 8, 9)) + " |")
    L += ["", "## LRU misses per plain decode step (experts; batch 1 = one session, b16 = 16 sessions)", "",
          "| scenario | union | " + " | ".join(f"miss@{c}" for c in caps) + " |", "|---|---:|" + "---:|" * len(caps)]
    for key, r in res["sims"].items():
        if key.endswith("|w1"):
            L.append(f"| {key[:-3]} | {r['union']:.0f} | " + " | ".join(f"{r[f'miss@{c}']:.1f}" for c in caps) + " |")
    L += ["", "## Throughput vs plain decoding, cache 3000 experts (~14 GB): optimistic / pessimistic", "",
          "| scenario | plain step (gpu / pcie ms) | k | " + " | ".join(f"a={a}" for a in alphas) + " |",
          "|---|---|---:|" + "---:|" * len(alphas)]
    names = sorted({k.split("|")[0] for k in res["speedup"]})
    for n in names:
        for k in (1, 2, 4, 8):
            rows = [res["speedup"].get(f"{n}|cap3000|k{k}|a{a}") for a in alphas]
            if rows[0] is None:
                continue
            pl = f"{rows[0]['gpu_plain_ms']:.1f} / {rows[0]['pcie_plain_ms']:.1f}"
            L.append(f"| {n} | {pl} | {k} | " + " | ".join(f"{r['speedup_opt']:.2f} / {r['speedup_pes']:.2f}"
                                                         for r in rows) + " |")
    (out / "spec_gate.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()
