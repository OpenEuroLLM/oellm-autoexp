#!/usr/bin/env python3
"""Exposed-communication breakdown of a torch-profiler chrome trace (one rank).

    python3 scripts/korbi/ep_speed/trace_breakdown.py <trace.json.gz> [--grep routed,mhc]

Kernels are split into communication (NCCL / EP comm kernels) and compute (everything else).
Over the profiled window it reports
  - compute busy: the union of compute-kernel intervals over all streams,
  - exposed comm by kind: time a comm kernel runs while no compute kernel runs (pipeline p2p =
    SendRecv, DP = ReduceScatter / AllGather / AllReduce, EP = HybridEP / MoonEP / DeepEP),
  - idle: neither,
  - with --grep: the compute time of kernels whose name matches any of the patterns (e.g. the
    routed-residual read / write / router kernels).
Host python3 (3.6+), stdlib only.
"""
import argparse
import gzip
import json
import re

COMM = [
    ("pp_p2p", re.compile(r"SendRecv|ncclDevKernel_Send|ncclDevKernel_Recv")),
    ("dp", re.compile(r"ReduceScatter|AllGather|AllReduce")),
    ("ep", re.compile(r"hybrid_ep::(dispatch|combine)|moonep(dispatch|combine)|deep_ep|ep_dispatch|ep_combine")),
    ("nccl_other", re.compile(r"nccl", re.I)),
]


def union(intervals):
    out = []
    for s, e in sorted(tuple(i) for i in intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def total(iv):
    return sum(e - s for s, e in iv)


def subtract(a, b):
    """a minus b (both unions, sorted)."""
    out, j = [], 0
    for s, e in a:
        cur = s
        while j < len(b) and b[j][1] <= cur:
            j += 1
        k = j
        while k < len(b) and b[k][0] < e:
            if b[k][0] > cur:
                out.append([cur, b[k][0]])
            cur = max(cur, b[k][1])
            k += 1
        if cur < e:
            out.append([cur, e])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--grep", default="")
    args = ap.parse_args()
    opener = gzip.open if args.trace.endswith(".gz") else open
    with opener(args.trace, "rt") as f:
        ev = json.load(f)
    ev = ev["traceEvents"] if isinstance(ev, dict) else ev
    kern = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    steps = [e for e in ev if e.get("ph") == "X" and str(e.get("name", "")).startswith("ProfilerStep")]
    t0 = min(e["ts"] for e in kern)
    t1 = max(e["ts"] + e["dur"] for e in kern)
    nsteps = max(1, len(steps))
    comm = {k: [] for k, _ in COMM}
    compute = []
    pats = [p for p in args.grep.split(",") if p]
    grep_t = {p: 0.0 for p in pats}
    for e in kern:
        name = e.get("name", "")
        iv = [e["ts"], e["ts"] + e["dur"]]
        for kind, rx in COMM:
            if rx.search(name):
                comm[kind].append(iv)
                break
        else:
            compute.append(iv)
            for p in pats:
                if re.search(p, name, re.I):
                    grep_t[p] += e["dur"]
    cu = union(compute)
    window = t1 - t0
    print("window %.1f ms/step over %d step(s)" % (window / 1e3 / nsteps, nsteps))
    print("compute busy %.1f ms/step (%.0f %%)" % (total(cu) / 1e3 / nsteps, 100 * total(cu) / window))
    allcomm = []
    for kind, _ in COMM:
        u = union(comm[kind])
        allcomm += u
        exp = subtract(u, cu)
        print("%-11s kernel %8.1f ms/step, exposed (no compute running) %8.1f ms/step"
              % (kind, total(u) / 1e3 / nsteps, total(exp) / 1e3 / nsteps))
    busy = union(compute + allcomm)
    print("idle (no kernel at all) %.1f ms/step" % ((window - total(busy)) / 1e3 / nsteps))
    for p in pats:
        print("compute kernels ~ /%s/: %.1f ms/step" % (p, grep_t[p] / 1e3 / nsteps))


if __name__ == "__main__":
    main()
