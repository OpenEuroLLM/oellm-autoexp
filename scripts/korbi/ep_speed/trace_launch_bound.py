#!/usr/bin/env python3
"""How much of a profiled step the GPU waits on the CPU to launch kernels (what CUDA graphs remove).

    python3 scripts/korbi/ep_speed/trace_launch_bound.py <trace.json[.gz]> [--step N]

For one profiled step (default: the second, the first carries profiler warm-up) and every GPU
stream with kernels, each gap between consecutive kernels is attributed:
  - launch-bound: the next kernel's launch call (cudaLaunchKernel & co., linked by the correlation
    id) ended after the previous kernel had finished -- the GPU sat idle waiting for the CPU; the
    part of the gap after the previous kernel's end and before the launch returned counts;
  - other: the launch was already queued (the gap is a dependency: an event / stream wait on
    communication, a host sync, a copy).
Reported for the busiest stream and for the union over all streams (time no kernel runs at all),
with the kernel-duration distribution (many short kernels = more launch overhead per GPU second).
Host python3 (3.6+), stdlib only.
"""
import argparse
import bisect
import collections
import gzip
import json


def load(path):
    with (gzip.open(path, "rt") if path.endswith(".gz") else open(path)) as f:
        return json.load(f)["traceEvents"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--step", type=int, default=1, help="index of the profiled step (0-based)")
    args = ap.parse_args()
    ev = load(args.trace)
    steps = sorted((e for e in ev if e.get("ph") == "X" and str(e.get("name", "")).startswith("ProfilerStep#")),
                   key=lambda e: e["ts"])
    if not steps:
        raise SystemExit("no ProfilerStep# events")
    s = steps[min(args.step, len(steps) - 1)]
    t0, t1 = s["ts"], s["ts"] + s["dur"]
    kernels = [e for e in ev if e.get("ph") == "X" and e.get("cat") == "kernel" and t0 <= e["ts"] < t1]
    launches = {}
    for e in ev:
        if e.get("ph") == "X" and e.get("cat") == "cuda_runtime" and "aunch" in str(e.get("name", "")):
            c = e.get("args", {}).get("correlation")
            if c is not None:
                launches[c] = e["ts"] + e["dur"]
    print(f"{s['name']}: {s['dur'] / 1e6:.2f} s wall, {len(kernels)} kernels, "
          f"{len(launches)} launch calls in the trace")

    by_stream = collections.defaultdict(list)
    for k in kernels:
        by_stream[(k.get("pid"), k.get("tid"))].append(k)
    busiest = max(by_stream, key=lambda sid: sum(k["dur"] for k in by_stream[sid]))

    def gaps(ks):
        ks = sorted(ks, key=lambda k: k["ts"])
        lb = other = 0.0
        for prev, cur in zip(ks, ks[1:]):
            end = prev["ts"] + prev["dur"]
            gap = cur["ts"] - end
            if gap <= 0:
                continue
            launched = launches.get(cur.get("args", {}).get("correlation"))
            waited = min(gap, max(0.0, launched - end)) if launched is not None else 0.0
            lb += waited
            other += gap - waited
        busy = sum(k["dur"] for k in ks)
        return busy, lb, other

    busy, lb, other = gaps(by_stream[busiest])
    print(f"\nbusiest stream {busiest}: {len(by_stream[busiest])} kernels, busy {busy / 1e6:.2f} s "
          f"({100 * busy / s['dur']:.1f} % of the step)")
    print(f"  gaps: launch-bound {lb / 1e6:.3f} s ({100 * lb / s['dur']:.1f} % of the step), "
          f"other {other / 1e6:.3f} s ({100 * other / s['dur']:.1f} %)")

    # union over all streams: time no kernel runs anywhere, and how much of it waits on a launch
    iv = sorted((k["ts"], k["ts"] + k["dur"], k) for k in kernels)
    merged = []
    for a, b, k in iv:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b, k])
    idle = lb_all = 0.0
    for (a0, b0, _), (a1, b1, k1) in zip(merged, merged[1:]):
        gap = a1 - b0
        idle += gap
        launched = launches.get(k1.get("args", {}).get("correlation"))
        if launched is not None:
            lb_all += min(gap, max(0.0, launched - b0))
    print(f"\nall streams: no kernel running {idle / 1e6:.3f} s ({100 * idle / s['dur']:.1f} % of the "
          f"step), of which waiting on a launch {lb_all / 1e6:.3f} s ({100 * lb_all / s['dur']:.1f} %)")

    durs = sorted(k["dur"] for k in kernels)
    edges = [5, 10, 20, 50, 100, 1000]
    print("\nkernel durations (us): count / share of kernel time")
    tot = sum(durs) or 1.0
    lo = 0.0
    for hi in edges + [float("inf")]:
        i0, i1 = bisect.bisect_left(durs, lo), bisect.bisect_left(durs, hi)
        sel = durs[i0:i1]
        print(f"  [{lo:g}, {hi:g}): {len(sel):7d}  {100 * sum(sel) / tot:5.1f} %")
        lo = hi


if __name__ == "__main__":
    main()
