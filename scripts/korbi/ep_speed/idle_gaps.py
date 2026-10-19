#!/usr/bin/env python3
"""What the host is doing while the GPU is idle (torch-profiler chrome trace, one rank).

    python3 scripts/korbi/ep_speed/idle_gaps.py <trace.json.gz> [--min-us 500] [--top 25]

Finds the gaps where no GPU kernel / memcpy runs (longer than --min-us) and sums, per name, the
CPU events (cuda runtime calls and CPU ops) overlapping each gap, by overlap time; also the
innermost CPU op covering the gap's start (the one that made the GPU wait).
"""
import argparse, gzip, json, collections

ap = argparse.ArgumentParser()
ap.add_argument("trace"); ap.add_argument("--min-us", type=float, default=500); ap.add_argument("--max-us", type=float, default=1e12); ap.add_argument("--top", type=int, default=25)
a = ap.parse_args()
ev = json.load(gzip.open(a.trace, "rt") if a.trace.endswith(".gz") else open(a.trace))
ev = ev["traceEvents"] if isinstance(ev, dict) else ev
gpu = sorted((e["ts"], e["ts"] + e["dur"]) for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset"))
cpu = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("cuda_runtime", "cpu_op", "python_function", "cuda_driver")
       and e.get("dur", 0) < 2e5]  # skip step-long annotations
gaps, end = [], gpu[0][1]
for s, e in gpu[1:]:
    if a.min_us < s - end < a.max_us:
        gaps.append((end, s))
    end = max(end, e)
tot = sum(e - s for s, e in gaps)
print("%d gaps > %.0f us, %.1f ms total" % (len(gaps), a.min_us, tot / 1e3))
cpu.sort(key=lambda e: e["ts"])
over = collections.Counter(); starter = collections.Counter()
import bisect
starts = [e["ts"] for e in cpu]
for gs, ge in gaps:
    i = bisect.bisect_left(starts, gs - 5e4)
    best = None
    for e in cpu[i:]:
        if e["ts"] > ge:
            break
        es, ee = e["ts"], e["ts"] + e["dur"]
        ov = min(ee, ge) - max(es, gs)
        if ov > 0:
            key = (e.get("cat"), e.get("name", "")[:90])
            over[key] += ov
            if es <= gs <= ee and (best is None or e["dur"] < best["dur"]):
                best = e
    if best is not None:
        starter[(best.get("cat"), best.get("name", "")[:90])] += ge - gs
print("\n-- CPU events overlapping the gaps (ms):")
for (cat, n), v in over.most_common(a.top):
    print("%9.1f  %-14s %s" % (v / 1e3, cat, n))
print("\n-- innermost CPU event at the gap start (gap ms attributed):")
for (cat, n), v in starter.most_common(a.top):
    print("%9.1f  %-14s %s" % (v / 1e3, cat, n))
