"""What is live at the allocator peak of a memory trace (training.py MEM_TRACE_DIR snapshots).

Replays the trace's alloc / free events, finds the peak of live allocated bytes, and groups the
allocations live there by call site: the innermost frame inside megatron/ (or a frame matching
--key), plus the size class. Allocations from before the trace started (parameters, optimizer
state, the checkpointed activations of micro-batches already in flight) are not in the trace;
they show up as "pre-trace" from the snapshot's segments.

    python scripts/korbi/hybrid_speed/mem_peak.py mem_rank0_iter1.pickle [--top 40] [--depth 2]
"""

import argparse
import collections
import pickle

GENERIC = ("torch/", "/autograd/", "transformer_engine", "triton", "contextlib", "functools",
           "megatron/core/tensor_parallel/random.py", "megatron/core/utils.py")


def site(frames, depth):
    out = []
    for fr in frames:
        f = fr.get("filename", "")
        if "megatron/" not in f or any(g in f for g in GENERIC):
            continue
        out.append(f"{f.split('megatron/', 1)[1]}:{fr.get('line')} {fr.get('name')}")
        if len(out) == depth:
            break
    return " <- ".join(out) if out else (frames[0].get("filename", "?") if frames else "?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--device", type=int, default=None)
    a = ap.parse_args()
    snap = pickle.load(open(a.snapshot, "rb"))
    traces = snap["device_traces"]
    dev = a.device if a.device is not None else max(range(len(traces)), key=lambda d: len(traces[d]))
    ev = traces[dev]

    # live bytes at trace start: everything allocated now that the trace never saw allocated
    seen = {e["addr"] for e in ev if e["action"] == "alloc"}
    pre = 0
    for seg in snap["segments"]:
        if seg["device"] != dev:
            continue
        for b in seg["blocks"]:
            if b["state"] == "active_allocated" and b["address"] not in seen:
                pre += b["size"]
    # frees in the trace of blocks allocated before it: they lower the live count from `pre`
    live, cur, peak, peak_i = {}, 0, 0, -1
    for i, e in enumerate(ev):
        act = e["action"]
        if act == "alloc":
            live[e["addr"]] = e
            cur += e["size"]
        elif act in ("free_completed",):
            x = live.pop(e["addr"], None)
            cur -= e["size"] if x is None else x["size"]
        if cur > peak:
            peak, peak_i = cur, i
    # re-run to the peak and collect what is live there
    live = {}
    for e in ev[: peak_i + 1]:
        if e["action"] == "alloc":
            live[e["addr"]] = e
        elif e["action"] == "free_completed":
            live.pop(e["addr"], None)
    groups = collections.defaultdict(lambda: [0, 0, collections.Counter()])
    for e in live.values():
        g = groups[site(e.get("frames", []), a.depth)]
        g[0] += e["size"]
        g[1] += 1
        g[2][e["size"]] += 1
    tot = sum(g[0] for g in groups.values())
    print(f"device {dev}: {len(ev)} events; pre-trace live {pre / 2**30:.2f} GiB; peak in trace "
          f"+{peak / 2**30:.2f} GiB at event {peak_i}; live at peak (trace allocs) {tot / 2**30:.2f} GiB")
    for k, (b, n, sizes) in sorted(groups.items(), key=lambda kv: -kv[1][0])[: a.top]:
        common = ", ".join(f"{s / 2**20:.0f}M x{c}" for s, c in sizes.most_common(3))
        print(f"{b / 2**30:7.2f} GiB {n:6d}  {k}   [{common}]")


if __name__ == "__main__":
    main()
