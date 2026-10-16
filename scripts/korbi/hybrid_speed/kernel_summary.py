"""Where the GPU time of a profiled speed-suite arm goes.

    python scripts/korbi/hybrid_speed/kernel_summary.py <arm dir or trace.json.gz> [--top 25]
        [--compare <reference arm dir>]

Reads the rank-0 torch-profiler chrome trace (``<arm>/torch_profile/rank-0.json.gz``, written
by ``aux.profile=true``), sums the GPU kernel time per kernel name over the profiled steps
and per category (GEMM, attention, linear-mixer kernels, elementwise, communication, ...),
and reports the GPU-busy fraction of the window (idle = host or launch bound). With
``--compare`` the categories of a reference arm are shown alongside, per step.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path

CATEGORIES = [  # first match wins
    ("comm", r"nccl|allreduce|all_gather|reduce_scatter|alltoall|sendrecv"),
    ("attention", r"flash|fmha|cudnn.*(sdpa|attn)|attn_fwd|attn_bwd|_attention"),
    ("gemm", r"gemm|cutlass|sm90_xmma|nvjet|wgmma|cublas|ampere_.*_tn|_nn_|_tn_|_nt_"),
    ("linear_mixer", r"chunk|kda|gdn|delta|recurrent|fwd_h|bwd_dh|wy|solve_tril|power|retention|"
                     r"triadic|joint|mlstm|ssd|mamba|selective|conv1d|short_conv"),
    ("moe", r"permute|unpermute|topk|router|moe|sort|histogram|scatter|gather"),
    ("optimizer", r"adam|muon|newton|schulz|multi_tensor|l2norm_kernel"),
    ("norm", r"norm|rms"),
    ("loss", r"cross_entropy|softmax|log_softmax"),
    ("elementwise", r"elementwise|vectorized|unrolled|triton_poi|fused_|copy|fill|cat|index"),
]


def load(path: Path) -> dict:
    if path.is_dir():
        path = path / "torch_profile" / "rank-0.json.gz"
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        return json.load(f)


def category(name: str) -> str:
    low = name.lower()
    for cat, pattern in CATEGORIES:
        if re.search(pattern, low):
            return cat
    return "other"


def summarise(trace: dict) -> dict:
    """Kernel time per name / category over the profiled steps, without the first one
    (profiler start-up makes it several times slower). Steps are the GPU-side
    ``ProfilerStep#`` annotations, each counted once."""
    events = trace["traceEvents"] if isinstance(trace, dict) else trace
    steps = sorted(
        (e for e in events
         if e.get("name", "").startswith("ProfilerStep#") and e.get("cat") == "gpu_user_annotation"),
        key=lambda e: e["ts"],
    )
    if len(steps) > 1:
        steps = steps[1:]
    if steps:
        t0, t1 = steps[0]["ts"], steps[-1]["ts"] + steps[-1]["dur"]
    else:  # no GPU step annotations: the whole trace
        t0, t1 = float("-inf"), float("inf")
    kernels = [
        e for e in events
        if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e and t0 <= e["ts"] < t1
    ]
    if not kernels:
        raise SystemExit("no GPU kernels in the profiled steps")
    if not steps:
        t0 = min(e["ts"] for e in kernels)
        t1 = max(e["ts"] + e["dur"] for e in kernels)
    by_name, by_cat = defaultdict(float), defaultdict(float)
    count = defaultdict(int)
    for e in kernels:
        by_name[e["name"]] += e["dur"]
        count[e["name"]] += 1
        by_cat[category(e["name"])] += e["dur"]
    # busy time: union of kernel intervals on all streams
    busy, end = 0.0, t0
    for start, stop in sorted((e["ts"], e["ts"] + e["dur"]) for e in kernels):
        if stop <= end:
            continue
        busy += stop - max(start, end)
        end = stop
    return {"window_us": t1 - t0, "busy_us": busy, "steps": max(len(steps), 1),
            "by_name": by_name, "count": count, "by_cat": by_cat}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trace", type=Path)
    p.add_argument("--top", type=int, default=25)
    p.add_argument("--compare", type=Path)
    args = p.parse_args()
    s = summarise(load(args.trace))
    ref = summarise(load(args.compare)) if args.compare else None
    n = s["steps"]
    total = sum(s["by_cat"].values())
    print(f"{args.trace}: {n} profiled step(s), window {s['window_us'] / n / 1e3:.1f} ms/step, "
          f"GPU busy {s['busy_us'] / s['window_us']:.0%}, kernel time {total / n / 1e3:.1f} ms/step\n")
    print("| category | ms/step | share |" + (" ref ms/step |" if ref else ""))
    print("|---|---:|---:|" + ("---:|" if ref else ""))
    for cat, dur in sorted(s["by_cat"].items(), key=lambda kv: -kv[1]):
        extra = f" {ref['by_cat'].get(cat, 0.0) / ref['steps'] / 1e3:.2f} |" if ref else ""
        print(f"| {cat} | {dur / n / 1e3:.2f} | {dur / total:.1%} |{extra}")
    print(f"\n| kernel (top {args.top}) | category | calls/step | ms/step | share |\n|---|---|---:|---:|---:|")
    for name, dur in sorted(s["by_name"].items(), key=lambda kv: -kv[1])[: args.top]:
        short = name if len(name) <= 90 else name[:87] + "..."
        print(f"| `{short}` | {category(name)} | {s['count'][name] / n:.0f} | {dur / n / 1e3:.3f} | {dur / total:.1%} |")


if __name__ == "__main__":
    main()
