#!/usr/bin/env python3
"""Aggregate RESULT lines from a GPFS contention benchmark log into a per-(arm,
concurrency) table.

Shows, at each concurrency level, the AGGREGATE reads/s + MiB/s
delivered by the data path and the per-worker rate + fault latency -- so
the "throughput breakdown" (per-worker rate collapses / latency balloons
as workers pile onto one GPFS file) and the node-local fix are both
visible.
"""

import re
import sys
from collections import defaultdict

RE = re.compile(
    r"RESULT arm=(\S+) procs=(\d+) node=(\d+) rank=(\d+) reads=(\d+) bytes=(\d+) "
    r"elapsed=([\d.]+) p50_ms=([\d.]+) p99_ms=([\d.]+) mean_ms=([\d.]+)"
)


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: parse_contention_bench.py <slurm.log>")
        sys.exit(1)
    groups = defaultdict(list)
    with open(sys.argv[1], errors="ignore") as f:
        for line in f:
            m = RE.search(line)
            if m:
                arm, procs = m.group(1), int(m.group(2))
                groups[(procs, arm)].append(
                    dict(
                        reads=int(m.group(5)),
                        bytes=int(m.group(6)),
                        elapsed=float(m.group(7)),
                        p50=float(m.group(8)),
                        p99=float(m.group(9)),
                    )
                )

    arm_order = {"direct-global": 0, "direct-window": 1, "mirror-window": 2}
    print(
        f"{'procs/node':>10} {'total':>6} {'arm':<14} {'agg reads/s':>12} {'agg MiB/s':>10} "
        f"{'per-wkr r/s':>11} {'p50 ms':>8} {'p99 ms':>9}  speedup"
    )
    print("-" * 96)
    # baseline (direct-global) per-worker rate at each concurrency, for the speedup column
    base = {}
    for (procs, arm), rows in groups.items():
        if arm == "direct-global":
            el = sum(r["elapsed"] for r in rows) / len(rows)
            base[procs] = (sum(r["reads"] for r in rows) / el) / len(rows)
    for procs, arm in sorted(groups, key=lambda k: (k[0], arm_order.get(k[1], 9))):
        rows = groups[(procs, arm)]
        n = len(rows)
        el = sum(r["elapsed"] for r in rows) / n
        agg_reads = sum(r["reads"] for r in rows) / el
        agg_mibs = sum(r["bytes"] for r in rows) / el / (1 << 20)
        per_wkr = agg_reads / n
        p50 = sorted(r["p50"] for r in rows)[n // 2]
        p99 = max(r["p99"] for r in rows)
        sp = per_wkr / base[procs] if base.get(procs) else float("nan")
        print(
            f"{procs:>10} {n:>6} {arm:<14} {agg_reads:>12,.0f} {agg_mibs:>10.1f} "
            f"{per_wkr:>11,.0f} {p50:>8.3f} {p99:>9.2f}  {sp:>5.1f}x"
        )
    print(
        "\nspeedup = per-worker reads/s vs direct-global at the same concurrency "
        "(so 1.0x for direct-global by definition)."
    )


if __name__ == "__main__":
    main()
