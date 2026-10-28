#!/usr/bin/env python3
"""Which nodes a pool host drops after its pre-run node check (pool setting precheck_command).

Reads the check's output: one or more lines per node in scripts/nodecheck.sh's format,

    [nodecheck] FAIL <node>: <reason>
    [nodecheck] OK <node> <metrics...>        (a metric gemm_tflops=<x> marks the node's speed)

and prints the nodes to exclude, one per line:
  * every node that reported FAIL, or nothing at all (hung, or its task never ran);
  * then, as far as the spare nodes allow (usable - need - failed), the slowest nodes whose
    gemm_tflops is more than --slow-tolerance below the median of the reporting nodes.
A slow node is only dropped while the run still fits, so the check never stops a run that a
plain launch would have started. Stdlib only (runs with the system python3 in a batch shell).
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys

LINE = re.compile(r"\[nodecheck\] (?P<verdict>OK|FAIL|WARN|SLOW) +(?P<node>[\w.-]+)(?P<rest>.*)")
GEMM = re.compile(r"\bgemm_tflops=(?P<value>[0-9.]+)")


def verdict(lines, nodes, need, tolerance):
    failed, speed, seen = {}, {}, set()
    for line in lines:
        m = LINE.search(line)
        if not m or m["node"] not in nodes:
            continue
        node = m["node"]
        seen.add(node)
        if m["verdict"] == "FAIL":
            failed.setdefault(node, m["rest"].lstrip(": ").strip() or "FAIL")
        g = GEMM.search(m["rest"])
        if g:
            speed[node] = min(float(g["value"]), speed.get(node, float("inf")))
    for node in nodes:
        if node not in seen:
            failed.setdefault(node, "no report (hung or not started)")
    exclude = [(n, f"FAIL: {failed[n]}") for n in nodes if n in failed]
    spare = len(nodes) - need - len(exclude)
    healthy = {n: s for n, s in speed.items() if n not in failed}
    if spare > 0 and len(healthy) >= 3:
        median = statistics.median(healthy.values())
        slow = sorted((s, n) for n, s in healthy.items() if s < median * (1 - tolerance))
        for s, n in slow[:spare]:
            exclude.append((n, f"slow: gemm {s:.0f} TFLOP/s vs median {median:.0f}"))
    return exclude


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", required=True)
    ap.add_argument("--nodes", required=True, help="comma-separated nodes that were checked")
    ap.add_argument("--need", type=int, required=True, help="nodes the run needs")
    ap.add_argument("--slow-tolerance", type=float, default=0.10)
    args = ap.parse_args()
    nodes = [n for n in args.nodes.split(",") if n]
    with open(args.log, errors="replace") as fh:
        lines = fh.readlines()
    for node, why in verdict(lines, nodes, args.need, args.slow_tolerance):
        print(node)
        print(f"[precheck] exclude {node}: {why}", file=sys.stderr)


if __name__ == "__main__":
    main()
