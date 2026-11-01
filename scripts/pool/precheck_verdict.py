#!/usr/bin/env python3
"""Which nodes a pool host drops after its pre-run node check (pool setting precheck_command).

Reads the check's output: one or more lines per node in scripts/nodecheck.sh's format,

    [nodecheck] FAIL <node>: <reason>
    [nodecheck] OK <node> <metrics...>        (speed metrics: gemm_tflops=<x>, ib_gbps=<x>)

and prints the nodes to exclude, one per line:
  * every node that reported FAIL, or nothing at all (hung, or its task never ran);
  * then, as far as the spare nodes allow (usable - need - failed), the slowest nodes: those
    with any speed metric more than --slow-tolerance below that metric's median, worst first.
The check never stops a run that a plain launch would have started: a slow node is only dropped
while the run still fits, and a check that FAILs (or hears nothing from) more nodes than there
are spares is not trusted at all -- a collective probe that breaks takes every node down with it
(the run then starts without the check, and a warning says so). Stdlib only.
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys

LINE = re.compile(r"\[nodecheck\] (?P<verdict>OK|FAIL|WARN|SLOW) +(?P<node>[\w.-]+)(?P<rest>.*)")
METRICS = ("gemm_tflops", "ib_gbps")
METRIC = re.compile(r"\b(?P<key>gemm_tflops|ib_gbps)=(?P<value>[0-9.]+)")


def verdict(lines, nodes, need, tolerance, warn=None):
    failed, speed, seen = {}, {m: {} for m in METRICS}, set()
    for line in lines:
        m = LINE.search(line)
        if not m or m["node"] not in nodes:
            continue
        node = m["node"]
        seen.add(node)
        if m["verdict"] == "FAIL":
            failed.setdefault(node, m["rest"].lstrip(": ").strip() or "FAIL")
        for g in METRIC.finditer(m["rest"]):
            per = speed[g["key"]]
            per[node] = min(float(g["value"]), per.get(node, float("inf")))
    for node in nodes:
        if node not in seen:
            failed.setdefault(node, "no report (hung or not started)")
    exclude = [(n, f"FAIL: {failed[n]}") for n in nodes if n in failed]
    spare = len(nodes) - need - len(exclude)
    if spare < 0:
        if warn:
            warn(
                f"check not trusted: {len(exclude)} of {len(nodes)} node(s) FAILed or did not "
                f"report, only {len(nodes) - need} spare(s) -- excluding nothing"
            )
        return []
    deficit = {}  # node -> (worst relative shortfall, description)
    for metric, values in speed.items():
        healthy = {n: v for n, v in values.items() if n not in failed}
        if len(healthy) < 3:
            continue
        median = statistics.median(healthy.values())
        for n, v in healthy.items():
            short = 1 - v / median if median > 0 else 0.0
            if short > tolerance and short > deficit.get(n, (0.0, ""))[0]:
                deficit[n] = (short, f"slow: {metric} {v:.1f} vs median {median:.1f}")
    for n, (_, why) in sorted(deficit.items(), key=lambda kv: -kv[1][0])[:max(spare, 0)]:
        exclude.append((n, why))
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
    warn = lambda msg: print(f"[precheck] WARNING {msg}", file=sys.stderr)  # noqa: E731
    for node, why in verdict(lines, nodes, args.need, args.slow_tolerance, warn):
        print(node)
        print(f"[precheck] exclude {node}: {why}", file=sys.stderr)


if __name__ == "__main__":
    main()
