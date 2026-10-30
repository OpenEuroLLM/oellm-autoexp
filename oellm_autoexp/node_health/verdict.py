"""Verdict over a node-health benchmark log (oellm_autoexp/node_health/bench.py): pass, exclude
nodes, or error (no usable result).

The benchmark's ``[node_health] {json}`` line is read; logs of the earlier one-off benchmark
(scripts/korbi/pool/fsdp_comm_bench.py) only have the ``[fsdp_bench]`` lines, which are read
as a fallback::

    [fsdp_bench] group gpu2 (125 ranks, 256 MB): all_gather 22.4 ms (11.9 GB/s), reduce_scatter 22.6 ms (11.8 GB/s)
    [fsdp_bench] slowest receivers (from -> to, GB/s): as01r5b17->as01r5b19:2=8.1, as07r3b32->as01r5b17:2=8.1, ...

Checks, in order (the first that finds something decides; the pool repeats the benchmark
without the excluded nodes):

* compute: a node with a GPU below ``ratio`` x the median matmul TFLOP/s;
* interconnect: a GPU group below ``ratio`` x the median group bandwidth. A degraded NIC (one
  GPU's rail) shows as ONE group far below the others (MareNostrum 5, 2026-10-27 and -28:
  5.7 / 11.9 vs 22 GB/s), and the node behind it as the node common to the two slowest ring
  pairs of that GPU index -- sender of one, receiver of the other. Its ring neighbours are
  slowed too, so the single slowest receiver can be a neighbour (as02r4b08 was excluded by
  mistake that way, 2026-10-27).
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, field

from oellm_autoexp.node_health.bench import PREFIX

GROUP_RE = re.compile(
    r"\[fsdp_bench\] group gpu(\d+) \([^)]*\): all_gather (?:[\d.]+ ms )?\(([\d.]+) GB/s\), "
    r"reduce_scatter (?:[\d.]+ ms )?\(([\d.]+) GB/s\)"
)
PAIRS_RE = re.compile(r"\[fsdp_bench\] slowest receivers \(from -> to, GB/s\): (.*)")
PAIR_RE = re.compile(r"([\w.-]+)->([\w.-]+):(\d+)=([\d.]+)")


@dataclass
class Verdict:
    status: str  # "pass" | "exclude" | "error"
    exclude: list[str] = field(default_factory=list)
    why: str = ""
    groups: dict[int, float] = field(default_factory=dict)


# the name the pool used first
PreflightVerdict = Verdict


@dataclass
class BenchResult:
    groups: dict[int, float]  # GPU index -> min(all-gather, reduce-scatter) GB/s
    pairs: list[tuple[str, str, int, float]]  # (from, to, GPU index, GB/s)
    compute: dict[str, list[float]] = field(default_factory=dict)  # node -> TFLOP/s per GPU


def parse(text: str) -> BenchResult | None:
    """The benchmark result in a log (the last one, if it ran more than once); None if none."""
    for line in reversed(text.splitlines()):
        i = line.find(PREFIX)
        if i < 0:
            continue
        try:
            r = json.loads(line[i + len(PREFIX) :])
        except json.JSONDecodeError:
            continue
        return BenchResult(
            groups={
                int(g): min(v["all_gather"], v["reduce_scatter"]) for g, v in r["groups"].items()
            },
            pairs=[(a, b, int(g), float(bw)) for a, b, g, bw in r.get("pairs", [])],
            compute={k: [float(x) for x in v] for k, v in r.get("compute", {}).items()},
        )
    groups, pairs = parse_bench(text)
    return BenchResult(groups, pairs) if groups else None


def parse_bench(text: str) -> tuple[dict[int, float], list[tuple[str, str, int, float]]]:
    """The ``[fsdp_bench]`` lines: (bandwidth per GPU group, the lower of all-gather /
    reduce-scatter; slowest ring pairs)."""
    groups: dict[int, float] = {}
    pairs: list[tuple[str, str, int, float]] = []
    for line in text.splitlines():
        m = GROUP_RE.search(line)
        if m:
            groups[int(m.group(1))] = min(float(m.group(2)), float(m.group(3)))
            continue
        m = PAIRS_RE.search(line)
        if m:
            pairs = [(a, b, int(g), float(v)) for a, b, g, v in PAIR_RE.findall(m.group(1))]
    return groups, pairs


def judge_result(r: BenchResult, ratio: float = 0.75) -> Verdict:
    flops = [x for v in r.compute.values() for x in v]
    if flops:
        med = statistics.median(flops)
        slow = sorted(n for n, v in r.compute.items() if min(v) < ratio * med)
        if slow:
            worst = min(min(r.compute[n]) for n in slow)
            return Verdict(
                "exclude",
                exclude=slow,
                why=f"GPU matmul down to {worst:.0f} TFLOP/s vs median {med:.0f}",
                groups=r.groups,
            )
    if not r.groups:
        return Verdict("pass", why="single node: no interconnect to test", groups={})
    med = statistics.median(r.groups.values())
    worst_g = min(r.groups, key=r.groups.get)
    if r.groups[worst_g] >= ratio * med:
        return Verdict(
            "pass", why=f"all GPU groups >= {ratio:.2f} x median ({med:.1f} GB/s)", groups=r.groups
        )
    why = f"GPU group {worst_g} at {r.groups[worst_g]:.1f} GB/s vs median {med:.1f}"
    on_g = sorted((p for p in r.pairs if p[2] == worst_g), key=lambda p: p[3])
    if not on_g:
        return Verdict("error", why=why + "; no ring pairs of that GPU index", groups=r.groups)
    if len(on_g) >= 2:
        common = sorted({on_g[0][0], on_g[0][1]} & {on_g[1][0], on_g[1][1]})
        if common:
            return Verdict(
                "exclude",
                exclude=common,
                why=why + "; node common to the two slowest pairs",
                groups=r.groups,
            )
    return Verdict(
        "exclude",
        exclude=sorted({on_g[0][0], on_g[0][1]}),
        why=why + "; both ends of the slowest pair",
        groups=r.groups,
    )


def judge(text: str, ratio: float = 0.75) -> Verdict:
    """The verdict over a benchmark log; "error" if it holds no result."""
    r = parse(text)
    if r is None:
        return Verdict("error", why="no benchmark result in the log")
    return judge_result(r, ratio)
