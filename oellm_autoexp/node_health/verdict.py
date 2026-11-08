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
    # the profile's verdict thresholds the benchmark recorded (bench.PROFILES); empty for a
    # light-era result: then the caller's ratio applies to compute and groups
    thresholds: dict = field(default_factory=dict)


def result_from_dict(r: dict) -> BenchResult:
    """A BenchResult from bench.measure()'s dict (the ``[node_health]`` line)."""
    return BenchResult(
        groups={int(g): min(v["all_gather"], v["reduce_scatter"]) for g, v in r["groups"].items()},
        pairs=[(a, b, int(g), float(bw)) for a, b, g, bw in r.get("pairs", [])],
        compute={k: [float(x) for x in v] for k, v in r.get("compute", {}).items()},
        thresholds=dict(r.get("thresholds") or {}),
    )


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
        return result_from_dict(r)
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
    """Thresholds: the ones the benchmark recorded (its profile), else ``ratio`` for compute and
    GPU groups, no absolute floor and no ring check (the light check's original verdict)."""
    th = r.thresholds or {}
    compute_ratio = float(th.get("compute_ratio", ratio))
    group_ratio = float(th.get("group_ratio", ratio))
    min_group_gbs = float(th.get("min_group_gbs", 0.0))
    ring_ratio = float(th.get("ring_ratio", 0.0))
    max_exclude = int(th.get("max_exclude", 0))
    tag = f" [{th['profile']}]" if th.get("profile") else ""

    flops = [x for v in r.compute.values() for x in v]
    if flops:
        med = statistics.median(flops)
        slow = sorted(n for n, v in r.compute.items() if min(v) < compute_ratio * med)
        if slow:
            worst = min(min(r.compute[n]) for n in slow)
            why = f"GPU matmul down to {worst:.0f} TFLOP/s vs median {med:.0f} (< {compute_ratio:.2f} x)"
            if max_exclude > 0 and len(slow) > max_exclude:
                # more slow nodes than the cap: the worst go, the rest are reported
                ranked = sorted(slow, key=lambda n: min(r.compute[n]))
                kept = ranked[max_exclude:]
                slow = sorted(ranked[:max_exclude])
                why += f"; {len(kept)} more below the threshold, not excluded (cap {max_exclude}): " + " ".join(
                    f"{n}={min(r.compute[n]) / med:.2f}" for n in kept
                )
            return Verdict("exclude", exclude=slow, why=why + tag, groups=r.groups)
    if not r.groups:
        return Verdict("pass", why="single node: no interconnect to test" + tag, groups={})
    med = statistics.median(r.groups.values())
    worst_g = min(r.groups, key=r.groups.get)
    group_bad = r.groups[worst_g] < group_ratio * med
    # an absolute floor: one group below it is located like a relative outlier; ALL groups below
    # it (an allocation-wide slow network, no node to blame) is reported, not excluded
    floor_bad = min_group_gbs > 0 and r.groups[worst_g] < min_group_gbs
    if floor_bad and not group_bad and all(v < min_group_gbs for v in r.groups.values()):
        nodes = _ring_slow_nodes(r, ring_ratio)
        if nodes:
            return Verdict("exclude", exclude=nodes, why=f"all GPU groups below {min_group_gbs:.0f} "
                           f"GB/s (median {med:.1f}); ring-slow nodes{tag}", groups=r.groups)
        return Verdict(
            "pass",
            why=f"WARNING all GPU groups below {min_group_gbs:.0f} GB/s (median {med:.1f}): "
            f"allocation-wide, no node to exclude{tag}",
            groups=r.groups,
        )
    if group_bad or floor_bad:
        why = f"GPU group {worst_g} at {r.groups[worst_g]:.1f} GB/s vs median {med:.1f}{tag}"
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
    nodes = _ring_slow_nodes(r, ring_ratio)
    if nodes:
        return Verdict(
            "exclude",
            exclude=nodes,
            why=f"node(s) slow in the ring as receiver and sender (< {ring_ratio:.2f} x median){tag}",
            groups=r.groups,
        )
    return Verdict(
        "pass", why=f"all GPU groups >= {group_ratio:.2f} x median ({med:.1f} GB/s){tag}", groups=r.groups
    )


def _ring_slow_nodes(r: BenchResult, ring_ratio: float) -> list[str]:
    """Nodes whose slowest ring transfer is below ``ring_ratio`` x the median both as receiver and
    as sender: a bad NIC slows both directions of its node, its ring neighbours only one."""
    if ring_ratio <= 0 or not r.pairs:
        return []
    med = statistics.median(p[3] for p in r.pairs)
    recv: dict[str, float] = {}
    send: dict[str, float] = {}
    for a, b, _g, bw in r.pairs:
        recv[b] = min(recv.get(b, bw), bw)
        send[a] = min(send.get(a, bw), bw)
    lim = ring_ratio * med
    slow = sorted((max(recv[n], send.get(n, lim)), n) for n in recv if recv[n] < lim and send.get(n, lim) < lim)
    cap = int((r.thresholds or {}).get("max_exclude", 0))
    return sorted(n for _, n in (slow[:cap] if cap > 0 else slow))


def judge_dict(result: dict, ratio: float = 0.75) -> Verdict:
    """The verdict over bench.measure()'s result dict (the production pre-check)."""
    return judge_result(result_from_dict(result), ratio)


def judge(text: str, ratio: float = 0.75) -> Verdict:
    """The verdict over a benchmark log; "error" if it holds no result."""
    r = parse(text)
    if r is None:
        return Verdict("error", why="no benchmark result in the log")
    return judge_result(r, ratio)
