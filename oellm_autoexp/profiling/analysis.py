"""Provider-neutral timeline analysis."""

from __future__ import annotations

import re
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from oellm_autoexp.profiling.adapters.base import ProfileAdapter
from oellm_autoexp.profiling.models import (
    IterationRecord,
    ProfileArtifacts,
    ProfilingError,
    SteadyWindow,
)
from oellm_autoexp.profiling.taxonomy import classify_kernel


DEFAULT_SHORT_THRESHOLDS_US = (10, 50, 100, 250, 1000)
ITERATION_RE = re.compile(
    r"\[(?P<timestamp>[^\]]+)\]\s+iteration\s+(?P<iteration>\d+)/"
    r".*?elapsed time per iteration \(ms\):\s*(?P<elapsed_ms>[0-9.]+)"
    r".*?TFLOP/s/GPU\):\s*(?P<tflops>[0-9.]+)"
    r".*?Tok/s/GPU\):\s*(?P<tokens_per_second>[0-9.]+)"
)


@dataclass
class Aggregate:
    category: str
    count: int = 0
    total_ns: int = 0
    min_ns: int = 2**63 - 1
    max_ns: int = 0

    def add(self, duration_ns: int) -> None:
        self.count += 1
        self.total_ns += duration_ns
        self.min_ns = min(self.min_ns, duration_ns)
        self.max_ns = max(self.max_ns, duration_ns)

    def row(self, total_ns: int) -> dict[str, Any]:
        return {
            "category": self.category,
            "count": self.count,
            "total_ms": self.total_ns / 1_000_000,
            "share_pct": 100.0 * self.total_ns / total_ns if total_ns else 0.0,
            "mean_us": self.total_ns / self.count / 1_000 if self.count else 0.0,
            "min_us": self.min_ns / 1_000 if self.count else 0.0,
            "max_us": self.max_ns / 1_000 if self.count else 0.0,
        }


@dataclass
class IntervalUnion:
    intervals: list[list[int]] = field(default_factory=list)
    last_start: int = -1
    dirty: bool = False

    def add(self, start: int, end: int) -> None:
        if end <= start:
            return
        if self.dirty or start < self.last_start:
            self.dirty = True
            self.intervals.append([start, end])
            self.last_start = max(self.last_start, start)
            return
        self.last_start = start
        if not self.intervals or start > self.intervals[-1][1]:
            self.intervals.append([start, end])
        elif end > self.intervals[-1][1]:
            self.intervals[-1][1] = end

    def merged(self) -> list[list[int]]:
        if not self.dirty:
            return self.intervals
        result: list[list[int]] = []
        for start, end in sorted(self.intervals):
            if not result or start > result[-1][1]:
                result.append([start, end])
            elif end > result[-1][1]:
                result[-1][1] = end
        self.intervals = result
        self.dirty = False
        return result

    def duration_ns(self) -> int:
        return sum(end - start for start, end in self.merged())


def interval_intersection_ns(left: IntervalUnion, right: IntervalUnion) -> int:
    left_intervals = left.merged()
    right_intervals = right.merged()
    left_index = right_index = duration = 0
    while left_index < len(left_intervals) and right_index < len(right_intervals):
        left_start, left_end = left_intervals[left_index]
        right_start, right_end = right_intervals[right_index]
        duration += max(0, min(left_end, right_end) - max(left_start, right_start))
        if left_end < right_end:
            left_index += 1
        else:
            right_index += 1
    return duration


def parse_iteration_log(path: Path) -> list[IterationRecord]:
    records: list[IterationRecord] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = ITERATION_RE.search(line)
            if not match:
                continue
            records.append(
                IterationRecord(
                    iteration=int(match.group("iteration")),
                    timestamp=datetime.fromisoformat(match.group("timestamp")),
                    elapsed_ms=float(match.group("elapsed_ms")),
                    tflops_per_gpu=float(match.group("tflops")),
                    tokens_per_second_per_gpu=float(match.group("tokens_per_second")),
                )
            )
    if not records:
        raise ProfilingError(f"No iteration records found in {path}")
    return records


def select_steady_window(
    records: Sequence[IterationRecord],
    start_iteration: int,
    end_iteration: int | None = None,
) -> SteadyWindow:
    by_iteration = {record.iteration: record for record in records}
    final_iteration = end_iteration or max(by_iteration)
    selected = [
        by_iteration[index]
        for index in range(start_iteration, final_iteration + 1)
        if index in by_iteration
    ]
    if not selected:
        raise ProfilingError(
            f"No iterations in requested steady window {start_iteration}-{final_iteration}"
        )
    boundary = by_iteration.get(start_iteration - 1)
    duration_ms = (
        (selected[-1].timestamp - boundary.timestamp).total_seconds() * 1000
        if boundary
        else sum(record.elapsed_ms for record in selected)
    )
    return SteadyWindow(
        start_iteration=start_iteration,
        end_iteration=selected[-1].iteration,
        step_count=len(selected),
        duration_ns=round(duration_ms * 1_000_000),
        profile_metrics={
            "median_step_ms": statistics.median(record.elapsed_ms for record in selected),
            "median_tflops_per_gpu": statistics.median(
                record.tflops_per_gpu for record in selected
            ),
            "median_tokens_per_second_per_gpu": statistics.median(
                record.tokens_per_second_per_gpu for record in selected
            ),
        },
    )


def _rollup(categories: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups = {
        "communication": {"communication"},
        "routing_permutation": {"routing_permutation"},
        "ambiguous_gemm": {"gemm_ambiguous"},
        "identified_model_compute": {
            "gated_delta_net",
            "attention",
            "normalization",
            "reduction_softmax",
        },
        "other": {"elementwise", "optimizer", "memory", "other"},
    }
    return [
        {
            "bucket": bucket,
            "count": sum(int(row["count"]) for row in categories if row["category"] in members),
            "total_ms": sum(
                float(row["total_ms"]) for row in categories if row["category"] in members
            ),
            "share_pct": sum(
                float(row["share_pct"]) for row in categories if row["category"] in members
            ),
        }
        for bucket, members in groups.items()
    ]


def recommendations(summary: Mapping[str, Any]) -> list[str]:
    categories = {row["category"]: row for row in summary["categories"]}
    result: list[str] = []
    communication = float(categories.get("communication", {}).get("share_pct", 0))
    routing = float(categories.get("routing_permutation", {}).get("share_pct", 0))
    if communication >= 15:
        result.append(
            "Communication is a major exposed cost; test overlap with an identical baseline."
        )
    if summary["overlap"]["communication_overlap_pct"] < 5 and communication >= 5:
        result.append(
            "Communication is almost fully serialized with compute; inspect scheduling and dependencies."
        )
    if routing >= 10:
        result.append(
            "Routing/permutation is material; benchmark dispatcher and permutation implementations."
        )
    under_100 = next(
        (
            row["share_of_dispatches_pct"]
            for row in summary["short_kernels"]
            if row["threshold_us"] == 100
        ),
        0,
    )
    if under_100 >= 50:
        result.append(
            "Most kernels are shorter than 100 us; prioritize fusion and launch-count reduction."
        )
    if "gemm_ambiguous" in categories:
        result.append(
            "Use markers and hardware counters before assigning generic GEMMs to model components."
        )
    return result


def analyze(
    adapter: ProfileAdapter,
    artifacts: ProfileArtifacts,
    *,
    steady_window: SteadyWindow | None = None,
    baseline_records: Sequence[IterationRecord] | None = None,
    short_thresholds_us: Sequence[int] = DEFAULT_SHORT_THRESHOLDS_US,
    top: int = 30,
) -> dict[str, Any]:
    """Analyze normalized events emitted by a provider adapter."""

    trace_end_ns = max((event.end_ns for event in adapter.iter_kernels(artifacts)), default=0)
    if not trace_end_ns:
        raise ProfilingError(f"No kernel events found in {artifacts.primary}")
    trace_start_ns = (
        trace_end_ns - steady_window.duration_ns
        if steady_window and steady_window.duration_ns
        else 0
    )

    by_name: dict[str, Aggregate] = {}
    by_category: dict[str, Aggregate] = {}
    category_unions: dict[str, IntervalUnion] = defaultdict(IntervalUnion)
    all_union = IntervalUnion()
    communication_union = IntervalUnion()
    compute_union = IntervalUnion()
    short_counts = {threshold: 0 for threshold in short_thresholds_us}
    dispatch_count = 0
    first_start_ns: int | None = None

    for event in adapter.iter_kernels(artifacts):
        if event.end_ns <= trace_start_ns or event.start_ns > trace_end_ns:
            continue
        start_ns = max(event.start_ns, trace_start_ns)
        end_ns = min(event.end_ns, trace_end_ns)
        duration_ns = end_ns - start_ns
        if duration_ns <= 0:
            continue
        first_start_ns = start_ns if first_start_ns is None else min(first_start_ns, start_ns)
        category = classify_kernel(event.name)
        by_name.setdefault(event.name, Aggregate(category)).add(duration_ns)
        by_category.setdefault(category, Aggregate(category)).add(duration_ns)
        category_unions[category].add(start_ns, end_ns)
        all_union.add(start_ns, end_ns)
        (communication_union if category == "communication" else compute_union).add(
            start_ns, end_ns
        )
        dispatch_count += 1
        for threshold in short_counts:
            if duration_ns < threshold * 1000:
                short_counts[threshold] += 1

    if not dispatch_count or first_start_ns is None:
        raise ProfilingError("No kernel events fall inside the selected window")

    total_direct_ns = sum(aggregate.total_ns for aggregate in by_category.values())
    category_rows: list[dict[str, Any]] = []
    for category, aggregate in by_category.items():
        row = aggregate.row(total_direct_ns)
        row["union_ms"] = category_unions[category].duration_ns() / 1_000_000
        category_rows.append(row)
    category_rows.sort(key=lambda row: (-row["total_ms"], row["category"]))

    kernel_rows = []
    for name, aggregate in by_name.items():
        row = aggregate.row(total_direct_ns)
        row["name"] = name
        kernel_rows.append(row)
    kernel_rows.sort(key=lambda row: (-row["total_ms"], row["name"]))

    window_ns = (
        steady_window.duration_ns
        if steady_window and steady_window.duration_ns
        else trace_end_ns - first_start_ns
    )
    busy_ns = all_union.duration_ns()
    communication_ns = communication_union.duration_ns()
    compute_ns = compute_union.duration_ns()
    overlap_ns = interval_intersection_ns(communication_union, compute_union)
    exposed_smaller_domain_ns = busy_ns - max(communication_ns, compute_ns)
    ideal_window_ns = max(window_ns - exposed_smaller_domain_ns, 1)

    baseline: dict[str, Any] = {}
    if baseline_records:
        baseline_window = select_steady_window(
            baseline_records,
            steady_window.start_iteration if steady_window else 5,
            steady_window.end_iteration if steady_window else None,
        )
        baseline = dict(baseline_window.profile_metrics)
        profile_tokens = (
            steady_window.profile_metrics.get("median_tokens_per_second_per_gpu")
            if steady_window
            else None
        )
        baseline_tokens = baseline.get("median_tokens_per_second_per_gpu")
        if profile_tokens is not None and baseline_tokens:
            baseline["profile_throughput_delta_pct"] = (
                100 * (profile_tokens - baseline_tokens) / baseline_tokens
            )

    summary: dict[str, Any] = {
        "schema_version": 1,
        "provider": artifacts.provider,
        "artifacts": {
            "root": str(artifacts.root),
            "primary": str(artifacts.primary),
            "stem": artifacts.stem,
            "files": {key: str(path) for key, path in artifacts.files.items()},
        },
        "window": {
            "start_iteration": steady_window.start_iteration if steady_window else None,
            "end_iteration": steady_window.end_iteration if steady_window else None,
            "step_count": steady_window.step_count if steady_window else None,
            "duration_ms": window_ns / 1_000_000,
            "trace_start_ns": trace_start_ns,
            "trace_end_ns": trace_end_ns,
        },
        "profile": steady_window.profile_metrics if steady_window else {},
        "baseline": baseline,
        "dispatches": {
            "count": dispatch_count,
            "per_step": dispatch_count / steady_window.step_count
            if steady_window and steady_window.step_count
            else None,
        },
        "gpu": {
            "busy_ms": busy_ns / 1_000_000,
            "idle_ms": max(window_ns - busy_ns, 0) / 1_000_000,
            "busy_pct": 100 * busy_ns / window_ns if window_ns else 0,
            "summed_kernel_ms": total_direct_ns / 1_000_000,
        },
        "overlap": {
            "communication_union_ms": communication_ns / 1_000_000,
            "compute_union_ms": compute_ns / 1_000_000,
            "communication_compute_overlap_ms": overlap_ns / 1_000_000,
            "communication_overlap_pct": 100 * overlap_ns / communication_ns
            if communication_ns
            else 0,
            "perfect_hiding_speedup_ceiling": window_ns / ideal_window_ns,
        },
        "categories": category_rows,
        "moe_rollup": _rollup(category_rows),
        "top_kernels": kernel_rows[: max(top, 0)],
        "short_kernels": [
            {
                "threshold_us": threshold,
                "count": count,
                "share_of_dispatches_pct": 100 * count / dispatch_count,
            }
            for threshold, count in sorted(short_counts.items())
        ],
        "provider_stats": adapter.supplemental_stats(artifacts),
        "notes": [
            "Category durations are summed kernel time and can exceed wall time.",
            "Generic GEMMs are semantically ambiguous without markers.",
            "Hardware roofline status requires counters not present in timeline traces.",
        ],
        "_all_kernel_rows": kernel_rows,
    }
    summary["recommendations"] = recommendations(summary)
    return summary


__all__ = [
    "DEFAULT_SHORT_THRESHOLDS_US",
    "IntervalUnion",
    "analyze",
    "interval_intersection_ns",
    "parse_iteration_log",
    "recommendations",
    "select_steady_window",
]
