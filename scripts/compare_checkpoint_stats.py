#!/usr/bin/env python3
"""Build a cross-checkpoint and cross-run report from checkpoint scans.

This is the second stage for ``scan_checkpoint_stats.py``.  The scanner keeps
checkpoint I/O simple and auditable; this script joins those outputs, checks
coverage, rebases trajectories, ranks localized changes, and optionally makes
an overview plot.

Example::

    python3 scripts/compare_checkpoint_stats.py \\
      --scan flagship=checkpoint_stats/flagship \\
      --scan bf16=checkpoint_stats/cont4b \\
      --pair bf16:flagship --baseline-iteration 64000 \\
      --output-dir checkpoint_comparison

Each scan directory must contain ``tensor_stats.csv`` and may also contain
``channel_stats.csv``, ``skipped.csv``, and ``manifest.json``.  Run each model
trajectory through the scanner separately so its first checkpoint is its own
baseline.  The comparison stage recalculates baselines from raw statistics and
does not trust precomputed ratio columns.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


TENSOR_OUTPUT_FIELDS = [
    "run",
    "checkpoint",
    "iteration",
    "kind",
    "state",
    "tensor",
    "layer",
    "numel",
    "baseline_iteration",
    "previous_iteration",
    "rms",
    "rms_ratio",
    "rms_previous_ratio",
    "abs_max",
    "abs_max_ratio",
    "abs_max_previous_ratio",
    "tail_factor",
    "tail_factor_ratio",
    "finite_frac",
    "zero_frac",
    "sample_relative_delta_rms",
    "sample_cosine",
    "sample_sign_flip_frac",
]

CHANNEL_OUTPUT_FIELDS = [
    "run",
    "checkpoint",
    "iteration",
    "kind",
    "state",
    "tensor",
    "layer",
    "axis",
    "metric",
    "channels",
    "baseline_iteration",
    "previous_iteration",
    "min_to_median",
    "frac_below_dead_ratio",
    "q001",
    "q001_ratio",
    "q01",
    "q01_ratio",
    "median",
    "median_ratio",
    "q99",
    "q99_ratio",
    "q999",
    "q999_ratio",
]

CHANGE_FIELDS = [
    "run",
    "iteration",
    "kind",
    "state",
    "tensor",
    "layer",
    "axis",
    "metric",
    "value",
    "score",
    "interpretation",
]

PAIR_FIELDS = [
    "candidate_run",
    "reference_run",
    "baseline_iteration",
    "iteration",
    "kind",
    "state",
    "tensor",
    "layer",
    "axis",
    "metric",
    "reference_change",
    "candidate_change",
    "effect_ratio",
    "abs_log2_effect",
]

COVERAGE_FIELDS = [
    "run",
    "iteration",
    "checkpoint",
    "model_rows",
    "optimizer_rows",
    "model_numel",
    "nonfinite_rows",
    "missing_vs_run_baseline",
    "extra_vs_run_baseline",
    "skipped_records",
]

GAP_FIELDS = ["run", "iteration", "status", "kind", "state", "tensor", "layer"]

FAMILY_FIELDS = [
    "run",
    "iteration",
    "kind",
    "state",
    "tensor",
    "layers",
    "numel",
    "global_rms",
    "global_rms_ratio",
    "abs_max",
    "p95_abs_log2_rms_change",
    "p95_tail_factor",
    "p95_sample_relative_delta_rms",
    "min_sample_cosine",
    "max_sample_sign_flip_frac",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--scan",
        action="append",
        required=True,
        metavar="LABEL=DIR",
        help="label and output directory from scan_checkpoint_stats.py (repeatable)",
    )
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        metavar="CANDIDATE:REFERENCE",
        help="produce matched difference-in-change rows for a run pair (repeatable)",
    )
    parser.add_argument(
        "--baseline-iteration",
        type=int,
        help="common baseline for paired comparisons; default is earliest common iteration",
    )
    parser.add_argument("--top", type=int, default=30, help="rows per ranking in report")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("checkpoint_comparison")
    )
    parser.add_argument("--no-plot", action="store_true", help="skip overview.png")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.top < 1:
        parser.error("--top must be positive")
    return args


def split_spec(spec: str, separator: str, what: str) -> tuple[str, str]:
    if separator not in spec:
        raise ValueError(f"{what} must have the form NAME{separator}VALUE: {spec}")
    left, right = spec.split(separator, 1)
    if not left or not right:
        raise ValueError(f"{what} must have the form NAME{separator}VALUE: {spec}")
    return left, right


def as_float(row: dict[str, str], field: str, default: float = float("nan")) -> float:
    value = row.get(field, "")
    try:
        return float(value) if value != "" else default
    except (TypeError, ValueError):
        return default


def as_int(row: dict[str, str], field: str, default: int = -1) -> int:
    value = row.get(field, "")
    try:
        return int(value) if value != "" else default
    except (TypeError, ValueError):
        return default


def safe_ratio(value: float, baseline: float) -> float:
    if not math.isfinite(value) or not math.isfinite(baseline):
        return float("nan")
    if baseline == 0:
        return 1.0 if value == 0 else float("inf")
    return value / baseline


def abs_log2(value: float) -> float:
    if math.isnan(value):
        return float("nan")
    return abs(math.log2(value)) if value > 0 and math.isfinite(value) else float("inf")


def quantile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(value for value in values if math.isfinite(value))
    if not ordered:
        return float("nan")
    position = probability * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def csv_value(value: Any) -> Any:
    return f"{value:.9g}" if isinstance(value, float) else value


def write_csv(path: Path, fields: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field, "")) for field in fields})


def read_rows(scan_specs: Sequence[str], filename: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for spec in scan_specs:
        label, raw_dir = split_spec(spec, "=", "--scan")
        path = Path(raw_dir).expanduser().resolve() / filename
        if not path.exists():
            if filename == "tensor_stats.csv":
                raise FileNotFoundError(path)
            continue
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                row["run"] = label
                rows.append(row)
    return rows


def tensor_identity(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return row["kind"], row["state"], row["tensor"], int(row["layer"])


def channel_identity(row: dict[str, Any]) -> tuple[str, str, str, int, str]:
    return (*tensor_identity(row), row["axis"])


def trajectory_rows(raw_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, tuple[str, str, str, int]], list[dict[str, str]]] = defaultdict(list)
    for row in raw_rows:
        grouped[(row["run"], tensor_identity(row))].append(row)

    output: list[dict[str, Any]] = []
    for (run, _), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: (as_int(row, "iteration"), row.get("checkpoint", "")))
        baseline = rows[0]
        previous = baseline
        baseline_rms = as_float(baseline, "rms")
        baseline_max = as_float(baseline, "abs_max")
        baseline_tail = safe_ratio(as_float(baseline, "abs_q999"), baseline_rms)
        for row in rows:
            rms = as_float(row, "rms")
            abs_maximum = as_float(row, "abs_max")
            tail = safe_ratio(as_float(row, "abs_q999"), rms)
            output.append(
                {
                    "run": run,
                    "checkpoint": row.get("checkpoint", ""),
                    "iteration": as_int(row, "iteration"),
                    "kind": row["kind"],
                    "state": row["state"],
                    "tensor": row["tensor"],
                    "layer": as_int(row, "layer"),
                    "numel": as_int(row, "numel", 0),
                    "baseline_iteration": as_int(baseline, "iteration"),
                    "previous_iteration": as_int(previous, "iteration"),
                    "rms": rms,
                    "rms_ratio": safe_ratio(rms, baseline_rms),
                    "rms_previous_ratio": safe_ratio(rms, as_float(previous, "rms")),
                    "abs_max": abs_maximum,
                    "abs_max_ratio": safe_ratio(abs_maximum, baseline_max),
                    "abs_max_previous_ratio": safe_ratio(
                        abs_maximum, as_float(previous, "abs_max")
                    ),
                    "tail_factor": tail,
                    "tail_factor_ratio": safe_ratio(tail, baseline_tail),
                    "finite_frac": as_float(row, "finite_frac"),
                    "zero_frac": as_float(row, "zero_frac"),
                    "sample_relative_delta_rms": as_float(
                        row, "sample_relative_delta_rms"
                    ),
                    "sample_cosine": as_float(row, "sample_cosine"),
                    "sample_sign_flip_frac": as_float(row, "sample_sign_flip_frac"),
                }
            )
            previous = row
    return output


def channel_trajectory_rows(raw_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[str, tuple[str, str, str, int, str]], list[dict[str, str]]
    ] = defaultdict(list)
    for row in raw_rows:
        grouped[(row["run"], channel_identity(row))].append(row)

    output: list[dict[str, Any]] = []
    for (run, _), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: (as_int(row, "iteration"), row.get("checkpoint", "")))
        baseline = rows[0]
        previous = baseline
        for row in rows:
            result: dict[str, Any] = {
                "run": run,
                "checkpoint": row.get("checkpoint", ""),
                "iteration": as_int(row, "iteration"),
                "kind": row["kind"],
                "state": row["state"],
                "tensor": row["tensor"],
                "layer": as_int(row, "layer"),
                "axis": row["axis"],
                "metric": row.get("metric", ""),
                "channels": as_int(row, "channels", 0),
                "baseline_iteration": as_int(baseline, "iteration"),
                "previous_iteration": as_int(previous, "iteration"),
                "min_to_median": as_float(row, "min_to_median"),
                "frac_below_dead_ratio": as_float(row, "frac_below_dead_ratio"),
            }
            for metric in ("q001", "q01", "median", "q99", "q999"):
                value = as_float(row, metric)
                result[metric] = value
                result[f"{metric}_ratio"] = safe_ratio(value, as_float(baseline, metric))
            output.append(result)
            previous = row
    return output


def coverage_rows(
    raw_tensors: list[dict[str, str]], raw_skipped: list[dict[str, str]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in raw_tensors:
        grouped[(row["run"], as_int(row, "iteration"), row.get("checkpoint", ""))].append(row)
    skipped_counts: dict[tuple[str, int], int] = defaultdict(int)
    for row in raw_skipped:
        skipped_counts[(row["run"], as_int(row, "iteration"))] += 1

    baseline_ids: dict[str, set[tuple[str, str, str, int]]] = {}
    for (run, iteration, _), rows in sorted(grouped.items()):
        baseline_ids.setdefault(run, {tensor_identity(row) for row in rows})

    summaries: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    for (run, iteration, checkpoint), rows in sorted(grouped.items()):
        identities = {tensor_identity(row) for row in rows}
        missing = baseline_ids[run] - identities
        extra = identities - baseline_ids[run]
        for status, entries in (("missing", missing), ("extra", extra)):
            for kind, state, tensor, layer in sorted(entries):
                gaps.append(
                    {
                        "run": run,
                        "iteration": iteration,
                        "status": status,
                        "kind": kind,
                        "state": state,
                        "tensor": tensor,
                        "layer": layer,
                    }
                )
        summaries.append(
            {
                "run": run,
                "iteration": iteration,
                "checkpoint": checkpoint,
                "model_rows": sum(row["kind"].startswith("model") for row in rows),
                "optimizer_rows": sum(row["kind"] == "optimizer" for row in rows),
                "model_numel": sum(
                    as_int(row, "numel", 0) for row in rows if row["kind"].startswith("model")
                ),
                "nonfinite_rows": sum(as_float(row, "finite_frac", 1.0) < 1 for row in rows),
                "missing_vs_run_baseline": len(missing),
                "extra_vs_run_baseline": len(extra),
                "skipped_records": skipped_counts[(run, iteration)],
            }
        )
    return summaries, gaps


def family_summary_rows(tensors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in tensors:
        grouped[
            (row["run"], row["iteration"], row["kind"], row["state"], row["tensor"])
        ].append(row)

    baseline_rms: dict[tuple[str, str, str, str], float] = {}
    output: list[dict[str, Any]] = []
    for (run, iteration, kind, state, tensor), rows in sorted(grouped.items()):
        total_numel = sum(max(row["numel"], 0) for row in rows)
        global_rms = (
            math.sqrt(sum(row["numel"] * row["rms"] ** 2 for row in rows) / total_numel)
            if total_numel
            else float("nan")
        )
        baseline_key = (run, kind, state, tensor)
        baseline_rms.setdefault(baseline_key, global_rms)
        cosines = [row["sample_cosine"] for row in rows if math.isfinite(row["sample_cosine"])]
        sign_flips = [
            row["sample_sign_flip_frac"]
            for row in rows
            if math.isfinite(row["sample_sign_flip_frac"])
        ]
        output.append(
            {
                "run": run,
                "iteration": iteration,
                "kind": kind,
                "state": state,
                "tensor": tensor,
                "layers": len(rows),
                "numel": total_numel,
                "global_rms": global_rms,
                "global_rms_ratio": safe_ratio(global_rms, baseline_rms[baseline_key]),
                "abs_max": max(row["abs_max"] for row in rows),
                "p95_abs_log2_rms_change": quantile(
                    (abs_log2(row["rms_ratio"]) for row in rows), 0.95
                ),
                "p95_tail_factor": quantile((row["tail_factor"] for row in rows), 0.95),
                "p95_sample_relative_delta_rms": quantile(
                    (row["sample_relative_delta_rms"] for row in rows), 0.95
                ),
                "min_sample_cosine": min(cosines) if cosines else float("nan"),
                "max_sample_sign_flip_frac": max(sign_flips) if sign_flips else float("nan"),
            }
        )
    return output


def ranked_changes(
    tensors: list[dict[str, Any]], channels: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []

    def add(row: dict[str, Any], metric: str, value: float, score: float, text: str) -> None:
        if math.isnan(score):
            return
        changes.append(
            {
                "run": row["run"],
                "iteration": row["iteration"],
                "kind": row["kind"],
                "state": row["state"],
                "tensor": row["tensor"],
                "layer": row["layer"],
                "axis": row.get("axis", ""),
                "metric": metric,
                "value": value,
                "score": score,
                "interpretation": text,
            }
        )

    for row in tensors:
        if row["iteration"] == row["baseline_iteration"]:
            continue
        for field in ("rms_ratio", "abs_max_ratio", "tail_factor_ratio"):
            ratio = row[field]
            add(row, field, ratio, abs_log2(ratio), "absolute log2 fold-change")
        delta = row["sample_relative_delta_rms"]
        if math.isfinite(delta):
            add(row, "sample_relative_delta_rms", delta, delta, "sampled update / baseline RMS")
        cosine = row["sample_cosine"]
        if math.isfinite(cosine):
            add(row, "sample_cosine_loss", 1 - cosine, max(0.0, 1 - cosine), "1 - cosine")
        if row["finite_frac"] < 1:
            add(row, "nonfinite_fraction", 1 - row["finite_frac"], float("inf"), "non-finite")

    for row in channels:
        if row["iteration"] == row["baseline_iteration"]:
            continue
        for field in ("q001_ratio", "q01_ratio", "median_ratio", "q99_ratio", "q999_ratio"):
            ratio = row[field]
            add(row, field, ratio, abs_log2(ratio), "absolute log2 fold-change")
        dead = row["frac_below_dead_ratio"]
        if math.isfinite(dead) and dead > 0:
            add(row, "frac_below_dead_ratio", dead, dead, "fraction below peer threshold")
    return sorted(changes, key=lambda row: row["score"], reverse=True)


def paired_effect_rows(
    trajectories: list[dict[str, Any]],
    pair_specs: Sequence[str],
    requested_baseline: int | None,
    channel: bool = False,
) -> list[dict[str, Any]]:
    if not trajectories:
        return []
    identity_fn = channel_identity if channel else tensor_identity
    by_run: dict[str, dict[tuple[int, tuple[Any, ...]], dict[str, Any]]] = defaultdict(dict)
    iterations: dict[str, set[int]] = defaultdict(set)
    for row in trajectories:
        by_run[row["run"]][(row["iteration"], identity_fn(row))] = row
        iterations[row["run"]].add(row["iteration"])

    metrics = ("q001", "q01", "median", "q99", "q999") if channel else (
        "rms",
        "abs_max",
        "tail_factor",
    )
    output: list[dict[str, Any]] = []
    for spec in pair_specs:
        candidate, reference = split_spec(spec, ":", "--pair")
        if candidate not in by_run or reference not in by_run:
            raise ValueError(f"unknown run in --pair {spec}; available: {sorted(by_run)}")
        common_iterations = sorted(iterations[candidate] & iterations[reference])
        if not common_iterations:
            raise ValueError(f"no common iterations for --pair {spec}")
        baseline_iteration = requested_baseline or common_iterations[0]
        if baseline_iteration not in common_iterations:
            raise ValueError(
                f"baseline {baseline_iteration} is not common to both runs in --pair {spec}"
            )
        candidate_rows = by_run[candidate]
        reference_rows = by_run[reference]
        for iteration in common_iterations:
            if iteration == baseline_iteration:
                continue
            candidate_keys = {
                identity for it, identity in candidate_rows if it == iteration
            }
            reference_keys = {
                identity for it, identity in reference_rows if it == iteration
            }
            for identity in sorted(candidate_keys & reference_keys):
                candidate_row = candidate_rows[(iteration, identity)]
                reference_row = reference_rows[(iteration, identity)]
                candidate_base = candidate_rows.get((baseline_iteration, identity))
                reference_base = reference_rows.get((baseline_iteration, identity))
                if candidate_base is None or reference_base is None:
                    continue
                for metric in metrics:
                    candidate_change = safe_ratio(candidate_row[metric], candidate_base[metric])
                    reference_change = safe_ratio(reference_row[metric], reference_base[metric])
                    effect = safe_ratio(candidate_change, reference_change)
                    output.append(
                        {
                            "candidate_run": candidate,
                            "reference_run": reference,
                            "baseline_iteration": baseline_iteration,
                            "iteration": iteration,
                            "kind": candidate_row["kind"],
                            "state": candidate_row["state"],
                            "tensor": candidate_row["tensor"],
                            "layer": candidate_row["layer"],
                            "axis": candidate_row.get("axis", ""),
                            "metric": metric,
                            "reference_change": reference_change,
                            "candidate_change": candidate_change,
                            "effect_ratio": effect,
                            "abs_log2_effect": abs_log2(effect),
                        }
                    )
                if not channel:
                    direct_metrics = {
                        "sample_relative_delta_rms": (
                            candidate_row["sample_relative_delta_rms"],
                            reference_row["sample_relative_delta_rms"],
                        ),
                        "sample_cosine_loss": (
                            1 - candidate_row["sample_cosine"],
                            1 - reference_row["sample_cosine"],
                        ),
                        "sample_sign_flip_frac": (
                            candidate_row["sample_sign_flip_frac"],
                            reference_row["sample_sign_flip_frac"],
                        ),
                    }
                    for metric, (candidate_change, reference_change) in direct_metrics.items():
                        if not (
                            math.isfinite(candidate_change) and math.isfinite(reference_change)
                        ):
                            continue
                        effect = safe_ratio(candidate_change, reference_change)
                        output.append(
                            {
                                "candidate_run": candidate,
                                "reference_run": reference,
                                "baseline_iteration": baseline_iteration,
                                "iteration": iteration,
                                "kind": candidate_row["kind"],
                                "state": candidate_row["state"],
                                "tensor": candidate_row["tensor"],
                                "layer": candidate_row["layer"],
                                "axis": "",
                                "metric": metric,
                                "reference_change": reference_change,
                                "candidate_change": candidate_change,
                                "effect_ratio": effect,
                                "abs_log2_effect": abs_log2(effect),
                            }
                        )
    return sorted(output, key=lambda row: row["abs_log2_effect"], reverse=True)


def format_number(value: Any) -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return f"{value:.4g}"
    return str(value)


def markdown_table(fields: Sequence[str], rows: Sequence[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["_None._"]
    lines = ["| " + " | ".join(fields) + " |", "|" + "|".join("---" for _ in fields) + "|"]
    for row in rows:
        values = [format_number(row.get(field, "")).replace("|", "\\|") for field in fields]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_report(
    path: Path,
    scan_specs: Sequence[str],
    coverage: list[dict[str, Any]],
    gaps: list[dict[str, Any]],
    families: list[dict[str, Any]],
    changes: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    paired_channels: list[dict[str, Any]],
    top: int,
    plotted: bool,
) -> None:
    lines = [
        "# Cross-checkpoint parameter comparison",
        "",
        "## Inputs",
        "",
        *[f"- `{spec}`" for spec in scan_specs],
        "",
        "## Coverage",
        "",
        *markdown_table(
            [
                "run",
                "iteration",
                "model_rows",
                "optimizer_rows",
                "nonfinite_rows",
                "missing_vs_run_baseline",
                "skipped_records",
            ],
            coverage,
        ),
        "",
    ]
    if gaps:
        lines.extend(
            [
                f"Coverage warning: {len(gaps)} model/state identities differ "
                "from their run baseline.",
                "See `coverage_gaps.csv` before interpreting trends.",
                "",
            ]
        )
    latest_by_run = {
        run: max(row["iteration"] for row in families if row["run"] == run)
        for run in {row["run"] for row in families}
    }
    latest_families = [
        row for row in families if row["iteration"] == latest_by_run[row["run"]]
    ]
    latest_families.sort(
        key=lambda row: abs_log2(row["global_rms_ratio"]), reverse=True
    )
    lines.extend(
        [
            "## Parameter-family summary at each run's latest checkpoint",
            "",
            *markdown_table(
                [
                    "run",
                    "iteration",
                    "tensor",
                    "layers",
                    "global_rms_ratio",
                    "abs_max",
                    "p95_sample_relative_delta_rms",
                    "min_sample_cosine",
                ],
                latest_families[:top],
            ),
            "",
            "## Largest within-run changes",
            "",
            *markdown_table(
                ["run", "iteration", "tensor", "layer", "axis", "metric", "value", "score"],
                changes[:top],
            ),
            "",
            "## Largest paired effects",
            "",
            *markdown_table(
                [
                    "candidate_run",
                    "reference_run",
                    "iteration",
                    "tensor",
                    "layer",
                    "metric",
                    "effect_ratio",
                    "abs_log2_effect",
                ],
                paired[:top],
            ),
            "",
            "## Largest paired channel effects",
            "",
            *markdown_table(
                [
                    "candidate_run",
                    "reference_run",
                    "iteration",
                    "tensor",
                    "layer",
                    "axis",
                    "metric",
                    "effect_ratio",
                ],
                paired_channels[:top],
            ),
            "",
            "## Interpretation guardrails",
            "",
            "- Rankings are screening signals, not statistical significance tests.",
            "- Tensor quantiles are deterministic samples; extrema and RMS are exact.",
            "- Sampled elementwise drift is approximate but preserves element identity.",
            "- Anonymous optimizer buckets cannot be attributed to model parameters.",
            "- Compare fixed-data evaluation before calling parameter drift causal.",
        ]
    )
    if plotted:
        lines.extend(["", "![Overview](overview.png)"])
    path.write_text("\n".join(lines) + "\n")


def make_plot(path: Path, tensors: list[dict[str, Any]], channels: list[dict[str, Any]]) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("warning: matplotlib unavailable; skipping overview plot", file=sys.stderr)
        return False

    model = [row for row in tensors if row["kind"] == "model"]
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in model:
        grouped[(row["run"], row["iteration"])].append(row)
    channel_grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in channels:
        if row["kind"] == "model":
            channel_grouped[(row["run"], row["iteration"])].append(row)

    runs = sorted({row["run"] for row in model})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    panels = [
        ("global_rms", "Parameter-weighted global RMS / baseline"),
        ("rms_p95", "P95 |log2 tensor RMS change|"),
        ("sample_p95", "P95 sampled relative element drift"),
        ("channel_p95", "P95 |log2 channel q01 change|"),
    ]
    for run in runs:
        iterations = sorted(iteration for label, iteration in grouped if label == run)
        baseline_global = None
        series: dict[str, list[float]] = {name: [] for name, _ in panels}
        for iteration in iterations:
            rows = grouped[(run, iteration)]
            total_numel = sum(max(row["numel"], 0) for row in rows)
            global_rms = (
                math.sqrt(
                    sum(row["numel"] * row["rms"] ** 2 for row in rows) / total_numel
                )
                if total_numel
                else float("nan")
            )
            baseline_global = global_rms if baseline_global is None else baseline_global
            series["global_rms"].append(safe_ratio(global_rms, baseline_global))
            series["rms_p95"].append(
                quantile((abs_log2(row["rms_ratio"]) for row in rows), 0.95)
            )
            series["sample_p95"].append(
                quantile((row["sample_relative_delta_rms"] for row in rows), 0.95)
            )
            series["channel_p95"].append(
                quantile(
                    (
                        abs_log2(row["q01_ratio"])
                        for row in channel_grouped.get((run, iteration), [])
                    ),
                    0.95,
                )
            )
        for axis, (name, title) in zip(axes.flat, panels):
            axis.plot(iterations, series[name], marker="o", ms=3, label=run)
            axis.set_title(title)
            axis.set_xlabel("iteration")
            axis.grid(alpha=0.3)
    for axis in axes.flat:
        axis.legend(frameon=False)
    fig.suptitle("Cross-checkpoint parameter and channel drift")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    outputs = [
        output_dir / "tensor_trajectories.csv",
        output_dir / "channel_trajectories.csv",
        output_dir / "ranked_changes.csv",
        output_dir / "paired_tensor_effects.csv",
        output_dir / "paired_channel_effects.csv",
        output_dir / "coverage.csv",
        output_dir / "coverage_gaps.csv",
        output_dir / "family_summary.csv",
        output_dir / "report.md",
        output_dir / "overview.png",
    ]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise SystemExit(
            "output files already exist; choose another --output-dir or pass --overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        raw_tensors = read_rows(args.scan, "tensor_stats.csv")
        raw_channels = read_rows(args.scan, "channel_stats.csv")
        raw_skipped = read_rows(args.scan, "skipped.csv")
        tensors = trajectory_rows(raw_tensors)
        channels = channel_trajectory_rows(raw_channels)
        coverage, gaps = coverage_rows(raw_tensors, raw_skipped)
        families = family_summary_rows(tensors)
        changes = ranked_changes(tensors, channels)
        paired = paired_effect_rows(tensors, args.pair, args.baseline_iteration)
        paired_channels = paired_effect_rows(
            channels, args.pair, args.baseline_iteration, channel=True
        )
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error

    write_csv(output_dir / "tensor_trajectories.csv", TENSOR_OUTPUT_FIELDS, tensors)
    write_csv(output_dir / "channel_trajectories.csv", CHANNEL_OUTPUT_FIELDS, channels)
    write_csv(output_dir / "ranked_changes.csv", CHANGE_FIELDS, changes)
    write_csv(output_dir / "paired_tensor_effects.csv", PAIR_FIELDS, paired)
    write_csv(output_dir / "paired_channel_effects.csv", PAIR_FIELDS, paired_channels)
    write_csv(output_dir / "coverage.csv", COVERAGE_FIELDS, coverage)
    write_csv(output_dir / "coverage_gaps.csv", GAP_FIELDS, gaps)
    write_csv(output_dir / "family_summary.csv", FAMILY_FIELDS, families)
    plotted = not args.no_plot and make_plot(output_dir / "overview.png", tensors, channels)
    write_report(
        output_dir / "report.md",
        args.scan,
        coverage,
        gaps,
        families,
        changes,
        paired,
        paired_channels,
        args.top,
        plotted,
    )
    print(f"compared {len(tensors)} tensor rows and {len(channels)} channel rows")
    print(f"reports written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
