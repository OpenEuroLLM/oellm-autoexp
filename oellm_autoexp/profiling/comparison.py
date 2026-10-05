"""Stable comparison of model-independent profiling summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .models import ProfilingError


def load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) not in {1, 2}:
        raise ProfilingError(f"Unsupported profiling summary schema: {path}")
    return payload


def _get(summary: Mapping[str, Any], *paths: tuple[str, ...]) -> float | None:
    for path in paths:
        value: Any = summary
        for part in path:
            if not isinstance(value, Mapping) or part not in value:
                break
            value = value[part]
        else:
            if isinstance(value, (int, float)):
                return float(value)
    return None


def _metric(
    name: str,
    unit: str,
    baseline: float | None,
    candidate: float | None,
    direction: str,
) -> dict[str, Any]:
    delta = candidate - baseline if baseline is not None and candidate is not None else None
    delta_pct = (
        100 * delta / baseline
        if delta is not None and baseline not in {None, 0}
        else None
    )
    favorable = None
    if delta is not None and direction in {"higher", "lower"}:
        favorable = delta > 0 if direction == "higher" else delta < 0
    return {
        "metric": name,
        "unit": unit,
        "delta_unit": "pp" if unit == "%" else unit,
        "baseline": baseline,
        "candidate": candidate,
        "delta": delta,
        "delta_pct": delta_pct,
        "direction": direction,
        "favorable": favorable,
    }


def _categories(summary: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row["category"]): row for row in summary.get("categories", [])}


def _config(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    if summary.get("schema_version") == 2:
        return summary.get("workload", {}).get("config", {}).get("megatron", {})
    return summary.get("config", {}).get("megatron", {})


def compare_summaries(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    baseline_label: str = "baseline",
    candidate_label: str = "candidate",
) -> dict[str, Any]:
    metrics = [
        _metric(
            "median_step_time",
            "ms",
            _get(baseline, ("throughput", "median_step_ms"), ("profile", "median_step_ms")),
            _get(candidate, ("throughput", "median_step_ms"), ("profile", "median_step_ms")),
            "lower",
        ),
        _metric(
            "tokens_per_second_per_gpu",
            "tokens/s/GPU",
            _get(
                baseline,
                ("throughput", "median_tokens_per_second_per_gpu"),
                ("profile", "median_tokens_per_second_per_gpu"),
            ),
            _get(
                candidate,
                ("throughput", "median_tokens_per_second_per_gpu"),
                ("profile", "median_tokens_per_second_per_gpu"),
            ),
            "higher",
        ),
        _metric(
            "model_tflops_per_gpu",
            "TFLOP/s/GPU",
            _get(
                baseline,
                ("throughput", "median_tflops_per_gpu"),
                ("profile", "median_tflops_per_gpu"),
            ),
            _get(
                candidate,
                ("throughput", "median_tflops_per_gpu"),
                ("profile", "median_tflops_per_gpu"),
            ),
            "higher",
        ),
        _metric(
            "gpu_seconds_per_million_tokens",
            "GPU-s/1M tokens",
            _get(baseline, ("throughput", "gpu_seconds_per_million_tokens")),
            _get(candidate, ("throughput", "gpu_seconds_per_million_tokens")),
            "lower",
        ),
        _metric(
            "gpu_busy",
            "%",
            _get(baseline, ("device", "busy_pct"), ("gpu", "busy_pct")),
            _get(candidate, ("device", "busy_pct"), ("gpu", "busy_pct")),
            "neutral",
        ),
        _metric(
            "dispatches_per_step",
            "dispatches/step",
            _get(baseline, ("kernel_launches", "per_step"), ("dispatches", "per_step")),
            _get(candidate, ("kernel_launches", "per_step"), ("dispatches", "per_step")),
            "lower",
        ),
        _metric(
            "dispatches_per_token",
            "dispatches/token",
            _get(baseline, ("kernel_launches", "per_token")),
            _get(candidate, ("kernel_launches", "per_token")),
            "lower",
        ),
        _metric(
            "communication_exposed_per_step",
            "ms/step",
            _get(baseline, ("communication", "exposed_ms_per_step")),
            _get(candidate, ("communication", "exposed_ms_per_step")),
            "lower",
        ),
        _metric(
            "communication_overlap",
            "%",
            _get(
                baseline,
                ("communication", "overlap_pct"),
                ("overlap", "communication_overlap_pct"),
            ),
            _get(
                candidate,
                ("communication", "overlap_pct"),
                ("overlap", "communication_overlap_pct"),
            ),
            "higher",
        ),
    ]

    baseline_categories = _categories(baseline)
    candidate_categories = _categories(candidate)
    category_metrics = []
    for category in sorted(set(baseline_categories) | set(candidate_categories)):
        before = baseline_categories.get(category, {})
        after = candidate_categories.get(category, {})
        category_metrics.append(
            {
                "category": category,
                "baseline_share_pct": before.get("share_pct"),
                "candidate_share_pct": after.get("share_pct"),
                "share_delta_pp": (
                    float(after["share_pct"]) - float(before["share_pct"])
                    if before.get("share_pct") is not None
                    and after.get("share_pct") is not None
                    else None
                ),
                "baseline_direct_us_per_token": before.get("direct_us_per_token"),
                "candidate_direct_us_per_token": after.get("direct_us_per_token"),
                "baseline_direct_ms_per_step": before.get("direct_ms_per_step"),
                "candidate_direct_ms_per_step": after.get("direct_ms_per_step"),
            }
        )

    warnings: list[str] = []
    before_config = _config(baseline)
    after_config = _config(candidate)
    comparable_fields = (
        "num_layers",
        "hidden_size",
        "seq_length",
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "expert_model_parallel_size",
        "expert_tensor_parallel_size",
        "micro_batch_size",
        "global_batch_size",
    )
    for field in comparable_fields:
        if before_config.get(field) != after_config.get(field):
            warnings.append(
                f"{field} differs: {before_config.get(field)!r} vs {after_config.get(field)!r}"
            )

    return {
        "schema_version": 1,
        "baseline": baseline_label,
        "candidate": candidate_label,
        "metrics": metrics,
        "categories": category_metrics,
        "comparability_warnings": warnings,
    }


__all__ = ["compare_summaries", "load_summary"]
