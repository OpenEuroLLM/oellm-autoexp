#!/usr/bin/env python3
"""Compare two provider-neutral profiling summaries."""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path
from typing import Any

from oellm_autoexp.profiling.comparison import compare_summaries, load_summary
from oellm_autoexp.profiling.models import ProfilingError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--baseline-label")
    parser.add_argument("--candidate-label")
    parser.add_argument("--format", choices=["text", "json", "csv"], default="text")
    parser.add_argument("--output", type=Path)
    return parser


def _number(value: float | None, unit: str) -> str:
    if value is None:
        return "n/a"
    if unit == "%":
        return f"{value:.2f}%"
    if unit == "pp":
        return f"{value:+.2f} pp"
    if abs(value) >= 1000:
        return f"{value:,.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.4f}"


def render_text(comparison: dict[str, Any]) -> str:
    lines = [
        f"Baseline:  {comparison['baseline']}",
        f"Candidate: {comparison['candidate']}",
        "",
        f"{'Metric':<38} {'Baseline':>14} {'Candidate':>14} {'Delta':>14} {'Change':>10}",
        "-" * 94,
    ]
    for row in comparison["metrics"]:
        unit = row["unit"]
        delta_pct = (
            f"{row['delta_pct']:+.2f}%" if row["delta_pct"] is not None else "n/a"
        )
        lines.append(
            f"{row['metric']:<38} {_number(row['baseline'], unit):>14} "
            f"{_number(row['candidate'], unit):>14} "
            f"{_number(row['delta'], row['delta_unit']):>14} {delta_pct:>10}"
        )
    category_rows = [
        row
        for row in comparison["categories"]
        if row["baseline_share_pct"] is not None or row["candidate_share_pct"] is not None
    ]
    if category_rows:
        lines.extend(
            [
                "",
                "Kernel categories",
                (
                    f"{'Category':<25} {'Share B':>10} {'Share C':>10} {'Delta':>10} "
                    f"{'direct B':>12} {'direct C':>12}"
                ),
                "-" * 86,
            ]
        )
        for row in category_rows:
            lines.append(
                f"{row['category']:<25} "
                f"{_number(row['baseline_share_pct'], '%'):>10} "
                f"{_number(row['candidate_share_pct'], '%'):>10} "
                f"{_number(row['share_delta_pp'], 'pp'):>10} "
                f"{_number(row['baseline_direct_ms_per_step'], 'ms'):>12} "
                f"{_number(row['candidate_direct_ms_per_step'], 'ms'):>12}"
            )
    if comparison["comparability_warnings"]:
        lines.extend(["", "Comparability warnings:"])
        lines.extend(f"  - {warning}" for warning in comparison["comparability_warnings"])
    return "\n".join(lines) + "\n"


def render_csv(comparison: dict[str, Any]) -> str:
    output = io.StringIO()
    fields = (
        "section",
        "name",
        "unit",
        "delta_unit",
        "baseline",
        "candidate",
        "delta",
        "delta_pct",
    )
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for row in comparison["metrics"]:
        writer.writerow(
            {
                "section": "metric",
                "name": row["metric"],
                "unit": row["unit"],
                "delta_unit": row["delta_unit"],
                "baseline": row["baseline"],
                "candidate": row["candidate"],
                "delta": row["delta"],
                "delta_pct": row["delta_pct"],
            }
        )
    for row in comparison["categories"]:
        writer.writerow(
            {
                "section": "category_share",
                "name": row["category"],
                "unit": "percentage_points",
                "baseline": row["baseline_share_pct"],
                "candidate": row["candidate_share_pct"],
                "delta": row["share_delta_pp"],
            }
        )
    return output.getvalue()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        comparison = compare_summaries(
            load_summary(args.baseline),
            load_summary(args.candidate),
            baseline_label=args.baseline_label or str(args.baseline),
            candidate_label=args.candidate_label or str(args.candidate),
        )
        if args.format == "json":
            rendered = json.dumps(comparison, indent=2) + "\n"
        elif args.format == "csv":
            rendered = render_csv(comparison)
        else:
            rendered = render_text(comparison)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        else:
            print(rendered, end="")
        return 0
    except (OSError, ValueError, json.JSONDecodeError, ProfilingError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
