"""Deterministic profiling report writers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fieldnames} for row in rows)


def render_markdown(summary: Mapping[str, Any]) -> str:
    window = summary["window"]
    gpu = summary["gpu"]
    overlap = summary["overlap"]
    lines = [
        "# Profiling bottleneck report",
        "",
        f"- Provider: `{summary['provider']}`",
        f"- Trace: `{summary['artifacts']['primary']}`",
        (
            f"- Window: iterations {window['start_iteration']}–{window['end_iteration']} "
            f"({window['duration_ms'] / 1000:.3f} s)"
        ),
        f"- GPU busy: {gpu['busy_pct']:.2f}%",
        f"- Dispatches: {summary['dispatches']['count']:,}",
        f"- Communication overlap: {overlap['communication_overlap_pct']:.2f}%",
        f"- Perfect communication-hiding ceiling: {overlap['perfect_hiding_speedup_ceiling']:.3f}x",
        "",
        "## Kernel categories",
        "",
        "| Category | Count | Direct time (s) | Share | Union time (s) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary["categories"]:
        lines.append(
            f"| {row['category']} | {row['count']:,} | {row['total_ms'] / 1000:.3f} "
            f"| {row['share_pct']:.2f}% | {row['union_ms'] / 1000:.3f} |"
        )
    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {item}" for item in summary["recommendations"])
    lines.extend(["", "## Caveats", ""])
    lines.extend(f"- {note}" for note in summary["notes"])
    return "\n".join(lines) + "\n"


def _tsx(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def render_canvas(summary: Mapping[str, Any]) -> str:
    categories = [row["category"] for row in summary["categories"]]
    shares = [round(float(row["share_pct"]), 4) for row in summary["categories"]]
    top_rows = [
        [
            row["category"],
            row["name"],
            f"{int(row['count']):,}",
            f"{float(row['total_ms']) / 1000:.3f} s",
        ]
        for row in summary["top_kernels"][:12]
    ]
    recommendation_rows = [
        [str(index), item] for index, item in enumerate(summary["recommendations"], 1)
    ]
    window = summary["window"]
    diagnosis = (
        summary["recommendations"][0]
        if summary["recommendations"]
        else "No dominant bottleneck rule fired."
    )
    return f'''import {{
  BarChart, Callout, Grid, H1, H2, Stack, Stat, Table, Text,
}} from "cursor/canvas";

const categories = {_tsx(categories)};
const shares = {_tsx(shares)};
const topRows = {_tsx(top_rows)};
const recommendationRows = {_tsx(recommendation_rows)};

export default function ProfileAnalysis() {{
  return (
    <Stack gap={{18}} style={{{{ padding: 22, maxWidth: 1140, margin: "0 auto" }}}}>
      <Stack gap={{6}}>
        <H1>Cross-platform profiling analysis</H1>
        <Text tone="secondary">
          {_tsx(summary['provider'])} · iterations {window['start_iteration']}–{window['end_iteration']}
          {" "}· {window['duration_ms'] / 1000:.3f} seconds
        </Text>
      </Stack>
      <Callout tone="info" title="Diagnosis">{diagnosis}</Callout>
      <Grid columns={{4}} gap={{12}}>
        <Stat value="{summary['gpu']['busy_pct']:.1f}%" label="GPU busy" />
        <Stat value="{summary['overlap']['communication_overlap_pct']:.2f}%"
          label="Communication overlapped" tone="warning" />
        <Stat value="{summary['dispatches']['count']:,}" label="Kernel dispatches" />
        <Stat value="{summary['overlap']['perfect_hiding_speedup_ceiling']:.2f}×"
          label="Perfect-hiding ceiling" tone="info" />
      </Grid>
      <Stack gap={{8}}>
        <H2>Summed kernel duration by category</H2>
        <BarChart categories={{categories}}
          series={{[{{ name: "Kernel duration share", data: shares }}]}}
          horizontal height={{320}} valueSuffix="%" showValues />
        <Text size="small" tone="tertiary">
          x-axis: summed kernel duration (%) · y-axis: category.
          Source: {_tsx(summary['artifacts']['primary'])}.
        </Text>
      </Stack>
      <Stack gap={{8}}>
        <H2>Top kernels</H2>
        <Table headers={{["Category", "Kernel", "Count", "Direct time"]}}
          rows={{topRows}} columnAlign={{["left", "left", "right", "right"]}} striped />
      </Stack>
      <Stack gap={{8}}>
        <H2>Recommended next experiments</H2>
        <Table headers={{["Priority", "Action"]}} rows={{recommendationRows}}
          columnAlign={{["center", "left"]}} />
      </Stack>
      <Callout tone="warning" title="Interpretation limits">
        Generic GEMMs need semantic markers for model attribution. Hardware roofline
        classification requires a separate counter profile.
      </Callout>
    </Stack>
  );
}}
'''


def write_reports(summary: dict[str, Any], output_dir: Path, canvas_out: Path | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_kernels = summary.pop("_all_kernel_rows", summary["top_kernels"])
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    write_csv(
        output_dir / "categories.csv",
        summary["categories"],
        ("category", "count", "total_ms", "share_pct", "union_ms", "mean_us", "min_us", "max_us"),
    )
    write_csv(
        output_dir / "top_kernels.csv",
        all_kernels,
        ("category", "name", "count", "total_ms", "share_pct", "mean_us", "min_us", "max_us"),
    )
    write_csv(
        output_dir / "short_kernels.csv",
        summary["short_kernels"],
        ("threshold_us", "count", "share_of_dispatches_pct"),
    )
    write_csv(
        output_dir / "moe_summary.csv",
        summary["moe_rollup"],
        ("bucket", "count", "total_ms", "share_pct"),
    )
    (output_dir / "report.md").write_text(render_markdown(summary), encoding="utf-8")
    if canvas_out:
        canvas_out.parent.mkdir(parents=True, exist_ok=True)
        canvas_out.write_text(render_canvas(summary), encoding="utf-8")


__all__ = ["render_canvas", "render_markdown", "write_reports"]
