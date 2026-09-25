#!/usr/bin/env python3
"""Analyze rocprofv3 or Nsight Systems timelines with a common MoE report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from oellm_autoexp.profiling.adapters import available_adapters, get_adapter
from oellm_autoexp.profiling.analysis import analyze, parse_iteration_log, select_steady_window
from oellm_autoexp.profiling.models import ProfilingError
from oellm_autoexp.profiling.reporting import write_reports


def _manifest(path: Path) -> dict:
    manifest_path = path / "profile_manifest.json" if path.is_dir() else path.parent / "profile_manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return {}


def _provider(path: Path, explicit: str | None, manifest: dict) -> str:
    if explicit:
        return explicit
    if manifest.get("provider"):
        return str(manifest["provider"])
    if path.suffix in {".sqlite", ".db"} or (
        path.is_dir() and (list(path.glob("*.sqlite")) or list(path.glob("*.nsys-rep")))
    ):
        return "nsys"
    return "rocprofv3"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="Profiling directory or primary artifact")
    parser.add_argument("--provider", choices=available_adapters())
    parser.add_argument("--stdout", type=Path, help="Profiled stdout log")
    parser.add_argument("--config", type=Path, help="Resolved AutoExp config")
    parser.add_argument("--baseline-stdout", type=Path)
    parser.add_argument("--steady-start", type=int)
    parser.add_argument("--steady-end", type=int)
    parser.add_argument("--rank", default="rank0")
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--canvas-out", type=Path)
    return parser


def _config_summary(path: Path | None) -> dict:
    if path is None:
        return {}
    try:
        import yaml
    except ImportError:
        return {"path": str(path), "warning": "PyYAML unavailable"}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = payload.get("config", payload) if isinstance(payload, dict) else {}
    megatron = root.get("backend", {}).get("megatron", {})
    keys = (
        "num_layers",
        "hidden_size",
        "seq_length",
        "num_experts",
        "moe_router_topk",
        "moe_ffn_hidden_size",
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "expert_tensor_parallel_size",
        "expert_model_parallel_size",
        "micro_batch_size",
        "global_batch_size",
        "recompute_granularity",
        "moe_grouped_gemm",
        "moe_permute_fusion",
        "moe_shared_expert_overlap",
        "overlap_moe_expert_parallel_comm",
    )
    return {
        "path": str(path),
        "megatron": {key: megatron.get(key) for key in keys},
        "nodes": root.get("slurm", {}).get("sbatch", {}).get("nodes"),
    }


def _print_summary(summary: dict, top: int) -> None:
    print(f"Provider: {summary['provider']}")
    print(f"Trace: {summary['artifacts']['primary']}")
    print(
        f"Window {summary['window']['duration_ms'] / 1000:.3f} s; "
        f"GPU busy {summary['gpu']['busy_pct']:.2f}%; "
        f"{summary['dispatches']['count']:,} dispatches"
    )
    print(
        f"Communication overlap {summary['overlap']['communication_overlap_pct']:.2f}%; "
        f"perfect-hiding ceiling {summary['overlap']['perfect_hiding_speedup_ceiling']:.3f}x"
    )
    print("\nCategories:")
    for row in summary["categories"]:
        print(
            f"  {row['category']:<24} {row['total_ms'] / 1000:>9.3f} s "
            f"{row['share_pct']:>6.2f}% n={row['count']:,}"
        )
    print(f"\nTop {top} kernels:")
    for row in summary["top_kernels"][:top]:
        print(
            f"  {row['total_ms'] / 1000:>9.3f} s {row['share_pct']:>6.2f}% "
            f"n={row['count']:<9,} {row['name']}"
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = _manifest(args.trace)
        provider = _provider(args.trace, args.provider, manifest)
        adapter = get_adapter(provider)
        artifacts = adapter.discover(args.trace, args.rank)
        records = parse_iteration_log(args.stdout) if args.stdout else None
        analysis_defaults = manifest.get("analysis_options", {})
        steady_start = args.steady_start or int(
            analysis_defaults.get("steady_start_iteration", 5)
        )
        steady_end = (
            args.steady_end
            if args.steady_end is not None
            else analysis_defaults.get("steady_end_iteration")
        )
        window = (
            select_steady_window(records, steady_start, steady_end)
            if records
            else None
        )
        baseline = parse_iteration_log(args.baseline_stdout) if args.baseline_stdout else None
        summary = analyze(
            adapter,
            artifacts,
            steady_window=window,
            baseline_records=baseline,
            top=max(args.top, 0),
        )
        summary["config"] = _config_summary(args.config)
        output_dir = args.output_dir or artifacts.root / "profile_summary"
        canvas_out = args.canvas_out
        if canvas_out is None and analysis_defaults.get("canvas"):
            canvas_out = output_dir / "profile-analysis.canvas.tsx"
        write_reports(summary, output_dir, canvas_out)
        _print_summary(summary, max(args.top, 0))
        print(f"\nWrote analysis to {output_dir}")
        if canvas_out:
            print(f"Wrote Canvas to {canvas_out}")
        return 0
    except (ProfilingError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
