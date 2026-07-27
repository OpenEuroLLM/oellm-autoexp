#!/usr/bin/env python3
"""Gather speed-test results from an experiments folder (Megatron +
torchtitan).

Takes one or more folders (e.g. $OUTPUT_DIR/<experiment_group>), treats each
subfolder with logs as one experiment, pairs every slurm-<jobid>.log with its
dumped config-<jobid>.yaml, and prints one row per run with parameters from
the config (job name, nodes, global batch size, experts) and steady-state
speed from the log (mean + median tok/s/GPU over the last N throughput lines,
TFLOPS, MFU%, memory).

Cross-backend comparisons should use tok/s/GPU — the two backends'
FLOPs-per-token accounting differs by ~14% for MoE (see
config/experiments/korbi/qwen3_30B_A3B_speed_comp/REPORT.md).

Usage:
  python scripts/extract_speed_results.py "$OUTPUT_DIR/megatron_jupiter_speed"
  ssh jupiter 'cd ~/work/Projects/oellm-autoexp && python scripts/extract_speed_results.py \
      /e/scratch/projectnucleus/poeppel1/output/megatron_jupiter_speed'

Notes:
  - --last-n averages the last N throughput lines; use --last-n 3 for compiled
    titan runs (titan logs every metrics.log_freq steps and compile warmup can
    reach into a wider window).
  - For weak-scaling sweeps: flat tok/s/GPU across the nodes column = perfect
    weak scaling.
"""

import argparse
import csv
import re
import statistics
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # config columns become "-", speed extraction still works
    yaml = None

GH200_BF16_PEAK_TFLOPS = 989.4

MEG_ITER_RE = re.compile(
    r"elapsed time per iteration \(ms\): (?P<ms>[\d.]+).*?"
    r"throughput per GPU \(TFLOP/s/GPU\): (?P<tflops>[\d.]+)"
    r"(?:.*?Tokens per second per GPU \(Tok/s/GPU\): (?P<tps>[\d.]+))?"
)
MEG_MEM_RE = re.compile(r"mem usages: (?P<mem>[\d.]+)")
TITAN_STEP_RE = re.compile(
    r"step:\s+(?P<step>\d+).*?memory:\s+(?P<mem>[\d.]+)GiB\((?P<mempct>[\d.]+)%\)"
    r".*?tps: (?P<tps>[\d,]+).*?tflops: (?P<tflops>[\d.]+)(?:.*?mfu: (?P<mfu>[\d.]+)%)?"
)
JOBID_RE = re.compile(r"slurm-(\d+)")


def parse_log(path, last_n, peak_tflops):
    """Parse one log; return dict with speed stats or None."""
    meg, titan, mem_pcts = [], [], []
    seen_titan_steps = set()
    with open(path, errors="replace") as fh:
        for line in fh:
            m = MEG_ITER_RE.search(line)
            if m:
                meg.append(m)
                mm = MEG_MEM_RE.search(line)
                if mm:
                    mem_pcts.append(float(mm.group("mem")) * 100)
                continue
            t = TITAN_STEP_RE.search(line)
            if t:
                mem_pcts.append(float(t.group("mempct")))
                step = int(t.group("step"))
                if step not in seen_titan_steps:  # dedupe multi-rank output
                    seen_titan_steps.add(step)
                    titan.append(t)

    if meg:
        rows = meg[-last_n:]
        tps = [float(r.group("tps")) for r in rows if r.group("tps")]
        tflops = statistics.mean(float(r.group("tflops")) for r in rows)
        return {
            "backend": "megatron",
            "iters": len(meg),
            "ms_iter": statistics.mean(float(r.group("ms")) for r in rows),
            "tps_avg": statistics.mean(tps) if tps else None,
            "tps_med": statistics.median(tps) if tps else None,
            "tflops": tflops,
            "mfu": 100 * tflops / peak_tflops,
            "mem": max(mem_pcts) if mem_pcts else None,
        }
    if titan:
        rows = titan[-last_n:]
        tps = [float(r.group("tps").replace(",", "")) for r in rows]
        tflops = statistics.mean(float(r.group("tflops")) for r in rows)
        mfu_vals = [float(r.group("mfu")) for r in rows if r.group("mfu")]
        return {
            "backend": "titan",
            "iters": int(titan[-1].group("step")),
            "ms_iter": None,
            "tps_avg": statistics.mean(tps),
            "tps_med": statistics.median(tps),
            "tflops": tflops,
            "mfu": statistics.mean(mfu_vals) if mfu_vals else 100 * tflops / peak_tflops,
            "mem": max(mem_pcts),  # max across ranks — PP stages differ
        }
    return None


def dig(cfg, *keys):
    for k in keys:
        if not isinstance(cfg, dict) or k not in cfg:
            return None
        cfg = cfg[k]
    return cfg


def parse_config(path, extra_cols=()):
    """Extract (name, nodes, gbs, experts [, extra dotted paths]) from a dumped
    config-<jobid>.yaml."""
    if yaml is None or path is None:
        return {}
    try:
        doc = yaml.safe_load(open(path))
    except Exception as e:
        print(f"warning: cannot parse {path}: {e}", file=sys.stderr)
        return {}
    cfg = doc.get("config", doc)  # dumps wrap the config; tolerate raw ones
    out = {
        "name": dig(cfg, "job", "name"),
        "nodes": dig(cfg, "slurm", "sbatch", "nodes"),
    }
    meg = dig(cfg, "backend", "megatron")
    titan = dig(cfg, "backend", "titan")
    if meg:
        out["gbs"] = meg.get("global_batch_size")
        out["experts"] = meg.get("num_experts")
    elif titan:
        lbs = dig(titan, "training", "local_batch_size")
        shard = dig(titan, "parallelism", "data_parallel_shard_degree") or 1
        rep = dig(titan, "parallelism", "data_parallel_replicate_degree") or 1
        out["gbs"] = lbs * max(shard, 1) * max(rep, 1) if lbs else None
        out["experts"] = dig(titan, "model", "moe_num_experts")
    out["extra"] = {col: dig(cfg, *col.split(".")) for col in extra_cols}
    return out


def find_config(log_path, exp_dir):
    """config-<jobid>.yaml next to the log, else the newest config*.yaml
    around."""
    jobid = JOBID_RE.search(log_path.name)
    candidates = []
    for d in {log_path.parent, exp_dir}:
        if jobid:
            candidates += list(d.glob(f"config-{jobid.group(1)}.yaml"))
    if candidates:
        return candidates[0]
    generic = [p for d in {log_path.parent, exp_dir} for p in d.glob("config*.yaml")]
    return max(generic, key=lambda p: p.stat().st_mtime) if generic else None


def log_recency(path):
    """Sort key: SLURM job id if present (newer submissions = higher ids),
    else file mtime."""
    jobid = JOBID_RE.search(path.name)
    return (1, int(jobid.group(1))) if jobid else (0, path.stat().st_mtime)


def gather(root, all_runs=False):
    """Yield (experiment_dir, log_path) pairs under a folder.

    By default only the latest run per experiment dir is yielded
    (highest SLURM job id), so stale logs from failed/superseded runs
    don't clutter the table; all_runs=True yields every log.
    """
    root = Path(root)
    subdirs = [d for d in sorted(root.iterdir()) if d.is_dir()] if root.is_dir() else []
    exp_dirs = [d for d in subdirs if any(d.rglob("slurm-*.log"))] or [root]
    for d in exp_dirs:
        logs = sorted(d.rglob("slurm-*.log")) or sorted(
            f for pat in ("*.log", "*.out") for f in d.rglob(pat)
        )
        if not all_runs and logs:
            logs = [max(logs, key=log_recency)]
        for log in logs:
            yield d, log


def fmt(v, spec="", dash="-"):
    return format(v, spec) if v is not None else dash


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("folders", nargs="+", help="experiment-group folder(s)")
    ap.add_argument(
        "--last-n",
        type=int,
        default=10,
        help="average/median over the last N throughput lines (default 10)",
    )
    ap.add_argument(
        "--peak-tflops",
        type=float,
        default=GH200_BF16_PEAK_TFLOPS,
        help="per-GPU peak TFLOPS for MFU when the log has none (default GH200 bf16)",
    )
    ap.add_argument("--csv", metavar="FILE", help="also write results as CSV ('-' for stdout)")
    ap.add_argument(
        "--md", metavar="FILE", help="also write results as a markdown table ('-' for stdout)"
    )
    ap.add_argument(
        "--col",
        metavar="DOTTED.PATH",
        action="append",
        default=[],
        help="add a column extracted from the config, e.g. "
        "--col backend.megatron.expert_tensor_parallel_size (repeatable)",
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="show every run per experiment folder (default: only the latest "
        "job, so superseded/failed runs are hidden)",
    )
    args = ap.parse_args()

    # column label = last path segment, full path on collision
    last_segs = [c.split(".")[-1] for c in args.col]
    col_labels = {
        c: (seg if last_segs.count(seg) == 1 else c) for c, seg in zip(args.col, last_segs)
    }

    records = []
    for folder in args.folders:
        if not Path(folder).exists():
            print(f"warning: {folder} not found", file=sys.stderr)
            continue
        for exp_dir, log in gather(folder, args.all):
            stats = parse_log(log, args.last_n, args.peak_tflops)
            if not stats:
                continue
            cfg = parse_config(find_config(log, exp_dir), args.col)
            records.append(
                {
                    "experiment": exp_dir.name,
                    "job_name": cfg.get("name"),
                    "backend": stats["backend"],
                    "nodes": cfg.get("nodes"),
                    "gbs": cfg.get("gbs"),
                    "experts": cfg.get("experts"),
                    "iters": stats["iters"],
                    "ms_per_iter": round(stats["ms_iter"]) if stats["ms_iter"] else None,
                    "tok_s_gpu_avg": round(stats["tps_avg"]) if stats["tps_avg"] else None,
                    "tok_s_gpu_median": round(stats["tps_med"]) if stats["tps_med"] else None,
                    "tflops": round(stats["tflops"], 1),
                    "mfu_pct": round(stats["mfu"], 1),
                    "mem_pct": round(stats["mem"], 1) if stats["mem"] else None,
                    **{col_labels[c]: cfg.get("extra", {}).get(c) for c in args.col},
                }
            )

    if not records:
        print("no throughput lines found", file=sys.stderr)
        return 1

    extra_headers = tuple(col_labels[c] for c in args.col)
    header = (
        "experiment",
        "job name",
        "backend",
        "nodes",
        "GBS",
        "experts",
        "iters",
        "ms/iter",
        "tok/s/GPU",
        "median",
        "TFLOPS",
        "MFU%",
        "mem%",
    ) + extra_headers
    pretty = [
        (
            r["experiment"],
            r["job_name"] or "-",
            r["backend"],
            fmt(r["nodes"]),
            fmt(r["gbs"]),
            fmt(r["experts"]),
            str(r["iters"]),
            fmt(r["ms_per_iter"]),
            fmt(r["tok_s_gpu_avg"], ","),
            fmt(r["tok_s_gpu_median"], ","),
            fmt(r["tflops"]),
            fmt(r["mfu_pct"]),
            fmt(r["mem_pct"]),
        )
        + tuple("-" if r[h] is None else str(r[h]) for h in extra_headers)
        for r in records
    ]

    widths = [max(len(h), *(len(p[i]) for p in pretty)) for i, h in enumerate(header)]
    for line in (header, *pretty):
        print("  ".join(c.ljust(w) for c, w in zip(line, widths)))

    if args.csv:
        out = sys.stdout if args.csv == "-" else open(args.csv, "w", newline="")
        writer = csv.DictWriter(out, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
        if out is not sys.stdout:
            out.close()
            print(f"wrote {args.csv}", file=sys.stderr)

    if args.md:
        lines = [
            "| " + " | ".join(header) + " |",
            "|" + "|".join("---" for _ in header) + "|",
            *("| " + " | ".join(p) + " |" for p in pretty),
        ]
        text = "\n".join(lines) + "\n"
        if args.md == "-":
            print(text)
        else:
            Path(args.md).write_text(text)
            print(f"wrote {args.md}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
