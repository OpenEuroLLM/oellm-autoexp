#!/usr/bin/env python3
"""Extract TFLOPs / MFU / TPS / config from Titan and Megatron training logs.

Usage (run on JUPITER or locally if output dir is mounted):
    python scripts/extract_results.py \\
        --base-dirs \\
            /e/scratch/projectnucleus/poeppel1/output/titan_moe_30BA3B_comparable \\
            /e/scratch/projectnucleus/poeppel1/output/megatron_moe_30BA3B_comparable

Options:
    --warmup N      Skip first N iterations when computing medians (default: 10)
    --csv           Also write a CSV file next to the script
    --sort FIELD    Sort output by: tflops, tps, mfu, name (default: name)
"""

import argparse
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

# GH200 SXM BF16 peak TFLOPs — derived from Titan logs (tflops / mfu_frac).
GH200_PEAK_TFLOPS = 989.4

# ── regex patterns ────────────────────────────────────────────────────────────

# Titan per-step log line (one per rank; we filter to rank 0 only):
#   0: [titan] 2026-03-24 ... step: 80  loss: ...  memory:  5.71GiB(6.01%)  tps: 57,729  tflops: 78.76  mfu: 7.96%
TITAN_RE = re.compile(
    r"^0:.*step:\s*(\d+)"
    r".*?memory:\s*([\d.]+)GiB\(([\d.]+)%\)"
    r".*?tps:\s*([\d,]+)"
    r".*?tflops:\s*([\d.]+)"
    r".*?mfu:\s*([\d.]+)%"
)

# Megatron per-iteration log line (logged once per iteration from one rank):
#   [default3]: [2026-...] iteration  80/  100 | ... | elapsed time per iteration (ms): 1548.1 | mem usages: 0.8131 | throughput per GPU (TFLOP/s/GPU): 57.6 | Tokens per second per GPU (Tok/s/GPU): 2645.8 | ...
MEGATRON_RE = re.compile(
    r"iteration\s+(\d+)/\s*\d+"
    r".*?elapsed time per iteration \(ms\):\s*([\d.]+)"
    r".*?mem usages:\s*([\d.]+)"
    r".*?throughput per GPU \(TFLOP/s/GPU\):\s*([\d.]+)"
    r".*?Tokens per second per GPU \(Tok/s/GPU\):\s*([\d.]+)"
)

OOM_RE = re.compile(
    r"CUDA out of memory|OutOfMemoryError|calloc.*bytes.*failed|Failed to CUDA calloc",
    re.IGNORECASE,
)

# Titan timestamp + step (used to derive step time independently of reported tps/tflops).
# Example: "0: [titan] 2026-03-25 13:37:04,721 - root - INFO - step: 90  loss: ..."
TITAN_STEP_TS_RE = re.compile(
    r"^0:.*\[titan\]\s+(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+).*step:\s*(\d+)"
)

# ── config parsing ────────────────────────────────────────────────────────────


def parse_config(name: str, yaml_path: "Path | None" = None) -> dict:
    """Extract config fields from a saved config YAML, falling back to the
    directory name."""
    if yaml_path is not None and yaml_path.exists() and yaml is not None:
        try:
            return _parse_config_from_yaml(name, yaml_path)
        except Exception as e:
            print(f"[WARN] Could not parse {yaml_path}: {e}", file=sys.stderr)
    return _parse_config_from_name(name)


def _parse_config_from_yaml(name: str, yaml_path: Path) -> dict:
    """Read a saved config-{job_id}.yaml and extract parallelism / training
    fields."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    # The file is always wrapped under a top-level "config" key.
    conf = data.get("config", data)
    backend_cfg = conf.get("backend", {})

    cfg: dict = {"name": name}

    # n_gpus from slurm config (nodes × gpus_per_node).
    sbatch = conf.get("slurm", {}).get("sbatch", {})
    nodes = sbatch.get("nodes", 1)
    gpus_per_node = sbatch.get("gpus_per_node", 1)
    cfg["n_gpus"] = nodes * gpus_per_node

    if "titan" in backend_cfg:
        cfg["backend"] = "titan"
        titan = backend_cfg["titan"]
        par = titan.get("parallelism", {})
        cfg["ep"] = par.get("expert_parallel_degree")
        cfg["pp"] = par.get("pipeline_parallel_degree")
        cfg["tp"] = par.get("tensor_parallel_degree")
        cfg["dp_shard"] = par.get("data_parallel_shard_degree")
        cfg["bs"] = titan.get("training", {}).get("local_batch_size")
        cfg["ac"] = titan.get("activation_checkpoint", {}).get("mode")  # "full" / "none"

        cfg["seq_len"] = titan.get("training", {}).get("seq_len")

        # TORCHDYNAMO_DISABLE=1 in the env overrides compile.enable at runtime.
        env = backend_cfg.get("env", {})
        if str(env.get("TORCHDYNAMO_DISABLE", "0")) == "1":
            cfg["compile"] = "no"
        elif titan.get("compile", {}).get("enable"):
            cfg["compile"] = "yes"
        else:
            cfg["compile"] = None

    elif "megatron" in backend_cfg:
        cfg["backend"] = "megatron"
        meg = backend_cfg["megatron"]
        cfg["ep"] = meg.get("expert_model_parallel_size")
        cfg["pp"] = meg.get("pipeline_model_parallel_size")
        cfg["tp"] = meg.get("tensor_model_parallel_size")
        cfg["dp_shard"] = None
        cfg["bs"] = meg.get("micro_batch_size")

        cfg["seq_len"] = meg.get("seq_length")

        granularity = meg.get("recompute_granularity")
        if granularity == "full":
            cfg["ac"] = "full"
        elif granularity == "selective":
            cfg["ac"] = "selective"
        else:
            cfg["ac"] = "none"

        cfg["compile"] = None  # Megatron does not use torch.compile in our configs

    else:
        cfg["backend"] = "unknown"
        cfg["ep"] = cfg["pp"] = cfg["tp"] = cfg["dp_shard"] = cfg["bs"] = cfg["ac"] = cfg[
            "compile"
        ] = None

    return cfg


def _parse_config_from_name(name: str) -> dict:
    """Fallback: extract config fields by pattern-matching the job directory name."""
    cfg = {"name": name, "n_gpus": None, "seq_len": None, "dp_shard": None}

    m = re.search(r"_ep(\d+)", name)
    cfg["ep"] = int(m.group(1)) if m else None

    m = re.search(r"_pp(\d+)", name)
    cfg["pp"] = int(m.group(1)) if m else None

    m = re.search(r"_tp(\d+)", name)
    cfg["tp"] = int(m.group(1)) if m else None

    m = re.search(r"_(?:mbs|bs)(\d+)", name)
    cfg["bs"] = int(m.group(1)) if m else None

    if "_nocompile" in name:
        cfg["compile"] = "no"
    elif "_compile" in name:
        cfg["compile"] = "yes"
    else:
        cfg["compile"] = None

    if "acfull" in name or "full_recompute" in name:
        cfg["ac"] = "full"
    elif "acnone" in name or "no_recompute" in name:
        cfg["ac"] = "none"
    else:
        cfg["ac"] = None

    if name.startswith("titan_"):
        cfg["backend"] = "titan"
    elif name.startswith("meg_"):
        cfg["backend"] = "megatron"
    else:
        cfg["backend"] = "unknown"

    return cfg


# ── log parsing ───────────────────────────────────────────────────────────────


_TITAN_TS_FMT = "%Y-%m-%d %H:%M:%S,%f"


def parse_log(log_path: Path, backend: str, warmup: int) -> dict:
    """Parse a single log file.

    Returns a metrics dict including step_time_ms derived from log
    timestamps (Titan) or the elapsed-time field (Megatron), independent
    of framework TFLOPS.
    """
    tflops, tps, mfu, mem_gib, mem_pct, elapsed_ms = [], [], [], [], [], []
    titan_steps: list[tuple[int, datetime]] = []  # (step, timestamp) for Titan
    oom = False

    pattern = TITAN_RE if backend == "titan" else MEGATRON_RE

    with open(log_path, errors="replace") as f:
        for line in f:
            if OOM_RE.search(line):
                oom = True

            # Collect Titan timestamps for independent step-time derivation.
            if backend == "titan":
                ts_m = TITAN_STEP_TS_RE.match(line)
                if ts_m:
                    step = int(ts_m.group(2))
                    if step > warmup:
                        try:
                            ts = datetime.strptime(ts_m.group(1), _TITAN_TS_FMT)
                            titan_steps.append((step, ts))
                        except ValueError:
                            pass

            m = pattern.search(line)
            if not m:
                continue
            step = int(m.group(1))
            if step <= warmup:
                continue

            if backend == "titan":
                mem_gib.append(float(m.group(2)))
                mem_pct.append(float(m.group(3)))
                tps.append(float(m.group(4).replace(",", "")))
                tflops.append(float(m.group(5)))
                mfu.append(float(m.group(6)))
            else:  # megatron
                elapsed_ms.append(float(m.group(2)))
                mem_pct.append(float(m.group(3)) * 100.0)
                tflops.append(float(m.group(4)))
                tps.append(float(m.group(5)))

    if not tflops:
        return {"status": "oom" if oom else "no_data"}

    # Compute step_time_ms from timestamps / elapsed field.
    step_time_ms: float | None = None
    if backend == "megatron" and elapsed_ms:
        step_time_ms = statistics.median(elapsed_ms)
    elif backend == "titan" and len(titan_steps) >= 2:
        # Use consecutive logged entries; each pair spans (step_b - step_a) steps.
        diffs_ms = []
        for (step_a, t_a), (step_b, t_b) in zip(titan_steps, titan_steps[1:]):
            dt = (t_b - t_a).total_seconds() * 1000  # ms
            n = step_b - step_a
            if n > 0 and dt > 0:
                diffs_ms.append(dt / n)
        if diffs_ms:
            step_time_ms = statistics.median(diffs_ms)

    med_tflops = statistics.median(tflops)
    med_tps = statistics.median(tps) if tps else None
    result = {
        "status": "oom" if oom else "ok",
        "tflops": med_tflops,
        # Both Titan and Megatron report tok/s/GPU directly in logs — use as-is.
        "tok_per_s_per_gpu": med_tps,
        "step_time_ms": step_time_ms,
        "mfu_pct": statistics.median(mfu) if mfu else med_tflops / GH200_PEAK_TFLOPS * 100,
        "mem_gib": statistics.median(mem_gib) if mem_gib else None,
        "mem_pct": statistics.median(mem_pct) if mem_pct else None,
        "n_iters": len(tflops),
    }
    return result


def find_log(job_dir: Path) -> Path | None:
    """Find the most recent slurm log in a job directory."""
    logs = sorted(job_dir.glob("slurm-*.log"), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def find_config_yaml(log: Path) -> "Path | None":
    """Return the config-{job_id}.yaml saved alongside a slurm log, if
    present."""
    m = re.search(r"slurm-(\d+)", log.name)
    if not m:
        return None
    candidate = log.parent / f"config-{m.group(1)}.yaml"
    return candidate if candidate.exists() else None


def _maybe_compute_tok_per_s_per_gpu(cfg: dict, metrics: dict) -> dict:
    """Fallback: compute tok_per_s_per_gpu from step_time if not already set."""
    if metrics.get("tok_per_s_per_gpu") is not None:
        return metrics
    step_time_ms = metrics.get("step_time_ms")
    bs = cfg.get("bs")
    seq_len = cfg.get("seq_len")
    n_gpus = cfg.get("n_gpus")
    if step_time_ms and bs and seq_len and n_gpus and step_time_ms > 0:
        tok_per_s_per_gpu = bs * seq_len / (step_time_ms / 1000.0) / n_gpus
        metrics = {**metrics, "tok_per_s_per_gpu": tok_per_s_per_gpu}
    return metrics


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--log-file",
        nargs="+",
        default=[],
        metavar="FILE",
        help="Log files",
    )
    parser.add_argument(
        "--base-dirs",
        nargs="+",
        metavar="DIR",
        default=[],
        help="Output directories to scan (one or more)",
    )
    parser.add_argument(
        "--warmup", type=int, default=10, help="Iterations to skip at start (default: 10)"
    )
    parser.add_argument(
        "--csv", action="store_true", help="Print CSV instead of a human-readable table"
    )
    parser.add_argument(
        "--sort",
        default="name",
        choices=["tflops", "tok_per_s_per_gpu", "mfu", "name"],
        help="Sort column (default: name)",
    )
    args = parser.parse_args()

    rows = []
    for base in args.base_dirs:
        base_path = Path(base)
        if not base_path.is_dir():
            print(f"[WARN] Not a directory: {base}", file=sys.stderr)
            continue
        for job_dir in sorted(base_path.iterdir()):
            if not job_dir.is_dir():
                continue
            log = find_log(job_dir)
            cfg = parse_config(job_dir.name, find_config_yaml(log) if log else None)
            if log is None:
                metrics = {"status": "no_log"}
            else:
                m = re.search(r"slurm-(\d+)", log.name)
                cfg["job_id"] = m.group(1) if m else ""
                metrics = parse_log(log, cfg["backend"], args.warmup)
            rows.append({**cfg, **_maybe_compute_tok_per_s_per_gpu(cfg, metrics)})

    for fname in args.log_file:
        log = Path(fname)
        cfg = parse_config(log.name, find_config_yaml(log))
        m = re.search(r"slurm-(\d+)", log.name)
        cfg["job_id"] = m.group(1) if m else ""
        metrics = parse_log(log, cfg["backend"], args.warmup)
        rows.append({**cfg, **_maybe_compute_tok_per_s_per_gpu(cfg, metrics)})

    # Sort
    sort_key = {
        "tflops": lambda r: -(r.get("tflops") or 0),
        "tok_per_s_per_gpu": lambda r: -(r.get("tok_per_s_per_gpu") or 0),
        "mfu": lambda r: -(r.get("mfu_pct") or 0),
        "name": lambda r: r["name"],
    }[args.sort]
    rows.sort(key=sort_key)

    if args.csv:
        _print_csv(rows)
    else:
        _print_table(rows)


def _fmt(val, fmt=".1f", suffix="", na="—"):
    if val is None:
        return na
    return f"{val:{fmt}}{suffix}"


def _print_table(rows):
    # Header
    cols = [
        ("backend", 8, "backend"),
        ("ep", 4, "EP"),
        ("pp", 4, "PP"),
        ("compile", 8, "compile"),
        ("ac", 6, "AC"),
        ("bs", 4, "BS"),
        ("n_gpus", 6, "nGPUs"),
        ("status", 8, "status"),
        ("tok_per_s_per_gpu", 14, "tok/s/GPU"),
        ("tflops", 10, "TFLOPS/GPU"),
        ("mfu_pct", 8, "MFU%"),
        ("mem_gib", 10, "Mem(GiB)"),
        ("mem_pct", 8, "Mem%"),
        ("n_iters", 7, "iters"),
        ("job_id", 10, "job_id"),
    ]

    header = "  ".join(f"{h:{w}}" for _, w, h in cols)
    sep = "  ".join("-" * w for w, *_ in [(w,) for _, w, _ in cols])
    print(header)
    print(sep)

    for r in rows:

        def g(k):
            return r.get(k)

        na = "OOM" if g("status") == "oom" else "—"
        vals = [
            f"{g('backend') or '':8}",
            f"{str(g('ep') or ''):4}",
            f"{str(g('pp') or ''):4}",
            f"{g('compile') or '':8}",
            f"{g('ac') or '':6}",
            f"{str(g('bs') or ''):4}",
            f"{str(g('n_gpus') or ''):6}",
            f"{g('status') or '':8}",
            _fmt(g("tok_per_s_per_gpu"), ",.0f", "", na).rjust(14),
            _fmt(g("tflops"), ".1f", "", na).rjust(10),
            _fmt(g("mfu_pct"), ".1f", "%", na).rjust(8),
            _fmt(g("mem_gib"), ".1f", " GiB", "—").rjust(10),
            _fmt(g("mem_pct"), ".1f", "%", "—").rjust(8),
            f"{str(g('n_iters') or ''):7}",
            f"{g('job_id') or '':10}",
        ]
        print("  ".join(vals))


def _print_csv(rows):
    import csv

    fields = [
        "name",
        "backend",
        "ep",
        "pp",
        "compile",
        "ac",
        "bs",
        "seq_len",
        "n_gpus",
        "status",
        "tok_per_s_per_gpu",
        "tflops",
        "mfu_pct",
        "step_time_ms",
        "mem_gib",
        "mem_pct",
        "n_iters",
        "job_id",
    ]
    w = csv.DictWriter(sys.stdout, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)


if __name__ == "__main__":
    main()
