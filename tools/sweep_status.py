#!/usr/bin/env python3
"""
Sweep run status table for multilingual_scaling experiments.

Reads a sweep YAML config, discovers all expected runs under the results directory,
and prints a detailed per-Slurm-job table with training metrics, throughput, GPU
hours, and status.

Usage:
    python sweep_status.py <config.yaml> [options]

    # Override where to look for runs (useful when cluster paths differ from local mount):
    python sweep_status.py config/experiments/multilingual_scaling/0.1B_ne.yaml \\
        --results-dir /home/diana/mn5/multilingual_scaling/0.1B_ne/training

    # Also write a CSV:
    python sweep_status.py config/experiments/multilingual_scaling/0.1B_ne.yaml \\
        --csv status.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import yaml


# ── Regex patterns ────────────────────────────────────────────────────────────

# Megatron stdout "arguments" section: "[default0]:  param_name ..... value"
RE_ARG = re.compile(r"\[default\d+\]:\s{2}(\w+)\s+\.+\s+(.+)")

# Model parameter count lines
RE_TOTAL_PARAMS_B = re.compile(
    r"\[default0\]:Total number of parameters in billions:\s+([\d.]+)"
)
RE_TRANSFORMER_PARAMS_B = re.compile(
    r"\[default0\]:Number of parameters in transformer block in billions:\s+([\d.]+)"
)

# Throughput / iteration line
RE_ITER = re.compile(
    r"iteration\s+(\d+)/\s*(\d+)\s*\|"
    r".*?elapsed time per iteration \(ms\):\s*([\d.]+)"
    r".*?throughput per GPU \(TFLOP/s/GPU\):\s*([\d.]+)"
    r".*?Tokens per second per GPU \(Tok/s/GPU\):\s*([\d.]+)",
    re.DOTALL,
)

# Timestamp embedded in iteration lines: "[2026-05-07 20:32:47]"
RE_TS = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")

# Status detection patterns for stderr
RE_SEGFAULT = re.compile(r"Segmentation fault|signal 11\b", re.IGNORECASE)
RE_OOM = re.compile(r"OutOfMemoryError|CUDA out of memory", re.IGNORECASE)
RE_FATAL = re.compile(r"\bFATAL ERROR\b", re.IGNORECASE)
RE_TIME_LIMIT = re.compile(r"DUE TO TIME LIMIT", re.IGNORECASE)
RE_SIGTERM = re.compile(
    r"SignalException.*?sigval=Signals\.SIGTERM|signal: 15\b", re.IGNORECASE
)
RE_WANDB_SUMMARY = re.compile(
    r"wandb:\s+Run summary:|wandb:\s+Synced\s+\S", re.IGNORECASE
)

# SBATCH directives
RE_SBATCH_NODES = re.compile(r"^#SBATCH\s+--nodes[= ](\d+)", re.MULTILINE)
RE_SBATCH_GPUS_NODE = re.compile(r"^#SBATCH\s+--gpus-per-node[= ](\d+)", re.MULTILINE)
RE_SBATCH_GRES = re.compile(r"^#SBATCH\s+--gres=gpu[^:\n]*:(\d+)", re.MULTILINE)

# Token budget and stage encoded in run name
RE_BUDGET = re.compile(r"_(stable|decay)(\d+BT)$")

TS_FMT = "%Y-%m-%d %H:%M:%S"


# ── Hydra config loading (from validate_sweep_runs.py) ────────────────────────

def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError:
        return {}


def _get_package(path: Path) -> str:
    if not path.exists():
        return "_global_"
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s.startswith("# @package"):
                return s.split("@package", 1)[1].strip()
            if s and not s.startswith("#"):
                break
    return "_global_"


def _fill_missing(base: dict, overlay: dict) -> None:
    for k, v in overlay.items():
        if k not in base:
            base[k] = v
        elif isinstance(base[k], dict) and isinstance(v, dict):
            _fill_missing(base[k], v)


def _find_config_root(config_path: str) -> Path | None:
    for p in Path(config_path).resolve().parents:
        if p.name == "config":
            return p
        candidate = p / "config"
        if candidate.is_dir():
            return candidate
    return None


def _resolve_defaults(cfg: dict, config_path: str) -> None:
    config_root = _find_config_root(config_path)
    if config_root is None:
        return
    for entry in cfg.get("defaults", []):
        if not isinstance(entry, dict):
            continue
        for raw_key, value in entry.items():
            if not isinstance(value, str):
                continue
            key = raw_key.lstrip("/")
            at_target = None
            if "@" in key:
                group, at_target = key.split("@", 1)
                key = group.rstrip("/")
            file_path = config_root / key / f"{value}.yaml"
            sub = _load_yaml(file_path)
            if not sub:
                continue
            package = _get_package(file_path)
            if at_target:
                _fill_missing(cfg.setdefault(at_target, {}), sub)
            elif package == "_global_":
                _fill_missing(cfg, sub)
            else:
                target = cfg
                for part in package.split("."):
                    target = target.setdefault(part, {})
                _fill_missing(target, sub)


def _eval_token_set(s: str) -> set[int]:
    """Evaluate a Python set-literal string like '{6_000_000_000}' or 'set()' into a set of ints."""
    if not s:
        return set()
    try:
        result = eval(s)  # safe: these are only integer set literals from the YAML
        return set(int(x) for x in result) if isinstance(result, (set, frozenset)) else set()
    except Exception:
        return set()


def parse_config(config_path: str) -> dict:
    """Parse sweep config and return all valid (run_name, stage, tokens) combos.

    The sweep uses a per-combo filter: each Group-1 entry carries
    center/cross/diagonal_tokens_set; a decay stage is only valid for a combo
    when its token budget appears in the union of those three sets.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    _resolve_defaults(cfg, config_path)

    meg = cfg["backend"]["megatron"]
    aux = meg["aux"]

    params: dict[str, Any] = {
        "seq_length": int(meg["seq_length"]),
        "seed": int(meg.get("seed", 1234)),
    }

    raw_dir = cfg["job"]["base_output_dir"]
    params["base_dir_template"] = raw_dir.split("${job.name}")[0].rstrip("/")

    groups = cfg["sweep"]["groups"]
    job_name_tpl: str | None = None

    # Pass 1 – collect per-combo data (Group 1 entries) and all decay stage defs (Group 2).
    combos: list[dict] = []          # {lr, gbsz, stable_tokens, valid_decay_tokens: set[int]}
    all_decay_stages: OrderedDict[str, int] = OrderedDict()  # stage_name -> token_budget

    for group in groups:
        if group.get("type") != "list" or "configs" not in group:
            continue
        for entry in group["configs"]:
            if not isinstance(entry, dict):
                continue

            # Group 0: job name template
            if "job.name" in entry:
                job_name_tpl = entry["job.name"]
                continue

            # Group 1: (lr, gbsz) hyperparameter combo entries
            if (
                "backend.megatron.lr" in entry
                and "backend.megatron.global_batch_size" in entry
                and "stage" not in entry
                and entry.get("type") != "product"
                and entry.get("type") != "list"
            ):
                lr = float(entry["backend.megatron.lr"])
                gbsz = int(entry["backend.megatron.global_batch_size"])
                stable_tok = int(entry.get("backend.megatron.aux.tokens", aux["tokens"]))

                center = _eval_token_set(entry.get("backend.megatron.aux.center_tokens_set", "set()"))
                cross = _eval_token_set(entry.get("backend.megatron.aux.cross_tokens_set", "set()"))
                diagonal = _eval_token_set(entry.get("backend.megatron.aux.diagonal_tokens_set", "set()"))
                valid_decay_tokens = center | cross | diagonal

                combos.append({
                    "lr": lr,
                    "gbsz": gbsz,
                    "stable_tokens": stable_tok,
                    "valid_decay_tokens": valid_decay_tokens,
                })
                continue

            # Group 2 stable entry: skip (it just sets train_iters formulas)
            if entry.get("stage") == "stable":
                continue

            # Group 2 nested decay list: collect all possible decay stage names
            if entry.get("type") == "list" and "configs" in entry:
                for dc in entry["configs"]:
                    if isinstance(dc, dict) and "stage" in dc and "backend.megatron.aux.tokens" in dc:
                        stage_name = str(dc["stage"])
                        tok = int(dc["backend.megatron.aux.tokens"])
                        all_decay_stages[stage_name] = tok

    # Build token → stage_name reverse map
    tok_to_stage: dict[int, str] = {tok: name for name, tok in all_decay_stages.items()}

    params["job_name_tpl"] = job_name_tpl
    params["combos"] = combos
    params["all_decay_stages"] = all_decay_stages
    params["tok_to_stage"] = tok_to_stage
    return params


def _subst(template: str, ctx: dict) -> str:
    def _repl(m: re.Match) -> str:
        return str(ctx[m.group(1)]) if m.group(1) in ctx else m.group(0)
    return re.sub(r"\$\{([^}]+)\}", _repl, template)


def render_job_name(
    tpl: str | None,
    nexp: int, lr: float, gbsz: int, seed: int, stage: str,
    stable_tokens: int | None = None,
) -> str:
    if tpl is None:
        return f"nexp_{nexp}_lr{lr}_gbsz{gbsz}_seed{seed}_{stage}"
    suffix = (
        f"{stable_tokens // 1_000_000_000}BT"
        if stage == "stable" and stable_tokens is not None
        else ""
    )
    out = tpl
    for key, val in {
        "backend.megatron.num_experts": nexp,
        "backend.megatron.lr": lr,
        "backend.megatron.global_batch_size": gbsz,
        "backend.megatron.seed": seed,
        "stage": stage,
        "backend.megatron.aux.job_horizon_suffix": suffix,
    }.items():
        out = out.replace(f"\\${{{key}}}", str(val))
    return out


def resolve_base_dir(template: str, nexp: int, lr: float, gbsz: int, seed: int) -> Path:
    ctx = {
        "backend.megatron.global_batch_size": gbsz,
        "backend.megatron.num_experts": nexp,
        "backend.megatron.lr": lr,
        "backend.megatron.seed": seed,
    }
    return Path(_subst(template, ctx))


# ── SBATCH parsing ────────────────────────────────────────────────────────────

def parse_sbatch(path: Path) -> tuple[int | None, int | None]:
    """Return (nodes, gpus_per_node) from a job.sbatch file."""
    if not path.is_file():
        return None, None
    text = path.read_text(errors="replace")
    nodes = gpus = None
    m = RE_SBATCH_NODES.search(text)
    if m:
        nodes = int(m.group(1))
    m = RE_SBATCH_GPUS_NODE.search(text)
    if m:
        gpus = int(m.group(1))
    if gpus is None:
        m = RE_SBATCH_GRES.search(text)
        if m:
            gpus = int(m.group(1))
    return nodes, gpus


# ── Stdout log parsing ────────────────────────────────────────────────────────

def parse_stdout(log_path: Path, max_elapsed_ms: float, skip_first_iters: int, max_iters: int) -> dict:
    """Parse a Megatron stdout log for training params, model size, and throughput."""
    result: dict[str, Any] = {
        "global_batch_size": None,
        "lr": None,
        "micro_batch_size": None,
        "num_workers": None,
        "train_iters": None,
        "total_params_b": None,
        "transformer_params_b": None,
        "last_iter": None,
        "total_iters": None,
        "avg_tflop_per_gpu": None,
        "avg_tok_per_gpu": None,
        "n_iters_sampled": None,
        "first_ts": None,
        "last_ts": None,
    }
    if not log_path.is_file():
        return result

    text = log_path.read_text(errors="replace")

    # --- Training args section ---
    arg_map: dict[str, str] = {}
    for m in RE_ARG.finditer(text):
        arg_map[m.group(1)] = m.group(2).strip()

    def _int(key: str) -> int | None:
        v = arg_map.get(key)
        try:
            return int(v) if v is not None else None
        except (ValueError, TypeError):
            return None

    def _float(key: str) -> float | None:
        v = arg_map.get(key)
        try:
            return float(v) if v is not None else None
        except (ValueError, TypeError):
            return None

    result["global_batch_size"] = _int("global_batch_size")
    result["lr"] = _float("lr")
    result["micro_batch_size"] = _int("micro_batch_size")
    result["num_workers"] = _int("num_workers")
    result["train_iters"] = _int("train_iters")

    # --- Model parameter counts ---
    m = RE_TOTAL_PARAMS_B.search(text)
    if m:
        result["total_params_b"] = float(m.group(1))
    m = RE_TRANSFORMER_PARAMS_B.search(text)
    if m:
        result["transformer_params_b"] = float(m.group(1))

    # --- Throughput from iteration lines ---
    rows: list[tuple[int, int, float, float, float]] = []
    timestamps: list[datetime] = []

    for m in RE_ITER.finditer(text):
        it = int(m.group(1))
        total = int(m.group(2))
        et = float(m.group(3))
        tflop = float(m.group(4))
        tok = float(m.group(5))
        if et <= max_elapsed_ms:
            rows.append((it, total, et, tflop, tok))

    for m in RE_TS.finditer(text):
        try:
            timestamps.append(datetime.strptime(m.group(1), TS_FMT))
        except ValueError:
            pass

    if timestamps:
        result["first_ts"] = timestamps[0]
        result["last_ts"] = timestamps[-1]

    if rows:
        rows.sort(key=lambda x: x[0])
        result["last_iter"] = rows[-1][0]
        result["total_iters"] = rows[-1][1]

        stable = [r for r in rows if r[0] > skip_first_iters]
        if not stable:
            stable = rows[skip_first_iters:] if len(rows) > skip_first_iters else rows
        if len(stable) > max_iters:
            stable = stable[:max_iters]
        if stable:
            result["avg_tflop_per_gpu"] = mean(r[3] for r in stable)
            result["avg_tok_per_gpu"] = mean(r[4] for r in stable)
            result["n_iters_sampled"] = len(stable)

    return result


# ── Stderr log parsing ────────────────────────────────────────────────────────

def parse_stderr(log_path: Path) -> dict:
    """Parse stderr log for error patterns and wandb completion signal."""
    result: dict[str, Any] = {
        "segfault": False,
        "oom": False,
        "fatal": False,
        "time_limit": False,
        "sigterm": False,
        "wandb_synced": False,
        "errors": [],
    }
    if not log_path.is_file():
        return result

    text = log_path.read_text(errors="replace")

    if RE_SEGFAULT.search(text):
        result["segfault"] = True
        result["errors"].append("Segmentation fault")
    if RE_OOM.search(text):
        result["oom"] = True
        result["errors"].append("Out of memory")
    if RE_FATAL.search(text):
        result["fatal"] = True
        result["errors"].append("Fatal error")
    if RE_TIME_LIMIT.search(text):
        result["time_limit"] = True
    if RE_SIGTERM.search(text):
        result["sigterm"] = True
    if RE_WANDB_SUMMARY.search(text):
        result["wandb_synced"] = True

    return result


# ── GPU hours via sacct ───────────────────────────────────────────────────────

def _parse_elapsed(s: str) -> float:
    """Convert sacct elapsed 'D-HH:MM:SS' or 'HH:MM:SS' to hours."""
    if "-" in s:
        days, rest = s.split("-", 1)
        h, mm, sec = rest.split(":")
        return int(days) * 24 + int(h) + int(mm) / 60 + int(sec) / 3600
    h, mm, sec = s.split(":")
    return int(h) + int(mm) / 60 + int(sec) / 3600


def query_sacct(job_ids: list[str]) -> dict[str, dict]:
    """Query sacct for job state, elapsed time, and GPU count. Returns {} on error."""
    if not job_ids:
        return {}
    try:
        result = subprocess.run(
            [
                "sacct", "-j", ",".join(job_ids),
                "--format=JobID,State,Elapsed,AllocTRES%80",
                "--noheader", "--parsable2",
            ],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            return {}
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {}

    info: dict[str, dict] = {}
    for line in result.stdout.splitlines():
        parts = line.split("|")
        if len(parts) < 4:
            continue
        jid_field, state, elapsed, alloc_tres = parts[:4]
        if "." in jid_field:
            continue
        gpus = 0
        m = re.search(r"gres/gpu=(\d+)", alloc_tres)
        if m:
            gpus = int(m.group(1))
        info[jid_field.strip()] = {
            "state": state.strip(),
            "elapsed": elapsed.strip(),
            "gpus": gpus,
        }
    return info


def gpu_hours_from_timestamps(
    first_ts: datetime | None,
    last_ts: datetime | None,
    nodes: int | None,
    gpus_per_node: int | None,
) -> float | None:
    if first_ts is None or last_ts is None:
        return None
    total_gpus = (nodes or 0) * (gpus_per_node or 0)
    if total_gpus == 0:
        return None
    elapsed_h = (last_ts - first_ts).total_seconds() / 3600
    return elapsed_h * total_gpus


# ── Job discovery ─────────────────────────────────────────────────────────────

def find_job_ids(run_dir: Path) -> list[str]:
    """Return all Slurm job IDs found in logs/, sorted ascending."""
    logs_dir = run_dir / "logs"
    if not logs_dir.is_dir():
        return []
    ids: set[str] = set()
    for f in logs_dir.iterdir():
        m = re.match(r"(?:stdout|stderr)-(\d+)\.log$", f.name)
        if m:
            ids.add(m.group(1))
    return sorted(ids)


# ── Token budget from run name ─────────────────────────────────────────────────

def extract_token_budget(run_name: str) -> str:
    m = RE_BUDGET.search(run_name)
    if m:
        return f"{m.group(1)} {m.group(2)[:-2]}"  # strip "BT" suffix
    return ""


# ── Status determination ──────────────────────────────────────────────────────

def determine_status(
    job_id: str,
    all_job_ids: list[str],
    stdout_data: dict,
    stderr_data: dict,
    sacct_info: dict[str, dict],
    is_latest_job: bool,
) -> tuple[str, str, str]:
    """
    Returns (emoji, status_word, error_description).

    Status logic:
    - DONE     : last_iter == train_iters (training completed)
    - TRAINING : latest job, no errors, still iterating
    - FAILED   : hard error (segfault, OOM, fatal)
    - RESTARTED: job ended cleanly (SIGTERM / time-limit) and a newer job exists
    - CANCELLED: job cancelled explicitly
    - QUEUED   : no stdout yet
    - UNKNOWN  : can't determine
    """
    sacct = sacct_info.get(job_id, {})
    sacct_state = sacct.get("state", "")

    errors = list(stderr_data.get("errors", []))
    error_str = "; ".join(errors) if errors else ""

    # DONE: completed training (last iter == train_iters, and wandb synced)
    last_iter = stdout_data.get("last_iter")
    train_iters = stdout_data.get("train_iters")
    if last_iter is not None and train_iters is not None and last_iter >= train_iters:
        return "✅", "DONE", ""
    if stderr_data.get("wandb_synced"):
        return "✅", "DONE", ""

    # Hard errors
    if stderr_data.get("segfault"):
        return "⚠️", "FAILED", "Segmentation fault"
    if stderr_data.get("oom"):
        return "⚠️", "FAILED", "Out of memory (CUDA OOM)"
    if stderr_data.get("fatal"):
        return "⚠️", "FAILED", "Fatal error"

    # sacct-based states (if sacct is available)
    if sacct_state:
        if "CANCEL" in sacct_state:
            return "🚫", "CANCELLED", ""
        if sacct_state in ("COMPLETED",):
            return "✅", "DONE", ""
        if sacct_state in ("FAILED",):
            return "⚠️", "FAILED", error_str or sacct_state
        if sacct_state in ("RUNNING",):
            return "⏳", "TRAINING", ""
        if sacct_state in ("PENDING",):
            return "🕒", "QUEUED", ""
        if sacct_state in ("TIMEOUT",):
            if not is_latest_job:
                return "🔁", "RESTARTED", "Time limit (normal restart)"
            return "⚠️", "TIMEOUT", "Time limit exceeded"

    # No stdout at all → queued or not started
    if stdout_data.get("last_iter") is None and stdout_data.get("train_iters") is None:
        return "🕒", "QUEUED", ""

    # Time limit / SIGTERM with a newer job → normal restart
    if (stderr_data.get("time_limit") or stderr_data.get("sigterm")) and not is_latest_job:
        return "🔁", "RESTARTED", "Time limit (normal restart)"

    # Latest job is still running
    if is_latest_job and last_iter is not None:
        return "⏳", "TRAINING", ""

    # Fell off the end of its time without a successor → suspended or unknown
    return "❓", "UNKNOWN", error_str


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Print a per-Slurm-job status table for a sweep config."
    )
    ap.add_argument("config", help="Path to the sweep config YAML")
    ap.add_argument(
        "--results-dir",
        default=None,
        help=(
            "Directory containing the run subdirectories. "
            "Defaults to the path in the config with '/gpfs/projects/ehpc533' "
            "replaced by '/home/diana/mn5'."
        ),
    )
    ap.add_argument(
        "--prefix-remap",
        default="/gpfs/projects/ehpc533:/home/diana/mn5",
        help="Colon-separated old:new prefix substitution for the config path (default: %(default)s)",
    )
    ap.add_argument(
        "--max-elapsed-ms",
        type=float,
        default=6000.0,
        help="Drop throughput iterations with longer elapsed time (default: %(default)s)",
    )
    ap.add_argument(
        "--skip-first-iters",
        type=int,
        default=50,
        help="Skip first N iterations for throughput average (warmup, default: %(default)s)",
    )
    ap.add_argument(
        "--max-iters",
        type=int,
        default=500,
        help="Maximum iterations to average throughput over (default: %(default)s)",
    )
    ap.add_argument(
        "--csv",
        default=None,
        help="Optional CSV output path",
    )
    args = ap.parse_args()

    os.chdir(Path(__file__).resolve().parent.parent)

    cfg = parse_config(args.config)
    base_dir_template = cfg["base_dir_template"]
    seed = cfg["seed"]
    combos = cfg["combos"]
    tok_to_stage = cfg["tok_to_stage"]

    # Remap cluster path to local mount
    if args.results_dir is None:
        old_prefix, new_prefix = args.prefix_remap.split(":", 1)
        local_template = base_dir_template.replace(old_prefix, new_prefix)
    else:
        local_template = str(args.results_dir)
        local_template = local_template.split("${job.name}")[0].rstrip("/")

    # Build list of (run_name, stage, tokens) applying the per-combo filter.
    # For each combo:
    #   - stable: always 1 run  → ..._stable{max_tokens_BT}BT
    #   - decays: only token budgets in (center | cross | diagonal) token sets
    run_specs: list[tuple[str, str, int]] = []
    for combo in combos:
        lr, gbsz = combo["lr"], combo["gbsz"]
        stable_tok = combo["stable_tokens"]
        valid_decay_toks = combo["valid_decay_tokens"]

        name = render_job_name(cfg["job_name_tpl"], 1, lr, gbsz, seed, "stable", stable_tok)
        run_specs.append((name, "stable", stable_tok))

        # Decay runs: only the token budgets permitted by the filter
        for decay_tok in sorted(valid_decay_toks):
            stage_name = tok_to_stage.get(decay_tok)
            if stage_name is None:
                continue  # token budget not in the sweep decay list
            name = render_job_name(cfg["job_name_tpl"], 1, lr, gbsz, seed, stage_name)
            run_specs.append((name, stage_name, decay_tok))

    # Resolve the local results base directory (template typically has no per-combo vars here)
    sample_ctx: dict[str, Any] = {"backend.megatron.seed": seed}
    if combos:
        sample_ctx.update({
            "backend.megatron.global_batch_size": combos[0]["gbsz"],
            "backend.megatron.num_experts": 1,
            "backend.megatron.lr": combos[0]["lr"],
        })
    resolved_base = Path(_subst(local_template, sample_ctx))
    if not resolved_base.is_dir():
        resolved_base = Path(local_template)

    # Collect all Slurm job IDs across all run dirs to batch-query sacct
    all_job_ids: list[str] = []
    run_job_map: dict[str, list[str]] = {}
    for run_name, _stage, _tok in run_specs:
        run_dir = resolved_base / run_name
        jids = find_job_ids(run_dir)
        run_job_map[run_name] = jids
        all_job_ids.extend(jids)

    sacct_info = query_sacct(all_job_ids)
    sacct_available = bool(sacct_info) or bool(all_job_ids)  # mark as available if sacct returned anything

    # ── Collect rows ────────────────────────────────────────────────────────
    rows: list[dict[str, Any]] = []

    for run_name, stage, tokens in run_specs:
        run_dir = resolved_base / run_name
        token_budget = extract_token_budget(run_name) or f"{stage} {tokens // 1_000_000_000}"

        # Latest checkpoint
        ckpt_file = run_dir / "checkpoints" / "latest_checkpointed_iteration.txt"
        last_ckpt: int | None = None
        if ckpt_file.is_file():
            try:
                last_ckpt = int(ckpt_file.read_text().strip())
            except (ValueError, OSError):
                pass

        # sbatch info
        sbatch_nodes, sbatch_gpus_per_node = parse_sbatch(run_dir / "job.sbatch")
        total_gpus = (sbatch_nodes or 0) * (sbatch_gpus_per_node or 0)

        job_ids = run_job_map.get(run_name, [])

        if not job_ids:
            # No logs at all → likely queued
            rows.append({
                "run_name": run_name,
                "job_id": "",
                "stage": stage,
                "token_budget": token_budget,
                "tokens_b": tokens / 1e9,
                "nodes": sbatch_nodes,
                "transformer_params_b": None,
                "total_params_b": None,
                "global_batch_size": None,
                "lr": None,
                "micro_batch_size": None,
                "num_workers": None,
                "train_iters": None,
                "last_ckpt": last_ckpt,
                "avg_tflop_per_gpu": None,
                "avg_tok_per_gpu": None,
                "n_iters_sampled": None,
                "gpu_hours": None,
                "sacct_state": "",
                "sacct_elapsed": "",
                "status_emoji": "🕒",
                "status_word": "QUEUED",
                "error_desc": "",
            })
            continue

        for job_id in job_ids:
            logs_dir = run_dir / "logs"
            stdout_log = logs_dir / f"stdout-{job_id}.log"
            stderr_log = logs_dir / f"stderr-{job_id}.log"
            is_latest = job_id == job_ids[-1]

            stdout_data = parse_stdout(
                stdout_log,
                args.max_elapsed_ms,
                args.skip_first_iters,
                args.max_iters,
            )
            stderr_data = parse_stderr(stderr_log)

            # GPU hours: prefer sacct, fall back to log timestamps
            sacct_entry = sacct_info.get(job_id, {})
            sacct_state = sacct_entry.get("state", "")
            sacct_elapsed = sacct_entry.get("elapsed", "")
            gpu_hours: float | None = None
            if sacct_elapsed:
                elapsed_h = _parse_elapsed(sacct_elapsed)
                gpus = sacct_entry.get("gpus", 0) or total_gpus
                gpu_hours = elapsed_h * gpus
            else:
                gpu_hours = gpu_hours_from_timestamps(
                    stdout_data.get("first_ts"),
                    stdout_data.get("last_ts"),
                    sbatch_nodes,
                    sbatch_gpus_per_node,
                )

            emoji, status_word, error_desc = determine_status(
                job_id, job_ids, stdout_data, stderr_data, sacct_info, is_latest
            )

            rows.append({
                "run_name": run_name,
                "job_id": job_id,
                "stage": stage,
                "token_budget": token_budget,
                "tokens_b": tokens / 1e9,
                "nodes": sbatch_nodes,
                "transformer_params_b": stdout_data.get("transformer_params_b"),
                "total_params_b": stdout_data.get("total_params_b"),
                "global_batch_size": stdout_data.get("global_batch_size"),
                "lr": stdout_data.get("lr"),
                "micro_batch_size": stdout_data.get("micro_batch_size"),
                "num_workers": stdout_data.get("num_workers"),
                "train_iters": stdout_data.get("train_iters"),
                "last_ckpt": last_ckpt,
                "avg_tflop_per_gpu": stdout_data.get("avg_tflop_per_gpu"),
                "avg_tok_per_gpu": stdout_data.get("avg_tok_per_gpu"),
                "n_iters_sampled": stdout_data.get("n_iters_sampled"),
                "gpu_hours": gpu_hours,
                "sacct_state": sacct_state,
                "sacct_elapsed": sacct_elapsed,
                "status_emoji": emoji,
                "status_word": status_word,
                "error_desc": error_desc,
            })

    # ── Print table ─────────────────────────────────────────────────────────

    def _fmt(v: Any, fmt: str = "") -> str:
        if v is None:
            return "N/A"
        if fmt:
            return format(v, fmt)
        return str(v)

    # Column widths
    W_NAME = max((len(r["run_name"]) for r in rows), default=30) + 1
    W_NAME = min(W_NAME, 60)

    HEADER = (
        f"{'Run':<{W_NAME}} "
        f"{'JobID':>10} "
        f"{'N_ne(B)':>8} "
        f"{'N(B)':>8} "
        f"{'D(B)':>6} "
        f"{'C(10^18)':>8} "
        f"{'stage':>7} "
        f"{'TrainIter':>10} "
        f"{'LastCkpt':>9} "
        f"{'LR':>8} "
        f"{'GBS':>5} "
        f"{'MBS':>4} "
        f"{'Workers':>7} "
        f"{'Nodes':>6} "
        f"{'TFLOP/s/GPU':>12} "
        f"{'Tok/s/GPU':>10} "
        f"{'GPU-h':>10} "
        f"{'St':>2} "
        f"{'Status':>12} "
        f"{'Error':>10}"
    )
    SEP = "─" * len(HEADER)

    print()
    print(f"Config:  {args.config}")
    print(f"Results: {resolved_base}")
    print(SEP)
    print(HEADER)
    print(SEP)

    status_colors = {
        "DONE":      "\033[92m",
        "TRAINING":  "\033[94m",
        "FAILED":    "\033[91m",
        "CANCELLED": "\033[91m",
        "TIMEOUT":   "\033[91m",
        "QUEUED":    "\033[93m",
        "RESTARTED": "\033[90m",
        "UNKNOWN":   "\033[90m",
    }
    RESET = "\033[0m"

    prev_run = None
    for r in rows:
        name_display = r["run_name"] if r["run_name"] != prev_run else ""
        prev_run = r["run_name"]
        if len(name_display) > W_NAME:
            name_display = name_display[: W_NAME - 1] + "…"

        color = status_colors.get(r["status_word"], "")

        tflop_str = f"{r['avg_tflop_per_gpu']:.1f}" if r["avg_tflop_per_gpu"] is not None else "N/A"
        tok_str = f"{r['avg_tok_per_gpu']:.0f}" if r["avg_tok_per_gpu"] is not None else "N/A"
        gpu_h_str = f"{r['gpu_hours']:.1f}" if r["gpu_hours"] is not None else "N/A"
        trans_str = f"{r['transformer_params_b']:.2f}" if r["transformer_params_b"] is not None else "N/A"
        total_str = f"{r['total_params_b']:.2f}" if r["total_params_b"] is not None else "N/A"
        lr_str = f"{r['lr']:.4f}" if r["lr"] is not None else "N/A"
        ckpt_str = str(r["last_ckpt"]) if r["last_ckpt"] is not None else "N/A"
        ti_str = str(r["train_iters"]) if r["train_iters"] is not None else "N/A"
        gbs_str = str(r["global_batch_size"]) if r["global_batch_size"] is not None else "N/A"
        mbs_str = str(r["micro_batch_size"]) if r["micro_batch_size"] is not None else "N/A"
        wkr_str = str(r["num_workers"]) if r["num_workers"] is not None else "N/A"

        error_disp = r["error_desc"][:28] if r["error_desc"] else ""

        d_str = f"{int(r['tokens_b'])}" if r.get("tokens_b") is not None else "N/A"
        if r["transformer_params_b"] is not None and r.get("tokens_b") is not None:
            c_str = f"{6.0 * r['transformer_params_b'] * r['tokens_b']:.2f}"
        else:
            c_str = "N/A"
        m_budget = RE_BUDGET.search(r["run_name"])
        stage_disp = m_budget.group(1) if m_budget else ("stable" if r["stage"] == "stable" else "decay")
        nodes_str = str(r["nodes"]) if r.get("nodes") is not None else "N/A"

        print(
            f"{name_display:<{W_NAME}} "
            f"{r['job_id']:>10} "
            f"{trans_str:>8} "
            f"{total_str:>8} "
            f"{d_str:>6} "
            f"{c_str:>8} "
            f"{stage_disp:>7} "
            f"{ti_str:>10} "
            f"{ckpt_str:>9} "
            f"{lr_str:>8} "
            f"{gbs_str:>5} "
            f"{mbs_str:>4} "
            f"{wkr_str:>7} "
            f"{nodes_str:>6} "
            f"{tflop_str:>12} "
            f"{tok_str:>10} "
            f"{gpu_h_str:>10} "
            f"{r['status_emoji']} "
            f"{color}{r['status_word']:<12}{RESET} "
            f"{error_disp:<30}"
        )

    print(SEP)

    # Summary counts
    from collections import Counter
    counts = Counter(r["status_word"] for r in rows)
    print()
    print("Summary:")
    for status, cnt in sorted(counts.items()):
        print(f"  {status:<14} {cnt}")

    # ── CSV output ──────────────────────────────────────────────────────────
    if args.csv:
        csv_fields = [
            "run_name", "job_id", "stage", "token_budget",
            "transformer_params_b", "total_params_b",
            "global_batch_size", "lr", "micro_batch_size", "num_workers",
            "train_iters", "last_ckpt",
            "avg_tflop_per_gpu", "avg_tok_per_gpu", "n_iters_sampled",
            "gpu_hours", "sacct_state", "sacct_elapsed",
            "status_emoji", "status_word", "error_desc",
        ]
        csv_path = Path(args.csv)
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
            writer.writeheader()
            for r in rows:
                row_out = {k: r.get(k, "") for k in csv_fields}
                # Format floats
                for fk in ("transformer_params_b", "total_params_b", "lr",
                           "avg_tflop_per_gpu", "avg_tok_per_gpu", "gpu_hours"):
                    v = row_out[fk]
                    if v is not None and v != "":
                        row_out[fk] = round(float(v), 4) if fk == "lr" else round(float(v), 2)
                writer.writerow(row_out)
        print(f"\nWrote {len(rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()
