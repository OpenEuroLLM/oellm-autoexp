#!/usr/bin/env python3
"""Sweep run status table for multilingual_scaling experiments.

Reads a sweep YAML config, discovers all expected runs under the results directory,
and prints a detailed per-Slurm-job table with training metrics, throughput, GPU
hours, and status.

Usage:
    python progress_tracker.py <config.yaml> [options]

    # Override where to look for runs (useful when cluster paths differ from local mount):
    python progress_tracker.py config/experiments/multilingual_scaling/0.1B_ne.yaml \\
        --results-dir /home/diana/mn5/multilingual_scaling/0.1B_ne/training

    # Also write a CSV:
    python progress_tracker.py config/experiments/multilingual_scaling/0.1B_ne.yaml \\
        --csv status.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

import yaml

# ── Sibling-module imports ─────────────────────────────────────────────────────
_tools_dir = str(Path(__file__).parent)
_scripts_dir = str(Path(__file__).parent.parent / "scripts")
for _d in (_tools_dir, _scripts_dir):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from gpu_hours import (  # noqa: E402
    gpu_billing_divisor as _gpu_billing_divisor,
    collect_job_ids as _gpu_collect_job_ids,
    query_sacct as _gpu_query_sacct,
    parse_elapsed as _gpu_parse_elapsed,
    merge_requeued as _gpu_merge_requeued,
)
from low_throughput_analysis import analyze_job as _analyze_low_throughput_job  # noqa: E402
from megatron_throughput_from_logs import load_or_compute_throughput as _compute_throughput  # noqa: E402
from megatron_throughput_from_logs import set_cache_root  # noqa: E402
from write_guard import guard_write  # noqa: E402
from validate_sweep_runs import (  # noqa: E402
    _resolve_defaults,
    render_job_name,
    substitute_omegaconf_path_vars as _subst,
)


# ── Regex patterns ────────────────────────────────────────────────────────────

# Megatron stdout "arguments" section.  Two line prefixes occur:
#   "[default0]:  param_name ..... value"   (torchrun, MN5 / Leonardo)
#   " 0:   param_name ..... value"            (srun --label, LUMI)
# The original pattern only accepted the first, which left LR/GBS/MBS blank on LUMI.
RE_ARG = re.compile(
    r"^(?:\[default\d+\]:|\s*\d+:)\s{2,}(\w+)\s+\.{3,}\s+(.+)$", re.MULTILINE
)

# Megatron dataset loading: one line per mix component, e.g.
#   "INFO:megatron.core.datasets.indexed_dataset:Load the _IndexReader from /data/.../nno_Latn.idx"
RE_INDEX_READER = re.compile(r"Load the _IndexReader from (\S+?)\.idx\b")

# data_args_path / save inside a config-{id}.yaml (fallback when no stdout log exists)
RE_CFG_DATA_ARGS = re.compile(r"^\s*data_args_path:\s*(\S+)\s*$", re.MULTILINE)
RE_CFG_SAVE = re.compile(r"^\s*save:\s*(\S+)\s*$", re.MULTILINE)

# Clusters the tracker knows about.  --machine must be one of these: in
# --accumulate mode rows are replaced per machine, so a wrong tag would erase
# another cluster's rows from the shared CSV.
MACHINES = ("LUMI", "LEO", "MN5")

# Cluster a job is attributed to when its log sits in this machine's run dir
# but local sacct does not know it (logs synced from elsewhere) or reports a
# different job (ID collision).  MN5 and Leonardo share a Slurm ID space and
# sync runs to each other; the hybrid runs on LUMI were continued from Leonardo.
DEFAULT_FOREIGN_MACHINE = {"LEO": "MN5", "MN5": "LEO", "LUMI": "LEO"}

# Model-size label in run names, e.g. qwen3_dense_0.1B_ne_lr... -> "0.1B_ne"
RE_SIZE = re.compile(r"(?:^|_)(\d+(?:\.\d+)?B_ne)(?:_|$)")

# First line the launcher writes to stdout: "Launching job <slurm job name>"
RE_LAUNCH = re.compile(r"^Launching job (\S+)", re.MULTILINE)

# Model parameter count lines
RE_TOTAL_PARAMS_B = re.compile(r"\[default0\]:Total number of parameters in billions:\s+([\d.]+)")
RE_TRANSFORMER_PARAMS_B = re.compile(
    r"\[default0\]:Number of parameters in transformer block in billions:\s+([\d.]+)"
)

# Training loss from iteration lines: "lm loss: 3.1416"
RE_TRAIN_LOSS = re.compile(r"iteration\s+\d+/\s*\d+\s*\|[^\n]*lm loss:\s*([\d.eE+\-]+)")

# Validation loss lines: "validation loss at iteration X | lm loss value: Y"
RE_VAL_LOSS = re.compile(r"validation loss at[^\n]*\|[^\n]*lm loss value:\s*([\d.eE+\-]+)")

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

# Timestamp co-located with iteration count: "[2026-05-07 20:32:47] iteration  N/  M"
RE_ITER_TS_NUM = re.compile(
    r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s+iteration\s+(\d+)/\s*(\d+)"
)

# Status detection patterns for stderr
RE_SEGFAULT = re.compile(r"Segmentation fault|signal 11\b", re.IGNORECASE)
RE_OOM = re.compile(r"OutOfMemoryError|CUDA out of memory", re.IGNORECASE)
RE_FATAL = re.compile(r"\bFATAL ERROR\b", re.IGNORECASE)
RE_TIME_LIMIT = re.compile(r"DUE TO TIME LIMIT", re.IGNORECASE)
RE_NODE_FAILURE = re.compile(r"DUE TO NODE FAILURE|Node failure on\b|NODE_FAIL", re.IGNORECASE)
RE_SIGTERM = re.compile(r"SignalException.*?sigval=Signals\.SIGTERM|signal: 15\b", re.IGNORECASE)
RE_WANDB_SUMMARY = re.compile(r"wandb:\s+Run summary:|wandb:\s+Synced\s+\S", re.IGNORECASE)

# SBATCH directives
RE_SBATCH_NODES = re.compile(r"^#SBATCH\s+--nodes[= ](\d+)", re.MULTILINE)
RE_SBATCH_GPUS_NODE = re.compile(r"^#SBATCH\s+--gpus-per-node[= ](\d+)", re.MULTILINE)
RE_SBATCH_GRES = re.compile(r"^#SBATCH\s+--gres=gpu[^:\n]*:(\d+)", re.MULTILINE)
RE_SBATCH_CKPT_STEP = re.compile(r"--ckpt-step\s+(\d+)")

# Token budget and stage encoded in run name
RE_BUDGET = re.compile(r"_(stable|decay)(\d+BT)$")

TS_FMT = "%Y-%m-%d %H:%M:%S"

TIER_SORT_ORDER = {"center": 0, "cross": 1, "diagonal": 2}

# Checkpoint save cadence: save_interval = SAVE_INTERVAL_NUM // global_batch_size
_SAVE_INTERVAL_NUM = 512_000
_DECAY_BUDGETS_TOK = (
    6_000_000_000,
    12_000_000_000,
    20_000_000_000,
    30_000_000_000,
    50_000_000_000,
    80_000_000_000,
    120_000_000_000,
    200_000_000_000,
    300_000_000_000,
)


def _summary_table_sort_key(
    exp_name: str,
    run_tier_map: dict[str, str],
    run_stage_map: dict[str, str],
) -> tuple:
    tier = run_tier_map.get(exp_name, "")
    stage = run_stage_map.get(exp_name, "")
    lr_m = re.search(r"_lr([0-9.e+\-]+)_", exp_name)
    gbsz_m = re.search(r"_gbsz(\d+)_", exp_name)
    tok_m = re.search(r"(?:stable|decay)(\d+)BT", exp_name)
    lr = float(lr_m.group(1)) if lr_m else 0.0
    gbsz = int(gbsz_m.group(1)) if gbsz_m else 0
    tok = int(tok_m.group(1)) if tok_m else 0
    return (
        TIER_SORT_ORDER.get(tier, 99),
        lr,
        gbsz,
        0 if stage == "stable" else 1,
        tok,
    )


def _build_summary_exp_gpu_h(
    run_specs: list[tuple[str, str, int, str]],
    exp_gpu_h: dict[str, float],
    rows: list[dict[str, Any]],
) -> dict[str, float | None]:
    """GPU-h per experiment for the summary table.

    Includes every run from *run_specs*.  Prefer *exp_gpu_h* (sacct via
    gpu_hours.collect_job_ids); fill gaps from per-row GPU-h in *rows*.
    """
    row_gpu_h: dict[str, float] = {}
    for r in rows:
        if r["gpu_hours"] is not None:
            rn = r["run_name"]
            row_gpu_h[rn] = row_gpu_h.get(rn, 0.0) + r["gpu_hours"]

    summary: dict[str, float | None] = {}
    for run_name, _stage, _tok, _tier in run_specs:
        if run_name in exp_gpu_h:
            summary[run_name] = exp_gpu_h[run_name]
        elif run_name in row_gpu_h:
            summary[run_name] = row_gpu_h[run_name]
        else:
            summary[run_name] = None
    return summary


# ── Monitor-state event classification ───────────────────────────────────────

_CLEAN_RESTART_EVENTS = frozenset({"time_limit", "inactive"})
_HARD_ERROR_EVENTS = frozenset(
    {"segmentation_fault", "error", "bus_error", "nan_loss", "connection_failure"}
)
_FINISH_EVENTS = frozenset({"finished_training", "finish"})

_EVENT_ERROR_DESC: dict[str, str] = {
    "segmentation_fault": "Segmentation fault",
    "error": "Exit code 1",
    "bus_error": "Bus error",
    "nan_loss": "NaN loss",
    "connection_failure": "Connection failure",
}

# Seconds after a job's sacct End time within which a monitor event can still
# be attributed to that job (accounts for monitoring-loop processing latency).
_EVENT_MATCH_TOLERANCE_S = 300.0


def _eval_token_set(s: str) -> set[int]:
    """Evaluate a Python set-literal string like '{6_000_000_000}' or 'set()'
    into a set of ints."""
    if not s:
        return set()
    try:
        result = eval(s)  # safe: these are only integer set literals from the YAML
        return {int(x) for x in result} if isinstance(result, (set, frozenset)) else set()
    except Exception:
        return set()


# Billing divisor used where no sacct TRES string is available (sbatch-derived
# GPU counts).  Set from --machine at startup; sacct-derived figures read the
# GPU model directly and do not depend on this.
_MACHINE_BILLING_DIVISOR: float = 1.0

# GPU model implied by each cluster, for the sbatch fallback above.
_MACHINE_GPU_MODEL = {"LUMI": "mi250"}


def _set_machine_billing_divisor(machine: str) -> None:
    global _MACHINE_BILLING_DIVISOR
    _MACHINE_BILLING_DIVISOR = _gpu_billing_divisor(
        _MACHINE_GPU_MODEL.get((machine or "").strip().upper(), "")
    )


def _redirect_output(path: Path, args: argparse.Namespace) -> Path:
    """Map *path* under --cache-dir when set, so nothing is written into the
    results tree.

    The results directory frequently belongs to another user on a shared
    project.  With --cache-dir the tracker treats it as strictly read-only and
    mirrors the absolute path beneath the cache root, so derived artifacts
    (throughput cache, gpu_hours.csv) land in the caller's own space instead.
    """
    root = getattr(args, "cache_dir", None)
    if not root:
        return path
    path = path.resolve()
    out = Path(root).expanduser() / path.relative_to(path.anchor)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _as_bool(v: Any) -> bool:
    """Coerce a YAML scalar to bool, tolerating the string forms OmegaConf
    leaves behind ('true', 'False', ...)."""
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"true", "yes", "1"}
    return bool(v)


def parse_config(config_path: str) -> dict:
    """Parse sweep config and return all valid (run_name, stage, tokens)
    combos.

    The sweep uses a per-combo filter: each Group-1 entry carries
    center/cross/diagonal_tokens_set; a decay stage is only valid for a combo
    when its token budget appears in the union of those three sets.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    _resolve_defaults(cfg, config_path)

    meg = cfg["backend"]["megatron"]
    aux = meg.get("aux", {})

    params: dict[str, Any] = {
        "seq_length": int(meg["seq_length"]),
        "seed": int(meg.get("seed", 1234)),
    }

    raw_dir = cfg["job"]["base_output_dir"]
    params["base_dir_template"] = raw_dir.split("${job.name}")[0].rstrip("/")

    groups = cfg.get("sweep", {}).get("groups", [])
    job_name_tpl: str | None = None

    # Pass 1 – collect per-combo data (Group 1 entries) and all decay stage defs (Group 2).
    combos: list[dict] = []  # {lr, gbsz, stable_tokens, valid_decay_tokens: set[int]}
    all_decay_stages: OrderedDict[str, int] = OrderedDict()  # stage_name -> token_budget

    # Used by the "new format" where stable entries carry an explicit stage name (e.g.
    # "stable12BT") and are immediately followed by their associated decay list.
    _pending_stable_combo: dict | None = None

    for group in groups:
        if group.get("type") != "list" or "configs" not in group:
            continue
        _pending_stable_combo = None  # reset per group
        # A group may carry a `defaults:` mapping supplying keys omitted from
        # its individual entries (e.g. aux.skip_stable_launch in the LUMI v2
        # sweeps).  Entry-level keys win.
        group_defaults = group.get("defaults") or {}
        if not isinstance(group_defaults, dict):
            group_defaults = {}
        for entry in group["configs"]:
            if not isinstance(entry, dict):
                continue
            if group_defaults:
                entry = {**group_defaults, **entry}

            # Group 0: job name template
            if "job.name" in entry:
                job_name_tpl = entry["job.name"]
                continue

            # Group 1 (old format): pure (lr, gbsz) combo entries without 'stage'
            if (
                "backend.megatron.lr" in entry
                and "backend.megatron.global_batch_size" in entry
                and "stage" not in entry
                and entry.get("type") != "product"
                and entry.get("type") != "list"
            ):
                lr = float(entry["backend.megatron.lr"])
                gbsz = int(entry["backend.megatron.global_batch_size"])
                # Not entry.get(key, aux["tokens"]): the default is evaluated
                # eagerly and the merged-grid configs carry tokens only per entry.
                stable_tok = int(
                    entry["backend.megatron.aux.tokens"]
                    if "backend.megatron.aux.tokens" in entry
                    else aux["tokens"]
                )

                center = _eval_token_set(
                    entry.get("backend.megatron.aux.center_tokens_set", "set()")
                )
                cross = _eval_token_set(entry.get("backend.megatron.aux.cross_tokens_set", "set()"))
                diagonal = _eval_token_set(
                    entry.get("backend.megatron.aux.diagonal_tokens_set", "set()")
                )
                valid_decay_tokens = center | cross | diagonal

                combos.append(
                    {
                        "lr": lr,
                        "gbsz": gbsz,
                        "stable_tokens": stable_tok,
                        "valid_decay_tokens": valid_decay_tokens,
                        "center_tokens": center,
                        "cross_tokens": cross,
                        "diagonal_tokens": diagonal,
                        "stable_launch_tier": str(
                            entry.get("backend.megatron.aux.stable_launch_tier", "")
                        ),
                        # Combos whose CSV row is checkpoint_status=reuse_checkpoint
                        # with no stable row of its own: an existing checkpoint
                        # already covers every branch point their decays need, so
                        # the sweep filter never submits a stable for them.  The
                        # stable run is therefore not expected to exist.
                        "skip_stable_launch": _as_bool(
                            entry.get("backend.megatron.aux.skip_stable_launch", False)
                        ),
                        "stable_stage_name": None,  # old format: use "stable" + job_horizon_suffix
                    }
                )
                continue

            # New format: stable phase entry with an explicit stage name (e.g. "stable12BT").
            # The adjacent decay list (next type:list entry) provides valid_decay_tokens.
            _stage_val = entry.get("stage", "")
            if (
                isinstance(_stage_val, str)
                and _stage_val.startswith("stable")
                and "backend.megatron.lr" in entry
                and "backend.megatron.global_batch_size" in entry
                and "backend.megatron.aux.tokens" in entry
                and entry.get("type") not in ("product", "list")
            ):
                _pending_stable_combo = {
                    "lr": float(entry["backend.megatron.lr"]),
                    "gbsz": int(entry["backend.megatron.global_batch_size"]),
                    "stable_tokens": int(entry["backend.megatron.aux.tokens"]),
                    "valid_decay_tokens": set(),
                    "center_tokens": set(),
                    "cross_tokens": set(),
                    "diagonal_tokens": set(),
                    "stable_launch_tier": str(
                        entry.get("backend.megatron.aux.stable_launch_tier", "")
                    ),
                    "skip_stable_launch": _as_bool(
                        entry.get("backend.megatron.aux.skip_stable_launch", False)
                    ),
                    "stable_stage_name": _stage_val,  # new format: pass full name (e.g. "stable12BT")
                }
                continue

            # Group 2 stable entry (old format, stage == "stable"): skip
            if entry.get("stage") == "stable":
                continue

            # Decay list: collect stage names; if preceded by a new-format stable entry,
            # associate these decay tokens with that combo.
            if entry.get("type") == "list" and "configs" in entry:
                _decay_toks: set[int] = set()
                for dc in entry["configs"]:
                    if (
                        isinstance(dc, dict)
                        and "stage" in dc
                        and "backend.megatron.aux.tokens" in dc
                    ):
                        stage_name = str(dc["stage"])
                        tok = int(dc["backend.megatron.aux.tokens"])
                        all_decay_stages[stage_name] = tok
                        _decay_toks.add(tok)
                if _pending_stable_combo is not None:
                    _pending_stable_combo["valid_decay_tokens"] = _decay_toks
                    _pending_stable_combo["center_tokens"] = _decay_toks
                    combos.append(_pending_stable_combo)
                    _pending_stable_combo = None

    # Build token → stage_name reverse map
    tok_to_stage: dict[int, str] = {tok: name for name, tok in all_decay_stages.items()}

    params["job_name_tpl"] = job_name_tpl
    params["job_name"] = cfg.get("job", {}).get("name")
    params["combos"] = combos
    params["all_decay_stages"] = all_decay_stages
    params["tok_to_stage"] = tok_to_stage
    params["adam_beta2"] = float(meg.get("adam_beta2", 0.95))
    params["cooldown_decay_fraction"] = float(aux.get("cooldown_decay_fraction", 0.2))
    # Detect single-run (non-sweep) configs: no sweep groups were found.
    params["is_single_run"] = not combos
    return params


# ── Checkpoint storage ─────────────────────────────────────────────────────────


def _save_interval(gbsz: int) -> int:
    return _SAVE_INTERVAL_NUM // gbsz


def _train_iters(tokens: int, gbsz: int, seq_length: int) -> int:
    return (tokens + seq_length * gbsz - 1) // (seq_length * gbsz)


def _count_checkpoint_iters(checkpoints_dir: Path) -> int:
    if not checkpoints_dir.is_dir():
        return 0
    return sum(1 for p in checkpoints_dir.iterdir() if p.is_dir() and p.name.startswith("iter_"))


def measure_checkpoint_storage_gb(checkpoints_dir: Path, *, timeout_s: int = 180) -> float | None:
    """Return total checkpoint-directory size in GiB, or None if
    unavailable."""
    if not checkpoints_dir.is_dir():
        return None
    try:
        proc = subprocess.run(
            ["du", "-sb", str(checkpoints_dir)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            nbytes = int(proc.stdout.split()[0])
            return nbytes / (1024**3)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    return None


def _expected_stable_checkpoint_iters(
    stable_tokens: int,
    gbsz: int,
    seq_length: int,
    cooldown_decay_fraction: float,
) -> set[int]:
    """Unique iteration indices saved by a stable run at completion."""
    train_iters = _train_iters(stable_tokens, gbsz, seq_length)
    save_int = _save_interval(gbsz)
    iters: set[int] = set()
    step = save_int
    while step <= train_iters:
        iters.add(step)
        step += save_int
    for budget in _DECAY_BUDGETS_TOK:
        if budget > stable_tokens:
            break
        budget_iters = _train_iters(budget, gbsz, seq_length)
        iters.add(int(budget_iters * (1.0 - cooldown_decay_fraction)))
        iters.add(budget_iters)
    return iters


def _expected_decay_checkpoint_iters(
    decay_tokens: int,
    gbsz: int,
    seq_length: int,
    cooldown_decay_fraction: float,
) -> set[int]:
    """Unique iteration indices saved by a decay run at completion."""
    full_iters = _train_iters(decay_tokens, gbsz, seq_length)
    start_iter = int(full_iters * (1.0 - cooldown_decay_fraction))
    save_int = _save_interval(gbsz)
    iters: set[int] = set()
    step = ((start_iter // save_int) + 1) * save_int
    while step <= full_iters:
        iters.add(step)
        step += save_int
    # Short decay phases (< save_interval) still persist a final checkpoint.
    if full_iters > start_iter:
        iters.add(full_iters)
    return iters


def _is_stable_stage(stage: str) -> bool:
    return stage == "stable" or stage.startswith("stable")


def _expected_checkpoint_count(
    stage: str,
    tokens: int,
    gbsz: int,
    seq_length: int,
    cooldown_decay_fraction: float,
) -> int:
    if _is_stable_stage(stage):
        return len(
            _expected_stable_checkpoint_iters(tokens, gbsz, seq_length, cooldown_decay_fraction)
        )
    return len(_expected_decay_checkpoint_iters(tokens, gbsz, seq_length, cooldown_decay_fraction))


def measure_all_checkpoint_storage_gb(
    resolved_base: Path,
    run_names: list[str],
    *,
    max_workers: int = 8,
) -> dict[str, float | None]:
    """Measure checkpoint storage for many runs in parallel (du -sb per
    run)."""
    results: dict[str, float | None] = {}

    def _du_one(name: str) -> tuple[str, float | None]:
        ckpt_dir = resolved_base / name / "checkpoints"
        return name, measure_checkpoint_storage_gb(ckpt_dir)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_du_one, name): name for name in run_names}
        for fut in as_completed(futures):
            name, gb = fut.result()
            results[name] = gb
    return results


def build_run_checkpoint_storage(
    run_specs: list[tuple[str, str, int, str]],
    run_to_gbsz: dict[str, int],
    run_stage_map: dict[str, str],
    resolved_base: Path,
    rows: list[dict[str, Any]],
    seq_length: int,
    cooldown_decay_fraction: float,
) -> dict[str, tuple[float | None, float | None]]:
    """Per-run checkpoint storage in GiB: (measured_gb, remaining_gb).

    measured_gb is the current ``checkpoints/`` size (du -sb) when present.
    remaining_gb is an estimated storage still needed at completion, derived
    from median GiB/checkpoint of finished runs and expected checkpoint count.
    """
    run_names = [name for name, _stage, _tok, _tier in run_specs]
    measured_gb = measure_all_checkpoint_storage_gb(resolved_base, run_names)
    iter_counts = {
        name: _count_checkpoint_iters(resolved_base / name / "checkpoints") for name in run_names
    }

    gb_per_ckpt_samples: list[float] = []
    for run_name, _stage, _tok, _tier in run_specs:
        gb = measured_gb.get(run_name)
        n_ckpt = iter_counts.get(run_name, 0)
        latest = next((r for r in reversed(rows) if r["run_name"] == run_name), None)
        if (
            gb is not None
            and gb > 0
            and n_ckpt > 0
            and latest is not None
            and latest.get("status_word") == "DONE"
        ):
            gb_per_ckpt_samples.append(gb / n_ckpt)

    ref_gb_per_ckpt = median(gb_per_ckpt_samples) if gb_per_ckpt_samples else None

    run_storage: dict[str, tuple[float | None, float | None]] = {}
    for run_name, stage, tokens, _tier in run_specs:
        gb = measured_gb.get(run_name)
        n_ckpt = iter_counts.get(run_name, 0)
        latest = next((r for r in reversed(rows) if r["run_name"] == run_name), None)
        status = latest.get("status_word", "") if latest else ""

        measured_val: float | None = gb if gb is not None and gb > 0 and n_ckpt > 0 else None

        remaining_val: float | None = None
        gbsz = run_to_gbsz.get(run_name)
        if ref_gb_per_ckpt is not None and gbsz is not None:
            exp_n = _expected_checkpoint_count(
                stage, tokens, gbsz, seq_length, cooldown_decay_fraction
            )
            if exp_n > 0:
                if status == "DONE" and measured_val is not None:
                    remaining_val = None
                else:
                    remaining_ckpts = max(0, exp_n - n_ckpt)
                    if remaining_ckpts > 0:
                        remaining_val = ref_gb_per_ckpt * remaining_ckpts

        run_storage[run_name] = (measured_val, remaining_val)

    return run_storage


CKPT_GB_PLACEHOLDER = "--"


def _format_ckpt_measured_cell(
    run_checkpoint_storage: dict[str, tuple[float | None, float | None]],
    exp_name: str,
    *,
    compute: bool,
    md: bool = False,
) -> str:
    if not compute:
        return "" if md else CKPT_GB_PLACEHOLDER
    measured, _remaining = run_checkpoint_storage.get(exp_name, (None, None))
    if measured is None:
        return CKPT_GB_PLACEHOLDER if not md else ""
    return f"{measured:.1f}"


def _format_ckpt_remaining_cell(
    run_checkpoint_storage: dict[str, tuple[float | None, float | None]],
    exp_name: str,
    *,
    compute: bool,
    md: bool = False,
) -> str:
    if not compute:
        return "" if md else CKPT_GB_PLACEHOLDER
    _measured, remaining = run_checkpoint_storage.get(exp_name, (None, None))
    if remaining is None:
        return CKPT_GB_PLACEHOLDER if not md else ""
    return f"{remaining:.1f}"


def _format_ckpt_gb_total(total: float, *, compute: bool, md: bool = False) -> str:
    if not compute:
        return "" if md else CKPT_GB_PLACEHOLDER
    return f"{total:.1f}" if total else CKPT_GB_PLACEHOLDER if not md else ""


# ── SBATCH parsing ────────────────────────────────────────────────────────────


def parse_sbatch(path: Path) -> tuple[int | None, int | None, int | None]:
    """Return (nodes, gpus_per_node, ckpt_step) from a job.sbatch file.

    ckpt_step is the value of --ckpt-step, present only in decay jobs to
    indicate the absolute iteration from which the decay phase starts.
    """
    if not path.is_file():
        return None, None, None
    text = path.read_text(errors="replace")
    nodes = gpus = ckpt_step = None
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
    m = RE_SBATCH_CKPT_STEP.search(text)
    if m:
        ckpt_step = int(m.group(1))
    return nodes, gpus, ckpt_step


# ── Stdout log parsing ────────────────────────────────────────────────────────


def data_root(prefixes) -> str | None:
    """Common root directory of tokenized-data prefixes (``.../name`` without
    ``.bin``/``.idx``).  A single prefix yields its own directory."""
    dirs = {os.path.dirname(p) for p in prefixes if p and p.startswith("/")}
    if not dirs:
        return None
    try:
        return os.path.commonpath(sorted(dirs))
    except ValueError:
        return None


def size_label(run_name: str, results_dir: str | Path | None = None) -> str:
    """Model-size label (``0.1B_ne``) from the run name, else from a
    ``.../<size>/training`` results path; "" when neither carries it."""
    m = RE_SIZE.search(run_name or "")
    if m:
        return m.group(1)
    if results_dir:
        for part in reversed(Path(str(results_dir)).parts):
            m = RE_SIZE.search(part)
            if m:
                return m.group(1)
    return ""


def config_locations(run_dir: Path, job_id: str) -> dict[str, str | None]:
    """data_args_path / save read from ``config-{job_id}.yaml`` (run dir or
    logs/).  Used for jobs without a stdout log (config-only rows) and when the
    log lacks the Megatron arguments block."""
    out: dict[str, str | None] = {"data_mix_path": None, "output_path": None}
    for p in (run_dir / f"config-{job_id}.yaml", run_dir / "logs" / f"config-{job_id}.yaml"):
        if not p.is_file():
            continue
        try:
            text = p.read_text(errors="replace")
        except OSError:
            continue
        m = RE_CFG_DATA_ARGS.search(text)
        if m and m.group(1) not in ("null", "None"):
            out["data_mix_path"] = m.group(1)
        m = RE_CFG_SAVE.search(text)
        if m and m.group(1) not in ("null", "None"):
            out["output_path"] = re.sub(r"/checkpoints$", "", m.group(1).rstrip("/"))
        break
    return out


def parse_stdout(log_path: Path) -> dict:
    """Parse a Megatron stdout log for training params, model size, and
    iteration tracking.

    Throughput metrics (avg_tflop_per_gpu, avg_tok_per_gpu) are NOT
    computed here; call load_or_compute_throughput() separately so
    caching is applied consistently.
    """
    result: dict[str, Any] = {
        "global_batch_size": None,
        "lr": None,
        "micro_batch_size": None,
        "num_workers": None,
        "train_iters": None,
        "total_params_b": None,
        "transformer_params_b": None,
        "first_iter": None,
        "last_iter": None,
        "total_iters": None,
        "first_ts": None,
        "last_ts": None,
        "first_iter_ts": None,
        "last_train_loss": None,
        "last_val_loss": None,
        "data_mix_path": None,
        "data_path": None,
        "output_path": None,
        "launch_name": None,
    }
    if not log_path.is_file():
        return result

    text = log_path.read_text(errors="replace")

    # Slurm job name the log itself was written under (cross-checked against sacct)
    m = RE_LAUNCH.search(text, 0, 4096)
    result["launch_name"] = m.group(1) if m else None

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

    # Megatron printed its arguments: the job got as far as starting training
    result["has_args"] = bool(arg_map)

    result["global_batch_size"] = _int("global_batch_size")
    result["lr"] = _float("lr")
    result["micro_batch_size"] = _int("micro_batch_size")
    result["num_workers"] = _int("num_workers")
    result["train_iters"] = _int("train_iters")

    # --- Data / output locations ---
    def _path_arg(key: str) -> str | None:
        v = arg_map.get(key)
        return v if v and v != "None" else None

    result["data_mix_path"] = _path_arg("data_args_path") or _path_arg("per_split_data_args_path")
    result["data_path"] = data_root(m.group(1) for m in RE_INDEX_READER.finditer(text))
    if result["data_path"] is None:
        # Blend given inline via --data-path "w1 prefix1 w2 prefix2 ..."
        result["data_path"] = data_root(
            re.findall(r"(/[^\s,'\"\]\[]+)", arg_map.get("data_path", ""))
        )
    save = _path_arg("save")
    if save:
        result["output_path"] = re.sub(r"/checkpoints$", "", save.rstrip("/"))

    # --- Model parameter counts ---
    m = RE_TOTAL_PARAMS_B.search(text)
    if m:
        result["total_params_b"] = float(m.group(1))
    m = RE_TRANSFORMER_PARAMS_B.search(text)
    if m:
        result["transformer_params_b"] = float(m.group(1))

    # --- Training and validation losses ---
    train_loss_matches = RE_TRAIN_LOSS.findall(text)
    if train_loss_matches:
        try:
            result["last_train_loss"] = float(train_loss_matches[-1])
        except (ValueError, TypeError):
            pass

    val_loss_matches = RE_VAL_LOSS.findall(text)
    if val_loss_matches:
        try:
            result["last_val_loss"] = float(val_loss_matches[-1])
        except (ValueError, TypeError):
            pass

    # --- Iteration tracking (first/last iter, total iters) ---
    # Throughput averaging is handled by load_or_compute_throughput (with CSV cache).
    iter_rows: list[tuple[int, int]] = []
    timestamps: list[datetime] = []

    for m in RE_ITER.finditer(text):
        iter_rows.append((int(m.group(1)), int(m.group(2))))

    for m in RE_TS.finditer(text):
        try:
            timestamps.append(datetime.strptime(m.group(1), TS_FMT))
        except ValueError:
            pass

    if timestamps:
        result["first_ts"] = timestamps[0]
        result["last_ts"] = timestamps[-1]

    # --- First iteration timestamp (for TTFI) ---
    iter_ts_pairs: list[tuple[int, datetime]] = []
    for m in RE_ITER_TS_NUM.finditer(text):
        try:
            ts = datetime.strptime(m.group(1), TS_FMT)
            it = int(m.group(2))
            iter_ts_pairs.append((it, ts))
        except ValueError:
            pass
    if iter_ts_pairs:
        iter_ts_pairs.sort(key=lambda x: x[0])
        result["first_iter_ts"] = iter_ts_pairs[0][1]

    if iter_rows:
        iter_rows.sort(key=lambda x: x[0])
        result["first_iter"] = iter_rows[0][0]
        result["last_iter"] = iter_rows[-1][0]
        result["total_iters"] = iter_rows[-1][1]

    # Infer submitting user from paths like /gpfs/projects/.../users/<name>/
    m = re.search(r"/users/([^/\s]+)/", text)
    result["log_user"] = m.group(1) if m else None

    return result


# ── Stderr log parsing ────────────────────────────────────────────────────────


def parse_stderr(log_path: Path) -> dict:
    """Parse stderr log for error patterns and wandb completion signal."""
    result: dict[str, Any] = {
        "segfault": False,
        "oom": False,
        "fatal": False,
        "time_limit": False,
        "node_failure": False,
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
    if RE_NODE_FAILURE.search(text):
        result["node_failure"] = True
        result["errors"].append("Node failure")
    if RE_SIGTERM.search(text):
        result["sigterm"] = True
    if RE_WANDB_SUMMARY.search(text):
        result["wandb_synced"] = True

    return result


# ── GPU hours via sacct ───────────────────────────────────────────────────────


def _parse_sacct_ts(s: str) -> float | None:
    """Parse a sacct timestamp string to a Unix timestamp float, or None."""
    if not s or s.strip() in ("Unknown", "None", "N/A", ""):
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s.strip(), fmt).timestamp()
        except ValueError:
            pass
    return None


def query_sacct(job_ids: list[str]) -> dict[str, dict]:
    """Query sacct for job state, elapsed time, GPU count, and start/end
    timestamps.

    Returns {} on error.
    """
    if not job_ids:
        return {}
    try:
        result = subprocess.run(
            [
                "sacct",
                "-j",
                ",".join(job_ids),
                "--format=JobID,State,Elapsed,AllocTRES%80,Start,End,User,JobName",
                # Every instance of requeued jobs, merged per ID below
                # (gpu_hours.merge_requeued explains why).
                "--duplicates",
                "--noheader",
                "--parsable2",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            return {}
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {}

    # Instances per job ID, grouped by job name.  A requeued job keeps its name,
    # so same-name instances are merged; Slurm can also reuse an ID for an
    # unrelated job, which shows up under another name and must stay separate.
    groups: dict[str, dict[str, dict]] = {}
    latest: dict[str, str] = {}
    for line in result.stdout.splitlines():
        parts = line.split("|")
        if len(parts) < 6:
            continue
        jid_field, state, elapsed, alloc_tres, start, end = parts[:6]
        user = parts[6].strip() if len(parts) > 6 else ""
        name = "|".join(parts[7:]).strip()
        if "." in jid_field:
            continue
        gpus = 0
        m = re.search(r"gres/gpu(?::([^=]+))?=(\d+)", alloc_tres)
        _model = ""
        if m:
            _model = (m.group(1) or "").strip().lower()
            gpus = int(m.group(2))
        _start_ts = _parse_sacct_ts(start)
        rec = {
            "state": state.strip(),
            "elapsed": elapsed.strip(),
            # Billed GPUs: LUMI reports GCDs but bills per 2-GCD MI250x module.
            "gpus_billed": gpus / _gpu_billing_divisor(_model),
            "gpu_model": _model,
            "gpus": gpus,
            "start_ts": _start_ts,
            "end_ts": _parse_sacct_ts(end),
            # "None"/"Unknown" for instances that never started
            "start_str": start.strip() if _start_ts is not None else "",
            "user": user,
            "name": name,
        }
        jid = jid_field.strip()
        g = groups.setdefault(jid, {})
        g[name] = _gpu_merge_requeued(g[name], rec) if name in g else rec
        latest[jid] = name
    # Top level: the latest job under each ID; "by_name" keeps every job, for
    # sacct_for_run() to pick the one that belongs to a given run.
    return {jid: {**g[latest[jid]], "by_name": g} for jid, g in groups.items()}


def job_name_matches(name: str | None, run_name: str) -> bool:
    """True if Slurm job *name* belongs to *run_name*: the run name itself, or
    the run name plus the ``_<index>`` suffix some launchers append."""
    if not name:
        return False
    return name == run_name or re.fullmatch(re.escape(run_name) + r"_\d+", name) is not None


def sacct_for_run(sacct_info: dict[str, dict], job_id: str, run_name: str) -> dict:
    """sacct record of *job_id* as seen from *run_name*'s directory.

    When Slurm reused the ID, prefer the job whose name belongs to the run;
    otherwise return the latest job under that ID.
    """
    entry = sacct_info.get(job_id)
    if not entry:
        return {}
    for name, rec in entry.get("by_name", {}).items():
        if job_name_matches(name, run_name):
            return rec
    return entry


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
    # job.sbatch carries no GPU model, so fall back to the divisor implied by
    # --machine (see _set_machine_billing_divisor).  Keeps this fallback in the
    # same billing units as the sacct-derived figures.
    return elapsed_h * total_gpus / _MACHINE_BILLING_DIVISOR


def load_external_gpu_h(
    csv_path: str, machine: str | None = None
) -> dict[tuple[str, str], float]:
    """Load per-job GPU-h from a CSV previously produced by this script (--csv
    output).

    Expects at minimum columns: Run, JobID, GPU-h.
    Returns {(run_name, job_id): gpu_hours}.  Rows with blank JobID are skipped.
    With *machine*, only rows tagged with that Machine are read: the shared
    all-cluster CSV holds job IDs from several Slurm ID spaces.
    """
    result: dict[tuple[str, str], float] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_name = row.get("Run", "").strip()
            job_id = row.get("JobID", "").strip()
            gpu_h_s = row.get("GPU-h", "").strip()
            if not run_name or not job_id or not gpu_h_s:
                continue
            if machine and row.get("Machine", "").strip() not in ("", machine):
                continue
            try:
                result[(run_name, job_id)] = float(gpu_h_s)
            except ValueError:
                pass
    return result


# ── Job discovery ─────────────────────────────────────────────────────────────


def _run_id_sets(run_dir: Path) -> tuple[set[str], set[str]]:
    """Return (config_ids, log_ids) for a run directory.

    config_ids: IDs from config-{id}.yaml (job submitted via pipeline).
    log_ids:    IDs from every log-naming convention, unioned:
                  - logs/stdout-{id}.log, logs/stderr-{id}.log  (MN5/Leonardo:
                    separate SBATCH --output/--error)
                  - logs/slurm-{id}.log                         (LUMI: single
                    combined --output, no --error)
                  - slurm-{id}.log at the run-dir root          (older layout)
                These must be unioned rather than tried in order: a run
                migrated between clusters carries both conventions in the same
                logs/ directory, and preferring one silently drops the jobs
                that used the other.
    A manually restarted run has log_ids that are not in config_ids (jobs
    submitted directly without the config-creation step).
    """
    config_ids: set[str] = set()
    log_ids: set[str] = set()
    if run_dir.is_dir():
        for f in run_dir.iterdir():
            m = re.match(r"config-(\d+)\.yaml$", f.name)
            if m:
                config_ids.add(m.group(1))
            # Combined slurm-{id}.log at the run-dir root
            m = re.match(r"slurm-(\d+)\.log$", f.name)
            if m:
                log_ids.add(m.group(1))
    logs_dir = run_dir / "logs"
    if logs_dir.is_dir():
        for f in logs_dir.iterdir():
            m = re.match(r"(?:stdout|stderr|slurm)-(\d+)\.log$", f.name)
            if m:
                log_ids.add(m.group(1))
            # Single-run configs store config-*.yaml inside logs/
            m = re.match(r"config-(\d+)\.yaml$", f.name)
            if m:
                config_ids.add(m.group(1))
    return config_ids, log_ids


def find_job_ids(run_dir: Path) -> list[str]:
    """Return all Slurm job IDs found in logs/ or config-*.yaml files, sorted
    ascending.

    config-{id}.yaml is written at submission time; stdout/stderr logs
    only appear once the job starts. Including config-based IDs lets us
    detect jobs that are submitted (or queued) but haven't started
    writing logs yet.
    """
    config_ids, log_ids = _run_id_sets(run_dir)
    return sorted(config_ids | log_ids)


# ── Token budget from run name ─────────────────────────────────────────────────


def extract_token_budget(run_name: str) -> str:
    m = RE_BUDGET.search(run_name)
    if m:
        return f"{m.group(1)} {m.group(2)[:-2]}"  # strip "BT" suffix
    return ""


# ── Monitor-state loading ─────────────────────────────────────────────────────


def load_run_monitor_events(run_name: str, monitor_dirs: list[Path]) -> list[dict] | None:
    """Load all monitor-state sessions for *run_name* from *monitor_dirs*.

    Scans each directory for ``{run_name}_*.job.json`` files and extracts the
    ``action_state`` events from their ``runtime`` block.

    Returns a list of session dicts, or None if no matching files exist (signals
    that stderr log parsing should be used as fallback).
    """
    sessions: list[dict] = []
    for d in monitor_dirs:
        for f in sorted(d.glob(f"{run_name}_*.job.json")):
            try:
                data = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            rt = data.get("runtime", {})
            events: list[dict] = []
            for key, val in rt.get("action_state", {}).items():
                parts = key.split(":")
                # key format: "log:<event_name>:<log_events_index>"
                if len(parts) >= 2 and parts[0] == "log":
                    events.append(
                        {
                            "name": parts[1],
                            "ts": val.get("last_action_ts"),
                        }
                    )
            sessions.append(
                {
                    "session_id": d.name,
                    "events": events,
                    "last_status": rt.get("last_status"),
                    "final_state": rt.get("final_state"),
                    "runtime_job_id": rt.get("runtime_job_id"),
                }
            )
    return sessions if sessions else None


def map_events_to_jobs(
    job_ids: list[str],
    sacct_info: dict[str, dict],
    monitor_sessions: list[dict],
) -> dict[str, set[str]]:
    """Assign monitor-state events to specific Slurm job IDs via sacct
    timestamps.

    For each event's ``last_action_ts``, we find the job whose sacct time window
    [Start, End + _EVENT_MATCH_TOLERANCE_S] contains the timestamp.

    Returns ``{job_id: set_of_event_names}``.  Jobs with no sacct start-time data
    get an empty set (caller should treat that as "no evidence either way").
    """
    job_events: dict[str, set[str]] = {jid: set() for jid in job_ids}

    all_events = [
        ev for session in monitor_sessions for ev in session["events"] if ev.get("ts") is not None
    ]
    if not all_events:
        return job_events

    for job_id in job_ids:
        sacct = sacct_info.get(job_id, {})
        start_ts = sacct.get("start_ts")
        end_ts = sacct.get("end_ts")
        if start_ts is None:
            continue
        deadline = (end_ts if end_ts is not None else float("inf")) + _EVENT_MATCH_TOLERANCE_S
        for ev in all_events:
            if start_ts <= ev["ts"] <= deadline:
                job_events[job_id].add(ev["name"])

    return job_events


# ── Status determination ──────────────────────────────────────────────────────


def _compute_restart_action(
    job_id: str,
    job_has_config: bool,
    is_latest_job: bool,
    all_job_ids: list[str],
    run_config_ids: set[str],
    run_log_ids: set[str],
) -> str:
    """Return the restart action for a failed/cancelled job based on what
    followed it.

    Looks at the next job in sequence that actually ran (has logs) and classifies:
      AUTO_RESTARTED:    next running job has a config  (normal autoexp restart)
      MANUALLY_RESTARTED: next running job has no config (manually submitted)
      NEW_SESSION:       current job has no config but next running job has one
                         (manual series ended, new autoexp session started)
      NONE:              latest job, or no subsequent running job found
    """
    if is_latest_job:
        return "NONE"
    # Submitted but never ran — no action applies
    if job_id not in run_log_ids:
        return ""
    try:
        idx = all_job_ids.index(job_id)
    except ValueError:
        return "NONE"
    # Find the next job that actually ran (has a log file)
    next_running: str | None = None
    for later_id in all_job_ids[idx + 1 :]:
        if later_id in run_log_ids:
            next_running = later_id
            break
    if next_running is None:
        return "NONE"
    next_has_config = next_running in run_config_ids
    if next_has_config:
        return "NEW_SESSION" if not job_has_config else "AUTO_RESTARTED"
    return "MANUALLY_RESTARTED"


def determine_status(
    job_id: str,
    all_job_ids: list[str],
    stdout_data: dict,
    stderr_data: dict,
    sacct_info: dict[str, dict],
    is_latest_job: bool,
    job_monitor_events: set[str] | None = None,
    run_config_ids: set[str] | None = None,
    run_log_ids: set[str] | None = None,
) -> tuple[str, str, str, str]:
    """Returns (emoji, status_word, error_description, action_word).

    Status words:  NOT_LAUNCHED | QUEUED | DONE | FAILED | CANCELLED | TRAINING
    Action words:  AUTO_RESTARTED | MANUALLY_RESTARTED | NEW_SESSION | NONE | ""
      - AUTO_RESTARTED:    next running job has a config (normal autoexp restart)
      - MANUALLY_RESTARTED: next running job has no config (manually submitted)
      - NEW_SESSION:       current job was log-only (manual) but next running job has
                           a config (new autoexp session started after manual series)
      - NONE:              latest job or no subsequent running job
      - "":                not applicable (DONE, TRAINING, QUEUED, etc.)

    Error source priority:
      1. *job_monitor_events* (from monitor-state files) when not None.
      2. stderr log patterns (fallback for manually launched runs).
    """
    sacct = sacct_info.get(job_id, {})
    sacct_state = sacct.get("state", "")

    # Pre-compute restart action (used for all FAILED/CANCELLED paths below)
    _cfg_ids = run_config_ids if run_config_ids is not None else set()
    _log_ids = run_log_ids if run_log_ids is not None else set()
    _has_cfg = job_id in _cfg_ids
    _restart = _compute_restart_action(
        job_id, _has_cfg, is_latest_job, all_job_ids, _cfg_ids, _log_ids
    )

    # ── Classify events from monitor state (primary) or stderr (fallback) ────
    use_monitor = job_monitor_events is not None
    if use_monitor:
        triggered_hard = job_monitor_events & _HARD_ERROR_EVENTS
        triggered_clean = job_monitor_events & _CLEAN_RESTART_EVENTS
        triggered_done = job_monitor_events & _FINISH_EVENTS
        hard_error_desc = "; ".join(_EVENT_ERROR_DESC.get(n, n) for n in sorted(triggered_hard))
    else:
        # stderr-derived booleans (legacy path)
        triggered_hard = set()
        triggered_clean = set()
        triggered_done = set()
        hard_error_desc = ""

    # ── DONE ─────────────────────────────────────────────────────────────────
    last_iter = stdout_data.get("last_iter")
    train_iters = stdout_data.get("train_iters")
    if last_iter is not None and train_iters is not None and last_iter >= train_iters:
        return "✅", "DONE", "", ""
    if stderr_data.get("wandb_synced"):
        return "✅", "DONE", "", ""
    if use_monitor and triggered_done:
        return "✅", "DONE", "", ""

    # ── Hard errors ───────────────────────────────────────────────────────────
    if use_monitor:
        if triggered_hard:
            return "⚠️", "FAILED", hard_error_desc, _restart
    else:
        if stderr_data.get("segfault"):
            return "⚠️", "FAILED", "Segmentation fault", _restart
        if stderr_data.get("oom"):
            return "⚠️", "FAILED", "Out of memory (CUDA OOM)", _restart
        if stderr_data.get("fatal"):
            return "⚠️", "FAILED", "Fatal error", _restart
        if stderr_data.get("node_failure"):
            return "⚠️", "FAILED", "Node failure", _restart

    # ── sacct-based states ────────────────────────────────────────────────────
    if sacct_state:
        if "CANCEL" in sacct_state:
            action = _restart if not is_latest_job else ""
            return "🚫", "CANCELLED", "", action
        if sacct_state == "COMPLETED":
            return "✅", "DONE", "", ""
        if sacct_state == "FAILED":
            err = hard_error_desc or "; ".join(stderr_data.get("errors", [])) or "UNKNOWN"
            return "⚠️", "FAILED", err, _restart
        if sacct_state == "RUNNING":
            return "⏳", "TRAINING", "", ""
        if sacct_state == "PENDING":
            return "🕒", "QUEUED", "", ""
        if sacct_state == "TIMEOUT":
            return "⚠️", "FAILED", "Time limit exceeded", _restart
        if sacct_state == "NODE_FAIL":
            return "⚠️", "FAILED", "Node failure", _restart

    # ── No stdout → queued / not started ─────────────────────────────────────
    if stdout_data.get("last_iter") is None and stdout_data.get("train_iters") is None:
        return "🕒", "QUEUED", "", ""

    # ── Clean restart events ──────────────────────────────────────────────────
    if use_monitor:
        if triggered_clean and not is_latest_job:
            return "⚠️", "FAILED", "Time limit (normal restart)", _restart
    else:
        if (stderr_data.get("time_limit") or stderr_data.get("sigterm")) and not is_latest_job:
            return "⚠️", "FAILED", "Time limit (normal restart)", _restart

    # ── Latest job still running ──────────────────────────────────────────────
    if is_latest_job and last_iter is not None:
        return "⏳", "TRAINING", "", ""

    # ── Fallback ──────────────────────────────────────────────────────────────
    error_str = "; ".join(stderr_data.get("errors", []))
    return "⚠️", "FAILED", error_str, _restart


def compute_progress(
    last_iter: int | None, ckpt_step: int | None, total_iters: int | None
) -> float | None:
    """Return training progress [0–100] for a single training phase.

    ckpt_step is the absolute iteration where training started (0 for stable runs,
    --ckpt-step value from job.sbatch for decay runs). Using ckpt_step rather than
    first_iter from the log is correct for decay jobs that span multiple Slurm
    segments: each restart resumes at a different first_iter, but ckpt_step is always
    the fixed decay-phase origin.
    """
    if last_iter is None or total_iters is None:
        return None
    start_iter = ckpt_step if ckpt_step is not None else 0
    denom = total_iters - start_iter
    if denom <= 0:
        return None
    return 100.0 * (last_iter - start_iter) / denom


# ── Main ──────────────────────────────────────────────────────────────────────


def _lw_last_iter(run_dir: Path) -> tuple[int | None, int | None, float | None]:
    """Return (last_iter, total_iters, last_train_loss) by scanning the most recent stdout log
    backwards in chunks until the last iteration line is found."""
    logs_dir = run_dir / "logs"
    # Union every log-naming convention (see _run_id_sets): a run migrated
    # between clusters has both stdout-*.log and slurm-*.log side by side, so
    # preferring one would miss the newest jobs.  Pick by mtime rather than
    # name — job IDs are only monotonic within a single cluster, and a lexical
    # sort would order every "slurm-" before every "stdout-".
    candidates: list[Path] = list(run_dir.glob("slurm-*.log"))
    if logs_dir.is_dir():
        candidates += list(logs_dir.glob("stdout-*.log"))
        candidates += list(logs_dir.glob("slurm-*.log"))
    if not candidates:
        return None, None, None
    log_path = max(candidates, key=lambda p: p.stat().st_mtime)
    try:
        chunk = 262144  # 256 KB per read
        max_read = 8 * 1024 * 1024  # give up after 8 MB
        with open(log_path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            offset = size
            accumulated = b""
            while offset > 0:
                read_size = min(chunk, offset)
                offset -= read_size
                f.seek(offset)
                accumulated = f.read(read_size) + accumulated
                text = accumulated.decode("utf-8", errors="replace")
                matches = list(RE_ITER.finditer(text))
                if matches:
                    m = matches[-1]
                    last_iter = int(m.group(1))
                    total_iters = int(m.group(2))
                    # average of last 10 training losses from the same region
                    loss_matches = list(RE_TRAIN_LOSS.finditer(text))
                    last_loss = (
                        sum(float(m.group(1)) for m in loss_matches[-10:]) / len(loss_matches[-10:])
                        if loss_matches else None
                    )
                    return last_iter, total_iters, last_loss
                if size - offset >= max_read:
                    break
    except OSError:
        pass
    return None, None, None


def _lw_max_checkpoint(run_dir: Path) -> int | None:
    """Return the highest checkpoint iteration saved on disk, or None.

    Handles both plain-digit dirs (e.g. 91553) and iter_XXXXXXX dirs.
    """
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        return None
    iters: list[int] = []
    for d in ckpt_dir.iterdir():
        if not d.is_dir():
            continue
        if d.name.isdigit():
            iters.append(int(d.name))
        elif d.name.startswith("iter_") and d.name[5:].isdigit():
            iters.append(int(d.name[5:]))
    return max(iters) if iters else None


def _squeue_active(job_ids: list[str]) -> set[str]:
    """Return the set of job IDs currently in the SLURM queue (running or pending)."""
    if not job_ids:
        return set()
    try:
        out = subprocess.check_output(
            ["squeue", "--noheader", "-o", "%i", "-j", ",".join(job_ids)],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return {line.strip() for line in out.splitlines() if line.strip()}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return set()


def _lw_status(
    run_dir: Path,
    last_iter: int | None,
    total_iters: int | None,
    active_jobs: set[str],
    latest_job: str | None,
    max_ckpt: int | None,
    has_any_logs: bool,
    has_config: bool,
) -> str:
    """Derive status using pre-computed local signals + squeue for ambiguous cases only."""
    if not run_dir.is_dir():
        return "NOT_LAUNCHED"
    # DONE: checkpoint or iteration count reached total
    if total_iters and max_ckpt is not None and max_ckpt >= total_iters:
        return "DONE"
    if total_iters and last_iter is not None and last_iter >= total_iters:
        return "DONE"
    # QUEUED/TRAINING: only if squeue confirms the job is active
    if latest_job and latest_job in active_jobs:
        return "TRAINING" if (last_iter is not None or max_ckpt is not None) else "QUEUED"
    # Not in squeue, not done — resolve from local evidence
    if not has_any_logs and not has_config:
        return "NOT_LAUNCHED"
    if has_any_logs:
        logs_dir = run_dir / "logs"
        # Error patterns live in stderr-*.log where stdout/stderr are split
        # (MN5/Leonardo) but in the combined slurm-*.log on LUMI, which writes
        # no stderr file at all.  Scan both, newest by mtime, or LUMI failures
        # would all fall through to STOPPED.
        stderr_candidates: list[Path] = list(run_dir.glob("slurm-*.log"))
        if logs_dir.is_dir():
            stderr_candidates += list(logs_dir.glob("stderr-*.log"))
            stderr_candidates += list(logs_dir.glob("slurm-*.log"))
        if stderr_candidates:
            try:
                newest = max(stderr_candidates, key=lambda p: p.stat().st_mtime)
                with open(newest, "rb") as f:
                    f.seek(0, 2)
                    size = f.tell()
                    f.seek(max(0, size - 8192))
                    tail = f.read().decode("utf-8", errors="replace")
                if RE_TIME_LIMIT.search(tail):
                    return "TIMEOUT"
                if RE_NODE_FAILURE.search(tail):
                    return "NODE_FAILURE"
                if RE_OOM.search(tail) or RE_SEGFAULT.search(tail) or RE_FATAL.search(tail):
                    return "FAILED"
            except OSError:
                pass
        return "STOPPED"
    # Config exists but no logs and not in squeue
    return "NOT_LAUNCHED"


def _lightweight_main(args: argparse.Namespace) -> None:
    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)

    cfg = parse_config(args.config)
    base_dir_template = cfg["base_dir_template"]
    seed = cfg["seed"]
    combos = cfg["combos"]
    tok_to_stage = cfg["tok_to_stage"]
    adam_beta2 = cfg.get("adam_beta2", 0.95)

    def _render(stage: str, lr: float, gbsz: int, stable_tok: int | None = None) -> str:
        raw = render_job_name(cfg["job_name_tpl"], 1, lr, gbsz, seed, stage, stable_tok)
        return raw.replace("\\${backend.megatron.adam_beta2}", str(adam_beta2))

    if args.results_dir is None:
        old_prefix, new_prefix = args.prefix_remap.split(":", 1)
        local_template = base_dir_template.replace(old_prefix, new_prefix)
    else:
        local_template = str(args.results_dir).split("${job.name}")[0].rstrip("/")

    run_specs: list[tuple[str, str, int, str]] = []
    run_to_gbsz: dict[str, int] = {}
    for combo in combos:
        lr, gbsz = combo["lr"], combo["gbsz"]
        stable_tok = combo["stable_tokens"]
        stable_stage = combo.get("stable_stage_name") or "stable"
        stable_name = _render(
            stable_stage, lr, gbsz, stable_tok if stable_stage == "stable" else None
        )
        # skip_stable_launch stables are tracked when their run dir exists
        # (finished stable in the merged grids); see main().
        if not combo.get("skip_stable_launch") or (Path(local_template) / stable_name).is_dir():
            run_specs.append((stable_name, "stable", stable_tok, combo["stable_launch_tier"]))
            run_to_gbsz[stable_name] = gbsz
        for decay_tok in sorted(combo["valid_decay_tokens"]):
            stage_name = tok_to_stage.get(decay_tok)
            if stage_name is None:
                continue
            decay_name = _render(stage_name, lr, gbsz)
            if decay_tok in combo["center_tokens"]:
                decay_tier = "center"
            elif decay_tok in combo["cross_tokens"]:
                decay_tier = "cross"
            else:
                decay_tier = "diagonal"
            run_specs.append((decay_name, stage_name, decay_tok, decay_tier))
            run_to_gbsz[decay_name] = gbsz

    if not run_specs and cfg.get("is_single_run") and cfg.get("job_name"):
        run_specs = [(cfg["job_name"], "stable", 0, "")]

    _tier_order = {"center": 0, "cross": 1, "diagonal": 2, "": 3}

    def _lw_sort_key(spec: tuple[str, str, int, str]) -> tuple:
        run_name, stage, tokens, tier = spec
        # Extract lr and gbsz from run_name for stable ordering within tier
        m_lr = re.search(r"lr([\d.]+)", run_name)
        m_gbsz = re.search(r"gbsz(\d+)", run_name)
        lr_val = float(m_lr.group(1)) if m_lr else 0.0
        gbsz_val = int(m_gbsz.group(1)) if m_gbsz else 0
        is_decay = 0 if stage.startswith("stable") else 1
        return (_tier_order.get(tier, 3), lr_val, gbsz_val, is_decay, tokens)

    run_specs.sort(key=_lw_sort_key)

    seq_length = cfg["seq_length"]
    sample_ctx: dict[str, Any] = {"backend.megatron.seed": seed}
    if combos:
        sample_ctx.update(
            {
                "backend.megatron.global_batch_size": combos[0]["gbsz"],
                "backend.megatron.num_experts": 1,
                "backend.megatron.lr": combos[0]["lr"],
            }
        )
    resolved_base = Path(_subst(local_template, sample_ctx))
    if not resolved_base.is_dir():
        resolved_base = Path(local_template)

    status_colors = {
        "DONE": "\033[92m",
        "TRAINING": "\033[94m",
        "STOPPED": "\033[33m",
        "FAILED": "\033[91m",
        "TIMEOUT": "\033[33m",
        "NODE_FAILURE": "\033[33m",
        "QUEUED": "\033[93m",
        "NOT_LAUNCHED": "\033[90m",
    }
    RESET = "\033[0m"

    # First pass: gather local signals (fast, no network)
    run_info: list[tuple[str, str, int, str, Path, str | None, int | None, int | None, int | None, bool, bool]] = []
    needs_squeue: list[str] = []
    for run_name, stage, tokens, tier in run_specs:
        run_dir = resolved_base / run_name
        job_ids = find_job_ids(run_dir)
        latest_job = job_ids[-1] if job_ids else None
        last_iter, total_iters, last_loss = _lw_last_iter(run_dir)
        if total_iters is None and tokens:
            gbsz = run_to_gbsz.get(run_name)
            if gbsz:
                total_iters = _train_iters(tokens, gbsz, seq_length)
        max_ckpt = _lw_max_checkpoint(run_dir)
        config_ids, log_ids = _run_id_sets(run_dir)
        has_any_logs = bool(log_ids)
        # Only query squeue for runs that are ambiguous (not clearly done or not launched)
        clearly_done = bool(
            (total_iters and max_ckpt is not None and max_ckpt >= total_iters)
            or (total_iters and last_iter is not None and last_iter >= total_iters)
        )
        clearly_not_launched = not run_dir.is_dir() or not bool(config_ids or log_ids)
        if latest_job and not clearly_done and not clearly_not_launched:
            needs_squeue.append(latest_job)
        run_info.append((run_name, stage, tokens, tier, run_dir, latest_job, last_iter, total_iters, max_ckpt, has_any_logs, bool(config_ids), last_loss))

    # Single squeue call only for ambiguous jobs
    active_jobs = _squeue_active(needs_squeue)

    rows: list[dict] = []
    for run_name, stage, tokens, tier, run_dir, latest_job, last_iter, total_iters, max_ckpt, has_any_logs, has_config, last_loss in run_info:
        status = _lw_status(run_dir, last_iter, total_iters, active_jobs, latest_job, max_ckpt, has_any_logs, has_config)
        progress = (
            f"{100.0 * last_iter / total_iters:.1f}%" if last_iter and total_iters else "—"
        )
        remaining = (
            f"{100.0 * (total_iters - last_iter) / total_iters:.1f}%"
            if last_iter and total_iters and last_iter < total_iters
            else ("0.0%" if status == "DONE" else "—")
        )
        display_iter = last_iter if last_iter is not None else max_ckpt
        iter_str = (
            f"{display_iter} / {total_iters}" if display_iter is not None and total_iters else
            (f"— / {total_iters}" if total_iters else "—")
        )
        loss_str = f"{last_loss:.4f}" if last_loss is not None else "—"
        rows.append(
            {
                "run_name": run_name,
                "job_id": latest_job or "—",
                "status": status,
                "iter_str": iter_str,
                "progress": progress,
                "remaining": remaining,
                "tier": tier,
                "loss": loss_str,
            }
        )

    W_NAME = max((len(r["run_name"]) for r in rows), default=30)
    W_NAME = min(W_NAME, 60)
    header = (
        f"{'Run':<{W_NAME}}  {'Job ID':>12}  {'Tier':<8}  {'Iter / Total':>22}  {'Done':>6}  {'Remaining':>9}  {'Loss':>8}  Status"
    )
    sep = "─" * len(header)
    print()
    print(f"Config:  {args.config}")
    print(f"Results: {resolved_base}")
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        name = r["run_name"] if len(r["run_name"]) <= W_NAME else r["run_name"][: W_NAME - 1] + "…"
        color = status_colors.get(r["status"], "")
        print(
            f"{name:<{W_NAME}}  {r['job_id']:>12}  {r['tier']:<8}  {r['iter_str']:>22}  "
            f"{r['progress']:>6}  {r['remaining']:>9}  {r['loss']:>8}  {color}{r['status']}{RESET}"
        )
    print(sep)
    done = sum(1 for r in rows if r["status"] == "DONE")
    training = sum(1 for r in rows if r["status"] == "TRAINING")
    queued = sum(1 for r in rows if r["status"] == "QUEUED")
    stopped = sum(1 for r in rows if r["status"] == "STOPPED")
    failed = sum(1 for r in rows if r["status"] in ("FAILED", "TIMEOUT", "NODE_FAILURE"))
    not_launched = sum(1 for r in rows if r["status"] == "NOT_LAUNCHED")
    print(
        f"  {len(rows)} runs total — "
        f"\033[92m{done} done\033[0m  "
        f"\033[94m{training} training\033[0m  "
        f"\033[93m{queued} queued\033[0m  "
        f"\033[33m{stopped} stopped\033[0m  "
        f"\033[91m{failed} failed\033[0m  "
        f"\033[90m{not_launched} not launched\033[0m"
    )
    print()


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
    ap.add_argument(
        "--md",
        default=None,
        help="Optional Markdown table output path",
    )
    ap.add_argument(
        "--monitor-dirs",
        nargs="*",
        default=None,
        metavar="DIR",
        help=(
            "Monitor-state session directories (each containing *.job.json files). "
            "If omitted, auto-discovered from <project_root>/monitor_state/*/ ."
        ),
    )
    ap.add_argument(
        "--compute-storage",
        action="store_true",
        help=(
            "Measure checkpoint storage (du -sb on each run's checkpoints/ directory). "
            "Slow on large trees; without this flag the Ckpt-GB columns are left blank."
        ),
    )
    ap.add_argument(
        "--machine",
        type=str.upper,
        choices=MACHINES,
        default=None,
        help="Cluster this tracker runs on (required unless --lightweight). Jobs "
        "found in local sacct are tagged with it; jobs whose logs are present but "
        "that ran elsewhere are tagged with --foreign-machine.",
    )
    ap.add_argument(
        "--foreign-machine",
        type=str.upper,
        choices=MACHINES,
        default=None,
        help="Cluster to attribute jobs to whose logs sit in the local run dirs "
        "but that local sacct does not know (or reports a different job for). "
        f"Default per --machine: {DEFAULT_FOREIGN_MACHINE}.",
    )
    ap.add_argument(
        "--accumulate",
        action="store_true",
        help=(
            "Treat --csv as the shared all-cluster file: keep rows of other "
            "machines and other model sizes, replace only this machine's rows "
            "for this size, then re-render --md from the merged CSV. Run once "
            "per cluster (git pull first, commit + push after)."
        ),
    )
    ap.add_argument(
        "--size",
        default=None,
        help="Model-size label stored in the Size column (e.g. 0.1B_ne). Default: "
        "parsed from the run names, else from the results dir.",
    )
    ap.add_argument(
        "--lightweight",
        action="store_true",
        help=(
            "Quick summary: show job name, latest job ID, status and progress for each run. "
            "Skips log parsing, sacct, throughput and GPU-h calculation."
        ),
    )
    ap.add_argument(
        "--cache-dir",
        default=None,
        metavar="DIR",
        help=(
            "Treat the results tree as strictly read-only: write every derived "
            "artifact under DIR instead, mirroring each path beneath it. Covers "
            "the per-run throughput cache (<run_dir>/throughput/*.csv) and the "
            "gpu_hours.csv summary, both of which otherwise land in the results "
            "directory. Use this whenever the results belong to another user on "
            "a shared project. Note --csv/--md are always written where you "
            "point them and are unaffected by this flag."
        ),
    )
    args = ap.parse_args()

    if args.cache_dir:
        set_cache_root(args.cache_dir)

    if args.lightweight:
        _set_machine_billing_divisor(args.machine or "")
        _lightweight_main(args)
        return

    if args.machine is None:
        ap.error("--machine {LUMI,LEO,MN5} is required")
    if args.accumulate and not args.csv:
        ap.error("--accumulate needs --csv (the shared progress.csv)")
    _foreign_cluster = args.foreign_machine or DEFAULT_FOREIGN_MACHINE[args.machine]
    if _foreign_cluster == args.machine:
        ap.error("--foreign-machine must differ from --machine")
    _set_machine_billing_divisor(args.machine)

    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)

    # Discover monitor-state directories
    if args.monitor_dirs is None:
        monitor_root = project_root / "monitor_state"
        monitor_dirs: list[Path] = (
            sorted(p for p in monitor_root.iterdir() if p.is_dir()) if monitor_root.is_dir() else []
        )
    else:
        monitor_dirs = [Path(d) for d in args.monitor_dirs]

    cfg = parse_config(args.config)
    base_dir_template = cfg["base_dir_template"]
    seed = cfg["seed"]
    combos = cfg["combos"]
    tok_to_stage = cfg["tok_to_stage"]
    adam_beta2 = cfg.get("adam_beta2", 0.95)

    def _render(stage: str, lr: float, gbsz: int, stable_tok: int | None = None) -> str:
        """render_job_name wrapper that also substitutes adam_beta2 and similar
        extras.

        render_job_name leaves un-substituted keys as '\\${key}'
        (backslash kept), so we must match the same prefix when
        replacing.
        """
        raw = render_job_name(cfg["job_name_tpl"], 1, lr, gbsz, seed, stage, stable_tok)
        return raw.replace("\\${backend.megatron.adam_beta2}", str(adam_beta2))

    # Remap cluster path to local mount
    if args.results_dir is None:
        old_prefix, new_prefix = args.prefix_remap.split(":", 1)
        local_template = base_dir_template.replace(old_prefix, new_prefix)
    else:
        local_template = str(args.results_dir)
        local_template = local_template.split("${job.name}")[0].rstrip("/")

    # Build list of (run_name, stage, tokens) applying the per-combo filter.
    # For each combo:
    #   - stable: always 1 run  → ..._stable{max_tokens_BT}BT  (or ..._stable12BT for new format)
    #   - decays: only token budgets in (center | cross | diagonal) token sets
    run_specs: list[tuple[str, str, int, str]] = []
    decay_to_stable: dict[str, str] = {}  # decay run_name → its paired stable run_name
    run_to_gbsz: dict[str, int] = {}
    no_more_training: set[str] = set()  # skip_stable_launch stables that are tracked
    for combo in combos:
        lr, gbsz = combo["lr"], combo["gbsz"]
        stable_tok = combo["stable_tokens"]
        valid_decay_toks = combo["valid_decay_tokens"]

        # New format: stable_stage_name is the full name (e.g. "stable12BT"); old format: None.
        stable_stage = combo.get("stable_stage_name") or "stable"
        stable_name = _render(
            stable_stage, lr, gbsz, stable_tok if stable_stage == "stable" else None
        )
        # skip_stable_launch: the sweep never submits this stable again.  Either
        # it reuses another run's checkpoint (no run dir of its own), or, in the
        # merged grids, it already finished — then its dir and GPU-h are real
        # and must be tracked.  Decide by whether the run dir exists.  The name
        # is computed either way: decays resolve their checkpoint through
        # decay_to_stable.
        if not combo.get("skip_stable_launch") or (Path(local_template) / stable_name).is_dir():
            run_specs.append((stable_name, "stable", stable_tok, combo["stable_launch_tier"]))
            run_to_gbsz[stable_name] = gbsz
            if combo.get("skip_stable_launch"):
                # Deliberately not trained further: no remaining GPU-h, whatever
                # its iteration count says (stopped early, or a dir that only
                # holds checkpoints transferred from another cluster).
                no_more_training.add(stable_name)

        # Decay runs: only the token budgets permitted by the filter
        for decay_tok in sorted(valid_decay_toks):
            stage_name = tok_to_stage.get(decay_tok)
            if stage_name is None:
                continue  # token budget not in the sweep decay list
            if decay_tok in combo["center_tokens"]:
                tier = "center"
            elif decay_tok in combo["cross_tokens"]:
                tier = "cross"
            else:
                tier = "diagonal"
            name = _render(stage_name, lr, gbsz)
            run_specs.append((name, stage_name, decay_tok, tier))
            decay_to_stable[name] = stable_name
            run_to_gbsz[name] = gbsz

    # Non-sweep config: treat the single job itself as the only run.
    if not run_specs and cfg.get("is_single_run") and cfg.get("job_name"):
        _single_name = cfg["job_name"]
        run_specs = [(_single_name, "stable", 0, "")]
        run_to_gbsz[_single_name] = 0

    # Merged grids can list sibling entries of one (lr, gbsz) that render the
    # same run name; track each run once (first entry wins).
    _seen_specs: set[str] = set()
    run_specs = [
        s for s in run_specs if not (s[0] in _seen_specs or _seen_specs.add(s[0]))
    ]

    run_tier_map: dict[str, str] = {name: tier for name, _stage, _tok, tier in run_specs}
    run_stage_map: dict[str, str] = {name: stage for name, stage, _tok, _tier in run_specs}

    # Resolve the local results base directory (template typically has no per-combo vars here)
    sample_ctx: dict[str, Any] = {"backend.megatron.seed": seed}
    if combos:
        sample_ctx.update(
            {
                "backend.megatron.global_batch_size": combos[0]["gbsz"],
                "backend.megatron.num_experts": 1,
                "backend.megatron.lr": combos[0]["lr"],
            }
        )
    resolved_base = Path(_subst(local_template, sample_ctx))
    if not resolved_base.is_dir():
        resolved_base = Path(local_template)

    # Collect all Slurm job IDs across all run dirs to batch-query sacct
    all_job_ids: list[str] = []
    run_job_map: dict[str, list[str]] = {}
    for run_name, _stage, _tok, _tier in run_specs:
        run_dir = resolved_base / run_name
        jids = find_job_ids(run_dir)
        run_job_map[run_name] = jids
        all_job_ids.extend(jids)

    sacct_info = query_sacct(all_job_ids)
    if args.accumulate and all_job_ids and not sacct_info:
        # Without sacct every job would be tagged --machine, including jobs
        # synced from other clusters, and those rows would then shadow the
        # authoritative rows in the shared CSV.
        sys.exit(
            "sacct returned nothing for this machine's jobs; --accumulate must run "
            f"on {args.machine} itself (login node), not off-cluster."
        )
    # Each run's view of sacct: a reused job ID resolves to the job named after the run.
    run_sacct: dict[str, dict[str, dict]] = {
        rn: {jid: sacct_for_run(sacct_info, jid, rn) for jid in jids}
        for rn, jids in run_job_map.items()
    }

    # If sacct is unavailable and --csv points to an existing file, read back the
    # GPU-h column from that file so the values are preserved when it is rewritten.
    external_job_gpu_h: dict[tuple[str, str], float] = {}
    if not sacct_info and args.csv and Path(args.csv).is_file():
        external_job_gpu_h = load_external_gpu_h(args.csv, machine=args.machine)
        print(
            f"sacct unavailable – loaded {len(external_job_gpu_h)} per-job GPU-h entries from {args.csv}"
        )

    # Load monitor-state events for every run and map them to Slurm job IDs.
    # run_monitor_sessions[run_name] = None  → no monitor state (use stderr fallback)
    # run_monitor_sessions[run_name] = [...]  → sessions found; use action_state events
    run_monitor_sessions: dict[str, list[dict] | None] = {
        run_name: load_run_monitor_events(run_name, monitor_dirs)
        for run_name, _stage, _tok, _tier in run_specs
    }
    # For runs that have monitor state, map each event to the Slurm job whose
    # sacct time window contains the event timestamp.
    run_job_monitor_events: dict[str, dict[str, set[str]]] = {}
    for run_name, _stage, _tok, _tier in run_specs:
        sessions = run_monitor_sessions[run_name]
        if sessions is not None:
            run_job_monitor_events[run_name] = map_events_to_jobs(
                run_job_map.get(run_name, []), run_sacct.get(run_name, {}), sessions
            )

    # ── Collect rows ────────────────────────────────────────────────────────
    rows: list[dict[str, Any]] = []

    for run_name, stage, tokens, tier in run_specs:
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

        # sbatch info; fall back to script/job.sbatch when the primary file lacks GPU directives
        sbatch_nodes, sbatch_gpus_per_node, sbatch_ckpt_step = parse_sbatch(run_dir / "job.sbatch")
        if sbatch_gpus_per_node is None:
            _n2, _g2, _c2 = parse_sbatch(run_dir / "script" / "job.sbatch")
            sbatch_nodes = sbatch_nodes or _n2
            sbatch_gpus_per_node = _g2
            sbatch_ckpt_step = sbatch_ckpt_step or _c2
        total_gpus = (sbatch_nodes or 0) * (sbatch_gpus_per_node or 0)

        # Per-job config/log sets, used by _compute_restart_action
        run_config_ids, run_log_ids = _run_id_sets(run_dir)

        job_ids = run_job_map.get(run_name, [])

        if not job_ids:
            if run_dir.is_dir():
                # Check monitor session: if every session for this run ended as cancelled,
                # show CANCELLED rather than misleading the user into thinking it's queued.
                _sessions = run_monitor_sessions.get(run_name) or []
                _all_cancelled = bool(_sessions) and all(
                    s.get("final_state") == "cancelled" for s in _sessions
                )
                # No job IDs means no config-{id}.yaml and no logs.  A bare
                # directory is not evidence of a submission: transfers create
                # run dirs holding only checkpoints from another cluster, and
                # some decay dirs hold an empty checkpoints/.
                _submitted = (run_dir / "job.sbatch").is_file() or (
                    run_dir / "script" / "job.sbatch"
                ).is_file()
                _has_ckpts = (run_dir / "checkpoints").is_dir() and any(
                    p.name.startswith("iter_") for p in (run_dir / "checkpoints").iterdir()
                )
                if _all_cancelled:
                    s_emoji, s_word = "🚫", "CANCELLED"
                elif _submitted:
                    # Submitted by hand (job.sbatch, no config), not started yet
                    s_emoji, s_word = "🕒", "QUEUED"
                elif _has_ckpts:
                    # Checkpoints copied here; trained on another cluster
                    s_emoji, s_word = "📦", "CKPT_ONLY"
                else:
                    s_emoji, s_word = "⚪", "NOT_LAUNCHED"
            else:
                # Run directory doesn't exist → this tier has not been launched at all
                s_emoji, s_word = "⚪", "NOT_LAUNCHED"
            rows.append(
                {
                    "run_name": run_name,
                    "job_id": "",
                    "stage": stage,
                    "tier": tier,
                    "token_budget": token_budget,
                    "tokens_b": tokens / 1e9,
                    "nodes": sbatch_nodes,
                    "transformer_params_b": None,
                    "total_params_b": None,
                    "global_batch_size": None,
                    "lr": None,
                    "micro_batch_size": None,
                    "num_workers": None,
                    "ttfi_min": None,
                    "ttfi_gpu_h": None,
                    "train_iters": None,
                    "last_iter": None,
                    "ckpt_step": sbatch_ckpt_step,
                    "progress": None,
                    "last_train_loss": None,
                    "last_val_loss": None,
                    "last_ckpt": last_ckpt,
                    "avg_tflop_per_gpu": None,
                    "avg_tok_per_gpu": None,
                    "n_iters_sampled": None,
                    "gpu_hours": None,
                    "time_lost_h": None,
                    "gpu_h_lost": None,
                    "overhead_time_h": None,
                    "overhead_gpu_h": None,
                    "overhead_pct": None,
                    "sacct_state": "",
                    "sacct_elapsed": "",
                    "sacct_start_str": "",
                    "owner": "",
                    # Placeholder rows describe what this machine's sweep plans,
                    # so they belong to the local machine.
                    "cluster": args.machine,
                    "source": "local",
                    "output_path": str(run_dir) if run_dir.is_dir() else "",
                    "data_mix_path": "",
                    "data_path": "",
                    "status_emoji": s_emoji,
                    "status_word": s_word,
                    "action_word": "",
                    "error_desc": "",
                    "has_log": False,
                    "is_collision": False,
                }
            )
            continue

        for job_id in job_ids:
            logs_dir = run_dir / "logs"
            stdout_log = logs_dir / f"stdout-{job_id}.log"
            stderr_log = logs_dir / f"stderr-{job_id}.log"
            # Fallback: combined slurm-{id}.log — check logs/ first, then run dir root
            if not stdout_log.is_file():
                slurm_log = logs_dir / f"slurm-{job_id}.log"
                if not slurm_log.is_file():
                    slurm_log = run_dir / f"slurm-{job_id}.log"
                if slurm_log.is_file():
                    stdout_log = slurm_log
                    stderr_log = slurm_log
            has_log = stdout_log.is_file()
            is_latest = job_id == job_ids[-1]

            stdout_data = parse_stdout(stdout_log)
            _tp = _compute_throughput(
                stdout_log,
                max_elapsed_ms=args.max_elapsed_ms,
                skip_first_iters=args.skip_first_iters,
                max_iters_used=args.max_iters,
            )

            # Eval-only run: no training iteration lines, but args section and
            # checkpoint file confirm the run finished.  Patch stdout_data so
            # progress / status downstream computations see a completed run.
            if (
                stdout_data.get("last_iter") is None
                and stdout_data.get("total_iters") is None
                and stdout_data.get("train_iters") is not None
                and last_ckpt is not None
                and last_ckpt >= stdout_data["train_iters"]
            ):
                stdout_data["last_iter"] = stdout_data["train_iters"]
                stdout_data["total_iters"] = stdout_data["train_iters"]

            stderr_data = parse_stderr(stderr_log)

            # GPU hours: prefer sacct (authoritative), then external CSV
            sacct_entry = run_sacct[run_name].get(job_id, {})
            sacct_state = sacct_entry.get("state", "")
            sacct_elapsed = sacct_entry.get("elapsed", "")
            gpu_hours: float | None = None
            # gpus stays in raw GCD units: it feeds cross-cluster collision
            # detection (which compares sacct GCDs against sbatch GCDs) and the
            # per-GPU throughput normalisation.  GPU-hours, the billed cost, is
            # derived from the billed count instead — LUMI bills per 2-GCD
            # MI250x module, so raw GCD-hours double the real spend.
            gpus = total_gpus
            _gpus_billed = total_gpus / _MACHINE_BILLING_DIVISOR
            if sacct_elapsed:
                elapsed_h = _gpu_parse_elapsed(sacct_elapsed)
                gpus = sacct_entry.get("gpus", 0) or total_gpus
                _gpus_billed = sacct_entry.get("gpus_billed") or (
                    gpus / _MACHINE_BILLING_DIVISOR
                )
                gpu_hours = elapsed_h * _gpus_billed
            elif (run_name, job_id) in external_job_gpu_h:
                gpu_hours = external_job_gpu_h[(run_name, job_id)]

            # Cross-cluster collision detection:
            # MN5 and Leonardo share a numeric Slurm job ID space.  A job ID from
            # MN5 that also exists on Leonardo as a different job returns wrong
            # timestamps and GPU counts.  The job name decides first:
            #   - sacct's job name belongs to this run → a local job of the run,
            #     whatever its GPU count (resumes may use fewer nodes than
            #     job.sbatch declares);
            #   - the log was launched under this run's name but sacct reports
            #     another name → collision.
            # Otherwise (no name evidence) fall back to two GPU heuristics:
            #   1. sacct reports 0 GPUs but sbatch declares GPUs → CPU-only collision
            #   2. sacct reports GPUs but fewer than half of what sbatch declares →
            #      a different user's small job collided (e.g. 1 GPU vs expected 32)
            _sacct_gpus = sacct_entry.get("gpus", 0)
            _sacct_name = sacct_entry.get("name", "")
            _name_is_run = job_name_matches(_sacct_name, run_name)
            _name_conflict = (
                bool(_sacct_name)
                and not _name_is_run
                and job_name_matches(stdout_data.get("launch_name"), run_name)
            )
            _is_collision = (
                bool(sacct_elapsed)
                and not _name_is_run
                and (
                    _name_conflict
                    or (sacct_entry and _sacct_gpus == 0 and total_gpus > 0)
                    or (sacct_entry and total_gpus > 0 and 0 < _sacct_gpus < total_gpus // 2)
                )
            )
            cluster = ""
            source = "local"
            if _is_collision:
                cluster = _foreign_cluster
                source = "inferred"
                sacct_entry = {}
                sacct_state = ""
                sacct_elapsed = ""
                gpu_hours = external_job_gpu_h.get((run_name, job_id))
                gpus = total_gpus
            elif sacct_entry:
                cluster = args.machine
            elif not sacct_info:
                # sacct unavailable altogether: every job is assumed local
                # (GPU-h, if any, came from the previous CSV of this machine).
                cluster = args.machine
            else:
                # sacct works but has never seen this job: its log or its
                # config-{id}.yaml was synced here from the cluster it was
                # submitted on.  (A local submission is known to sacct even
                # while pending; an array task with this raw ID is listed under
                # <array>_<task>, so it does not count as known.)
                cluster = _foreign_cluster
                source = "inferred"

            # Low-throughput analysis (reuses the throughput cache written above).
            # Billed GPU count so gpu_h_lost lands in the same units as gpu_hours.
            _job_lt = _analyze_low_throughput_job(
                stdout_log,
                num_gpus=_gpus_billed or None,
                max_elapsed_ms=args.max_elapsed_ms,
                skip_first_iters=args.skip_first_iters,
                max_iters_used=args.max_iters,
            )

            # TTFI: time from Slurm job start to first iteration in this log.
            # Fallback hierarchy (each fires only when previous conditions fail):
            #   1. sacct start + first_iter_ts  (normal case)
            #   2. sacct start + sacct end, no iter  (sacct available, iter never reached)
            #   3. gpu_hours / total_gpus, no iter  (sacct unavailable; gpu_hours from external CSV)
            ttfi_min: float | None = None
            sacct_start = sacct_entry.get("start_ts")
            sacct_end = sacct_entry.get("end_ts")
            first_iter_ts = stdout_data.get("first_iter_ts")
            # Validate first_iter_ts against the sacct time window: if the log
            # timestamp falls outside [start-60s, end+300s] the log was written by
            # a different job (stale filename or cross-cluster log mismatch).
            if first_iter_ts is not None and sacct_start is not None and sacct_end is not None:
                _ts = first_iter_ts.timestamp()
                if _ts < sacct_start - 60 or _ts > sacct_end + 300:
                    first_iter_ts = None
            if sacct_start is not None and first_iter_ts is not None:
                ttfi_min = (first_iter_ts.timestamp() - sacct_start) / 60.0
            elif sacct_start is not None and first_iter_ts is None and sacct_end is not None:
                ttfi_min = (sacct_end - sacct_start) / 60.0
            elif first_iter_ts is None and gpu_hours is not None and _gpus_billed > 0:
                ttfi_min = (gpu_hours / _gpus_billed) * 60.0
            ttfi_gpu_h: float | None = (
                (ttfi_min / 60.0) * _gpus_billed
                if ttfi_min is not None and _gpus_billed > 0
                else None
            )

            # Sanity check: TTFI cannot physically exceed total job elapsed time.
            # If it does, the timestamps belong to a different job (cross-cluster
            # collision where sacct returned a Leo job with the same ID and GPUs).
            if ttfi_gpu_h is not None and gpu_hours is not None and ttfi_gpu_h > gpu_hours:
                ttfi_min = None
                ttfi_gpu_h = None

            # Total overhead = TTFI + low-throughput (both in hours / GPU-h)
            _lt_time_v = _job_lt.get("time_lost_h")
            _lt_gpu_h_v = _job_lt.get("gpu_h_lost")
            overhead_time_h: float | None = (
                (ttfi_min / 60.0 if ttfi_min is not None else 0.0) + (_lt_time_v or 0.0)
                if ttfi_min is not None or _lt_time_v is not None
                else None
            )
            overhead_gpu_h: float | None = (
                (ttfi_gpu_h or 0.0) + (_lt_gpu_h_v or 0.0)
                if ttfi_gpu_h is not None or _lt_gpu_h_v is not None
                else None
            )
            # Sanity check: overhead cannot exceed total job GPU-h.  If it does,
            # the LowTP analysis read a longer log than the actual job duration
            # (cross-cluster log collision, same job ID sharing a log file).
            if overhead_gpu_h is not None and gpu_hours is not None and overhead_gpu_h > gpu_hours:
                overhead_time_h = None
                overhead_gpu_h = None
            overhead_pct: float | None = (
                overhead_gpu_h / gpu_hours * 100.0
                if overhead_gpu_h is not None and gpu_hours and gpu_hours > 0
                else None
            )

            # Per-job monitor events: set[str] if monitor state exists, else None
            job_events_map = run_job_monitor_events.get(run_name)
            job_monitor_events = (
                job_events_map.get(job_id)  # may be empty set
                if job_events_map is not None
                else None  # no monitor state → use stderr
            )

            emoji, status_word, error_desc, action_word = determine_status(
                job_id,
                job_ids,
                stdout_data,
                stderr_data,
                {job_id: sacct_entry},
                is_latest,
                job_monitor_events=job_monitor_events,
                run_config_ids=run_config_ids,
                run_log_ids=run_log_ids,
            )

            _progress = compute_progress(
                stdout_data.get("last_iter"),
                sbatch_ckpt_step,
                stdout_data.get("total_iters"),
            )
            if _progress is not None and _progress >= 100.0 and status_word != "DONE":
                emoji, status_word, error_desc, action_word = "✅", "DONE", "", ""

            # Data / output locations: prefer the Megatron args in the log, fall
            # back to the submission config (jobs that never wrote a log).
            _cfg_loc = (
                config_locations(run_dir, job_id)
                if not (stdout_data.get("data_mix_path") and stdout_data.get("output_path"))
                else {"data_mix_path": None, "output_path": None}
            )

            rows.append(
                {
                    "run_name": run_name,
                    "job_id": job_id,
                    "stage": stage,
                    "token_budget": token_budget,
                    "tokens_b": tokens / 1e9,
                    "tier": tier,
                    "nodes": sbatch_nodes,
                    "transformer_params_b": stdout_data.get("transformer_params_b"),
                    "total_params_b": stdout_data.get("total_params_b"),
                    "global_batch_size": stdout_data.get("global_batch_size"),
                    "lr": stdout_data.get("lr"),
                    "micro_batch_size": stdout_data.get("micro_batch_size"),
                    "num_workers": stdout_data.get("num_workers"),
                    "ttfi_min": ttfi_min,
                    "ttfi_gpu_h": ttfi_gpu_h,
                    "train_iters": stdout_data.get("train_iters"),
                    "last_iter": stdout_data.get("last_iter"),
                    "ckpt_step": sbatch_ckpt_step,
                    "progress": _progress,
                    "last_train_loss": stdout_data.get("last_train_loss"),
                    "last_val_loss": stdout_data.get("last_val_loss"),
                    "last_ckpt": last_ckpt,
                    "avg_tflop_per_gpu": _tp["avg_tflop_per_gpu"] if _tp else None,
                    "avg_tok_per_gpu": _tp["avg_tok_per_gpu"] if _tp else None,
                    "n_iters_sampled": _tp["n_iters"] if _tp else None,
                    "gpu_hours": gpu_hours,
                    "time_lost_h": _job_lt.get("time_lost_h"),
                    "gpu_h_lost": _job_lt.get("gpu_h_lost"),
                    "overhead_time_h": overhead_time_h,
                    "overhead_gpu_h": overhead_gpu_h,
                    "overhead_pct": overhead_pct,
                    "sacct_state": sacct_state,
                    "sacct_elapsed": sacct_elapsed,
                    "sacct_start_str": sacct_entry.get("start_str", ""),
                    # sacct user for local jobs; fall back to path-based inference
                    # from log for foreign-cluster jobs (sacct entry is discarded).
                    "owner": sacct_entry.get("user", "") or stdout_data.get("log_user", "") or "",
                    # Foreign jobs have no sacct start: fall back to the first log
                    # timestamp so jobs can still be ordered across clusters.
                    "start_str": sacct_entry.get("start_str", "")
                    or (
                        stdout_data["first_ts"].strftime("%Y-%m-%dT%H:%M:%S")
                        if stdout_data.get("first_ts") is not None
                        else ""
                    ),
                    "cluster": cluster,
                    "source": source,
                    "output_path": stdout_data.get("output_path")
                    or _cfg_loc["output_path"]
                    or str(run_dir),
                    "data_mix_path": stdout_data.get("data_mix_path")
                    or _cfg_loc["data_mix_path"]
                    or "",
                    "data_path": stdout_data.get("data_path") or "",
                    "status_emoji": emoji,
                    "status_word": status_word,
                    "action_word": action_word,
                    "error_desc": error_desc,
                    "has_log": has_log,
                    "is_collision": _is_collision,
                    "log_has_args": bool(stdout_data.get("has_args")),
                    # Local job that ended without ever starting (sacct has no
                    # start time and the job is no longer pending)
                    "never_started": bool(sacct_entry)
                    and sacct_entry.get("start_ts") is None
                    and not sacct_state.startswith(("PENDING", "REQUEUED")),
                }
            )

    # ── Fill-forward static fields for eval-only runs ───────────────────────
    # Fields that are constant per run but may be absent from an eval-only log
    # (Megatron may not print the full args/param-count block in eval mode).
    _FILL_FIELDS = [
        "transformer_params_b",
        "total_params_b",
        "global_batch_size",
        "lr",
        "micro_batch_size",
        "num_workers",
        "train_iters",
        "avg_tflop_per_gpu",
        "avg_tok_per_gpu",
        "last_train_loss",
    ]
    _run_last_known: dict[str, dict[str, Any]] = {}
    # Data root comes from dataset-loading lines; a restart that resumed without
    # them (or an eval-only job) inherits it from the previous job of the run on
    # the same cluster.
    _data_known: dict[tuple[str, str], str] = {}
    for r in rows:
        key = (r["run_name"], r.get("cluster", ""))
        if r.get("data_path"):
            _data_known[key] = r["data_path"]
        elif r.get("has_log") and key in _data_known:
            r["data_path"] = _data_known[key]
    for r in rows:
        rn = r["run_name"]
        known = _run_last_known.setdefault(rn, {})
        for fld in _FILL_FIELDS:
            # Only inject into rows that actually ran — config-only rows (no log
            # file) must stay blank so CANCELLED/QUEUED stubs don't inherit the
            # previous job's loss or throughput values.
            if r.get(fld) is None and fld in known and r.get("has_log"):
                r[fld] = known[fld]
            if r.get(fld) is not None:
                known[fld] = r[fld]
        # Track last known iter/progress for QUEUED fill-forward below.
        if r.get("last_iter") is not None:
            known["last_iter"] = r["last_iter"]
        if r.get("progress") is not None:
            known["progress"] = r["progress"]
        # After filling train_iters, recompute progress if last_iter is still None
        # but the checkpoint confirms training completed.
        if r.get("last_iter") is None and r.get("last_ckpt") is not None:
            ti = r.get("train_iters")
            if ti is not None and r["last_ckpt"] >= ti:
                r["last_iter"] = ti
                r["progress"] = 100.0

    # For QUEUED rows with no progress, carry forward last_iter/progress from the
    # previous job of the same run: a queued job hasn't started yet so its progress
    # is the same as where the previous job left off.
    _run_queued_known: dict[str, dict[str, Any]] = {}
    for r in rows:
        rn = r["run_name"]
        known = _run_queued_known.setdefault(rn, {})
        if r["status_word"] == "QUEUED" and r.get("last_iter") is None:
            if "last_iter" in known:
                r["last_iter"] = known["last_iter"]
            if "progress" in known:
                r["progress"] = known["progress"]
        else:
            if r.get("last_iter") is not None:
                known["last_iter"] = r["last_iter"]
            if r.get("progress") is not None:
                known["progress"] = r["progress"]

    # Drop config-only rows that never produced a log and carry no useful status:
    # - CANCELLED stubs with no log are always noise.
    # - Collision-detected rows have an unreliable status (the sacct data belongs
    #   to a different job on the local cluster); keep them only when the synced
    #   log itself shows a training run.  The cluster the job ran on reports it.
    # - Jobs that ended without ever starting (no log, no sacct start) ran nothing.
    def _keep_row(r: dict[str, Any]) -> bool:
        if r.get("is_collision"):
            return bool(r.get("has_log") and r.get("log_has_args"))
        if r.get("has_log"):
            return True
        return r["status_word"] != "CANCELLED" and not r.get("never_started")

    _kept_rows = [r for r in rows if _keep_row(r)]
    # A run whose every job was dropped above must not vanish: the CSV (and the
    # Markdown rendered from it) only knows runs that have a row.  Keep one
    # job-less CANCELLED placeholder for it instead.
    _kept_runs = {r["run_name"] for r in _kept_rows}
    _placeholders: dict[str, dict[str, Any]] = {}
    for r in rows:
        if r["run_name"] not in _kept_runs and r["run_name"] not in _placeholders:
            _placeholders[r["run_name"]] = {
                **r,
                "job_id": "",
                "gpu_hours": None,
                "ttfi_min": None,
                "ttfi_gpu_h": None,
                "overhead_time_h": None,
                "overhead_gpu_h": None,
                "overhead_pct": None,
                "sacct_state": "",
                "sacct_elapsed": "",
                "sacct_start_str": "",
                "start_str": "",
                "cluster": args.machine,
                "source": "local",
                "data_path": "",
                "status_emoji": "🚫",
                "status_word": "CANCELLED",
                "action_word": "",
                "error_desc": "",
            }
    rows = _kept_rows + list(_placeholders.values())

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
        f"{'Tier':>8} "
        f"{'Stage':>7} "
        f"{'TotIter':>9} "
        f"{'CurIter':>9} "
        f"{'LastCkpt':>9} "
        f"{'Prog%':>6} "
        f"{'TrnLoss':>8} "
        f"{'ValLoss':>8} "
        f"{'LR':>8} "
        f"{'GBS':>5} "
        f"{'MBS':>4} "
        f"{'Nodes':>6} "
        f"{'Workers':>7} "
        f"{'TFLOP/s/GPU':>12} "
        f"{'Tok/s/GPU':>10} "
        f"{'TTFI(min)':>9} "
        f"{'TTFI-GPU-h':>10} "
        f"{'LowTP-time(h)':>14} "
        f"{'LowTP-GPU-h':>12} "
        f"{'Overhead-time(h)':>16} "
        f"{'Overhead-GPU-h':>14} "
        f"{'GPU-h':>10} "
        f"{'Overhead%':>9} "
        f"{'Cluster':>7} "
        f"{'Emoji':>2} "
        f"{'Status':>12} "
        f"{'Action':>14} "
        f"{'Error':>10}"
    )
    SEP = "─" * len(HEADER)

    print()
    print(f"Config:  {args.config}")
    print(f"Results: {resolved_base}")
    print(SEP)
    print(HEADER)
    print(SEP)

    _RESTART_ACTIONS = {"AUTO_RESTARTED", "MANUALLY_RESTARTED", "NEW_SESSION"}
    status_colors = {
        "DONE": "\033[92m",
        "TRAINING": "\033[94m",
        "FAILED": "\033[91m",
        "FAILED+AUTO_RESTARTED": "\033[33m",
        "FAILED+MANUALLY_RESTARTED": "\033[35m",
        "FAILED+NEW_SESSION": "\033[36m",
        "CANCELLED": "\033[91m",
        "CANCELLED+MANUALLY_RESTARTED": "\033[35m",
        "CANCELLED+NEW_SESSION": "\033[36m",
        "QUEUED": "\033[93m",
        "CKPT_ONLY": "\033[90m",
        "NOT_LAUNCHED": "\033[90m",
    }
    RESET = "\033[0m"

    prev_run = None
    for r in rows:
        name_display = r["run_name"] if r["run_name"] != prev_run else ""
        prev_run = r["run_name"]
        if len(name_display) > W_NAME:
            name_display = name_display[: W_NAME - 1] + "…"

        if r["action_word"] in _RESTART_ACTIONS and r["status_word"] in ("FAILED", "CANCELLED"):
            color_key = f"{r['status_word']}+{r['action_word']}"
        else:
            color_key = r["status_word"]
        color = status_colors.get(color_key, "")

        tflop_str = f"{r['avg_tflop_per_gpu']:.1f}" if r["avg_tflop_per_gpu"] is not None else "N/A"
        tok_str = f"{r['avg_tok_per_gpu']:.0f}" if r["avg_tok_per_gpu"] is not None else "N/A"
        gpu_h_str = f"{r['gpu_hours']:.1f}" if r["gpu_hours"] is not None else "N/A"
        lt_time_str = f"{r['time_lost_h']:.3f}" if r.get("time_lost_h") is not None else "N/A"
        lt_gpu_h_str = f"{r['gpu_h_lost']:.2f}" if r.get("gpu_h_lost") is not None else "N/A"
        trans_str = (
            f"{r['transformer_params_b']:.2f}" if r["transformer_params_b"] is not None else "N/A"
        )
        total_str = f"{r['total_params_b']:.2f}" if r["total_params_b"] is not None else "N/A"
        lr_str = f"{r['lr']:.4f}" if r["lr"] is not None else "N/A"
        ckpt_str = str(r["last_ckpt"]) if r["last_ckpt"] is not None else "N/A"
        ti_str = str(r["train_iters"]) if r["train_iters"] is not None else "N/A"
        ci_str = str(r["last_iter"]) if r.get("last_iter") is not None else "N/A"
        prog_str = f"{r['progress']:.1f}%" if r.get("progress") is not None else "N/A"
        trn_loss_str = (
            f"{r['last_train_loss']:.4f}" if r.get("last_train_loss") is not None else "N/A"
        )
        val_loss_str = f"{r['last_val_loss']:.4f}" if r.get("last_val_loss") is not None else "N/A"
        gbs_str = str(r["global_batch_size"]) if r["global_batch_size"] is not None else "N/A"
        mbs_str = str(r["micro_batch_size"]) if r["micro_batch_size"] is not None else "N/A"
        wkr_str = str(r["num_workers"]) if r["num_workers"] is not None else "N/A"
        ttfi_str = f"{r['ttfi_min']:.1f}" if r.get("ttfi_min") is not None else "N/A"
        ttfi_gpu_h_str = f"{r['ttfi_gpu_h']:.2f}" if r.get("ttfi_gpu_h") is not None else "N/A"
        oh_time_str = (
            f"{r['overhead_time_h']:.3f}" if r.get("overhead_time_h") is not None else "N/A"
        )
        oh_gpu_h_str = (
            f"{r['overhead_gpu_h']:.2f}" if r.get("overhead_gpu_h") is not None else "N/A"
        )
        oh_pct_str = f"{r['overhead_pct']:.1f}%" if r.get("overhead_pct") is not None else "N/A"

        error_disp = r["error_desc"][:28] if r["error_desc"] else ""

        d_str = f"{int(r['tokens_b'])}" if r.get("tokens_b") is not None else "N/A"
        if r["transformer_params_b"] is not None and r.get("tokens_b") is not None:
            c_str = f"{6.0 * r['transformer_params_b'] * r['tokens_b']:.2f}"
        else:
            c_str = "N/A"
        m_budget = RE_BUDGET.search(r["run_name"])
        stage_disp = (
            m_budget.group(1) if m_budget else ("stable" if r["stage"] == "stable" else "decay")
        )
        nodes_str = str(r["nodes"]) if r.get("nodes") is not None else "N/A"

        print(
            f"{name_display:<{W_NAME}} "
            f"{r['job_id']:>10} "
            f"{trans_str:>8} "
            f"{total_str:>8} "
            f"{d_str:>6} "
            f"{c_str:>8} "
            f"{r['tier']:>8} "
            f"{stage_disp:>7} "
            f"{ti_str:>9} "
            f"{ci_str:>9} "
            f"{ckpt_str:>9} "
            f"{prog_str:>6} "
            f"{trn_loss_str:>8} "
            f"{val_loss_str:>8} "
            f"{lr_str:>8} "
            f"{gbs_str:>5} "
            f"{mbs_str:>4} "
            f"{nodes_str:>6} "
            f"{wkr_str:>7} "
            f"{tflop_str:>12} "
            f"{tok_str:>10} "
            f"{ttfi_str:>9} "
            f"{ttfi_gpu_h_str:>10} "
            f"{lt_time_str:>14} "
            f"{lt_gpu_h_str:>12} "
            f"{oh_time_str:>16} "
            f"{oh_gpu_h_str:>14} "
            f"{gpu_h_str:>10} "
            f"{oh_pct_str:>9} "
            f"{r.get('cluster', ''):>7} "
            f"{'🔁' if r['action_word'] == 'AUTO_RESTARTED' else '🔄' if r['action_word'] == 'MANUALLY_RESTARTED' else '🆕' if r['action_word'] == 'NEW_SESSION' else ''}{r['status_emoji']} "
            f"{color}{r['status_word']:<12}{RESET} "
            f"{r['action_word']:<14} "
            f"{error_disp:<30}"
        )

    print(SEP)

    # ── GPU hours CSV (always written, same format as gpu_hours.py) ─────────
    exp_gpu_h: dict[str, float] = {}
    grand_gpu_total = 0.0
    gpu_csv_rows: list[dict] = []

    if _gpu_collect_job_ids is not None:
        try:
            gpu_exp_jobs = _gpu_collect_job_ids(str(resolved_base))
            all_gpu_jids = [jid for jids in gpu_exp_jobs.values() for jid in jids]
            gpu_sacct_data = _gpu_query_sacct(all_gpu_jids) if all_gpu_jids else {}
            for exp_name, job_ids in sorted(gpu_exp_jobs.items()):
                for job_id in job_ids:
                    if job_id not in gpu_sacct_data:
                        continue
                    d = gpu_sacct_data[job_id]
                    gpu_h = _gpu_parse_elapsed(d["elapsed"]) * d.get("gpus_billed", d["gpus"])
                    exp_gpu_h[exp_name] = exp_gpu_h.get(exp_name, 0.0) + gpu_h
                    grand_gpu_total += gpu_h
                    gpu_csv_rows.append(
                        {
                            "experiment": exp_name,
                            "job_id": job_id,
                            "state": d["state"],
                            "elapsed": d["elapsed"],
                            "gpus": d["gpus"],
                            "gpu_hours": round(gpu_h, 1),
                        }
                    )
        except SystemExit:
            pass

    # sacct unavailable but external GPU-h CSV provided: aggregate from rows
    if not exp_gpu_h and external_job_gpu_h:
        for r in rows:
            if r["gpu_hours"] is not None:
                rn = r["run_name"]
                exp_gpu_h[rn] = exp_gpu_h.get(rn, 0.0) + r["gpu_hours"]
                grand_gpu_total += r["gpu_hours"]

    # Infer GPU-h for foreign-cluster jobs from log data (sacct on this machine
    # returns 0 for jobs that ran on the other cluster).
    # GPU-h = efficient training GPU-h (from throughput) + LowTP GPU-h (from logs).
    # LowTP is log-based and valid; efficient GPU-h covers the remaining training time.
    # (_foreign_cluster is resolved from --foreign-machine / --machine at startup.)
    _run_prev_last_iter: dict[str, int] = {}
    for r in rows:
        rn = r["run_name"]
        if r.get("gpu_hours") is None and r.get("cluster") == _foreign_cluster:
            avg_tp = r.get("avg_tok_per_gpu")
            tok_b = r.get("tokens_b")
            tr_iters = r.get("train_iters")
            last_it = r.get("last_iter")
            if avg_tp and avg_tp > 0 and tr_iters and tr_iters > 0 and tok_b and last_it:
                tokens_per_iter = tok_b * 1e9 / tr_iters
                start_iter = _run_prev_last_iter.get(rn, 0)
                job_tokens = (last_it - start_iter) * tokens_per_iter
                if job_tokens > 0:
                    efficient_gpu_h = job_tokens / (avg_tp * 3600)
                    lowtp_gpu_h = r.get("gpu_h_lost") or 0.0
                    r["gpu_hours"] = efficient_gpu_h + lowtp_gpu_h
        if r.get("cluster") == _foreign_cluster and r.get("last_iter") is not None:
            _run_prev_last_iter[rn] = r["last_iter"]

    # For foreign-cluster rows: TTFI requires sacct job-start time so always null it.
    # LowTP is log-based and kept. Recompute overhead without TTFI component.
    # For rows still without gpu_hours (couldn't infer), null overhead% but keep LowTP.
    # Also null TTFI/overhead for undetected foreign jobs (gpu_hours=None) in foreign runs.
    _foreign_run_names = {r["run_name"] for r in rows if r.get("cluster") == _foreign_cluster}
    for r in rows:
        is_foreign_row = r.get("cluster") == _foreign_cluster
        is_undetected_foreign = r["run_name"] in _foreign_run_names and r.get("gpu_hours") is None
        if is_foreign_row or is_undetected_foreign:
            # TTFI always unknown without sacct start time
            r["ttfi_min"] = None
            r["ttfi_gpu_h"] = None
            if r.get("gpu_hours") is not None:
                # Recompute overhead as LowTP only (no TTFI)
                ltp = r.get("gpu_h_lost") or 0.0
                r["overhead_gpu_h"] = ltp if ltp > 0 else None
                r["overhead_time_h"] = r.get("time_lost_h")
                r["overhead_pct"] = (ltp / r["gpu_hours"] * 100) if ltp > 0 else None
            else:
                r["overhead_time_h"] = None
                r["overhead_gpu_h"] = None
                r["overhead_pct"] = None

    # Add inferred foreign-cluster GPU-h into exp_gpu_h so the summary table
    # reflects real usage (sacct returned 0 for these jobs).
    for r in rows:
        if r.get("cluster") == _foreign_cluster and r.get("gpu_hours") is not None:
            rn = r["run_name"]
            exp_gpu_h[rn] = exp_gpu_h.get(rn, 0.0) + r["gpu_hours"]
            grand_gpu_total += r["gpu_hours"]

    summary_gpu_h = _build_summary_exp_gpu_h(run_specs, exp_gpu_h, rows)
    summary_grand_total = sum(h for h in summary_gpu_h.values() if h is not None)

    # Per-run GPU-h split by cluster, aggregated from per-job rows.
    _run_gpu_h_leo: dict[str, float] = {}
    _run_gpu_h_mn5: dict[str, float] = {}
    _foreign_job_seen = False
    for _r in rows:
        _rn = _r["run_name"]
        _cl = _r.get("cluster", "")
        if _cl == _foreign_cluster:
            _foreign_job_seen = True
            if _r.get("gpu_hours") is not None:
                _run_gpu_h_mn5[_rn] = _run_gpu_h_mn5.get(_rn, 0.0) + _r["gpu_hours"]
        elif _cl and _r.get("gpu_hours") is not None:
            _run_gpu_h_leo[_rn] = _run_gpu_h_leo.get(_rn, 0.0) + _r["gpu_hours"]
    _any_cluster_split = _foreign_job_seen

    # ── Checkpoint storage (GiB) ──────────────────────────────────────────────
    run_checkpoint_storage: dict[str, tuple[float | None, float | None]] = {}
    if args.compute_storage:
        print("\nMeasuring checkpoint storage (du -sb on checkpoints/ per run)…", flush=True)
        run_checkpoint_storage = build_run_checkpoint_storage(
            run_specs,
            run_to_gbsz,
            run_stage_map,
            resolved_base,
            rows,
            cfg["seq_length"],
            cfg.get("cooldown_decay_fraction", 0.2),
        )

    # ── Remaining GPU-h estimation ────────────────────────────────────────────
    # Per-run throughput: mean of avg_tok_per_gpu across all jobs for that run.
    # Each per-job value is already filtered (max_elapsed_ms / skip_first_iters /
    # max_iters in load_or_compute_throughput), so the mean is robust to outliers.
    _run_tp_vals: dict[str, list[float]] = {}
    for r in rows:
        if r.get("avg_tok_per_gpu") is not None:
            _run_tp_vals.setdefault(r["run_name"], []).append(r["avg_tok_per_gpu"])
    _run_avg_tok: dict[str, float] = {rn: mean(v) for rn, v in _run_tp_vals.items()}

    # Global fallback: mean across all runs that have throughput data.
    _ref_avg_tok_per_gpu: float | None = mean(list(_run_avg_tok.values())) if _run_avg_tok else None

    # Per-gbsz average throughput from stable runs (used to estimate NOT_LAUNCHED runs).
    _gbsz_stable_toks: dict[int, list[float]] = {}
    for _rn_s, _avg_s in _run_avg_tok.items():
        if run_stage_map.get(_rn_s, "") == "stable":
            _gbsz_s = run_to_gbsz.get(_rn_s)
            if _gbsz_s is not None:
                _gbsz_stable_toks.setdefault(_gbsz_s, []).append(_avg_s)
    _gbsz_stable_avg_tok: dict[int, float] = {
        gbsz: mean(vals) for gbsz, vals in _gbsz_stable_toks.items()
    }

    # Per-stage branch-point token count for decay runs.
    # A decay run starts from a stable checkpoint and runs to decay_tok total tokens,
    # so it only processes (decay_tok - ckpt_tokens) new tokens.
    # ckpt_tokens = ckpt_step × gbsz × seq_length is the same across all combos of
    # the same stage (they branch at the same training-token budget), so we take the
    # first available value from any launched decay row of that stage.
    _seq_len = cfg["seq_length"]
    _stage_ckpt_tokens: dict[str, int] = {}  # stage_name → branch-point raw token count
    for r in rows:
        _rs = run_stage_map.get(r["run_name"], "")
        if (
            _rs != "stable"
            and r.get("ckpt_step") is not None
            and r.get("global_batch_size") is not None
            and _rs not in _stage_ckpt_tokens
        ):
            _stage_ckpt_tokens[_rs] = r["ckpt_step"] * r["global_batch_size"] * _seq_len

    # (value, is_estimated) per run
    #   is_estimated=False → extrapolated from actual spend: h_spent × (1−p)/p
    #   is_estimated=True  → derived from throughput of corresponding stable run
    #                        (or global mean if that too is unavailable)
    run_remaining: dict[str, tuple[float | None, bool]] = {}
    for _rn, _rs, _rt, _ in run_specs:
        _run_rows_r = [r for r in rows if r["run_name"] == _rn]
        _latest_r = _run_rows_r[-1] if _run_rows_r else {}
        _status_r = _latest_r.get("status_word", "NOT_LAUNCHED")
        _h_r = summary_gpu_h.get(_rn)
        _prog_r = _latest_r.get("progress")
        # Reference throughput: for NOT_LAUNCHED runs prefer gbsz-matched stable avg,
        # otherwise prefer stable sibling, then own data, then global mean.
        _stable_rn = decay_to_stable.get(_rn)  # None for stable runs
        _gbsz_rn = run_to_gbsz.get(_rn)
        if _status_r == "NOT_LAUNCHED":
            _ref_tok = (
                (_gbsz_stable_avg_tok.get(_gbsz_rn) if _gbsz_rn is not None else None)
                or _run_avg_tok.get(_stable_rn)
                or _run_avg_tok.get(_rn)
                or _ref_avg_tok_per_gpu
            )
        else:
            _ref_tok = _run_avg_tok.get(_stable_rn) or _run_avg_tok.get(_rn) or _ref_avg_tok_per_gpu
        # Effective token count: for decay runs subtract the stable branch-point tokens.
        # For stable runs (or decay runs with no ckpt_tokens info), use full budget.
        _ckpt_toks = _stage_ckpt_tokens.get(_rs, 0) if _rs != "stable" else 0
        _eff_tokens = max(0, _rt - _ckpt_toks)
        # Remaining compute is always "work left x measured throughput", never an
        # extrapolation of GPU-h burned so far.  Cumulative GPU-h includes every
        # failed and cancelled restart, so extrapolating it charges wasted compute
        # again for the work still to do — on runs with many restarts that
        # overstated the remainder several-fold.
        #
        # Throughput comes from the LUMI logs themselves (_run_avg_tok, measured
        # tok/s/GPU), preferring this run, then its stable sibling, then a
        # gbsz-matched average, then the global mean.  GPU-h is independent of the
        # GPU count here: per-GPU throughput already divides it out.
        _cur_it = _latest_r.get("last_iter")
        _tot_it = _latest_r.get("train_iters")
        _gbsz_tokens = (_gbsz_rn or 0) * _seq_len
        if _status_r == "DONE" or _rn in no_more_training:
            run_remaining[_rn] = (0.0, False)
        elif (
            _ref_tok
            and _cur_it is not None
            and _tot_it is not None
            and _tot_it > _cur_it
            and _gbsz_tokens > 0
        ):
            # Partially trained: bill only the iterations still outstanding.
            # _ref_tok is per-GCD throughput from the logs, so tokens/(_ref_tok*3600)
            # yields GCD-hours; divide by the billing divisor to match the billed
            # GPU-hours burned (LUMI bills per 2-GCD module).
            _rem_tokens = (_tot_it - _cur_it) * _gbsz_tokens
            run_remaining[_rn] = (
                _rem_tokens / (_ref_tok * 3600.0 * _MACHINE_BILLING_DIVISOR),
                False,
            )
        elif _ref_tok is not None and _eff_tokens > 0:
            # Not started: the full budget, less any stable branch-point already trained.
            run_remaining[_rn] = (
                _eff_tokens / (_ref_tok * 3600.0 * _MACHINE_BILLING_DIVISOR),
                True,
            )
        else:
            run_remaining[_rn] = (None, False)

    if summary_gpu_h:
        run_latest_row: dict[str, dict] = {}
        for run_name, _stage, _tok, _tier in run_specs:
            run_rows = [r for r in rows if r["run_name"] == run_name]
            if run_rows:
                run_latest_row[run_name] = run_rows[-1]

        W_EXP = max(len(e) for e in summary_gpu_h) + 2
        gpu_sep = "─" * (W_EXP + 172)
        sorted_exps = sorted(
            summary_gpu_h.items(),
            key=lambda item: _summary_table_sort_key(item[0], run_tier_map, run_stage_map),
        )
        print()
        print("Training progress summary:")
        if _ref_avg_tok_per_gpu is not None:
            print(
                f"  (~ estimates use stable-run throughput per combo; global fallback: {_ref_avg_tok_per_gpu:,.0f} Tok/s/GPU)"
            )
        if args.compute_storage:
            print(
                "  (Ckpt-GB: measured checkpoints/ size; Ckpt-GB-remaining: estimated storage still needed)"
            )
        _local = args.machine
        _foreign = _foreign_cluster
        _gpu_h_hdr = (
            f"  {'GPU-h(' + _local + ')':>10}  {'GPU-h(' + _foreign + ')':>10}  {'GPU-h':>8}"
            if _any_cluster_split
            else f"  {'GPU-h':>8}"
        )
        print(
            f"  {'T#':>3}  {'#':>3}  {'Run':<{W_EXP}}  {'Tier':<9}  {'Progress':>9}  {'':>2} {'Status':<12}  {'Clusters':>8}"
            f"  {'TTFI-GPU-h':>10}  {'LowTP-GPU-h':>12}  {'Overhead-GPU-h':>14}"
            f"{_gpu_h_hdr}  {'Overhead%':>9}  {'Remaining-GPU-h':>16}"
            f"  {'Ckpt-GB':>10}  {'Ckpt-GB-rem':>12}"
        )
        print(gpu_sep)
        grand_lt_time = 0.0
        grand_lt_gpu_h = 0.0
        grand_ttfi_gpu_h = 0.0
        grand_oh_gpu_h = 0.0
        grand_remaining_gpu_h = 0.0
        grand_ckpt_gb = 0.0
        grand_ckpt_rem_gb = 0.0
        prev_tier: str | None = None
        tier_idx = 0
        _tier_ttfi_gpu_h = 0.0
        _tier_lt_gpu_h = 0.0
        _tier_oh_gpu_h = 0.0
        _tier_gpu_h = 0.0
        _tier_gpu_h_leo = 0.0
        _tier_gpu_h_mn5 = 0.0
        _tier_rem_gpu_h = 0.0
        _tier_ckpt_gb = 0.0
        _tier_ckpt_rem_gb = 0.0
        _tier_name_cur = ""
        grand_gpu_h_leo = 0.0
        grand_gpu_h_mn5 = 0.0
        for idx, (exp_name, h) in enumerate(sorted_exps, start=1):
            tier_str = run_tier_map.get(exp_name, "")
            if tier_str != prev_tier:
                tier_idx = 1
                prev_tier = tier_str
                _tier_name_cur = tier_str
            else:
                tier_idx += 1
            latest = run_latest_row.get(exp_name, {})
            prog = latest.get("progress")
            prog_str = f"{prog:.1f}%" if prog is not None else "N/A"
            emoji = latest.get("status_emoji", "")
            status = latest.get("status_word", "")
            action = latest.get("action_word", "")
            if action in _RESTART_ACTIONS and status in ("FAILED", "CANCELLED"):
                _ck = f"{status}+{action}"
            else:
                _ck = status
            color = status_colors.get(_ck, "")
            run_rows = [r for r in rows if r["run_name"] == exp_name]
            _run_cls = {r.get("cluster", "") for r in run_rows if r.get("cluster")}
            _cluster_tag = (
                "MIX" if len(_run_cls) > 1 else (next(iter(_run_cls)) if _run_cls else "")
            )
            # TTFI GPU-h: sum across all jobs for this run
            ttfi_gpu_h_vals = [r["ttfi_gpu_h"] for r in run_rows if r.get("ttfi_gpu_h") is not None]
            ttfi_gpu_h_sum = sum(ttfi_gpu_h_vals) if ttfi_gpu_h_vals else None
            ttfi_gpu_h_str = f"{ttfi_gpu_h_sum:.2f}" if ttfi_gpu_h_sum is not None else "N/A"
            if ttfi_gpu_h_sum is not None:
                grand_ttfi_gpu_h += ttfi_gpu_h_sum
                _tier_ttfi_gpu_h += ttfi_gpu_h_sum
            # Low-throughput stats: sum across all jobs for this run
            lt_time_vals = [r["time_lost_h"] for r in run_rows if r.get("time_lost_h") is not None]
            lt_gpu_h_vals = [r["gpu_h_lost"] for r in run_rows if r.get("gpu_h_lost") is not None]
            lt_time = sum(lt_time_vals) if lt_time_vals else None
            lt_gpu_h = sum(lt_gpu_h_vals) if lt_gpu_h_vals else None
            lt_gpu_h_str = f"{lt_gpu_h:.2f}" if lt_gpu_h is not None else "N/A"
            if lt_time is not None:
                grand_lt_time += lt_time
            if lt_gpu_h is not None:
                grand_lt_gpu_h += lt_gpu_h
                _tier_lt_gpu_h += lt_gpu_h
            # Overhead (TTFI + LowTP) GPU-h and percentage for this run.
            # Only include jobs with known GPU-h so the denominator and numerator
            # are from the same cluster (avoids inflated % from MN5 LowTP with
            # only LEO GPU-h in the denominator).
            oh_gpu_h_vals = [
                r["overhead_gpu_h"]
                for r in run_rows
                if r.get("overhead_gpu_h") is not None and r.get("gpu_hours") is not None
            ]
            oh_gpu_h_sum = sum(oh_gpu_h_vals) if oh_gpu_h_vals else None
            oh_gpu_h_str = f"{oh_gpu_h_sum:.2f}" if oh_gpu_h_sum is not None else "N/A"
            oh_pct_val = (
                oh_gpu_h_sum / h * 100.0
                if oh_gpu_h_sum is not None and h is not None and h > 0
                else None
            )
            oh_pct_str = f"{oh_pct_val:.1f}%" if oh_pct_val is not None else "N/A"
            gpu_h_str = f"{h:.1f}" if h is not None else "N/A"
            _h_leo = _run_gpu_h_leo.get(exp_name)
            _h_mn5 = _run_gpu_h_mn5.get(exp_name)
            _h_leo_str = f"{_h_leo:.1f}" if _h_leo is not None else "N/A"
            _h_mn5_str = f"{_h_mn5:.1f}" if _h_mn5 is not None else "N/A"
            if oh_gpu_h_sum is not None:
                grand_oh_gpu_h += oh_gpu_h_sum
                _tier_oh_gpu_h += oh_gpu_h_sum
            if h is not None:
                _tier_gpu_h += h
            if _h_leo is not None:
                _tier_gpu_h_leo += _h_leo
                grand_gpu_h_leo += _h_leo
            if _h_mn5 is not None:
                _tier_gpu_h_mn5 += _h_mn5
                grand_gpu_h_mn5 += _h_mn5
            rem_val, rem_est = run_remaining.get(exp_name, (None, False))
            rem_str = (
                (f"~{rem_val:.1f}" if rem_est else f"{rem_val:.1f}")
                if rem_val is not None
                else "N/A"
            )
            if rem_val is not None:
                grand_remaining_gpu_h += rem_val
                _tier_rem_gpu_h += rem_val
            ckpt_str = _format_ckpt_measured_cell(
                run_checkpoint_storage, exp_name, compute=args.compute_storage
            )
            ckpt_rem_str = _format_ckpt_remaining_cell(
                run_checkpoint_storage, exp_name, compute=args.compute_storage
            )
            if args.compute_storage:
                ckpt_val, ckpt_rem_val = run_checkpoint_storage.get(exp_name, (None, None))
                if ckpt_val is not None:
                    grand_ckpt_gb += ckpt_val
                    _tier_ckpt_gb += ckpt_val
                if ckpt_rem_val is not None:
                    grand_ckpt_rem_gb += ckpt_rem_val
                    _tier_ckpt_rem_gb += ckpt_rem_val
            _gpu_h_cols = (
                f"  {_h_leo_str:>10}  {_h_mn5_str:>10}  {gpu_h_str:>8}"
                if _any_cluster_split
                else f"  {gpu_h_str:>8}"
            )
            print(
                f"  {tier_idx:>3}  {idx:>3}  {exp_name:<{W_EXP}}  {tier_str:<9}  {prog_str:>9}  {emoji} {color}{status:<12}{RESET}  {_cluster_tag:>8}"
                f"  {ttfi_gpu_h_str:>10}  {lt_gpu_h_str:>12}  {oh_gpu_h_str:>14}"
                f"{_gpu_h_cols}  {oh_pct_str:>9}  {rem_str:>16}"
                f"  {ckpt_str:>10}  {ckpt_rem_str:>12}"
            )
            is_last_entry = idx >= len(sorted_exps)
            next_tier_str = "" if is_last_entry else run_tier_map.get(sorted_exps[idx][0], "")
            if is_last_entry or next_tier_str != tier_str:
                _t_oh_pct = _tier_oh_gpu_h / _tier_gpu_h * 100.0 if _tier_gpu_h > 0 else None
                _t_oh_pct_s = f"{_t_oh_pct:.1f}%" if _t_oh_pct is not None else "N/A"
                _t_ttfi_s = f"{_tier_ttfi_gpu_h:.2f}" if _tier_ttfi_gpu_h else "N/A"
                _t_oh_s = f"{_tier_oh_gpu_h:.2f}" if _tier_oh_gpu_h else "N/A"
                _t_gpu_h_s = f"{_tier_gpu_h:.1f}" if _tier_gpu_h else "N/A"
                _t_leo_s = f"{_tier_gpu_h_leo:.1f}" if _tier_gpu_h_leo else "N/A"
                _t_mn5_s = f"{_tier_gpu_h_mn5:.1f}" if _tier_gpu_h_mn5 else "N/A"
                _t_rem_s = f"{_tier_rem_gpu_h:.1f}" if _tier_rem_gpu_h else "N/A"
                _t_ckpt_s = _format_ckpt_gb_total(_tier_ckpt_gb, compute=args.compute_storage)
                _t_ckpt_rem_s = _format_ckpt_gb_total(
                    _tier_ckpt_rem_gb, compute=args.compute_storage
                )
                _t_gpu_h_cols = (
                    f"  {_t_leo_s:>10}  {_t_mn5_s:>10}  {_t_gpu_h_s:>8}"
                    if _any_cluster_split
                    else f"  {_t_gpu_h_s:>8}"
                )
                print(
                    f"  {'':>3}  {'':>3}  {f'[{_tier_name_cur} total]':<{W_EXP}}  {_tier_name_cur:<9}  {'':>9}  {'':>2} {'':<12}  {'':>8}"
                    f"  {_t_ttfi_s:>10}  {_tier_lt_gpu_h:>12.2f}"
                    f"  {_t_oh_s:>14}{_t_gpu_h_cols}  {_t_oh_pct_s:>9}  {_t_rem_s:>16}"
                    f"  {_t_ckpt_s:>10}  {_t_ckpt_rem_s:>12}"
                )
                _tier_ttfi_gpu_h = 0.0
                _tier_lt_gpu_h = 0.0
                _tier_oh_gpu_h = 0.0
                _tier_gpu_h = 0.0
                _tier_gpu_h_leo = 0.0
                _tier_gpu_h_mn5 = 0.0
                _tier_rem_gpu_h = 0.0
                _tier_ckpt_gb = 0.0
                _tier_ckpt_rem_gb = 0.0
                print(gpu_sep)
        grand_ttfi_gpu_h_str = f"{grand_ttfi_gpu_h:.2f}" if grand_ttfi_gpu_h else "N/A"
        grand_oh_gpu_h_str = f"{grand_oh_gpu_h:.2f}" if grand_oh_gpu_h else "N/A"
        grand_oh_pct = (
            grand_oh_gpu_h / summary_grand_total * 100.0
            if grand_oh_gpu_h and summary_grand_total > 0
            else None
        )
        grand_oh_pct_str = f"{grand_oh_pct:.1f}%" if grand_oh_pct is not None else "N/A"
        grand_rem_str = f"{grand_remaining_gpu_h:.1f}" if grand_remaining_gpu_h else "N/A"
        grand_ckpt_str = _format_ckpt_gb_total(grand_ckpt_gb, compute=args.compute_storage)
        grand_ckpt_rem_str = _format_ckpt_gb_total(grand_ckpt_rem_gb, compute=args.compute_storage)
        _grand_leo_s = f"{grand_gpu_h_leo:.1f}" if grand_gpu_h_leo else "N/A"
        _grand_mn5_s = f"{grand_gpu_h_mn5:.1f}" if grand_gpu_h_mn5 else "N/A"
        _grand_gpu_h_cols = (
            f"  {_grand_leo_s:>10}  {_grand_mn5_s:>10}  {summary_grand_total:>8.1f}"
            if _any_cluster_split
            else f"  {summary_grand_total:>8.1f}"
        )
        print(
            f"  {'':>3}  {'':>3}  {'TOTAL':<{W_EXP}}  {'':<9}  {'':>9}  {'':>2} {'':<12}"
            f"  {grand_ttfi_gpu_h_str:>10}  {grand_lt_gpu_h:>12.2f}"
            f"  {grand_oh_gpu_h_str:>14}{_grand_gpu_h_cols}  {grand_oh_pct_str:>9}  {grand_rem_str:>16}"
            f"  {grand_ckpt_str:>10}  {grand_ckpt_rem_str:>12}"
        )

    # Summary counts
    from collections import Counter

    counts = Counter(r["status_word"] for r in rows)
    print()
    print("Status breakdown:")
    for status, cnt in sorted(counts.items()):
        print(f"  {status:<14} {cnt}")

    # ── CSV rows: fixed, machine-independent schema (progress_report.CSV_FIELDS) ──
    import progress_report as _pr

    _size = (
        args.size
        or next((s for s in (size_label(r["run_name"]) for r in rows) if s), "")
        or size_label("", resolved_base)
    )
    _updated_at = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    # Run-level estimates (remaining GPU-h, checkpoint storage) go on the run's
    # latest job row only.
    _csv_latest_job: dict[str, str] = {rn: jids[-1] for rn, jids in run_job_map.items() if jids}

    def _fmt_or_blank(v: Any, fmt: str) -> str:
        return format(v, fmt) if v is not None else ""

    csv_rows: list[dict[str, str]] = []
    for r in rows:
        mb = RE_BUDGET.search(r["run_name"])
        sd = mb.group(1) if mb else ("stable" if r["stage"] == "stable" else "decay")
        c_val = (
            f"{6.0 * r['transformer_params_b'] * r['tokens_b']:.2f}"
            if r["transformer_params_b"] is not None and r.get("tokens_b") is not None
            else ""
        )
        _is_latest = r["job_id"] == "" or r["job_id"] == _csv_latest_job.get(r["run_name"])
        _rem_v, _rem_est = (
            run_remaining.get(r["run_name"], (None, False)) if _is_latest else (None, False)
        )
        _ckpt_v, _ckpt_rem_v = (
            run_checkpoint_storage.get(r["run_name"], (None, None))
            if _is_latest and args.compute_storage
            else (None, None)
        )
        _gpu_h_s = _fmt_or_blank(r.get("gpu_hours"), ".1f")
        _row_cluster = r.get("cluster", "") or args.machine
        csv_rows.append(
            {
                "Size": _size,
                "Run": r["run_name"],
                "Machine": _row_cluster,
                "Source": r.get("source", "local"),
                "JobID": r["job_id"],
                "Owner": r.get("owner", ""),
                "StartTime": r.get("start_str", "") or r.get("sacct_start_str", ""),
                "Elapsed": r.get("sacct_elapsed", ""),
                "N_ne(B)": _fmt_or_blank(r["transformer_params_b"], ".2f"),
                "N(B)": _fmt_or_blank(r["total_params_b"], ".2f"),
                "D(B)": str(int(r["tokens_b"])) if r.get("tokens_b") is not None else "",
                "C(10^18)": c_val,
                "Tier": r.get("tier", ""),
                "Stage": sd,
                "TotIter": _fmt_or_blank(r["train_iters"], "d"),
                "CurIter": _fmt_or_blank(r.get("last_iter"), "d"),
                "LastCkpt": _fmt_or_blank(r["last_ckpt"], "d"),
                "Prog%": _fmt_or_blank(r.get("progress"), ".1f"),
                "TrainLoss": _fmt_or_blank(r.get("last_train_loss"), ".4f"),
                "ValLoss": _fmt_or_blank(r.get("last_val_loss"), ".4f"),
                "LR": _fmt_or_blank(r["lr"], ".4f"),
                "GBS": _fmt_or_blank(r["global_batch_size"], "d"),
                "MBS": _fmt_or_blank(r["micro_batch_size"], "d"),
                "Nodes": _fmt_or_blank(r.get("nodes"), "d"),
                "Workers": _fmt_or_blank(r["num_workers"], "d"),
                "TFLOP/s/GPU": _fmt_or_blank(r["avg_tflop_per_gpu"], ".1f"),
                "Tok/s/GPU": _fmt_or_blank(r["avg_tok_per_gpu"], ".0f"),
                "TTFI(min)": _fmt_or_blank(r.get("ttfi_min"), ".1f"),
                "TTFI-GPU-h": _fmt_or_blank(r.get("ttfi_gpu_h"), ".2f"),
                "LowTP-time(h)": _fmt_or_blank(r.get("time_lost_h"), ".4f"),
                "LowTP-GPU-h": _fmt_or_blank(r.get("gpu_h_lost"), ".4f"),
                "Overhead-time(h)": _fmt_or_blank(r.get("overhead_time_h"), ".4f"),
                "Overhead-GPU-h": _fmt_or_blank(r.get("overhead_gpu_h"), ".4f"),
                **{
                    f"GPU-h({m})": (_gpu_h_s if _row_cluster == m else "")
                    for m in MACHINES
                },
                "GPU-h": _gpu_h_s,
                "Overhead%": _fmt_or_blank(r.get("overhead_pct"), ".2f"),
                "Remaining-GPU-h": (
                    (f"~{_rem_v:.1f}" if _rem_est else f"{_rem_v:.1f}") if _rem_v is not None else ""
                ),
                "Ckpt-GB": _fmt_or_blank(_ckpt_v, ".1f"),
                "Ckpt-GB-remaining": _fmt_or_blank(_ckpt_rem_v, ".1f"),
                "Emoji": r["status_emoji"],
                "Status": r["status_word"],
                "Action": r["action_word"],
                "Error": r["error_desc"],
                "OutputPath": r.get("output_path", "") or "",
                "DataPath": r.get("data_path", "") or "",
                "DataMixPath": r.get("data_mix_path", "") or "",
                "Config": str(Path(args.config).resolve()),
                "UpdatedAt": _updated_at,
            }
        )

    md_rows = csv_rows
    if args.csv:
        csv_path = Path(args.csv)
        if args.accumulate:
            existing = _pr.read_csv(csv_path) if csv_path.is_file() else []
            md_rows, _merge_stats = _pr.merge_rows(
                existing,
                csv_rows,
                machine=args.machine,
                size=_size,
                tracked_runs={rn for rn, _s, _t, _ti in run_specs},
                carry_ckpt=not args.compute_storage,
            )
            _pr.write_csv(csv_path, md_rows)
            print(
                f"\nMerged {args.machine}/{_size} into {csv_path}: {_merge_stats} "
                f"-> {len(md_rows)} rows total"
            )
        else:
            _pr.write_csv(csv_path, csv_rows)
            print(f"\nWrote {len(csv_rows)} rows to {csv_path}")

    if gpu_csv_rows:
        gpu_csv_rows.append(
            {
                "experiment": "TOTAL",
                "job_id": "",
                "state": "",
                "elapsed": "",
                "gpus": "",
                "gpu_hours": round(grand_gpu_total, 1),
            }
        )
        # Single-run configs: resolved_base may be a shared, non-writable parent dir.
        # Fall back to the directory of --csv (or CWD) so we can always write.
        if cfg.get("is_single_run"):
            _csv_anchor = Path(args.csv).parent if args.csv else Path(".")
            gpu_csv_path = _csv_anchor / "gpu_hours.csv"
        else:
            gpu_csv_path = _redirect_output(resolved_base, args) / "gpu_hours.csv"
        guard_write(gpu_csv_path)
        gpu_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with gpu_csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["experiment", "job_id", "state", "elapsed", "gpus", "gpu_hours"]
            )
            writer.writeheader()
            writer.writerows(gpu_csv_rows)
        print(f"\nWrote {len(gpu_csv_rows) - 1} job rows (+1 total) to {gpu_csv_path}")

    # ── Markdown output (rendered from the CSV rows; merged rows with --accumulate) ──
    if args.md:
        md_path = Path(args.md)
        _pr.write_text(md_path, _pr.render_md(md_rows))
        print(f"\nWrote markdown to {md_path}")


if __name__ == "__main__":
    main()
