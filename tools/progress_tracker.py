#!/usr/bin/env python3
"""
Sweep run status table for multilingual_scaling experiments.

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
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

# ── Sibling-module imports ─────────────────────────────────────────────────────
_tools_dir = str(Path(__file__).parent)
_scripts_dir = str(Path(__file__).parent.parent / "scripts")
for _d in (_tools_dir, _scripts_dir):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from gpu_hours import (
    collect_job_ids as _gpu_collect_job_ids,
    query_sacct as _gpu_query_sacct,
    parse_elapsed as _gpu_parse_elapsed,
)
from validate_sweep_runs import (
    _resolve_defaults,
    render_job_name,
    substitute_omegaconf_path_vars as _subst,
)


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

# Training loss from iteration lines: "lm loss: 3.1416"
RE_TRAIN_LOSS = re.compile(
    r"iteration\s+\d+/\s*\d+\s*\|[^\n]*lm loss:\s*([\d.eE+\-]+)"
)

# Validation loss lines: "validation loss at iteration X | lm loss value: Y"
RE_VAL_LOSS = re.compile(
    r"validation loss at[^\n]*\|[^\n]*lm loss value:\s*([\d.eE+\-]+)"
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
RE_SBATCH_CKPT_STEP = re.compile(r"--ckpt-step\s+(\d+)")

# Token budget and stage encoded in run name
RE_BUDGET = re.compile(r"_(stable|decay)(\d+BT)$")

TS_FMT = "%Y-%m-%d %H:%M:%S"

# ── Monitor-state event classification ───────────────────────────────────────

_CLEAN_RESTART_EVENTS = frozenset({"time_limit", "inactive"})
_HARD_ERROR_EVENTS    = frozenset({"segmentation_fault", "error", "bus_error",
                                    "nan_loss", "connection_failure"})
_FINISH_EVENTS        = frozenset({"finished_training", "finish"})

_EVENT_ERROR_DESC: dict[str, str] = {
    "segmentation_fault": "Segmentation fault",
    "error":              "Exit code 1",
    "bus_error":          "Bus error",
    "nan_loss":           "NaN loss",
    "connection_failure": "Connection failure",
}

# Seconds after a job's sacct End time within which a monitor event can still
# be attributed to that job (accounts for monitoring-loop processing latency).
_EVENT_MATCH_TOLERANCE_S = 300.0



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
                    "center_tokens": center,
                    "cross_tokens": cross,
                    "diagonal_tokens": diagonal,
                    "stable_launch_tier": str(entry.get("backend.megatron.aux.stable_launch_tier", "")),
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



# ── SBATCH parsing ────────────────────────────────────────────────────────────

def parse_sbatch(path: Path) -> tuple[int | None, int | None, int | None]:
    """Return (nodes, gpus_per_node, ckpt_step) from a job.sbatch file.

    ckpt_step is the value of --ckpt-step, present only in decay jobs to indicate
    the absolute iteration from which the decay phase starts.
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
        "first_iter": None,
        "last_iter": None,
        "total_iters": None,
        "avg_tflop_per_gpu": None,
        "avg_tok_per_gpu": None,
        "n_iters_sampled": None,
        "first_ts": None,
        "last_ts": None,
        "last_train_loss": None,
        "last_val_loss": None,
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
        result["first_iter"] = rows[0][0]
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
    """Query sacct for job state, elapsed time, GPU count, and start/end timestamps.
    Returns {} on error."""
    if not job_ids:
        return {}
    try:
        result = subprocess.run(
            [
                "sacct", "-j", ",".join(job_ids),
                "--format=JobID,State,Elapsed,AllocTRES%80,Start,End",
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
        if len(parts) < 6:
            continue
        jid_field, state, elapsed, alloc_tres, start, end = parts[:6]
        if "." in jid_field:
            continue
        gpus = 0
        m = re.search(r"gres/gpu=(\d+)", alloc_tres)
        if m:
            gpus = int(m.group(1))
        info[jid_field.strip()] = {
            "state":    state.strip(),
            "elapsed":  elapsed.strip(),
            "gpus":     gpus,
            "start_ts": _parse_sacct_ts(start),
            "end_ts":   _parse_sacct_ts(end),
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
    """Return all Slurm job IDs found in logs/ or config-*.yaml files, sorted ascending.

    config-{id}.yaml is written at submission time; stdout/stderr logs only appear
    once the job starts. Including config-based IDs lets us detect jobs that are
    submitted (or queued) but haven't started writing logs yet.
    """
    ids: set[str] = set()
    logs_dir = run_dir / "logs"
    if logs_dir.is_dir():
        for f in logs_dir.iterdir():
            m = re.match(r"(?:stdout|stderr)-(\d+)\.log$", f.name)
            if m:
                ids.add(m.group(1))
    if run_dir.is_dir():
        for f in run_dir.iterdir():
            m = re.match(r"config-(\d+)\.yaml$", f.name)
            if m:
                ids.add(m.group(1))
    return sorted(ids)


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
                    events.append({
                        "name": parts[1],
                        "ts":   val.get("last_action_ts"),
                    })
            sessions.append({
                "session_id":     d.name,
                "events":         events,
                "last_status":    rt.get("last_status"),
                "final_state":    rt.get("final_state"),
                "runtime_job_id": rt.get("runtime_job_id"),
            })
    return sessions if sessions else None


def map_events_to_jobs(
    job_ids: list[str],
    sacct_info: dict[str, dict],
    monitor_sessions: list[dict],
) -> dict[str, set[str]]:
    """Assign monitor-state events to specific Slurm job IDs via sacct timestamps.

    For each event's ``last_action_ts``, we find the job whose sacct time window
    [Start, End + _EVENT_MATCH_TOLERANCE_S] contains the timestamp.

    Returns ``{job_id: set_of_event_names}``.  Jobs with no sacct start-time data
    get an empty set (caller should treat that as "no evidence either way").
    """
    job_events: dict[str, set[str]] = {jid: set() for jid in job_ids}

    all_events = [
        ev
        for session in monitor_sessions
        for ev in session["events"]
        if ev.get("ts") is not None
    ]
    if not all_events:
        return job_events

    for job_id in job_ids:
        sacct = sacct_info.get(job_id, {})
        start_ts = sacct.get("start_ts")
        end_ts   = sacct.get("end_ts")
        if start_ts is None:
            continue
        deadline = (end_ts if end_ts is not None else float("inf")) + _EVENT_MATCH_TOLERANCE_S
        for ev in all_events:
            if start_ts <= ev["ts"] <= deadline:
                job_events[job_id].add(ev["name"])

    return job_events


# ── Status determination ──────────────────────────────────────────────────────

def determine_status(
    job_id: str,
    all_job_ids: list[str],
    stdout_data: dict,
    stderr_data: dict,
    sacct_info: dict[str, dict],
    is_latest_job: bool,
    job_monitor_events: set[str] | None = None,
) -> tuple[str, str, str, str]:
    """
    Returns (emoji, status_word, error_description, action_word).

    Status words:  NOT_LAUNCHED | QUEUED | DONE | FAILED | CANCELLED | TRAINING
    Action words:  AUTO_RESTARTED | NONE | ""
      - AUTO_RESTARTED: a new job was automatically submitted after this one ended
      - NONE:           job failed and no automatic restart was taken
      - "":             not applicable (DONE, TRAINING, QUEUED, etc.)

    Error source priority:
      1. *job_monitor_events* (from monitor-state files) when not None.
      2. stderr log patterns (fallback for manually launched runs).
    """
    sacct = sacct_info.get(job_id, {})
    sacct_state = sacct.get("state", "")

    # ── Classify events from monitor state (primary) or stderr (fallback) ────
    use_monitor = job_monitor_events is not None
    if use_monitor:
        triggered_hard  = job_monitor_events & _HARD_ERROR_EVENTS
        triggered_clean = job_monitor_events & _CLEAN_RESTART_EVENTS
        triggered_done  = job_monitor_events & _FINISH_EVENTS
        hard_error_desc = "; ".join(
            _EVENT_ERROR_DESC.get(n, n) for n in sorted(triggered_hard)
        )
    else:
        # stderr-derived booleans (legacy path)
        triggered_hard  = set()
        triggered_clean = set()
        triggered_done  = set()
        hard_error_desc = ""

    # ── DONE ─────────────────────────────────────────────────────────────────
    last_iter   = stdout_data.get("last_iter")
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
            action = "AUTO_RESTARTED" if not is_latest_job else "NONE"
            return "⚠️", "FAILED", hard_error_desc, action
    else:
        if stderr_data.get("segfault"):
            action = "AUTO_RESTARTED" if not is_latest_job else "NONE"
            return "⚠️", "FAILED", "Segmentation fault", action
        if stderr_data.get("oom"):
            action = "AUTO_RESTARTED" if not is_latest_job else "NONE"
            return "⚠️", "FAILED", "Out of memory (CUDA OOM)", action
        if stderr_data.get("fatal"):
            action = "AUTO_RESTARTED" if not is_latest_job else "NONE"
            return "⚠️", "FAILED", "Fatal error", action

    # ── sacct-based states ────────────────────────────────────────────────────
    if sacct_state:
        if "CANCEL" in sacct_state:
            return "🚫", "CANCELLED", "", ""
        if sacct_state == "COMPLETED":
            return "✅", "DONE", "", ""
        if sacct_state == "FAILED":
            err = hard_error_desc or "; ".join(stderr_data.get("errors", [])) or "UNKNOWN"
            action = "AUTO_RESTARTED" if not is_latest_job else "NONE"
            return "⚠️", "FAILED", err, action
        if sacct_state == "RUNNING":
            return "⏳", "TRAINING", "", ""
        if sacct_state == "PENDING":
            return "🕒", "QUEUED", "", ""
        if sacct_state == "TIMEOUT":
            action = "AUTO_RESTARTED" if not is_latest_job else "NONE"
            return "⚠️", "FAILED", "Time limit exceeded", action

    # ── No stdout → queued / not started ─────────────────────────────────────
    if stdout_data.get("last_iter") is None and stdout_data.get("train_iters") is None:
        return "🕒", "QUEUED", "", ""

    # ── Clean restart events ──────────────────────────────────────────────────
    if use_monitor:
        if triggered_clean and not is_latest_job:
            return "⚠️", "FAILED", "Time limit (normal restart)", "AUTO_RESTARTED"
    else:
        if (stderr_data.get("time_limit") or stderr_data.get("sigterm")) and not is_latest_job:
            return "⚠️", "FAILED", "Time limit (normal restart)", "AUTO_RESTARTED"

    # ── Latest job still running ──────────────────────────────────────────────
    if is_latest_job and last_iter is not None:
        return "⏳", "TRAINING", "", ""

    # ── Fallback ──────────────────────────────────────────────────────────────
    error_str = "; ".join(stderr_data.get("errors", []))
    return "⚠️", "FAILED", error_str, "NONE"


def compute_progress(last_iter: int | None, ckpt_step: int | None, total_iters: int | None) -> float | None:
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
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)

    # Discover monitor-state directories
    if args.monitor_dirs is None:
        monitor_root = project_root / "monitor_state"
        monitor_dirs: list[Path] = (
            sorted(p for p in monitor_root.iterdir() if p.is_dir())
            if monitor_root.is_dir() else []
        )
    else:
        monitor_dirs = [Path(d) for d in args.monitor_dirs]

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
    run_specs: list[tuple[str, str, int, str]] = []
    for combo in combos:
        lr, gbsz = combo["lr"], combo["gbsz"]
        stable_tok = combo["stable_tokens"]
        valid_decay_toks = combo["valid_decay_tokens"]

        name = render_job_name(cfg["job_name_tpl"], 1, lr, gbsz, seed, "stable", stable_tok)
        run_specs.append((name, "stable", stable_tok, combo["stable_launch_tier"]))

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
            name = render_job_name(cfg["job_name_tpl"], 1, lr, gbsz, seed, stage_name)
            run_specs.append((name, stage_name, decay_tok, tier))

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
    for run_name, _stage, _tok, _tier in run_specs:
        run_dir = resolved_base / run_name
        jids = find_job_ids(run_dir)
        run_job_map[run_name] = jids
        all_job_ids.extend(jids)

    sacct_info = query_sacct(all_job_ids)
    sacct_available = bool(sacct_info) or bool(all_job_ids)  # mark as available if sacct returned anything

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
                run_job_map.get(run_name, []), sacct_info, sessions
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

        # sbatch info
        sbatch_nodes, sbatch_gpus_per_node, sbatch_ckpt_step = parse_sbatch(run_dir / "job.sbatch")
        total_gpus = (sbatch_nodes or 0) * (sbatch_gpus_per_node or 0)

        job_ids = run_job_map.get(run_name, [])

        if not job_ids:
            if run_dir.is_dir():
                # Directory exists but no log files → submitted to Slurm, not started yet
                s_emoji, s_word = "🕒", "QUEUED"
            else:
                # Run directory doesn't exist → this tier has not been launched at all
                s_emoji, s_word = "⚪", "NOT_LAUNCHED"
            rows.append({
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
                "train_iters": None,
                "last_iter": None,
                "progress": None,
                "last_train_loss": None,
                "last_val_loss": None,
                "last_ckpt": last_ckpt,
                "avg_tflop_per_gpu": None,
                "avg_tok_per_gpu": None,
                "n_iters_sampled": None,
                "gpu_hours": None,
                "sacct_state": "",
                "sacct_elapsed": "",
                "status_emoji": s_emoji,
                "status_word": s_word,
                "action_word": "",
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
                elapsed_h = _gpu_parse_elapsed(sacct_elapsed)
                gpus = sacct_entry.get("gpus", 0) or total_gpus
                gpu_hours = elapsed_h * gpus
            else:
                gpu_hours = gpu_hours_from_timestamps(
                    stdout_data.get("first_ts"),
                    stdout_data.get("last_ts"),
                    sbatch_nodes,
                    sbatch_gpus_per_node,
                )

            # Per-job monitor events: set[str] if monitor state exists, else None
            job_events_map = run_job_monitor_events.get(run_name)
            job_monitor_events = (
                job_events_map.get(job_id)   # may be empty set
                if job_events_map is not None
                else None                     # no monitor state → use stderr
            )

            emoji, status_word, error_desc, action_word = determine_status(
                job_id, job_ids, stdout_data, stderr_data, sacct_info, is_latest,
                job_monitor_events=job_monitor_events,
            )

            rows.append({
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
                "train_iters": stdout_data.get("train_iters"),
                "last_iter": stdout_data.get("last_iter"),
                "progress": compute_progress(
                    stdout_data.get("last_iter"),
                    sbatch_ckpt_step,
                    stdout_data.get("total_iters"),
                ),
                "last_train_loss": stdout_data.get("last_train_loss"),
                "last_val_loss": stdout_data.get("last_val_loss"),
                "last_ckpt": last_ckpt,
                "avg_tflop_per_gpu": stdout_data.get("avg_tflop_per_gpu"),
                "avg_tok_per_gpu": stdout_data.get("avg_tok_per_gpu"),
                "n_iters_sampled": stdout_data.get("n_iters_sampled"),
                "gpu_hours": gpu_hours,
                "sacct_state": sacct_state,
                "sacct_elapsed": sacct_elapsed,
                "status_emoji": emoji,
                "status_word": status_word,
                "action_word": action_word,
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
        f"{'Tier':>8} "
        f"{'Stage':>7} "
        f"{'TotIter':>9} "
        f"{'CurIter':>9} "
        f"{'Prog%':>6} "
        f"{'TrnLoss':>8} "
        f"{'ValLoss':>8} "
        f"{'LastCkpt':>9} "
        f"{'LR':>8} "
        f"{'GBS':>5} "
        f"{'MBS':>4} "
        f"{'Nodes':>6} "
        f"{'Workers':>7} "
        f"{'TFLOP/s/GPU':>12} "
        f"{'Tok/s/GPU':>10} "
        f"{'GPU-h':>10} "
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

    status_colors = {
        "DONE":                  "\033[92m",
        "TRAINING":              "\033[94m",
        "FAILED":                "\033[91m",
        "FAILED+AUTO_RESTARTED": "\033[33m",
        "CANCELLED":             "\033[91m",
        "QUEUED":                "\033[93m",
        "NOT_LAUNCHED":          "\033[90m",
    }
    RESET = "\033[0m"

    prev_run = None
    for r in rows:
        name_display = r["run_name"] if r["run_name"] != prev_run else ""
        prev_run = r["run_name"]
        if len(name_display) > W_NAME:
            name_display = name_display[: W_NAME - 1] + "…"

        color_key = "FAILED+AUTO_RESTARTED" if r["action_word"] == "AUTO_RESTARTED" and r["status_word"] == "FAILED" else r["status_word"]
        color = status_colors.get(color_key, "")

        tflop_str = f"{r['avg_tflop_per_gpu']:.1f}" if r["avg_tflop_per_gpu"] is not None else "N/A"
        tok_str = f"{r['avg_tok_per_gpu']:.0f}" if r["avg_tok_per_gpu"] is not None else "N/A"
        gpu_h_str = f"{r['gpu_hours']:.1f}" if r["gpu_hours"] is not None else "N/A"
        trans_str = f"{r['transformer_params_b']:.2f}" if r["transformer_params_b"] is not None else "N/A"
        total_str = f"{r['total_params_b']:.2f}" if r["total_params_b"] is not None else "N/A"
        lr_str = f"{r['lr']:.4f}" if r["lr"] is not None else "N/A"
        ckpt_str = str(r["last_ckpt"]) if r["last_ckpt"] is not None else "N/A"
        ti_str = str(r["train_iters"]) if r["train_iters"] is not None else "N/A"
        ci_str = str(r["last_iter"]) if r.get("last_iter") is not None else "N/A"
        prog_str = f"{r['progress']:.1f}%" if r.get("progress") is not None else "N/A"
        trn_loss_str = f"{r['last_train_loss']:.4f}" if r.get("last_train_loss") is not None else "N/A"
        val_loss_str = f"{r['last_val_loss']:.4f}" if r.get("last_val_loss") is not None else "N/A"
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
            f"{r['tier']:>8} "
            f"{stage_disp:>7} "
            f"{ti_str:>9} "
            f"{ci_str:>9} "
            f"{prog_str:>6} "
            f"{trn_loss_str:>8} "
            f"{val_loss_str:>8} "
            f"{ckpt_str:>9} "
            f"{lr_str:>8} "
            f"{gbs_str:>5} "
            f"{mbs_str:>4} "
            f"{nodes_str:>6} "
            f"{wkr_str:>7} "
            f"{tflop_str:>12} "
            f"{tok_str:>10} "
            f"{gpu_h_str:>10} "
            f"{'🔁' if r['action_word'] == 'AUTO_RESTARTED' else ''}{r['status_emoji']} "
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
                    gpu_h = _gpu_parse_elapsed(d["elapsed"]) * d["gpus"]
                    exp_gpu_h[exp_name] = exp_gpu_h.get(exp_name, 0.0) + gpu_h
                    grand_gpu_total += gpu_h
                    gpu_csv_rows.append({
                        "experiment": exp_name,
                        "job_id":     job_id,
                        "state":      d["state"],
                        "elapsed":    d["elapsed"],
                        "gpus":       d["gpus"],
                        "gpu_hours":  round(gpu_h, 1),
                    })
        except SystemExit:
            pass

    if exp_gpu_h:
        run_latest_row: dict[str, dict] = {}
        for run_name, _stage, _tok, _tier in run_specs:
            run_rows = [r for r in rows if r["run_name"] == run_name]
            if run_rows:
                run_latest_row[run_name] = run_rows[-1]

        W_EXP = max(len(e) for e in exp_gpu_h) + 2
        gpu_sep = "─" * (W_EXP + 38)
        print()
        print("Training progress summary:")
        print(f"  {'Run':<{W_EXP}}  {'GPU-h':>8}  {'Progress':>9}  {'':>2} {'Status':<12}")
        print(gpu_sep)
        for exp_name, h in sorted(exp_gpu_h.items()):
            latest = run_latest_row.get(exp_name, {})
            prog = latest.get("progress")
            prog_str = f"{prog:.1f}%" if prog is not None else "N/A"
            emoji = latest.get("status_emoji", "")
            status = latest.get("status_word", "")
            color = status_colors.get(
                "FAILED+AUTO_RESTARTED" if latest.get("action_word") == "AUTO_RESTARTED" and status == "FAILED" else status, ""
            )
            print(f"  {exp_name:<{W_EXP}}  {h:>8.1f}  {prog_str:>9}  {emoji} {color}{status:<12}{RESET}")
        print(gpu_sep)
        print(f"  {'TOTAL':<{W_EXP}}  {grand_gpu_total:>8.1f}")

    # Summary counts
    from collections import Counter
    counts = Counter(r["status_word"] for r in rows)
    print()
    print("Status breakdown:")
    for status, cnt in sorted(counts.items()):
        print(f"  {status:<14} {cnt}")

    # ── CSV output ──────────────────────────────────────────────────────────
    if args.csv:
        csv_fields = [
            "Run", "JobID", "N_ne(B)", "N(B)", "D(B)", "C(10^18)", "Tier", "Stage",
            "TotIter", "CurIter", "Prog%", "TrainLoss", "ValLoss", "LastCkpt", "LR", "GBS", "MBS", "Nodes", "Workers",
            "TFLOP/s/GPU", "Tok/s/GPU", "GPU-h", "Emoji", "Status", "Action", "Error",
        ]
        csv_path = Path(args.csv)
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            for r in rows:
                mb = RE_BUDGET.search(r["run_name"])
                sd = mb.group(1) if mb else ("stable" if r["stage"] == "stable" else "decay")
                c_val = (
                    f"{6.0 * r['transformer_params_b'] * r['tokens_b']:.2f}"
                    if r["transformer_params_b"] is not None and r.get("tokens_b") is not None
                    else ""
                )
                writer.writerow({
                    "Run":         r["run_name"],
                    "JobID":       r["job_id"],
                    "N_ne(B)":     f"{r['transformer_params_b']:.2f}" if r["transformer_params_b"] is not None else "",
                    "N(B)":        f"{r['total_params_b']:.2f}" if r["total_params_b"] is not None else "",
                    "D(B)":        int(r["tokens_b"]) if r.get("tokens_b") is not None else "",
                    "C(10^18)":    c_val,
                    "Tier":        r.get("tier", ""),
                    "Stage":       sd,
                    "TotIter":     r["train_iters"] if r["train_iters"] is not None else "",
                    "CurIter":     r.get("last_iter") if r.get("last_iter") is not None else "",
                    "Prog%":       f"{r['progress']:.1f}" if r.get("progress") is not None else "",
                    "TrainLoss":   f"{r['last_train_loss']:.4f}" if r.get("last_train_loss") is not None else "",
                    "ValLoss":     f"{r['last_val_loss']:.4f}" if r.get("last_val_loss") is not None else "",
                    "LastCkpt":    r["last_ckpt"] if r["last_ckpt"] is not None else "",
                    "LR":          f"{r['lr']:.4f}" if r["lr"] is not None else "",
                    "GBS":         r["global_batch_size"] if r["global_batch_size"] is not None else "",
                    "MBS":         r["micro_batch_size"] if r["micro_batch_size"] is not None else "",
                    "Nodes":       r["nodes"] if r.get("nodes") is not None else "",
                    "Workers":     r["num_workers"] if r["num_workers"] is not None else "",
                    "TFLOP/s/GPU": f"{r['avg_tflop_per_gpu']:.1f}" if r["avg_tflop_per_gpu"] is not None else "",
                    "Tok/s/GPU":   f"{r['avg_tok_per_gpu']:.0f}" if r["avg_tok_per_gpu"] is not None else "",
                    "GPU-h":       f"{r['gpu_hours']:.1f}" if r["gpu_hours"] is not None else "",
                    "Emoji":       r["status_emoji"],
                    "Status":      r["status_word"],
                    "Action":      r["action_word"],
                    "Error":       r["error_desc"],
                })
        print(f"\nWrote {len(rows)} rows to {csv_path}")

    if gpu_csv_rows:
        gpu_csv_rows.append({
            "experiment": "TOTAL",
            "job_id": "", "state": "", "elapsed": "", "gpus": "",
            "gpu_hours": round(grand_gpu_total, 1),
        })
        gpu_csv_path = resolved_base / "gpu_hours.csv"
        with gpu_csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["experiment", "job_id", "state", "elapsed", "gpus", "gpu_hours"]
            )
            writer.writeheader()
            writer.writerows(gpu_csv_rows)
        print(f"\nWrote {len(gpu_csv_rows) - 1} job rows (+1 total) to {gpu_csv_path}")

    # ── Markdown output ─────────────────────────────────────────────────────
    if args.md:
        md_path = Path(args.md)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with md_path.open("w") as f:
            f.write(f"_Updated: {now_str}_\n\n")
            f.write(f"# Sweep run status\n\n")
            f.write(f"**Config:** `{args.config}`  \n")
            f.write(f"**Results:** `{resolved_base}`\n\n")

            # Experiment summary GPU consumption + progress table
            f.write("## Training progress summary\n\n")
            f.write("| Experiment | GPU-h | Progress | Status |\n")
            f.write("| --- | --- | --- | --- |\n")
            for exp_name, h in sorted(exp_gpu_h.items()):
                run_rows = [r for r in rows if r["run_name"] == exp_name]
                latest = run_rows[-1] if run_rows else {}
                prog = latest.get("progress")
                prog_str = f"{prog:.1f}%" if prog is not None else "N/A"
                emoji = latest.get("status_emoji", "")
                status = latest.get("status_word", "")
                f.write(f"| {exp_name} | {h:.1f} | {prog_str} | {emoji} {status} |\n")
            f.write(f"| **TOTAL** | **{grand_gpu_total:.1f}** | | |\n")
            f.write("\n")

            # Status summary
            f.write("## Status breakdown\n\n")
            from collections import Counter
            counts = Counter(r["status_word"] for r in rows)
            for status, cnt in sorted(counts.items()):
                f.write(f"- **{status}**: {cnt}\n")
        print(f"\nWrote markdown to {md_path}")


if __name__ == "__main__":
    main()
