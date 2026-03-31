#!/usr/bin/env python3
"""Validate sweep runs against the config YAML that generated them.

Usage:
    python scripts/validate_sweep_runs.py <config.yaml>

Parses the sweep config to extract LRs, batch sizes, num_experts, decay
stages, and results directory, then validates every expected job.
"""

import argparse
import math
import os
import re
import sys
from collections import OrderedDict
from pathlib import Path

import yaml


# ── Config parsing ───────────────────────────────────────────────────────────

def parse_config(config_path: str) -> dict:
    """Extract sweep grid and training parameters from a config YAML."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    meg = cfg["backend"]["megatron"]
    aux = meg["aux"]

    params = {
        "seq_length": int(meg["seq_length"]),
        "stable_tokens": int(aux["tokens"]),
        "cooldown_decay_fraction": float(aux["cooldown_decay_fraction"]),
        "seed": int(meg.get("seed", 1234)),
    }

    # Parent of each job folder: strip ${job.name}; may still contain
    # ${backend.megatron.global_batch_size} etc. — resolved per combo in main().
    raw_dir = cfg["job"]["base_output_dir"]
    params["base_dir_template"] = raw_dir.split("${job.name}")[0].rstrip("/")

    # Walk sweep groups
    groups = cfg["sweep"]["groups"]
    lrs: list[float] = []
    gbszs: list[int] = []
    num_experts: list[int] = []
    decay_stages: OrderedDict[str, int] = OrderedDict()
    job_name_tpl: str | None = None

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
            # Product sub-group (lr × gbsz)
            if entry.get("type") == "product" and "params" in entry:
                p = entry["params"]
                lrs.extend(p.get("backend.megatron.lr", []))
                gbszs.extend(p.get("backend.megatron.global_batch_size", []))
                continue
            # Flat num_experts entry
            if "backend.megatron.num_experts" in entry and "stage" not in entry:
                num_experts.append(int(entry["backend.megatron.num_experts"]))
                continue
            # Stable phase → skip (tokens from base)
            if entry.get("stage") == "stable":
                continue
            # Nested decay list
            if entry.get("type") == "list" and "configs" in entry:
                for dc in entry["configs"]:
                    if isinstance(dc, dict) and "stage" in dc:
                        decay_stages[dc["stage"]] = int(
                            dc["backend.megatron.aux.tokens"]
                        )

    params["lrs"] = [float(v) for v in lrs]
    params["gbszs"] = [int(v) for v in gbszs]
    params["num_experts"] = num_experts
    params["decay_stages"] = decay_stages
    params["job_name_tpl"] = job_name_tpl
    return params


def render_job_name(
    tpl: str | None,
    nexp: int,
    lr: float,
    gbsz: int,
    seed: int,
    stage: str,
) -> str:
    """Substitute concrete values into the escaped-OmegaConf job name template."""
    if tpl is None:
        return f"nexp_{nexp}_lr{lr}_gbsz{gbsz}_seed{seed}_{stage}"
    out = tpl
    for key, val in {
        "backend.megatron.num_experts": nexp,
        "backend.megatron.lr": lr,
        "backend.megatron.global_batch_size": gbsz,
        "backend.megatron.seed": seed,
        "stage": stage,
    }.items():
        out = out.replace(f"\\${{{key}}}", str(val))
    return out


# ── Helpers ──────────────────────────────────────────────────────────────────

def substitute_omegaconf_path_vars(template: str, ctx: dict[str, object]) -> str:
    """Replace ${key} segments when key is in ctx; leave others unchanged."""

    def _repl(m: re.Match[str]) -> str:
        key = m.group(1)
        if key in ctx:
            return str(ctx[key])
        return m.group(0)

    return re.sub(r"\$\{([^}]+)\}", _repl, template)


def resolve_results_base_dir(template: str, nexp: int, lr: float, gbsz: int, seed: int) -> Path:
    """Hydra-style path prefix with sweep variables filled in."""
    ctx = {
        "backend.megatron.global_batch_size": gbsz,
        "backend.megatron.num_experts": nexp,
        "backend.megatron.lr": lr,
        "backend.megatron.seed": seed,
    }
    return Path(substitute_omegaconf_path_vars(template, ctx))


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def compute_train_iters(tokens: int, seq_length: int, gbsz: int) -> int:
    return ceil_div(tokens, seq_length * gbsz)


def compute_start_iter(tokens: int, seq_length: int, gbsz: int, cdf: float) -> int:
    return int(compute_train_iters(tokens, seq_length, gbsz) * (1 - cdf))


def compute_save_extra_steps(
    decay_stages: dict[str, int], seq_length: int, gbsz: int, cdf: float,
) -> list[int]:
    return [compute_start_iter(t, seq_length, gbsz, cdf) for t in decay_stages.values()]


# ── Log parsing ──────────────────────────────────────────────────────────────

def parse_log(log_path: Path) -> dict:
    """Extract key metrics from a Megatron log file."""
    info: dict = {
        "exists": log_path.exists(),
        "train_iters": None,
        "last_val_iter": None,
        "last_train_iter": None,
        "save_extra_steps": None,
        "ckpt_step": None,
        "val_loss": None,
        "errors": [],
    }
    if not info["exists"]:
        return info

    content = log_path.read_bytes().decode("utf-8", errors="replace")

    m = re.search(r"train_iters\s*\.+\s*(\d+)", content)
    if m:
        info["train_iters"] = int(m.group(1))

    m = re.search(r"save_extra_steps\s*\.+\s*\[([^\]]+)\]", content)
    if m:
        info["save_extra_steps"] = [int(x.strip()) for x in m.group(1).split(",")]

    m = re.search(r"--ckpt-step\s+(\d+)", content)
    if m:
        info["ckpt_step"] = int(m.group(1))

    for m in re.finditer(r"validation loss at iteration (\d+)", content):
        info["last_val_iter"] = int(m.group(1))

    for m in re.finditer(r" iteration\s+(\d+)/", content):
        info["last_train_iter"] = int(m.group(1))

    val_losses = re.findall(r"lm loss value:\s+([\d.E+-]+)", content)
    if val_losses:
        info["val_loss"] = float(val_losses[-1])

    for pat in ("FATAL ERROR", "OutOfMemoryError", "Traceback"):
        if pat in content:
            info["errors"].append(pat)

    return info


# ── Validation ───────────────────────────────────────────────────────────────

def validate_job(
    base_dir: Path,
    name: str,
    stage: str,
    tokens: int,
    seq_length: int,
    gbsz: int,
    cdf: float,
    decay_stages: dict[str, int],
) -> dict:
    """Validate a single job against expected parameters."""
    job_dir = base_dir / name
    log_path = job_dir / "current.log"
    expected_iters = compute_train_iters(tokens, seq_length, gbsz)
    is_stable = stage == "stable"

    result = {
        "name": name,
        "stage": stage,
        "dir_exists": job_dir.exists(),
        "expected_train_iters": expected_iters,
        "issues": [],
        "status": "UNKNOWN",
    }

    if not result["dir_exists"]:
        result["status"] = "MISSING"
        result["issues"].append("directory not found")
        return result

    log = parse_log(log_path)
    result["log_info"] = log

    if not log["exists"]:
        result["status"] = "NO_LOG"
        result["issues"].append("current.log not found")
        return result

    if log["errors"]:
        result["issues"].append(f"errors in log: {log['errors']}")

    if log["train_iters"] is not None and log["train_iters"] != expected_iters:
        result["issues"].append(
            f"train_iters mismatch: got {log['train_iters']}, expected {expected_iters}"
        )

    completed = False
    if log["last_val_iter"] is not None:
        if log["last_val_iter"] == expected_iters:
            completed = True
        elif log["last_val_iter"] < expected_iters:
            result["issues"].append(
                f"incomplete: last_val_iter={log['last_val_iter']}, expected={expected_iters}"
            )
        else:
            result["issues"].append(
                f"overshot: last_val_iter={log['last_val_iter']}, expected={expected_iters}"
            )

    if is_stable:
        expected_ses = compute_save_extra_steps(decay_stages, seq_length, gbsz, cdf)
        if log["save_extra_steps"] is not None:
            if log["save_extra_steps"] != expected_ses:
                result["issues"].append(
                    f"save_extra_steps mismatch: got {log['save_extra_steps']}, expected {expected_ses}"
                )
        else:
            result["issues"].append("save_extra_steps not found in log")
    else:
        expected_start = compute_start_iter(tokens, seq_length, gbsz, cdf)
        if log["ckpt_step"] is not None and log["ckpt_step"] != expected_start:
            result["issues"].append(
                f"ckpt_step mismatch: got {log['ckpt_step']}, expected {expected_start}"
            )

    if completed and not log["errors"]:
        result["status"] = "OK"
    elif log["errors"]:
        result["status"] = "ERROR"
    elif not completed and log["last_val_iter"] is not None:
        result["status"] = "INCOMPLETE"
    else:
        result["status"] = "UNCLEAR"

    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Validate sweep runs against config")
    parser.add_argument("config", help="Path to the sweep config YAML")
    args = parser.parse_args()

    os.chdir(Path(__file__).resolve().parent.parent)

    cfg = parse_config(args.config)
    base_dir_template = cfg["base_dir_template"]
    seq = cfg["seq_length"]
    cdf = cfg["cooldown_decay_fraction"]
    seed = cfg["seed"]
    stable_tok = cfg["stable_tokens"]
    decay_stages = cfg["decay_stages"]

    n_stages = 1 + len(decay_stages)
    n_combos = len(cfg["lrs"]) * len(cfg["gbszs"]) * len(cfg["num_experts"])

    print("=" * 110)
    print(f"Config:  {args.config}")
    if cfg["lrs"] and cfg["gbszs"] and cfg["num_experts"]:
        _ex = resolve_results_base_dir(
            base_dir_template, cfg["num_experts"][0], cfg["lrs"][0], cfg["gbszs"][0], seed
        )
        print(f"Results: {base_dir_template}/")
        print(f"         (resolved e.g. {_ex}/)")
    else:
        print(f"Results: {base_dir_template}/")
    print(
        f"Grid:    {len(cfg['lrs'])} lr × {len(cfg['gbszs'])} gbsz × "
        f"{len(cfg['num_experts'])} nexp × {n_stages} stages = {n_combos * n_stages} jobs"
    )
    print("=" * 110)

    results = []
    for nexp in cfg["num_experts"]:
        for lr in cfg["lrs"]:
            for gbsz in cfg["gbszs"]:
                base_dir = resolve_results_base_dir(base_dir_template, nexp, lr, gbsz, seed)
                name = render_job_name(cfg["job_name_tpl"], nexp, lr, gbsz, seed, "stable")
                results.append(
                    validate_job(base_dir, name, "stable", stable_tok, seq, gbsz, cdf, decay_stages)
                )
                for stage_name, stage_tok in decay_stages.items():
                    name = render_job_name(cfg["job_name_tpl"], nexp, lr, gbsz, seed, stage_name)
                    results.append(
                        validate_job(base_dir, name, stage_name, stage_tok, seq, gbsz, cdf, decay_stages)
                    )

    total = len(results)
    ok = sum(1 for r in results if r["status"] == "OK")
    failed = [r for r in results if r["status"] != "OK"]

    header = f"{'Job Name':<70} {'Expected':>8} {'Actual':>8} {'Status':>10}"
    print(header)
    print("-" * 110)

    status_colors = {
        "OK": "\033[92m",
        "MISSING": "\033[91m",
        "ERROR": "\033[91m",
        "INCOMPLETE": "\033[93m",
        "UNCLEAR": "\033[93m",
        "NO_LOG": "\033[91m",
    }

    for r in results:
        actual = ""
        if "log_info" in r and r["log_info"]["last_val_iter"] is not None:
            actual = str(r["log_info"]["last_val_iter"])
        elif r["status"] == "MISSING":
            actual = "-"

        suffix = ""
        if "log_info" in r and r["log_info"].get("val_loss") is not None:
            suffix = f"  ppl={math.exp(r['log_info']['val_loss']):.1f}"

        color = status_colors.get(r["status"], "")
        reset = "\033[0m" if color else ""

        print(
            f"{r['name']:<70} {r['expected_train_iters']:>8} {actual:>8} "
            f"{color}{r['status']:>10}{reset}{suffix}"
        )

    if failed:
        print()
        print("=" * 110)
        print(f"ISSUES ({len(failed)} jobs):")
        print("=" * 110)
        for r in failed:
            print(f"\n  {r['name']}  [{r['status']}]")
            for issue in r["issues"]:
                print(f"    - {issue}")

    # Reference table
    print()
    print("-" * 110)
    print("Expected train_iters per (gbsz, stage):")
    for gbsz in cfg["gbszs"]:
        parts = [f"stable={compute_train_iters(stable_tok, seq, gbsz)}"]
        for sn, st in decay_stages.items():
            parts.append(f"{sn}={compute_train_iters(st, seq, gbsz)}")
        print(f"  gbsz={gbsz}: {', '.join(parts)}")

    print()
    print("Expected save_extra_steps (stable only):")
    for gbsz in cfg["gbszs"]:
        ses = compute_save_extra_steps(decay_stages, seq, gbsz, cdf)
        print(f"  gbsz={gbsz}: {ses}")

    print()
    print(f"{'=' * 110}")
    print(f"Result: {ok}/{total} jobs OK")
    print(f"{'=' * 110}")
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
