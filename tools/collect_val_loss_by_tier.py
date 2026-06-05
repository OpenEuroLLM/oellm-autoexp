#!/usr/bin/env python3
"""Collect final validation losses from a tiered scaling-law eval folder.

Point it at a directory that contains one subfolder per run, e.g.
    /leonardo_work/.../0.2B_ne/validation_leo
where each run folder looks like
    qwen3_dense_0.2B_ne_lr0.001_gbsz128_decay30BT/
        current.yaml           (-> resolved config-<jobid>.yaml)
        logs/stdout-<jobid>.log
        ...

For every run it writes one CSV row with columns:
    bsz, lr, tokens, stage, run_type, iteration, lm loss, run_name

Where:
  * bsz / lr            come from the resolved config (backend.megatron.*).
  * tokens              is the run's token budget in billions (aux.tokens // 1e9):
                        the decay budget for a decay run, the stable's
                        max_decay_tokens for a stable run.
  * stage               is "stable" or "decay".
  * run_type            is the tier (center / cross / diagonal):
                          - decay  -> the tier whose *_tokens_set contains aux.tokens
                          - stable -> aux.stable_launch_tier
                        This mirrors the sweep.filter tier routing, read straight
                        from each run's own resolved config, so it works for any
                        model size without re-parsing the sweep YAML.
  * iteration / lm loss come from the LAST "validation loss at iteration ..."
                        line in the run's stdout log (the metric is printed to
                        stdout, not stderr/current.log).

Usage:
    python tools/collect_val_loss_by_tier.py <runs_dir> [-o <output.csv>]

Requires PyYAML (the resolved configs are YAML).
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("PyYAML is required (pip install pyyaml); the resolved configs are YAML.")

# "[default3]: validation loss at iteration 57221 on validation set | lm loss value: 2.597687E+00 | lm loss PPL: ... |"
VAL_LOSS_RE = re.compile(
    r"validation loss at iteration\s+(?P<iteration>\d+)\s+on validation set\s*\|"
    r".*?lm loss value:\s*(?P<lm_loss>[\d.E+\-]+)"
)

# Fallback parse from the run folder name, e.g.
#   qwen3_dense_0.2B_ne_lr0.001_gbsz128_decay30BT
#   qwen3_dense_0.2B_ne_lr0.002_gbsz512_stable300BT
RUN_NAME_RE = re.compile(
    r"lr(?P<lr>[\d.eE+\-]+)_gbsz(?P<bsz>\d+)_(?P<stage>stable|decay)(?P<bt>\d+)BT"
)

TIERS = ("center", "cross", "diagonal")

OUT_COLS = ["index", "tier_index", "bsz", "lr", "tokens", "stage", "run_type", "iteration", "lm loss", "run_name"]


def parse_token_set(value: object) -> set[int]:
    """Parse a Python-set-literal string like '{6_000_000_000, 12_000_000_000}' or 'set()'."""
    if value is None:
        return set()
    s = str(value).strip()
    if s in ("set()", "", "{}", "None"):
        return set()
    s = s.strip().lstrip("{").rstrip("}")
    out: set[int] = set()
    for part in s.split(","):
        part = part.strip().replace("_", "")
        if part:
            try:
                out.add(int(part))
            except ValueError:
                pass
    return out


def load_config(run_dir: Path) -> dict | None:
    """Load the run's resolved config (current.yaml, else newest config-*.yaml)."""
    cfg_path: Path | None = None
    cur = run_dir / "current.yaml"
    if cur.exists():
        cfg_path = cur
    else:
        candidates = sorted(run_dir.glob("config-*.yaml"), key=lambda p: p.stat().st_mtime, reverse=True)
        cfg_path = candidates[0] if candidates else None
    if cfg_path is None:
        return None
    try:
        with cfg_path.open(errors="replace") as f:
            root = yaml.safe_load(f)
    except Exception:
        return None
    # Resolved configs are nested under a top-level "config:" key.
    if isinstance(root, dict) and "config" in root and isinstance(root["config"], dict):
        return root["config"]
    return root if isinstance(root, dict) else None


def fields_from_config(cfg: dict) -> dict | None:
    """Pull bsz/lr/tokens/stage/run_type out of a resolved config dict."""
    try:
        mg = cfg["backend"]["megatron"]
        aux = mg["aux"]
    except (KeyError, TypeError):
        return None

    stage_raw = str(cfg.get("stage", "")).strip()
    stage = "stable" if stage_raw.startswith("stable") else ("decay" if stage_raw else "")

    tokens_raw = aux.get("tokens")
    tokens_bt = int(tokens_raw) // 1_000_000_000 if tokens_raw is not None else None

    if stage == "stable":
        run_type = str(aux.get("stable_launch_tier", "")).strip() or "unknown"
    elif stage == "decay" and tokens_raw is not None:
        run_type = "unknown"
        for tier in TIERS:
            if int(tokens_raw) in parse_token_set(aux.get(f"{tier}_tokens_set")):
                run_type = tier
                break
    else:
        run_type = "unknown"

    return {
        "bsz": mg.get("global_batch_size"),
        "lr": mg.get("lr"),
        "tokens": tokens_bt,
        "stage": stage,
        "run_type": run_type,
    }


def fields_from_name(name: str) -> dict | None:
    """Fallback: parse bsz/lr/tokens/stage from the folder name (run_type unknown)."""
    m = RUN_NAME_RE.search(name)
    if not m:
        return None
    return {
        "bsz": int(m.group("bsz")),
        "lr": float(m.group("lr")),
        "tokens": int(m.group("bt")),
        "stage": m.group("stage"),
        "run_type": "unknown",
    }


def candidate_logs(run_dir: Path) -> list[Path]:
    """All .log files for the run (logs/ subdir + run dir), newest first."""
    logs: set[Path] = set()
    logs_dir = run_dir / "logs"
    if logs_dir.is_dir():
        logs.update(logs_dir.glob("*.log"))
    logs.update(run_dir.glob("*.log"))
    return sorted((p for p in logs if p.is_file()), key=lambda p: p.stat().st_mtime, reverse=True)


def last_val_loss(log_path: Path) -> tuple[int, float] | None:
    last = None
    with log_path.open(errors="replace") as f:
        for line in f:
            m = VAL_LOSS_RE.search(line)
            if m:
                last = (int(m.group("iteration")), float(m.group("lm_loss")))
    return last


def find_val_loss(run_dir: Path) -> tuple[int, float] | None:
    """Return (iteration, lm_loss) from the most recent log that has a match."""
    for log_path in candidate_logs(run_dir):
        result = last_val_loss(log_path)
        if result is not None:
            return result
    return None


def collect(runs_dir: Path) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    warnings: list[str] = []
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        cfg = load_config(run_dir)
        meta = fields_from_config(cfg) if cfg else None
        if meta is None:
            meta = fields_from_name(run_dir.name)
            if meta is None:
                warnings.append(f"skip (no config & unrecognised name): {run_dir.name}")
                continue
            warnings.append(f"no config, used folder name (run_type=unknown): {run_dir.name}")

        loss = find_val_loss(run_dir)
        if loss is None:
            warnings.append(f"skip (no validation loss line): {run_dir.name}")
            continue
        iteration, lm_loss = loss

        rows.append({
            "bsz": meta["bsz"],
            "lr": meta["lr"],
            "tokens": meta["tokens"],
            "stage": meta["stage"],
            "run_type": meta["run_type"],
            "iteration": iteration,
            "lm loss": lm_loss,
            "run_name": run_dir.name,
        })

    # Sort: tier first (center -> cross -> diagonal -> unknown), then stable
    # before decay within a tier, then by bsz / lr / tokens.
    tier_rank = {"center": 0, "cross": 1, "diagonal": 2, "unknown": 3}

    def sort_key(r: dict):
        return (
            tier_rank.get(r["run_type"], 99),
            0 if r["stage"] == "stable" else 1,
            r["bsz"] if isinstance(r["bsz"], int) else 1 << 30,
            r["lr"] if isinstance(r["lr"], (int, float)) else 1e30,
            r["tokens"] if isinstance(r["tokens"], int) else 1 << 30,
        )

    rows.sort(key=sort_key)

    # Assign a global 1-based index and a per-tier index that resets each tier.
    tier_counter: dict[str, int] = {}
    for i, r in enumerate(rows, start=1):
        r["index"] = i
        tier_counter[r["run_type"]] = tier_counter.get(r["run_type"], 0) + 1
        r["tier_index"] = tier_counter[r["run_type"]]

    return rows, warnings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs_dir", type=Path, help="Directory containing per-run subfolders")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output CSV (default: <runs_dir>/validation_loss_by_tier.csv)",
    )
    args = parser.parse_args()

    if not args.runs_dir.is_dir():
        sys.exit(f"Error: {args.runs_dir} is not a directory")

    out_path = args.output or (args.runs_dir / "validation_loss_by_tier.csv")

    rows, warnings = collect(args.runs_dir)

    for w in warnings:
        print(f"  [warn] {w}", file=sys.stderr)

    if not rows:
        sys.exit("No validation results collected — nothing written.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_COLS)
        writer.writeheader()
        writer.writerows(rows)

    n_unknown = sum(1 for r in rows if r["run_type"] == "unknown")
    print(f"Wrote {len(rows)} rows to {out_path}"
          + (f" ({n_unknown} with run_type=unknown)" if n_unknown else ""))


if __name__ == "__main__":
    main()
