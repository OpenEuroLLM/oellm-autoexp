#!/usr/bin/env python3
"""Extract key TensorBoard metrics from bilingual training runs.

Usage:
    python scripts/extract_tb_metrics.py [--output /tmp/bilingual_tb_metrics.json]
"""

import argparse
import json
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

OUTPUT_DIR = Path("/home/risto.luukkonen@amd.com/rluukkon/oellm/oellm-autoexp/output")

AGG_TAGS = [
    "lm loss",
    "grad-norm",
    "learning-rate",
    "load_balancing_loss",
    "global_load_balancing_loss",
    "moe/expert_dead_count",
    "moe/expert_mean_entropy_pct",
    "moe/expert_min_entropy_pct",
    "moe/expert_max_frac",
    "moe/expert_logit_spread_mean",
    "moe/expert_logit_spread_max",
    "moe/expert_logit_spread_norm_max",
    "router_mean_max_prob",
    "router_logit_logsumexp",
    "moe/router_mean_token_entropy",
    "moe/router_mean_max_prob",
]


def extract_run(tb_dir: Path) -> dict:
    ea = EventAccumulator(str(tb_dir), size_guidance={"scalars": 0})
    ea.Reload()

    available = ea.Tags().get("scalars", [])
    run_data = {}
    for tag in AGG_TAGS:
        if tag not in available:
            continue
        events = ea.Scalars(tag)
        if not events:
            continue
        steps = [e.step for e in events]
        values = [e.value for e in events]
        d = {
            "n_points": len(events),
            "step_range": [min(steps), max(steps)],
            "first_val": values[0],
            "last_val": values[-1],
            "min_val": min(values),
            "max_val": max(values),
            "mean_val": sum(values) / len(values),
        }
        ckpt_vals = {}
        for target in [300, 600, 900, 1200, 1500]:
            closest = min(events, key=lambda e: abs(e.step - target))
            if abs(closest.step - target) <= 10:
                ckpt_vals[str(target)] = closest.value
        if ckpt_vals:
            d["at_ckpts"] = ckpt_vals
        run_data[tag] = d
    return run_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", "-o",
        default="/tmp/bilingual_tb_metrics.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--pattern",
        default="qwen3_5_35B_A3B_bilingual_*",
        help="Glob pattern for run directories",
    )
    args = parser.parse_args()

    runs = sorted(OUTPUT_DIR.glob(args.pattern))
    results = {}

    for run_dir in runs:
        tb_dir = run_dir / "tensorboard"
        if not tb_dir.exists():
            continue

        run_name = run_dir.name.replace("qwen3_5_35B_A3B_bilingual_", "")
        print(f"\n=== {run_name} ===")

        run_data = extract_run(tb_dir)
        results[run_name] = run_data

        for tag in AGG_TAGS:
            if tag in run_data:
                d = run_data[tag]
                fv = d["first_val"]
                lv = d["last_val"]
                mn = d["min_val"]
                mx = d["max_val"]
                ckpt_str = ""
                if "at_ckpts" in d:
                    parts = []
                    for k, v in sorted(d["at_ckpts"].items()):
                        parts.append(f"{k}:{v:.4f}")
                    ckpt_str = " | ckpts: " + ", ".join(parts)
                print(f"  {tag}: first={fv:.6f} last={lv:.6f} min={mn:.6f} max={mx:.6f}{ckpt_str}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
