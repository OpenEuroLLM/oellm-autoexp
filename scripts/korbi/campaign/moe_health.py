"""MoE router / expert health per run from TensorBoard (docs/hybrid_moe_3b_campaign.md §10.3).

    uv run --with tensorboard python scripts/korbi/campaign/moe_health.py \
        ~/.cache/campaign_3b_tb/screen3b_s1 [...] [--tail 0.2] [--csv out.csv]

Megatron's moe_per_layer_logging + moe_expert_viability_metrics (moe_health.py) write the
router metrics to TensorBoard / wandb only (not stdout), under moe/*. Per run (<suite>/<arm>/tb):
- entropy: dispatched expert-load entropy (Megatron logs it normalised by log E: 1 = uniform
  load), mean over layers, over the
  last ``--tail`` of the run, and its minimum over layers (the worst layer);
- max_vio: selected-expert max violation (max load / mean load - 1), worst layer, tail mean and
  whole-run max;
- dead: dead experts (zero load) summed over layers, tail mean; persistent near-dead
  expert-layer slots (< 10 % of uniform for >= 100 steps), at the end;
- logit spread: router logit spread, worst layer, whole-run max (router logits running away);
- router entropy: mean router score entropy, normalised by log E (1 = uniform scores, low = a
  confident router), tail mean.
Fetch the TB logs with scripts/korbi/campaign/wandb_sync_tb.sh (its first step rsyncs every
run's tb/ into ~/.cache/campaign_3b_tb; DRY_RUN=1 rsyncs without uploading).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def scalars(tb: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out: dict[str, list] = {}
    for ev in sorted(tb.rglob("events.out.tfevents.*")):
        acc = EventAccumulator(str(ev), size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags()["scalars"]:
            if tag.startswith("moe/") or tag in ("lm loss", "num-experts"):
                out.setdefault(tag, []).extend((e.step, e.value) for e in acc.Scalars(tag))
    res = {}
    for tag, sv in out.items():
        d = dict(sv)  # a resumed run rewrites steps: keep the last value per step
        s = np.array(sorted(d))
        res[tag] = (s, np.array([d[k] for k in s]))
    return res


def summary(sc: dict, tail: float) -> dict:
    if not any(t.startswith("moe/") for t in sc):
        return {}
    last = max(s[-1] for s, _ in sc.values())
    t0 = (1 - tail) * last

    def tail_mean(tag):
        if tag not in sc:
            return float("nan")
        s, v = sc[tag]
        return float(np.mean(v[s >= t0])) if (s >= t0).any() else float("nan")

    def run_max(tag):
        return float(np.max(sc[tag][1])) if tag in sc else float("nan")

    def end(tag):
        return float(sc[tag][1][-1]) if tag in sc else float("nan")

    return {
        "steps": int(last),
        "entropy": tail_mean("moe/dispatched_expert_load_entropy_mean"),
        "entropy_min": tail_mean("moe/dispatched_expert_load_entropy_min"),
        "max_vio": tail_mean("moe/selected_expert_max_vio_max"),
        "max_vio_peak": run_max("moe/selected_expert_max_vio_max"),
        "dead": tail_mean("moe/expert_dead_count"),
        "near_dead_persistent": end("moe/persistent_selected_near_dead_expert_layer_slots_100"),
        "logit_spread_peak": run_max("moe/expert_logit_spread_max"),
        "router_entropy": tail_mean("moe/router_mean_score_entropy_mean"),
        "loss": tail_mean("lm loss"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("suites", nargs="+", type=Path, help="<suite> dirs with <arm>/tb/")
    ap.add_argument("--tail", type=float, default=0.2, help="fraction of the run averaged (end)")
    ap.add_argument("--csv", type=Path)
    a = ap.parse_args()
    rows = []
    for suite in a.suites:
        for tb in sorted(suite.glob("*/tb")):
            rundir = tb.parent
            s = summary(scalars(tb), a.tail)
            if s:
                rows.append(dict(suite=suite.name, arm=rundir.name, **s))
    if not rows:
        print("no MoE metrics found (moe_per_layer_logging off, or TB logs not fetched)")
        return
    cols = ["steps", "loss", "entropy", "entropy_min", "max_vio", "max_vio_peak", "dead",
            "near_dead_persistent", "logit_spread_peak", "router_entropy"]
    print("| suite | arm | " + " | ".join(cols) + " |")
    print("|---|---|" + "---:|" * len(cols))
    for r in rows:
        print(f"| {r['suite']} | {r['arm']} | " + " | ".join(
            f"{r[c]:.0f}" if c in ("steps", "near_dead_persistent") else f"{r[c]:.3f}" for c in cols) + " |")
    if a.csv:
        with a.csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
