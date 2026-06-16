"""Cross-run overlay of the OFFLINE-computable MoE metrics — no job, no reload.

Reads each run's `saturation.json` (the richest cached file: per-step, per-layer
it carries `expert_token_counts`, `compare_saturation`, and the scalar
`expert_coactivation`) and overlays all runs on shared axes for the five
metrics we can compute without re-running the model:

    load CV (%)         load imbalance (sharp)            <- from token counts
    load entropy (%)    "effective experts used"          <- from token counts
    dead experts        zero-token experts                <- from token counts
    saturation (%)      top-k overlap vs final            <- compare_saturation
    coactivation        per-step scalar (perm-invariant)  <- expert_coactivation

Each curve is the layer-mean per step. Colour = run family (aux-loss variant),
line style = expert count (solid=nexp_64, dashed=nexp_8) so the load-balancing
method comparison is readable within each expert-count group.

logit_spread / mean_max_prob are intentionally excluded: they need raw logits,
which are not cached.

Usage:
    python scripts/moe/compare_moe_runs.py \
        results/moe_analysis_auxloss results/moe_analysis_deepseek_bias \
        results/moe_analysis_global_batch_aux --out results/moe_compare
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_NEXP_RE = re.compile(r"nexp_(\d+)")


def load_metrics(counts: np.ndarray) -> Tuple[float, float, float, int]:
    """cv_pct, entropy_pct, max_frac, dead_count (== failcase_metrics defs)."""
    counts = np.asarray(counts, dtype=np.float64)
    tot, mean = counts.sum(), counts.mean()
    p = counts / tot if tot > 0 else np.zeros_like(counts)
    nz = p[p > 0]
    ent = (-(nz * np.log(nz)).sum() / np.log(len(counts)) * 100.0) if len(nz) else 0.0
    cv = (counts.std() / mean * 100.0) if mean > 0 else 0.0
    mf = (counts.max() / tot) if tot > 0 else 0.0
    dead = int((counts == 0).sum())
    return cv, ent, mf, dead


def collect_run(leaf: Path) -> Optional[dict]:
    sat_path = leaf / "saturation.json"
    if not sat_path.exists():
        return None
    sat = json.loads(sat_path.read_text())
    layers_by_step = sat.get("layers", {})
    steps = sorted((int(s) for s in layers_by_step), key=int)
    series: Dict[str, List[float]] = {k: [] for k in
                                      ("cv", "ent", "dead", "sat", "coact")}
    xs: List[int] = []
    for s in steps:
        per_layer = layers_by_step[str(s)]
        cvs, ents, deads, sats, coacts = [], [], [], [], []
        for entry in per_layer.values():
            counts = entry.get("expert_token_counts")
            if counts is not None:
                cv, ent, _mf, dead = load_metrics(counts)
                cvs.append(cv); ents.append(ent); deads.append(dead)
            if "compare_saturation" in entry:
                sats.append(entry["compare_saturation"] * 100.0)
            if "expert_coactivation" in entry:
                coacts.append(entry["expert_coactivation"])
        if not cvs:
            continue
        xs.append(s)
        series["cv"].append(float(np.mean(cvs)))
        series["ent"].append(float(np.mean(ents)))
        # dead experts: report the WORST layer, not the layer-mean. The mean
        # smears a localized deep-layer collapse into a misleading fraction
        # (e.g. 18 dead in 2 layers reads as "1.0 per layer"); max is honest.
        series["dead"].append(float(np.max(deads)) if deads else 0.0)
        series["sat"].append(float(np.mean(sats)) if sats else np.nan)
        series["coact"].append(float(np.mean(coacts)) if coacts else np.nan)
    nexp_m = _NEXP_RE.search(str(leaf))
    return {"steps": xs, "series": series,
            "nexp": int(nexp_m.group(1)) if nexp_m else None}


def plot_compare(runs: Dict[str, dict], out_dir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("cv", "load CV (%)"), ("ent", "load entropy (%)"),
              ("dead", "dead experts (max / layer)"),
              ("sat", "saturation vs final (%)"),
              ("coact", "coactivation (scalar)")]
    # stable colour per run-family
    families = sorted({lbl.split(" | ")[0] for lbl in runs})
    cmap = plt.get_cmap("tab10")
    fam_color = {f: cmap(i % 10) for i, f in enumerate(families)}

    # x-axis = % of training so runs of different length align for evolution.
    def xfrac(steps):
        s = np.asarray(steps, float)
        span = s[-1] - s[0]
        return (s - s[0]) / span * 100.0 if span > 0 else s * 0.0

    fig, axes = plt.subplots(2, 3, figsize=(17, 8))
    axes = axes.ravel()
    for ax, (key, title) in zip(axes, panels):
        for label, r in runs.items():
            fam = label.split(" | ")[0]
            ls = "-" if (r["nexp"] or 0) >= 64 else "--"
            ax.plot(xfrac(r["steps"]), r["series"][key], ls, lw=1.8,
                    color=fam_color[fam], label=label)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("% of training")
    axes[3].set_ylim(0, 108)
    axes[-1].axis("off")
    axes[-1].legend(*axes[0].get_legend_handles_labels(),
                    fontsize=8, loc="center", frameon=False)
    fig.suptitle("Offline-computable MoE metrics — EVOLUTION over training "
                 "(layer-mean; x=% of training; solid=nexp_64, dashed=nexp_8)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / "computable_metrics_compare.png"
    fig.savefig(fp, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return fp


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("results/moe_compare"))
    args = ap.parse_args()

    runs: Dict[str, dict] = {}
    for root in args.roots:
        for sat in sorted(root.rglob("saturation.json")):
            leaf = sat.parent
            r = collect_run(leaf)
            if r is None:
                continue
            tag = str(leaf).replace(str(root) + "/", "")
            label = f"{root.name.replace('moe_analysis_', '')} | {tag}"
            runs[label] = r

    if not runs:
        print("no saturation.json found under given roots"); return 1

    fp = plot_compare(runs, args.out)
    # final-step table
    print(f"wrote {fp}\n")
    hdr = f"{'run':<46} {'step':>7} {'cv%':>7} {'ent%':>7} {'dead_max':>9} {'sat%':>7} {'coact':>8}"
    print(hdr); print("-" * len(hdr))
    rows = []
    for label, r in runs.items():
        i = -1
        rows.append((label, r["steps"][i], r["series"]["cv"][i],
                     r["series"]["ent"][i], r["series"]["dead"][i],
                     r["series"]["sat"][i], r["series"]["coact"][i]))
    for label, st, cv, ent, dead, sat, co in sorted(rows):
        print(f"{label:<46} {st:>7} {cv:>7.1f} {ent:>7.1f} {dead:>9.0f} "
              f"{sat:>7.1f} {co:>8.4f}")
    (args.out / "computable_metrics_compare.json").write_text(
        json.dumps({lbl: r for lbl, r in runs.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
