"""Offline load-family MoE metrics for our runs — NO job, NO model reload.

Everything here is derived purely from the already-cached `token_counts.json`
(per-step, per-layer, post-top-k expert counts), so it runs in seconds on the
login node. Computes the load-family metrics, identical to the definitions in
`failcase_metrics.py`:

  * cv_pct       — coefficient of variation of per-expert load (sharp imbalance alarm)
  * entropy_pct  — normalised Shannon entropy of load ("effective experts used %")
  * max_frac     — busiest-expert fraction (dull at high top-k: capped near 1/k)
  * dead_count   — experts with zero tokens

across the FULL training trajectory (every cached step), per layer.

What is NOT here, and why:
  * logit_spread_norm and mean_max_prob need the RAW logits / per-token softmax,
    which we do not cache (token counts are already collapsed over top-k). Those
    require re-running the model — i.e. a SLURM job — so they are out of scope.
  * router saturation already lives in saturation.json (per step); co-activation
    in coactivation.json (final). We don't recompute them here.

Usage:
    python scripts/moe/load_metrics_offline.py \
        results/moe_analysis_auxloss results/moe_analysis_deepseek_bias \
        results/moe_analysis_global_batch_aux
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_LAYER_IDX = re.compile(r"layers?[._](\d+)")


# --- metric defs (kept in lock-step with failcase_metrics.load_metrics) ----- #
def load_metrics(counts: np.ndarray) -> Dict[str, float]:
    counts = np.asarray(counts, dtype=np.float64)
    tot = counts.sum()
    mean = counts.mean()
    p = counts / tot if tot > 0 else np.zeros_like(counts)
    nz = p[p > 0]
    ent_pct = (-(nz * np.log(nz)).sum() / np.log(len(counts)) * 100.0) if len(nz) else 0.0
    return {
        "cv_pct": (counts.std() / mean * 100.0) if mean > 0 else 0.0,
        "max_frac": (counts.max() / tot) if tot > 0 else 0.0,
        "entropy_pct": ent_pct,
        "dead_count": int((counts == 0).sum()),
    }


def _layer_idx(name: str) -> int:
    m = _LAYER_IDX.search(name)
    return int(m.group(1)) if m else 1_000_000


def _counts_of(entry) -> Optional[List[float]]:
    """token_counts.json layer value is either {counts, topk} (Megatron) or a
    bare list (HF pipeline). Return the counts list."""
    if isinstance(entry, dict):
        return entry.get("counts")
    if isinstance(entry, list):
        return entry
    return None


def process_run(leaf: Path) -> Optional[dict]:
    tc_path = leaf / "token_counts.json"
    if not tc_path.exists():
        return None
    tc = json.loads(tc_path.read_text())

    # Detect schema: per-step {step: {layer: ...}} vs single {_meta, layers}.
    if "layers" in tc and all(k in ("layers", "_meta") for k in tc):
        steps = {"final": tc["layers"]}
    else:
        steps = {s: tc[s] for s in tc if s != "_meta"}

    def step_key(s: str):
        try:
            return int(s)
        except ValueError:
            return 1 << 62

    ordered = sorted(steps, key=step_key)
    per_step: Dict[str, dict] = {}
    layer_names: set = set()
    for s in ordered:
        layers = steps[s]
        per_layer = {}
        for lname, entry in layers.items():
            counts = _counts_of(entry)
            if counts is None:
                continue
            per_layer[lname] = load_metrics(counts)
            layer_names.add(lname)
        per_step[s] = per_layer

    layer_names = sorted(layer_names, key=_layer_idx)
    return {
        "steps": ordered,
        "layers": layer_names,
        "per_step": per_step,
        "topk": _topk_of(steps[ordered[0]]) if ordered else None,
        "n_experts": _nexp_of(steps[ordered[0]]) if ordered else None,
    }


def _topk_of(layers: dict) -> Optional[int]:
    for entry in layers.values():
        if isinstance(entry, dict) and "topk" in entry:
            return int(entry["topk"])
    return None


def _nexp_of(layers: dict) -> Optional[int]:
    for entry in layers.values():
        c = _counts_of(entry)
        if c is not None:
            return len(c)
    return None


def _series(res: dict, metric: str) -> Tuple[List[int], np.ndarray]:
    """Return (numeric_steps, [n_steps x n_layers] matrix) for a metric."""
    steps_num, rows = [], []
    for s in res["steps"]:
        try:
            sv = int(s)
        except ValueError:
            sv = len(steps_num)
        steps_num.append(sv)
        rows.append([res["per_step"][s].get(ln, {}).get(metric, np.nan)
                     for ln in res["layers"]])
    return steps_num, np.asarray(rows, dtype=float)


def plot_run(res: dict, out_dir: Path, title: str) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # dead_count is reported as MAX over layers (the worst, honest signal);
    # cv/entropy as layer-mean. See note in compare_moe_runs.py.
    metrics = [("cv_pct", "load CV (%)", "mean"),
               ("entropy_pct", "load entropy (%)", "mean"),
               ("dead_count", "dead experts (max / layer)", "max")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (m, lbl, agg) in zip(axes, metrics):
        steps, mat = _series(res, m)
        # thin per-layer lines + bold aggregate
        ax.plot(steps, mat, color="#bbbbbb", lw=0.6, alpha=0.7)
        agg_vals = np.nanmax(mat, axis=1) if agg == "max" else np.nanmean(mat, axis=1)
        ax.plot(steps, agg_vals, color="#c44e52", lw=2.0,
                label=f"layer {agg}")
        ax.set_title(lbl, fontsize=10)
        ax.set_xlabel("training step")
        ax.legend(fontsize=8, loc="best")
    sub = f"top_k={res['topk']}  n_experts={res['n_experts']}  " \
          f"steps={len(res['steps'])}  layers={len(res['layers'])}"
    fig.suptitle(f"{title}\n{sub}", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fp = out_dir / "load_metrics_evolution.png"
    fig.savefig(fp, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return fp


def find_leaves(root: Path) -> List[Path]:
    return sorted({p.parent for p in root.rglob("token_counts.json")})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    summary_rows = []
    for root in args.roots:
        leaves = find_leaves(root)
        if not leaves:
            print(f"[skip] no token_counts.json under {root}")
            continue
        for leaf in leaves:
            res = process_run(leaf)
            if res is None:
                continue
            out = {
                "_meta": {"topk": res["topk"], "n_experts": res["n_experts"],
                          "steps": res["steps"], "layers": res["layers"]},
                "per_step": res["per_step"],
            }
            (leaf / "load_metrics.json").write_text(json.dumps(out, indent=2))
            rel = leaf.relative_to(root.parent if root.parent != Path('.') else root)
            tag = str(leaf).replace(str(root) + "/", "")
            if not args.no_plot:
                plot_run(res, leaf, f"{root.name} :: {tag}")
            # final-step layer-mean summary row
            last = res["steps"][-1]
            ps = res["per_step"][last]
            def mean_of(m):
                vals = [v[m] for v in ps.values()]
                return float(np.mean(vals)) if vals else float("nan")
            def max_of(m):
                vals = [v[m] for v in ps.values()]
                return float(np.max(vals)) if vals else float("nan")
            summary_rows.append((
                f"{root.name}/{tag}", last,
                mean_of("cv_pct"), mean_of("entropy_pct"),
                mean_of("max_frac"), max_of("dead_count"),
            ))
            print(f"wrote {leaf}/load_metrics.json  (+plot)")

    # cross-run final-step table
    print("\n=== final-step, layer-mean load metrics ===")
    hdr = f"{'run':<58} {'step':>8} {'cv%':>7} {'ent%':>7} {'maxfrac':>8} {'dead_max':>9}"
    print(hdr); print("-" * len(hdr))
    for name, step, cv, ent, mf, dead in summary_rows:
        print(f"{name:<58} {step:>8} {cv:>7.1f} {ent:>7.1f} {mf:>8.3f} {dead:>9.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
