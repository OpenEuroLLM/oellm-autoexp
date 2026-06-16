"""Aggregate per-revision HF MoE metrics into cross-revision (cross-step) plots.

The per-revision driver (`compute_moe_metrics_hf.py`) writes one snapshot per
OLMoE revision under

    <root>/<short_name>/<revision>/expert_activation_norms.json
                                  /token_counts.json
                                  /routing_maps.npz

where <revision> is a branch name like `step100000-tokens420B`. Coactivation
and token-count plots are inherently per-snapshot and already correct. The
activation-norm view, however, is meant to be read *across training time* (the
Megatron pipeline plots per-layer max & median vs checkpoint step, log-y) — a
single snapshot can't show that. This script stitches the per-revision snapshots
into that cross-step view, matching `compute_moe_metrics.py` exactly:

    * expert_activation_norms.png          — per-layer solid=max / dashed=median
    * expert_activation_max_over_median.png — per-layer max/median ratio

X-axis is training tokens (B) parsed from the revision name (falls back to step).

Usage:
    python scripts/moe/aggregate_hf_revisions.py \
        --root results/moe_analysis_hf/olmoe_1b_7b \
        [--out-dir <root>/_aggregate] [--x tokens|step]
"""

from __future__ import annotations

import argparse
import io
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm


_REV_RE = re.compile(r"step(\d+)-tokens(\d+)B", re.IGNORECASE)
_LAYER_RE = re.compile(r"layers?[._](\d+)")


def _parse_revision(name: str) -> Optional[Tuple[int, float]]:
    """(step, tokens_in_billions) from a revision dir name, or None for `main`."""
    m = _REV_RE.search(name)
    if not m:
        return None
    return int(m.group(1)), float(m.group(2))


def _layer_index(name: str) -> Optional[int]:
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def _layer_short_label(name: str) -> str:
    li = _layer_index(name)
    return f"L{li}" if li is not None else name.replace(".", "_")


def _natural_layer_sort(names: List[str]) -> List[str]:
    def key(n: str):
        li = _layer_index(n)
        return (li if li is not None else 1_000_000, n)

    return sorted(names, key=key)


def _collect(root: Path, x_axis: str) -> Tuple[Dict[float, Dict[str, List[float]]], str, str, str]:
    """Returns (by_x, vector_norm, token_reduce, x_label).

    by_x maps x-value (tokens-B or step) -> {layer_name -> [per-expert norms]}.
    """
    by_x: Dict[float, Dict[str, List[float]]] = {}
    vector_norm, token_reduce = "l2", "mean"
    for jpath in sorted(root.glob("*/expert_activation_norms.json")):
        rev = jpath.parent.name
        parsed = _parse_revision(rev)
        if parsed is None:
            # Skip `main` / unparseable: it duplicates the numbered final ckpt.
            continue
        step, tokens = parsed
        x = tokens if x_axis == "tokens" else float(step)
        data = json.loads(jpath.read_text())
        layers = data.get("layers", {})
        if not layers:
            continue
        meta = data.get("_meta", {})
        vector_norm = meta.get("vector_norm", vector_norm)
        token_reduce = meta.get("token_reduce", token_reduce)
        by_x[x] = layers
    x_label = "Training tokens (B)" if x_axis == "tokens" else "Checkpoint step"
    return by_x, vector_norm, token_reduce, x_label


def _series(by_x, xs_sorted, layer):
    """Per-layer (xs, max_vals, med_vals, mean_vals) dropping missing snapshots."""
    xs, max_vals, med_vals, mean_vals = [], [], [], []
    for x in xs_sorted:
        norms = by_x[x].get(layer)
        if not norms:
            continue
        t = np.asarray(norms, dtype=np.float64)
        t = t[~np.isnan(t)]
        if t.size == 0:
            continue
        xs.append(x)
        max_vals.append(float(t.max()))
        med_vals.append(float(np.median(t)))
        mean_vals.append(float(t.mean()))
    return xs, max_vals, med_vals, mean_vals


def plot_activation_norms(root: Path, out_dir: Path, x_axis: str) -> bool:
    by_x, vn, tr, x_label = _collect(root, x_axis)
    if len(by_x) < 1:
        print(f"[aggregate] no parseable revision snapshots under {root}")
        return False
    out_dir.mkdir(parents=True, exist_ok=True)
    xs_sorted = sorted(by_x.keys())
    layer_names = _natural_layer_sort(
        list({ln for entry in by_x.values() for ln in entry})
    )
    if not layer_names:
        print("[aggregate] no layers found in snapshots")
        return False

    cmap = cm.get_cmap("tab20" if len(layer_names) <= 20 else "hsv")

    # Plot 1: per-layer max (solid) & median (dashed) vs training time, log-y.
    plt.figure(figsize=(12, 7))
    for idx, ln in enumerate(layer_names):
        color = cmap(idx / max(len(layer_names) - 1, 1))
        xs, max_vals, med_vals, _mean_vals = _series(by_x, xs_sorted, ln)
        if not xs:
            continue
        short = _layer_short_label(ln)
        plt.plot(xs, max_vals, color=color, linewidth=2.0, label=f"{short} max")
        plt.plot(xs, med_vals, color=color, linewidth=1.4, linestyle="--",
                 label=f"{short} med")
    plt.yscale("log")
    plt.xlabel(x_label, fontweight="bold")
    plt.ylabel(f"Per-expert activation norm ({vn}/{tr})", fontweight="bold")
    plt.title("Expert activation norms — solid: max, dashed: median")
    plt.grid(True, which="both", alpha=0.25, linestyle="--")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left",
               fontsize=7, ncol=1, framealpha=0.95)
    plt.tight_layout()
    p1 = out_dir / "expert_activation_norms.png"
    plt.savefig(p1, dpi=200)
    plt.close()
    print(f"[aggregate] wrote {p1}  ({len(xs_sorted)} revisions, {len(layer_names)} layers)")

    # Plot 2: per-layer max/median ratio vs training time, log-y.
    plt.figure(figsize=(12, 6))
    for idx, ln in enumerate(layer_names):
        color = cmap(idx / max(len(layer_names) - 1, 1))
        xs, max_vals, med_vals, _mean_vals = _series(by_x, xs_sorted, ln)
        rxs, ratios = [], []
        for x, mx, md in zip(xs, max_vals, med_vals):
            if md > 0:
                rxs.append(x)
                ratios.append(mx / md)
        if rxs:
            plt.plot(rxs, ratios, color=color, linewidth=1.6,
                     label=_layer_short_label(ln))
    plt.yscale("log")
    plt.xlabel(x_label, fontweight="bold")
    plt.ylabel("max / median (per-expert activation norm)", fontweight="bold")
    plt.title("Max-to-median expert activation-norm ratio per layer")
    plt.grid(True, which="both", alpha=0.25, linestyle="--")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left",
               fontsize=7, ncol=1, framealpha=0.95)
    plt.tight_layout()
    p2 = out_dir / "expert_activation_max_over_median.png"
    plt.savefig(p2, dpi=200)
    plt.close()
    print(f"[aggregate] wrote {p2}")
    return True


def plot_per_layer_evolution(root: Path, out_dir: Path, x_axis: str,
                             write_individual: bool = True) -> bool:
    """One panel per layer: mean / median / max activation-norm vs training time.

    Produces a single 4-col subplot grid (`expert_activation_per_layer.png`) and,
    when `write_individual`, one PNG per layer under `per_layer/` for the dashboard.
    """
    by_x, vn, tr, x_label = _collect(root, x_axis)
    if len(by_x) < 1:
        print(f"[aggregate] no parseable revision snapshots under {root}")
        return False
    out_dir.mkdir(parents=True, exist_ok=True)
    xs_sorted = sorted(by_x.keys())
    layer_names = _natural_layer_sort(
        list({ln for entry in by_x.values() for ln in entry})
    )
    if not layer_names:
        print("[aggregate] no layers found in snapshots")
        return False

    ylabel = f"Activation norm ({vn}/{tr})"

    def _draw(ax, ln):
        xs, mx, md, _mn = _series(by_x, xs_sorted, ln)
        if not xs:
            return False
        ax.plot(xs, mx, color="#c44e52", linewidth=1.6, label="max")
        ax.plot(xs, md, color="#55a868", linewidth=1.6, linestyle="--", label="median")
        ax.set_yscale("log")
        ax.set_title(_layer_short_label(ln), fontsize=9)
        ax.grid(True, which="both", alpha=0.25, linestyle="--")
        return True

    # Grid: 4 columns, enough rows for every layer.
    ncol = 4
    nrow = (len(layer_names) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.6 * nrow),
                             squeeze=False, sharex=True)
    handles = labels = None
    for i, ln in enumerate(layer_names):
        ax = axes[i // ncol][i % ncol]
        if _draw(ax, ln) and handles is None:
            handles, labels = ax.get_legend_handles_labels()
    # Blank any unused panels.
    for j in range(len(layer_names), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.supxlabel(x_label, fontweight="bold")
    fig.supylabel(ylabel, fontweight="bold")
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=9, framealpha=0.95)
    fig.suptitle("Per-layer expert activation-norm evolution", fontweight="bold")
    fig.tight_layout(rect=(0.01, 0.01, 1.0, 0.97))
    pg = out_dir / "expert_activation_per_layer.png"
    fig.savefig(pg, dpi=150)
    plt.close(fig)
    print(f"[aggregate] wrote {pg}  ({len(xs_sorted)} revisions, {len(layer_names)} layers)")

    if write_individual:
        ind_dir = out_dir / "per_layer"
        ind_dir.mkdir(parents=True, exist_ok=True)
        for ln in layer_names:
            fig, ax = plt.subplots(figsize=(8, 5))
            if not _draw(ax, ln):
                plt.close(fig)
                continue
            ax.set_xlabel(x_label, fontweight="bold")
            ax.set_ylabel(ylabel, fontweight="bold")
            ax.legend(fontsize=9, framealpha=0.95)
            fig.tight_layout()
            p = ind_dir / f"expert_activation_{_layer_short_label(ln)}.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
        print(f"[aggregate] wrote {len(layer_names)} per-layer plots under {ind_dir}")
    return True


def _routing_layers(npz) -> Dict[str, str]:
    """Map dotted layer name -> npz key for every `*::routing` array in `npz`."""
    out: Dict[str, str] = {}
    for k in npz.files:
        if k.endswith("::routing"):
            dotted = k[: -len("::routing")].replace("__", ".")
            out[dotted] = k
    return out


def compute_router_saturation(root: Path, out_dir: Path,
                              x_axis: str = "tokens") -> bool:
    """Per-layer router saturation vs the final revision, from `routing_maps.npz`.

    Mirrors Megatron's `compare_saturation`: for every token, the fraction of its
    top-k routed experts that match the final-revision routing for that same token,
    averaged over tokens. All revisions are evaluated on the same input tokens in
    the same order (verified: identical N_tokens), so row i is comparable across
    revisions. Writes `router_saturation.json` + `router_saturation_vs_final.png`
    (per-layer, EMA-smoothed, x = training tokens, star at the final = 100%).
    """
    npz_paths: Dict[Tuple[int, float], Path] = {}
    for p in sorted(root.glob("*/routing_maps.npz")):
        parsed = _parse_revision(p.parent.name)
        if parsed is not None:
            npz_paths[parsed] = p
    if not npz_paths:
        print(f"[aggregate] no routing_maps.npz under {root}")
        return False

    revs_sorted = sorted(npz_paths.keys(), key=lambda st: (st[1], st[0]))  # by tokens
    ref_key = revs_sorted[-1]  # final = max training tokens
    print(f"[aggregate] saturation reference = step{ref_key[0]}-tokens{ref_key[1]:g}B")

    # Cache the reference routing maps (bool) once; reused for every revision.
    ref_npz = np.load(npz_paths[ref_key])
    ref_layers = _routing_layers(ref_npz)
    ref_maps = {ln: ref_npz[k].astype(bool) for ln, k in ref_layers.items()}
    topk = {ln: int(m.sum(axis=1)[:1024].max()) or 1 for ln, m in ref_maps.items()}

    # sat[layer][ (step,tokens) ] = saturation fraction
    sat: Dict[str, Dict[Tuple[int, float], float]] = {ln: {} for ln in ref_maps}
    for rev in revs_sorted:
        z = np.load(npz_paths[rev])
        layers = _routing_layers(z)
        for ln, ref_m in ref_maps.items():
            k = layers.get(ln)
            if k is None:
                continue
            cur = z[k].astype(bool)
            if cur.shape != ref_m.shape:
                continue
            inter = np.logical_and(cur, ref_m).sum(axis=1).astype(np.float64)
            sat[ln][rev] = float((inter / topk[ln]).mean())

    out_dir.mkdir(parents=True, exist_ok=True)
    layer_names = _natural_layer_sort(list(ref_maps.keys()))

    # Persist raw values for reuse / dashboard.
    payload = {
        "_meta": {
            "reference": f"step{ref_key[0]}-tokens{ref_key[1]:g}B",
            "metric": "compare_saturation (per-token top-k overlap vs final)",
            "topk": topk,
        },
        "layers": {
            ln: {f"step{st}-tokens{tk:g}B": sat[ln][(st, tk)]
                 for (st, tk) in revs_sorted if (st, tk) in sat[ln]}
            for ln in layer_names
        },
    }
    (out_dir / "router_saturation.json").write_text(json.dumps(payload, indent=2))

    # Plot: per-layer saturation (%) vs training time, EMA-smoothed.
    x_label = "Training tokens (B)" if x_axis == "tokens" else "Checkpoint step"
    cmap = cm.get_cmap("tab20" if len(layer_names) <= 20 else "hsv")
    plt.figure(figsize=(14, 7))
    for idx, ln in enumerate(layer_names):
        color = cmap(idx / max(len(layer_names) - 1, 1))
        xs, ys = [], []
        for (st, tk) in revs_sorted:
            if (st, tk) not in sat[ln]:
                continue
            xs.append(tk if x_axis == "tokens" else float(st))
            ys.append(sat[ln][(st, tk)] * 100.0)
        if not xs:
            continue
        ema, acc = [], None  # EMA alpha=0.3, matching Megatron
        for v in ys:
            acc = v if acc is None else 0.3 * v + 0.7 * acc
            ema.append(acc)
        plt.plot(xs, ema, color=color, linewidth=1.8,
                 label=_layer_short_label(ln))
    ref_x = ref_key[1] if x_axis == "tokens" else float(ref_key[0])
    plt.scatter([ref_x], [100.0], marker="*", s=180, color="black",
                edgecolors="white", linewidths=1.0, zorder=5,
                label="final (=ref, 100%)")
    plt.xlabel(x_label, fontsize=12, fontweight="bold")
    plt.ylabel("Router saturation (%)", fontsize=12, fontweight="bold")
    plt.title("Router saturation vs final revision  "
              f"(ref=step{ref_key[0]}-tokens{ref_key[1]:g}B)",
              fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.ylim(0, 108)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8,
               framealpha=0.95, edgecolor="black")
    plt.tight_layout(rect=(0, 0.02, 1, 1))
    p = out_dir / "router_saturation_vs_final.png"
    plt.savefig(p, dpi=200)
    plt.close()
    print(f"[aggregate] wrote {p}  ({len(revs_sorted)} revisions, {len(layer_names)} layers)")
    return True


def _collect_token_counts(root: Path) -> Tuple[Dict[Tuple[int, float], Dict[str, List[float]]], str]:
    """(by_rev, x_label): map (step, tokens-B) -> {layer -> per-expert counts}."""
    by_rev: Dict[Tuple[int, float], Dict[str, List[float]]] = {}
    for jpath in sorted(root.glob("*/token_counts.json")):
        parsed = _parse_revision(jpath.parent.name)
        if parsed is None:
            continue  # skip `main` (duplicates final numbered ckpt)
        data = json.loads(jpath.read_text())
        layers = data.get("layers", {})
        if layers:
            by_rev[parsed] = layers
    return by_rev, "Training tokens (B)"


def render_token_count_gifs(root: Path, out_dir: Path, frame_ms: int,
                            x_axis: str = "tokens") -> bool:
    """One animated GIF per layer: per-expert token-count bars over training.

    Each frame is one OLMoE revision (ordered by training tokens), mirroring the
    Megatron `_render_routing_gif` look — bar chart of per-expert counts with a
    dashed uniform-load line and a y-axis held fixed across frames so the
    redistribution of load over training is visible. Written under `routing/`.
    """
    try:
        from PIL import Image
    except ImportError:
        print("[aggregate] Pillow not available; skipping GIFs")
        return False

    by_rev, _ = _collect_token_counts(root)
    if not by_rev:
        print(f"[aggregate] no parseable token_counts snapshots under {root}")
        return False

    revs_sorted = sorted(by_rev.keys(), key=lambda st: (st[1], st[0]))  # by tokens
    layer_names = _natural_layer_sort(
        list({ln for entry in by_rev.values() for ln in entry})
    )
    gif_dir = out_dir / "routing"
    gif_dir.mkdir(parents=True, exist_ok=True)

    n_written = 0
    for ln in layer_names:
        # Per-layer fixed y-axis across all frames.
        y_max = 0.0
        for st in revs_sorted:
            counts = by_rev[st].get(ln)
            if counts:
                y_max = max(y_max, max(counts))
        if y_max <= 0:
            continue
        y_max *= 1.05
        li = _layer_index(ln)
        short = _layer_short_label(ln)

        frames = []
        for step, tokens in revs_sorted:
            counts = by_rev[(step, tokens)].get(ln)
            if not counts:
                continue
            n_experts = len(counts)
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.bar(range(n_experts), counts, color="#1f77b4")
            uniform = sum(counts) / n_experts if n_experts else 0.0
            ax.axhline(uniform, color="gray", linestyle="--", linewidth=1,
                       label="uniform load")
            ax.set_ylim(0, y_max)
            ax.set_xlabel("Expert ID")
            ax.set_ylabel("Token count")
            title_layer = f"Layer {li}" if li is not None else short
            ax.set_title(f"{title_layer} — {tokens:g}B tokens (step {step:,})")
            ax.legend(loc="upper right", fontsize=8)
            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            frames.append(Image.open(buf).convert("RGB"))

        if not frames:
            continue
        out_path = gif_dir / (
            f"expert_routing_layer_{li}.gif" if li is not None
            else f"expert_routing_{short}.gif"
        )
        frames[0].save(out_path, save_all=True, append_images=frames[1:],
                       duration=frame_ms, loop=0, optimize=False)
        n_written += 1
    print(f"[aggregate] wrote {n_written} per-layer routing GIFs under {gif_dir} "
          f"({len(revs_sorted)} frames each)")
    return n_written > 0


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True,
                   help="Per-model results dir, e.g. results/moe_analysis_hf/olmoe_1b_7b")
    p.add_argument("--out-dir", default=None,
                   help="Where to write aggregate plots (default: <root>/_aggregate)")
    p.add_argument("--x", default="tokens", choices=["tokens", "step"],
                   help="X-axis: training tokens (B) or checkpoint step.")
    p.add_argument("--gif-frame-ms", type=int, default=700,
                   help="Per-frame duration (ms) of the per-layer routing GIFs.")
    p.add_argument("--no-gif", action="store_true",
                   help="Skip the per-layer token-count GIFs.")
    p.add_argument("--no-saturation", action="store_true",
                   help="Skip router-saturation (needs routing_maps.npz; heavier).")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    root = Path(args.root)
    if not root.is_dir():
        print(f"[aggregate] root not found: {root}")
        return 1
    out_dir = Path(args.out_dir) if args.out_dir else root / "_aggregate"
    ok = plot_activation_norms(root, out_dir, args.x)
    ok = plot_per_layer_evolution(root, out_dir, args.x) and ok
    if not args.no_gif:
        ok = render_token_count_gifs(root, out_dir, args.gif_frame_ms, args.x) and ok
    if not args.no_saturation:
        ok = compute_router_saturation(root, out_dir, args.x) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
