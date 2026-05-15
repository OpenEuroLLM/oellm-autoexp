"""Re-render router_saturation_vs_final.png from existing saturation.json.

Used to retrofit the plot improvements (final-step star marker, stable->decay
boundary line) onto already-generated runs without re-evaluating checkpoints.

Usage:
    python3.11 scripts/moe/replot_saturation.py \\
        [--root results/moe_analysis] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm


_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
_LOG_DECAY_RE = re.compile(
    r"Discovered\s+(\d+)\s+stable\s+\+\s+(\d+)\s+decay\s+ckpts;\s*final=(\d+)"
    r"(?:\s+from\s+(\S+))?"
)


def _layer_short_label(name: str) -> str:
    m = _LAYER_RE.search(name)
    return f"L{m.group(1)}" if m else name


def _natural_layer_sort(names):
    def key(n):
        m = _LAYER_RE.search(n)
        return (int(m.group(1)) if m else 1_000_000, n)
    return sorted(names, key=key)


def _parse_decay_info(log_path: Path):
    """Returns (n_stable, n_decay, final_step, final_dir_basename) or None."""
    if not log_path.is_file():
        return None
    try:
        text = log_path.read_text(errors="ignore")
    except OSError:
        return None
    m = _LOG_DECAY_RE.search(text)
    if not m:
        return None
    return (
        int(m.group(1)),
        int(m.group(2)),
        int(m.group(3)),
        m.group(4) or "",
    )


def replot(bucket: Path, dry_run: bool = False) -> bool:
    sat_path = bucket / "saturation.json"
    if not sat_path.is_file():
        return False

    data = json.loads(sat_path.read_text())
    layers_data = data.get("layers", {})
    compare_steps = data.get("compare_steps") or sorted(int(s) for s in layers_data)
    final_ckpt = data.get("final_checkpoint")

    if not compare_steps:
        return False

    # Decay / final info: prefer JSON fields (new format), fall back to log parse.
    decay_steps = data.get("decay_steps") or []
    final_dir = data.get("final_dir") or ""
    final_dir_label = ""
    if final_dir:
        final_dir_label = Path(final_dir).parent.name

    if not decay_steps or not final_dir_label:
        info = _parse_decay_info(bucket / "current.log")
        if info is not None:
            n_stable, n_decay, log_final, log_final_dir = info
            if not decay_steps and n_decay > 0:
                steps_sorted_for_split = sorted(int(s) for s in compare_steps)
                # Stable came first; decay overrides on overlap and tends to be
                # the last contiguous chunk by step.
                decay_steps = steps_sorted_for_split[n_stable:]
            if not final_dir_label and log_final_dir:
                final_dir_label = log_final_dir
            if final_ckpt in (None, "release"):
                final_ckpt = log_final

    steps_sorted = sorted(int(s) for s in compare_steps)
    layer_names = set()
    for step in steps_sorted:
        layer_names.update(layers_data.get(str(step), {}).keys())
    layer_names = _natural_layer_sort(list(layer_names))
    if not layer_names:
        return False

    if dry_run:
        print(
            f"  would replot {bucket} | layers={len(layer_names)} | "
            f"final={final_ckpt} | decay_steps={decay_steps[:1]}..{decay_steps[-1:]} ({len(decay_steps)})"
        )
        return True

    plt.figure(figsize=(14, 7))
    cmap = cm.get_cmap("tab20" if len(layer_names) <= 20 else "hsv")
    colors = [cmap(i / max(len(layer_names) - 1, 1)) for i in range(len(layer_names))]

    for idx, layer_name in enumerate(layer_names):
        y_vals = []
        for step in steps_sorted:
            ld = layers_data.get(str(step), {}).get(layer_name)
            if ld is None:
                y_vals.append(float("nan"))
            else:
                y_vals.append(ld.get("compare_saturation", 0.0) * 100.0)
        ema_alpha = 0.3
        ema_vals = []
        acc = None
        for v in y_vals:
            if v != v:
                ema_vals.append(float("nan"))
                continue
            acc = v if acc is None else ema_alpha * v + (1 - ema_alpha) * acc
            ema_vals.append(acc)
        plt.plot(
            steps_sorted, ema_vals,
            label=_layer_short_label(layer_name),
            color=colors[idx], linewidth=1.8,
        )

    ref_x = final_ckpt if isinstance(final_ckpt, int) else None
    if ref_x is not None:
        plt.scatter(
            [ref_x], [100.0],
            marker="*", s=180,
            color="black", edgecolors="white", linewidths=1.0,
            zorder=5,
            label="final (=ref, 100%)",
        )

    if decay_steps:
        boundary = min(decay_steps)
        plt.axvline(
            boundary, color="gray", linestyle=":",
            linewidth=1.2, alpha=0.85, zorder=2,
        )
        plt.text(
            boundary, 102, " decay phase →",
            fontsize=9, color="gray", va="bottom", ha="left",
        )

    plt.xlabel("Checkpoint step", fontsize=12, fontweight="bold")
    plt.ylabel("Router saturation (%)", fontsize=12, fontweight="bold")
    suffix = ""
    if ref_x is not None:
        suffix = f"  (final={ref_x}{', decay' if decay_steps else ''})"
    plt.title(
        f"Router saturation vs final checkpoint{suffix}",
        fontsize=14, fontweight="bold",
    )
    if final_dir_label:
        plt.figtext(
            0.5, 0.005,
            f"Reference: {final_dir_label}",
            fontsize=8, color="gray", ha="center",
        )
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.ylim(0, 108)
    plt.legend(
        bbox_to_anchor=(1.05, 1), loc="upper left",
        fontsize=9, framealpha=0.95, edgecolor="black",
    )
    plt.tight_layout(rect=(0, 0.02, 1, 1))
    out = bucket / "router_saturation_vs_final.png"
    plt.savefig(out, dpi=200)
    plt.close()
    return True


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=repo_root / "results" / "moe_analysis")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    buckets = sorted(args.root.glob("bsz*/nexp_*/lr*/120BT"))
    print(f"Scanning {args.root} -> {len(buckets)} buckets")
    n_ok = 0
    n_decay = 0
    for b in buckets:
        ok = replot(b, dry_run=args.dry_run)
        if ok:
            n_ok += 1
            sat = json.loads((b / "saturation.json").read_text())
            if sat.get("decay_steps"):
                n_decay += 1
            else:
                # log-based detection
                if _parse_decay_info(b / "current.log"):
                    info = _parse_decay_info(b / "current.log")
                    if info and info[1] > 0:
                        n_decay += 1
    print(f"Replotted: {n_ok}/{len(buckets)} (decay-phase: {n_decay})")


if __name__ == "__main__":
    main()
