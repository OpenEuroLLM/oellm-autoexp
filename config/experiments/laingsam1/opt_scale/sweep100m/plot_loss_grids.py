"""Heatmaps of the final validation loss of a sweep: batch size x learning rate, one panel per budget.

Reads the Slurm logs of the cooldown jobs and writes, into --out:
  loss_grid_<optimizer>.png   one row: 1x | 2x | 4x
  loss_grid_all.png / .pdf    all optimizers stacked

Usage (matplotlib is not in the autoexp venv, so run it through uv):
  uv run --no-project --with matplotlib python plot_loss_grids.py \
      --logs $OPT_SCALE_BASE/logs --out $OPT_SCALE_BASE/figures/sweep100m

Colour (viridis, dark = low) is the gap to the lowest loss in the same panel, on a
square-root scale and saturating at --cap.
"""
import argparse
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm

OPTIMIZERS = [("adam", "AdamW"), ("muon", "Muon"), ("muown_classical", "Muown")]
BUDGETS = [("1", "1x budget"), ("2", "2x budget"), ("4", "4x budget")]
NAME = re.compile(r"_(adam|muon|muown_classical)_lr([0-9.e-]+)_(beta|mom)([0-9.]+)_bs(\d+)_cooldown(\d)x-(\d+)\.out$")
LOSS = re.compile(r"validation loss at iteration\s+(\d+) on validation set \| lm loss value: ([0-9.E+-]+)")
ITER = re.compile(r"iteration\s+(\d+)/\s*(\d+)")


def read_final(path):
    """Final validation loss of a finished job, else None."""
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        f.seek(max(0, f.tell() - 20000))
        tail = f.read().decode(errors="replace")
    its, losses = ITER.findall(tail), LOSS.findall(tail)
    if not its or not losses or its[-1][0] != its[-1][1] or losses[-1][0] != its[-1][1]:
        return None
    return float(losses[-1][1])


def collect(logs, prefix, hp):
    data = {}
    for path in sorted(glob.glob(os.path.join(logs, f"{prefix}_*cooldown*x-*.out"))):
        m = NAME.search(path)
        if not m or (hp is not None and float(m[4]) != hp):
            continue
        loss = read_final(path)
        if loss is not None:  # a later job id of the same run overwrites an earlier one
            data[(m[1], m[6], int(m[5]), float(m[2]))] = loss
    return data


def draw(ax, data, opt, budget, batches, lrs, cap, cmap):
    grid = np.array([[data.get((opt, budget, b, lr), np.nan) for lr in lrs] for b in batches])
    best = np.nanmin(grid)
    norm = PowerNorm(0.5, vmin=0, vmax=cap)
    im = ax.imshow(grid - best, cmap=cmap, norm=norm, aspect="equal")
    for i in range(len(batches)):
        for j in range(len(lrs)):
            v = grid[i, j]
            if np.isnan(v):
                continue
            r, g, b_, _ = cmap(norm(min(v - best, cap)))
            bright = 0.299 * r + 0.587 * g + 0.114 * b_ > 0.62
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9.5,
                    color="#1a1a1a" if bright else "white")
    ax.set_xticks(range(len(lrs)), [f"{v:g}" for v in lrs], fontsize=8.5)
    ax.set_yticks(range(len(batches)), batches, fontsize=8.5, rotation=90, va="center")
    ax.set_xticks(np.arange(0.5, len(lrs) - 1), minor=True)
    ax.set_yticks(np.arange(0.5, len(batches) - 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.6)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def figure(data, opts, model, batches, lrs, cap, mult):
    cmap = plt.get_cmap("viridis")
    n = len(opts)
    cell = 0.78  # inches per cell
    width = 3 * len(lrs) * cell + 2.6
    row_h = len(batches) * cell + 1.75
    fig, axes = plt.subplots(n, 3, figsize=(width, n * row_h), squeeze=False)
    fig.subplots_adjust(left=0.6 / width, right=1 - 1.25 / width, top=1 - 0.5 / (n * row_h),
                        bottom=1.05 / (n * row_h), hspace=1.75 / (len(batches) * cell), wspace=0.1)
    for r, (opt, name) in enumerate(opts):
        for c, (budget, blabel) in enumerate(BUDGETS):
            ax = axes[r, c]
            im = draw(ax, data, opt, budget, batches, lrs, cap, cmap)
            ax.set_title(rf"$D = {mult[c]}N$" if mult else blabel, fontsize=12, pad=7)
            ax.set_xlabel(r"Learning rate $\eta$", fontsize=10.5)
            if c == 0:
                ax.set_ylabel(r"Batch size $b$", fontsize=10.5)
        left, right = axes[r, 0].get_position(), axes[r, 2].get_position()
        fig.text((left.x0 + right.x1) / 2, left.y0 - 0.82 / (n * row_h),
                 f"({chr(97 + r)}) {name}, " + rf"$N = {model}$", ha="center", va="top", fontsize=15)
        cax = fig.add_axes([right.x1 + 0.22 / width, right.y0, 0.14 / width, right.height])
        ticks = [0, 0.01, 0.03, 0.1, 0.2, cap]
        cb = fig.colorbar(im, cax=cax, ticks=[t for t in ticks if t <= cap])
        cb.ax.set_yticklabels(["min"] + [f"+{t:g}" for t in ticks[1:] if t <= cap], fontsize=8.5)
        cb.ax.set_title("Loss", fontsize=10, pad=6)
        cb.outline.set_visible(False)
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--logs", required=True, help="directory with the Slurm .out files")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--prefix", default="hyb100m_sweep", help="job name prefix")
    p.add_argument("--model", default="0.1\\mathrm{B}", help="model size in the row captions (LaTeX math)")
    p.add_argument("--hp", type=float, default=0.95, help="beta / momentum value to plot")
    p.add_argument("--mult", default="20,40,80", help="tokens per parameter at the 1x,2x,4x budgets ('' to omit)")
    p.add_argument("--cap", type=float, default=0.3, help="loss gap at which the colour saturates")
    a = p.parse_args()

    data = collect(a.logs, a.prefix, a.hp)
    if not data:
        raise SystemExit("no finished cooldown logs found")
    batches = sorted({k[2] for k in data}, reverse=True)
    lrs = sorted({k[3] for k in data})
    opts = [(o, n) for o, n in OPTIMIZERS if any(k[0] == o for k in data)]
    mult = a.mult.split(",") if a.mult else None
    os.makedirs(a.out, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10", "DejaVu Serif"], "mathtext.fontset": "cm",
                         "axes.formatter.use_mathtext": True, "axes.unicode_minus": False})
    for opt in opts:
        fig = figure(data, [opt], a.model, batches, lrs, a.cap, mult)
        fig.savefig(os.path.join(a.out, f"loss_grid_{opt[0]}.png"), dpi=170, facecolor="white", bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
    fig = figure(data, opts, a.model, batches, lrs, a.cap, mult)
    fig.savefig(os.path.join(a.out, "loss_grid_all.png"), dpi=170, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    fig.savefig(os.path.join(a.out, "loss_grid_all.pdf"), facecolor="white", bbox_inches="tight", pad_inches=0.15)
    print(f"{len(data)} results -> {a.out}")


if __name__ == "__main__":
    main()
