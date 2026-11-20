"""Heatmaps of the final validation loss with beta / momentum as an axis, as one PDF.

Writes, into --out:
  beta_grid_<optimizer>.pdf   beta or momentum (rows) x learning rate (columns);
                              one panel per batch size (rows) and budget (columns)
  beta_grid_summary.pdf       batch size (rows) x beta or momentum (columns), one panel per
                              optimizer and budget; each cell is the best loss over the learning
                              rates that were run at every beta / momentum, with that rate below
  beta_grid_all.pdf           the pages above in one file

Usage (matplotlib is not in the autoexp venv, so run it through uv):
  uv run --no-project --with matplotlib python plot_beta_grids.py \
      --logs $OPT_SCALE_BASE/logs --out $OPT_SCALE_BASE/figures/sweep100m

The logs are read by make_sweep_page.collect, so a cooldown counts only if it loaded exactly the
checkpoint step it asked for, and a *_rerun replaces the first attempt (README.md, "Phase 2
reruns"). Empty cells are settings that were not run.

Colour (viridis, dark = low) is a loss gap on a square-root scale, saturating at --cap: the gap to
the lowest loss in the panel in the per-optimizer files, and the gap to the best beta / momentum
at the same batch size in the summary.
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import PowerNorm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_sweep_page import collect  # noqa: E402

OPTIMIZERS = [("adam", "AdamW", r"$\beta_1 = \beta_2$"), ("muon", "Muon", r"Momentum $\mu$"),
              ("muown_classical", "Muown", r"Momentum $\mu$")]
BUDGETS = ["1", "2", "4"]
CELL = 0.78          # inches per cell
EMPTY = "#ececec"


def hp_key(h):
    """Tied values in increasing order, the untied AdamW reference (0.9/0.95) last."""
    return (1, 0.0) if "/" in h else (0, float(h))


def hp_text(h):
    return r"$\beta_1,\beta_2$" + "\n" + h.replace("/", ", ") if "/" in h else h


def heat(ax, grid, gap, cap, cmap, sub=None, fontsize=9.5):
    """grid: losses (nan = not run); gap: what the colour shows; sub: optional second text line."""
    norm = PowerNorm(0.5, vmin=0, vmax=cap)
    ax.set_facecolor(EMPTY)
    im = ax.imshow(np.ma.masked_invalid(gap), cmap=cmap, norm=norm, aspect="equal")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if np.isnan(grid[i, j]):
                continue
            r, g, b, _ = cmap(norm(min(gap[i, j], cap)))
            ink = "#1a1a1a" if 0.299 * r + 0.587 * g + 0.114 * b > 0.62 else "white"
            if sub is None:
                ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", fontsize=fontsize, color=ink)
            else:
                ax.text(j, i - 0.13, f"{grid[i, j]:.3f}", ha="center", va="center", fontsize=fontsize, color=ink)
                ax.text(j, i + 0.22, sub[i][j], ha="center", va="center", fontsize=fontsize - 3, color=ink)
    ax.set_xticks(np.arange(0.5, grid.shape[1] - 1), minor=True)
    ax.set_yticks(np.arange(0.5, grid.shape[0] - 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.6)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def colorbar(fig, im, box, cap, title):
    cax = fig.add_axes(box)
    ticks = [t for t in (0, 0.003, 0.01, 0.03, 0.1, 0.2) if t < cap] + [cap]
    cb = fig.colorbar(im, cax=cax, ticks=ticks)
    cb.ax.set_yticklabels(["min"] + [f"+{t:g}" for t in ticks[1:]], fontsize=8.5)
    cb.ax.set_title(title, fontsize=9.5, pad=8)
    cb.outline.set_visible(False)


def title_of(c, mult):
    return rf"$D = {mult[c]}N$" if mult else f"{BUDGETS[c]}x budget"


def page_optimizer(data, opt, name, hp_name, model, cap, mult, source):
    """beta / momentum x learning rate; panels: batch size (rows) x budget (columns)."""
    keys = [k for k in data if k[0] == opt]
    batches = sorted({k[2] for k in keys})
    hps = sorted({k[3] for k in keys}, key=hp_key, reverse=True)
    hps = [h for h in hps if "/" not in h] + [h for h in hps if "/" in h]   # reference in the bottom row
    lrs = sorted({k[4] for k in keys})
    cmap = plt.get_cmap("viridis")
    nr, nc = len(batches), 3
    width = nc * len(lrs) * CELL + 3.2
    height = nr * (len(hps) * CELL + 0.95) + 1.6
    fig, axes = plt.subplots(nr, nc, figsize=(width, height), squeeze=False)
    fig.subplots_adjust(left=1.25 / width, right=1 - 1.3 / width, top=1 - 1.0 / height, bottom=0.75 / height,
                        hspace=0.95 / (len(hps) * CELL), wspace=0.08)
    for r, bs in enumerate(batches):
        for c, budget in enumerate(BUDGETS):
            ax = axes[r, c]
            grid = np.array([[data.get((opt, budget, bs, h, lr), np.nan) for lr in lrs] for h in hps])
            im = heat(ax, grid, grid - np.nanmin(grid), cap, cmap)
            ax.set_xticks(range(len(lrs)), [f"{v:g}" for v in lrs], fontsize=8.5)
            ax.set_yticks(range(len(hps)), [hp_text(h) for h in hps] if c == 0 else [], fontsize=8.5)
            if r == 0:
                ax.set_title(title_of(c, mult), fontsize=12, pad=7)
            if r == nr - 1:
                ax.set_xlabel(r"Learning rate $\eta$", fontsize=10.5)
            if c == 0:
                ax.set_ylabel(rf"$b = {bs}$" + "\n" + hp_name, fontsize=10.5)
        right = axes[r, nc - 1].get_position()
        colorbar(fig, im, [right.x1 + 0.25 / width, right.y0, 0.14 / width, right.height], cap, "Gap to\npanel min")
    fig.suptitle(f"{name}, " + rf"$N = {model}$" + f": final validation loss ({source})", fontsize=14, y=1 - 0.3 / height)
    return fig


def page_summary(data, opts, model, cap, mult, source):
    """batch size x beta / momentum, best over the learning rates run at every beta / momentum."""
    cmap = plt.get_cmap("viridis")
    info = []
    for opt, name, hp_name in opts:
        keys = [k for k in data if k[0] == opt and "/" not in k[3]]
        hps = sorted({k[3] for k in keys}, key=hp_key)
        if len(hps) < 2:
            continue
        batches = sorted({k[2] for k in keys}, reverse=True)
        common = sorted(set.intersection(*[{k[4] for k in keys if k[3] == h} for h in hps]))
        info.append((opt, name, hp_name, hps, batches, common))
    if not info:
        return None
    nr = len(info)
    ncell = max(len(i[3]) for i in info)
    width = 3 * ncell * CELL * 1.25 + 3.0
    row_h = max(len(i[4]) for i in info) * CELL + 1.75
    height = nr * row_h + 0.9
    fig, axes = plt.subplots(nr, 3, figsize=(width, height), squeeze=False)
    fig.subplots_adjust(left=0.75 / width, right=1 - 1.45 / width, top=1 - 1.25 / height, bottom=1.1 / height,
                        hspace=1.75 / (len(info[0][4]) * CELL), wspace=0.1)
    for r, (opt, name, hp_name, hps, batches, common) in enumerate(info):
        for c, budget in enumerate(BUDGETS):
            ax = axes[r, c]
            grid = np.full((len(batches), len(hps)), np.nan)
            sub = [["" for _ in hps] for _ in batches]
            for i, bs in enumerate(batches):
                for j, h in enumerate(hps):
                    got = [(data[(opt, budget, bs, h, lr)], lr) for lr in common if (opt, budget, bs, h, lr) in data]
                    if got:
                        grid[i, j], lr = min(got)
                        sub[i][j] = rf"$\eta$ = {lr:g}"
            gap = grid - np.nanmin(grid, axis=1, keepdims=True)
            im = heat(ax, grid, gap, cap, cmap, sub=sub)
            ax.set_xticks(range(len(hps)), hps, fontsize=8.5)
            ax.set_yticks(range(len(batches)), batches, fontsize=8.5, rotation=90, va="center")
            ax.set_title(title_of(c, mult), fontsize=12, pad=7)
            ax.set_xlabel(hp_name, fontsize=10.5)
            if c == 0:
                ax.set_ylabel(r"Batch size $b$", fontsize=10.5)
        left, right = axes[r, 0].get_position(), axes[r, 2].get_position()
        fig.text((left.x0 + right.x1) / 2, left.y0 - 0.72 / height,
                 f"({chr(97 + r)}) {name}, " + rf"$N = {model}$" + ", best of " +
                 r"$\eta \in \{$" + ", ".join(f"{v:g}" for v in common) + r"$\}$", ha="center", va="top", fontsize=13)
        colorbar(fig, im, [right.x1 + 0.25 / width, right.y0, 0.14 / width, right.height], cap, "Gap to best\nin the row")
    fig.suptitle(f"Effect of beta / momentum on the final validation loss ({source})", fontsize=14, y=1 - 0.3 / height)
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--logs", required=True, help="directory with the Slurm .out files")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--source", default="train", choices=["train", "fixed"],
                   help="train: in-training validation of the cooldown (all runs); "
                        "fixed: scoring jobs on the fixed validation set (only where they were run)")
    p.add_argument("--model", default="0.1\\mathrm{B}", help="model size in the captions (LaTeX math)")
    p.add_argument("--mult", default="20,40,80", help="tokens per parameter at the 1x,2x,4x budgets ('' to omit)")
    p.add_argument("--cap", type=float, default=0.1, help="loss gap at which the colour saturates")
    a = p.parse_args()

    got = collect(a.logs)
    data = {(r["opt"], r["budget"], r["bs"], r["hp"], r["lr"]): r["loss"] for r in got["rows"] if r["kind"] == a.source}
    if not data:
        raise SystemExit(f"no results for --source {a.source}")
    for label in ("excluded", "unfinished"):
        for x in got[label]:
            print(f"{label}: {x}")
    source = "in-training validation" if a.source == "train" else "fixed validation set"
    tag = "" if a.source == "train" else "_fixed"
    mult = a.mult.split(",") if a.mult else None
    opts = [o for o in OPTIMIZERS if any(k[0] == o[0] for k in data)]
    os.makedirs(a.out, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10", "DejaVu Serif"], "mathtext.fontset": "cm",
                         "axes.formatter.use_mathtext": True, "axes.unicode_minus": False})
    save = dict(facecolor="white", bbox_inches="tight", pad_inches=0.15)
    written = []
    with PdfPages(os.path.join(a.out, f"beta_grid_all{tag}.pdf")) as pdf:
        fig = page_summary(data, opts, a.model, a.cap, mult, source)
        if fig is not None:
            fig.savefig(os.path.join(a.out, f"beta_grid_summary{tag}.pdf"), **save)
            pdf.savefig(fig, **save)
            plt.close(fig)
            written.append(f"beta_grid_summary{tag}.pdf")
        for opt, name, hp_name in opts:
            if len({k[3] for k in data if k[0] == opt}) < 2:
                continue   # only one beta / momentum: plot_loss_grids.py shows everything there is
            fig = page_optimizer(data, opt, name, hp_name, a.model, a.cap, mult, source)
            fig.savefig(os.path.join(a.out, f"beta_grid_{opt}{tag}.pdf"), **save)
            pdf.savefig(fig, **save)
            plt.close(fig)
            written.append(f"beta_grid_{opt}{tag}.pdf")
    print(f"{len(data)} results -> {a.out}: beta_grid_all{tag}.pdf, " + ", ".join(written))


if __name__ == "__main__":
    main()
