"""Colour-coded results table shared by plot_hf.py and plot_reasoning.py: one
row per task, one column per checkpoint.

Bold = within 1 se of the row's best; shading = distance from the best
in se (darkest = best, lightest at 3 se or more), so statistical ties
look like ties.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LEGEND = (
    "± under each task: standard error.  Bold: within 1 se of the row's best.  "
    "Shading: distance from the best (lightest at 3 se or more)."
)


def draw(rows, models, out, title, footnote=None, sep_before=None):
    """rows: [(label, {model: value}, se, orange)]; orange labels mark caveated rows (see footnote).
    sep_before: index of a row (e.g. the mean) to separate from the rows above by a line."""
    note = LEGEND + (f"\n{footnote}" if footnote else "")
    h = 0.55 * len(rows) + 1.1 + 0.17 * (note.count("\n") + 1)
    fig, ax = plt.subplots(figsize=(1.55 * len(models) + 4.6, h))
    ax.set_xlim(-0.5, len(models) - 0.5)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    for i, (label, vals, se, orange) in enumerate(rows):
        best = max(vals.values())
        for j, m in enumerate(models):
            if m not in vals:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color="#eeeeee"))
                ax.text(j, i, "–", ha="center", va="center", color="#888888")
                continue
            d = (best - vals[m]) / se
            s = 0.15 + 0.6 * max(0.0, 1 - d / 3)
            ax.add_patch(
                plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color=plt.cm.Greens(s), ec="white", lw=2)
            )
            ax.text(
                j,
                i,
                f"{vals[m]:.3f}",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold" if d <= 1 else "normal",
                color="white" if s > 0.55 else "black",
            )
        ax.text(
            -0.62,
            i,
            f"{label}\n± {se:.3f}",
            ha="right",
            va="center",
            fontsize=9.5,
            color="#8a4b00" if orange else "black",
        )
    if sep_before is not None:
        ax.axhline(sep_before - 0.5, color="black", lw=1.5)

    ax.set_xticks(range(len(models)), models, fontsize=10)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    fig.text(0.01, 0.02, note, fontsize=8.5, color="#555555", va="bottom")
    fig.suptitle(title, fontsize=11, y=0.985)
    # row labels are ax.text, so tight_layout already leaves room for them
    fig.tight_layout(rect=(0.005, (0.15 + 0.16 * (note.count("\n") + 1)) / h, 1, 1 - 0.4 / h))
    fig.savefig(out, dpi=150)
    print(out)
