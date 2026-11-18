"""Campaign timeline: screens, main runs, key decisions (2026-10-14 .. 11-18)."""
from datetime import datetime as D, timedelta
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from fig_common import *

def d(s):
    return D.strptime("2026-" + s, "%Y-%m-%d %H:%M" if " " in s else "%Y-%m-%d")

# (lane, start, end, label, color, hatch)
LDM = "////"
bars = [
    # screens / experiments
    ("Screens &\nexperiments", "10-17", "10-19 12:00", "R1–R4", AQUA, None),
    ("Screens &\nexperiments", "10-19 12:00", "10-24", "Round 5 + follow-ups", AQUA, None),
    ("Screens &\nexperiments", "10-26", "10-28", "GQA\nrematch", AQUA, None),
    ("Screens &\nexperiments", "10-28 12:00", "11-01", "stab / QB / size / ctxlen", AQUA, None),
    ("Screens &\nexperiments", "11-05", "11-11", "sparse-attn & MTP (T1–T15)", AQUA, None),
    ("Screens &\nexperiments", "11-11", "11-16", "recall surgery", AQUA, None),
    ("Screens &\nexperiments", "11-18", "11-19", "", AQUA, None),
    # conservative
    ("Conservative\n(q35)", "10-24 18:00", "10-26 12:00", "run_2 (MHA)", ORANGE, None),
    ("Conservative\n(q35)", "10-27", "11-03 09:20", "main_gqa → 1.10T", ORANGE, None),
    # ambitious
    ("Ambitious\n(main run)", "10-24 18:00", "10-26 23:48", "v1 51B (MHA)", BLUE, None),
    ("Ambitious\n(main run)", "10-28 09:00", "10-29 12:00", "LDM", BLUE, LDM),
    ("Ambitious\n(main run)", "10-30 12:00", "10-31 20:00", "v2", BLUE, LDM),
    ("Ambitious\n(main run)", "10-31 20:25", "11-01 22:45", "v3", BLUE, LDM),
    ("Ambitious\n(main run)", "11-02 01:30", "11-09 20:40", "v4_2ph 84B (mHC-4), 8k context", BLUE, None),
    ("Ambitious\n(main run)", "11-09 20:40", "11-13 20:00", "64k → 36,000", "#1c5cab", None),
    ("Ambitious\n(main run)", "11-17", "11-19", "", BLUE, None),
    ("Ambitious\nside arms", "11-02", "11-04 10:00", "v4_ldm", VIOLET, LDM),
    ("Ambitious\nside arms", "11-09 12:00", "11-10 12:00", "", VIOLET, None),
    ("Ambitious\nside arms", "11-12 12:00", "11-13 06:00", "", VIOLET, None),
    ("Ambitious\nside arms", "11-16 04:00", "11-16 15:00", "", VIOLET, None),
    ("Ambitious\nside arms", "11-17", "11-18 00:30", "v4a", VIOLET, None),
]
lanes = ["Ambitious\n(main run)", "Ambitious\nside arms", "Conservative\n(q35)", "Screens &\nexperiments"]
y = {l: i for i, l in enumerate(lanes)}

decisions = [
    ("10-17", "CKDA always; angular MuOwn; 1k warm-up redo"),
    ("10-19", "mHC-4 > routed residual"),
    ("10-24", "Decision day: recipe frozen, both runs launched"),
    ("10-26", "GQA flag bug → both runs restart"),
    ("10-27", "LDM chosen (GQA rematch)"),
    ("10-30", "pab fixes expert collapse"),
    ("10-31", "84B-A2.9B-E80B vote; QB dropped"),
    ("11-01", "v3 LDM unstable → mHC-4 (v4)"),
    ("11-03", "No-Slurm-mutation policy"),
    ("11-04", "q35 ended; v4 → 256 nodes"),
    ("11-05", "spectral cap + z-loss; head dim 256"),
    ("11-09", "8k → 64k context switch at step 23,691"),
    ("11-10", "Engram row-norm cap 13→7 + own clip, lr ×1 (25k)"),
    ("11-16", "V-layer insertion (v4a)"),
    ("11-17", "v4b copy-init + gates; v4c ramp"),
    ("11-18", "sparse-attention audit"),
]

fig = plt.figure(figsize=(13, 7.6))
ax = fig.add_axes([0.09, 0.40, 0.89, 0.52])
H = 0.56
CH = 0.21  # days per character at this scale
for lane, s, e, lab, col, hatch in bars:
    x0, x1 = mdates.date2num(d(s)), mdates.date2num(d(e))
    ax.broken_barh([(x0, x1 - x0)], (y[lane] - H / 2, H), facecolors=col, edgecolor=SURFACE,
                   linewidth=2, hatch=hatch, alpha=0.92)
    if lab:
        dark = col in (BLUE, VIOLET, "#1c5cab")
        w = x1 - x0
        longest = max(len(t) for t in lab.split("\n"))
        if longest * CH + 0.25 < w and not hatch:
            ax.text(x0 + 0.12, y[lane], lab, va="center", ha="left", fontsize=8.2,
                    color="white" if dark else INK)
        else:
            ax.text(x0 + w / 2, y[lane] + H / 2 + 0.05, lab.replace("\n", " "), va="bottom", ha="center",
                    fontsize=7.8, color=INK2)

for s, lab in [("11-09 12:00", "ckda_branch"), ("11-12 12:00", "broad"), ("11-16 04:00", "DK03 (JUPITER)")]:
    ax.text(mdates.date2num(d(s)), y["Ambitious\nside arms"] - H / 2 - 0.05, lab, va="top", ha="left",
            fontsize=7.5, color=INK2)
ax.text(mdates.date2num(d("11-17")), y["Ambitious\n(main run)"] - H / 2 - 0.05, "v4b · v4c · v4_cont",
        va="top", ha="left", fontsize=7.8, color=INK2)
ax.text(mdates.date2num(d("11-18")), y["Screens &\nexperiments"] + H / 2 + 0.05, "t16", va="bottom",
        ha="left", fontsize=7.8, color=INK2)

top = len(lanes) - 0.5
for i, (s, lab) in enumerate(decisions):
    x = mdates.date2num(d(s)) + 0.5
    ax.axvline(x, color=GRID, lw=0.8, zorder=0)
    yy = top + 0.3 + (i % 2) * 0.42
    ax.plot(x, yy, marker="o", ms=13, color=INK, mec=SURFACE, mew=1.5, zorder=3)
    ax.text(x, yy, str(i + 1), va="center", ha="center", fontsize=7.2, color="white", zorder=4,
            fontweight="bold")

ax.set_yticks(range(len(lanes)))
ax.set_yticklabels(lanes)
ax.set_ylim(-0.75, top + 0.95)
ax.set_xlim(mdates.date2num(d("10-14")), mdates.date2num(d("11-21")))
ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 4, 7, 10, 13, 16, 19, 22, 25, 28]))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.grid(axis="y", visible=False)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.legend(handles=[Patch(facecolor=BLUE, label="ambitious main run (8k)"),
                   Patch(facecolor="#1c5cab", label="ambitious main run (64k)"),
                   Patch(facecolor=VIOLET, label="ambitious side arms (surgery / LDM)"),
                   Patch(facecolor=ORANGE, label="conservative q35"),
                   Patch(facecolor=AQUA, label="small-scale screens / tests"),
                   Patch(facecolor="white", edgecolor=INK2, hatch=LDM, label="LDM residual")],
          loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=6, frameon=False, fontsize=8.0)
ax.set_title("Nucleus hybrid campaign, 2026-10-14 → 11-18: runs and key decisions", loc="left",
             fontsize=11.5, color=INK, pad=8)

# numbered key
kx = fig.add_axes([0.09, 0.0, 0.89, 0.25]); kx.axis("off")
n = len(decisions); per = (n + 1) // 2
for i, (s, lab) in enumerate(decisions):
    col, row = divmod(i, per)
    xx, yy = 0.0 + col * 0.5, 1 - row / per
    kx.plot(xx + 0.008, yy, marker="o", ms=12, color=INK, transform=kx.transAxes, clip_on=False)
    kx.text(xx + 0.008, yy, str(i + 1), transform=kx.transAxes, va="center", ha="center", fontsize=7,
            color="white", fontweight="bold")
    kx.text(xx + 0.025, yy, f"{d(s):%b %d}  {lab}", transform=kx.transAxes, va="center", ha="left",
            fontsize=8.6, color=INK)
save(fig, "timeline.svg")
print("ok")
