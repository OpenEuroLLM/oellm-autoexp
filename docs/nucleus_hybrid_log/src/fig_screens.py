"""Small-scale screen results: loss delta (nats) per feature at s1 / s2 / s3, with the decision."""
from matplotlib.lines import Line2D
from fig_common import *

# feature, reference, (s1, s2, s3), outcome
rows = [
    ("XSA on attention (R1, AM)", "vs attention", (-0.018, -0.020, None), "kept"),
    ("CKDA 3:1 hybrid (R3/3C, AM)", "vs attention", (-0.022, -0.010, None), "kept (7:1)"),
    ("Qwen3.5-like GDN 3:1 (R5)", "vs CKDA 5:1 MoE", (0.023, 0.012, 0.004), "rejected"),
    ("all-attention MoE (R5)", "vs CKDA 5:1 MoE", (0.007, -0.004, None), "rejected"),
    ("mHC-4, mixer lr ×0.25 (R5)", "vs plain residual", (-0.051, -0.040, -0.037), "kept (final)"),
    ("Block AttnRes (R5)", "vs plain residual", (-0.002, 0.023, None), "rejected"),
    ("routed residual 1+1 of 8 (R5)", "vs plain residual", (-0.007, None, 0.010), "rejected"),
    ("LDM ×1 (GQA rematch)", "vs mHC-4", (0.0003, -0.0149, -0.0075), "chosen 10-27, dropped 11-04"),
    ("LatentMoE (R5)", "vs CKDA 5:1 MoE", (-0.008, -0.007, None), "never adopted"),
    ("Engram, 20 % of params (R5)", "vs CKDA 5:1 MoE", (-0.015, -0.018, -0.017), "kept (s3 = 34 %)"),
    ("pab bounded post-act norm", "vs base", (-0.003, 0.000, -0.005), "kept"),
    ("quantile balancing", "vs bias + aux", (-0.018, None, -0.015), "failed at 360 experts"),
]

fig, ax = plt.subplots(figsize=(10.5, 6.2))
cols = [SEQ["250"], SEQ["450"], SEQ["650"]]
marks = ["o", "s", "D"]
n = len(rows)
ax.axvspan(-0.004, 0.004, color="#f0efec", zorder=0)
ax.axvline(0, color=MUTED, lw=1, zorder=1)
for i, (name, ref, vals, outcome) in enumerate(rows):
    yy = n - 1 - i
    pts = [(v, k) for k, v in enumerate(vals) if v is not None]
    xs = [p[0] for p in pts]
    ax.plot([min(xs), max(xs)], [yy, yy], color=GRID, lw=3, zorder=2, solid_capstyle="round")
    for v, k in pts:
        ax.plot(v, yy, marker=marks[k], ms=8, color=cols[k], mec=SURFACE, mew=2, zorder=3, linestyle="none")
    ax.text(0.0415, yy, outcome, va="center", ha="left", fontsize=8.6, color=INK2)
ax.set_yticks(range(n))
ax.set_yticklabels([f"{r[0]}\n{r[1]}" for r in rows][::-1], fontsize=8.4)
ax.set_xlim(-0.06, 0.04); ax.set_ylim(-0.85, n - 0.4)
ax.set_xlabel("loss delta in nats at equal tokens (negative = better)")
ax.grid(axis="y", visible=False)
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.text(0.0045, -0.55, "± noise ≈ 0.004", ha="left", va="center", fontsize=7.8, color=MUTED)
ax.legend(handles=[Line2D([], [], marker=marks[k], color=cols[k], ls="none", ms=8, label=l)
                   for k, l in enumerate(["s1  ~85M active", "s2  ~230M active", "s3  ~490M active"])],
          loc="lower left", frameon=False, fontsize=8.4)
ax.set_title("Small-scale screens: feature effect by model size, and what we decided", loc="left",
             fontsize=11.5, color=INK, pad=10)
fig.text(0.01, -0.02, "Single seeds, 2–4k steps. MoE itself (16× sparsity vs dense at equal active size, R3B s1) is "
         "−0.144 nats, off this scale. Sources: docs/hybrid_moe_3b_campaign.md §7–§12, rematch, stabilization.",
         fontsize=7.8, color=MUTED, ha="left")
save(fig, "screen_deltas.svg")
print("ok")
