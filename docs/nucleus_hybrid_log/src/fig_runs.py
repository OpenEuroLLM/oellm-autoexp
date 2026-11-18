"""Main-run loss curves (MN5 logs, harvested read-only) and the V-arm paired comparison."""
import csv
import numpy as np
from matplotlib.lines import Line2D
from fig_common import *

C = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data") + "/"


def load(name):
    rows = [r for r in csv.DictReader((l for l in open(C + name + ".tsv") if not l.startswith("#")), delimiter="\t")]
    st = np.array([int(r["step"]) for r in rows])
    tok = np.array([float(r["tokens"]) for r in rows])
    loss = np.array([float(r["lm_loss"]) for r in rows])
    n = np.array([int(r["n"]) for r in rows])
    return st, tok, loss, n


def smooth(x, y, w):
    if w <= 1:
        return x, y
    k = np.ones(w) / w
    return x[w - 1:], np.convolve(y, k, mode="valid")


# ---------------------------------------------------------------- figure 1: main runs
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.0), gridspec_kw={"width_ratios": [1, 1.25]})

early = [  # name, label, color, smoothing window (rows)
    ("amb_v4_main_lineage", "v4_2ph (mHC-4, 84B) — kept", BLUE, 3),
    ("q35_main_gqa", "q35 conservative (GQA)", ORANGE, 3),
    ("amb1_e216_eng40", "v1 51B (mHC-4, MHA bug)", AQUA, 3),
    ("amb_v3", "v3 84B (LDM, 4k phase)", YELLOW, 10),
    ("amb_ldm_e216_eng40", "LDM 51B restart", MAGENTA, 10),
    ("amb_v4_ldm", "v4_ldm (LDM, 8k)", GREEN, 10),
]
for name, lab, col, w in early:
    st, tok, loss, n = load(name)
    m = tok <= 185e9
    x, yv = smooth(tok[m] / 1e9, loss[m], w)
    a1.plot(x, yv, color=col, lw=1.8, label=lab)
a1.set_xlim(0, 185)
a1.set_ylim(1.35, 2.2)
a1.set_xlabel("tokens (B)")
a1.set_ylabel("train lm loss (smoothed)")
a1.set_title("First 185B tokens: the runs that did not survive", loc="left", fontsize=10.5)
a1.annotate("LDM restart: grad spikes,\ncancelled at step 1,038", xy=(66, 1.80), xytext=(72, 1.95), fontsize=8,
            color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
a1.annotate("v3: front-of-network\nblow-up at step ~1,750", xy=(116, 1.585), xytext=(124, 1.70), fontsize=8,
            color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
a1.legend(frameon=False, fontsize=7.8, loc="upper right")

# right: full v4 lineage vs q35
st, tok, loss, n = load("amb_v4_main_lineage")
x, yv = smooth(tok / 1e12, loss, 5)
a2.plot(x, yv, color=BLUE, lw=1.8, label="ambitious v4 main line (opt81 mix, raw)")
st2, tok2, loss2, n2 = load("q35_main_gqa")
x2, y2 = smooth(tok2 / 1e12, loss2, 5)
a2.plot(x2, y2, color=ORANGE, lw=1.8, label="conservative q35 (flag-decon-v1, raw)")
rec = [(0.067, 1.639), (0.268, 1.416), (0.537, 1.344), (1.0, 1.290), (1.07, 1.284)]
a2.plot([r[0] for r in rec], [r[1] for r in rec], ls="none", marker="o", ms=7, color=BLUE, mfc=SURFACE, mew=2,
        label="v4 recalibrated to q35's data mix")
events = [(5909, "256 nodes"), (9815, "cap + z-loss,\nhead dim 256"), (23691, "64k phase"),
          (25000, "Engram\npackage"), (35900, "V-layer arms\nbranch here")]
for i, (s, lab) in enumerate(events):
    xx = s * 67108864 / 1e12
    a2.axvline(xx, color=GRID, lw=1, zorder=0)
    a2.text(xx + 0.02, 1.62 - (i % 2) * 0.07, lab, fontsize=7.6, color=INK2, va="top")
a2.text(2.61, 1.115, "v4_cont\n38,897", fontsize=7.6, color=INK2, va="bottom", ha="right")
a2.text(1.10, 1.462, "q35 ended\n1.10T", fontsize=7.6, color=INK2, va="bottom", ha="left")
a2.set_xlim(0, 2.7)
a2.set_ylim(1.05, 1.65)
a2.set_xlabel("tokens (T)")
a2.set_title("Full run: ambitious v4 line vs conservative baseline", loc="left", fontsize=10.5)
a2.legend(frameon=False, fontsize=7.8, loc="lower left")
fig.suptitle("Main runs on MN5 (train loss from the run logs)", x=0.01, ha="left", fontsize=11.5, y=1.02)
fig.text(0.01, -0.04, "Raw losses are not comparable across data mixes: v1 / LDM-restart / q35 train on flag-decon-v1 "
         "(tokenizer-256k), v3+ on v8ml / opt81 (agent tokenizer), which reads ~0.09–0.11 lower at equal tokens. "
         "The recalibrated points (docs/loss_extrapolation) put v4 on q35's mix: −0.168 at 1.07T.",
         fontsize=7.8, color=MUTED, ha="left")
fig.tight_layout()
save(fig, "main_runs_loss.svg")

# ---------------------------------------------------------------- figure 2: V arms
fig, (b1, b2) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.35, 1]})
W = 50


def windows(name):
    st, tok, loss, n = load(name)
    out = {}
    for s, l, k in zip(st, loss, n):
        if s <= 35900:
            continue
        w = (s - 35901) // W
        a = out.setdefault(w, [0.0, 0])
        a[0] += l * k
        a[1] += k
    return {w: v[0] / v[1] for w, v in out.items() if v[1] >= W * 0.8}


ref = windows("v4_cont_main")
arms = [("v4a_v6_pin", "v4a  zero-init V layers", MAGENTA),
        ("v4b_learned", "v4b  copy-init + learnable gates", BLUE),
        ("v4c_ramp", "v4c  copy-init + 100-step ramp", AQUA)]
for name, lab, col in arms:
    w = windows(name)
    ks = sorted(k for k in w if k in ref)
    xs = [35901 + k * W + W / 2 for k in ks]
    ys = [(w[k] - ref[k]) * 1000 for k in ks]
    b1.plot(xs, ys, color=col, lw=1.8, marker="o", ms=3.5, label=lab)
b1.axhline(0, color=MUTED, lw=1)
b1.set_ylim(-3, 14)
b1.set_xlabel("step (all arms branch from 35,900 on identical batches)")
b1.set_ylabel("train loss − v4_cont  (mnat, 50-step windows)")
b1.set_title("Paired training loss vs the vanilla control", loc="left", fontsize=10.5)
b1.annotate("v4a 35,901–36,000: +68 mnat\n(off scale; cold random branch)", xy=(36026, 10.9), xytext=(36150, 11.5),
            fontsize=8, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
b1.legend(frameon=False, fontsize=8, loc="center right")
b1.text(35905, 6.4, "v4c ramp\nwindow", fontsize=7.6, color=INK2, ha="left")

sets = ["all", "web_en", "multilingual", "code", "math", "reasoning", "long_64k"]


def val(name):
    rows = list(csv.DictReader(open(C + name + "_val.tsv"), delimiter="\t"))
    return {int(r["step"]): {s: float(r["val_" + s]) for s in sets} for r in rows}


vref = val("v4_cont_main")
mk = {36000: "o", 37000: "D"}
for j, (name, lab, col) in enumerate(arms):
    v = val(name)
    for stp in (36000, 37000):
        if stp not in v:
            continue
        xs = [(v[stp][s] - vref[stp][s]) * 1000 for s in sets]
        ys = [len(sets) - 1 - i + (j - 1) * 0.22 for i in range(len(sets))]
        b2.plot(xs, ys, ls="none", marker=mk[stp], ms=7, color=col, mec=SURFACE, mew=1.5,
                alpha=1.0 if stp == 37000 else 0.55)
b2.axvline(0, color=MUTED, lw=1)
b2.set_yticks(range(len(sets)))
b2.set_yticklabels(sets[::-1])
b2.set_xlabel("validation loss − v4_cont  (mnat)")
b2.set_title("Fixed validation sets at 36k (faint) and 37k", loc="left", fontsize=10.5)
b2.grid(axis="y", visible=False)
b2.tick_params(axis="y", length=0)
b2.legend(handles=[Line2D([], [], marker="o", ls="none", color=MUTED, alpha=0.55, ms=7, label="step 36,000"),
                   Line2D([], [], marker="D", ls="none", color=MUTED, ms=7, label="step 37,000 (v4c: n/a)")],
          frameon=False, fontsize=8, loc="lower right")
fig.suptitle("V-layer insertion arms (11-17 → 11-18): train loss reaches parity, long_64k improves",
             x=0.01, ha="left", fontsize=11.5, y=1.02)
fig.tight_layout()
save(fig, "v_arms.svg")
print("ok")
