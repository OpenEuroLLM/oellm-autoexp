#!/usr/bin/env python3
"""Recalibrated training loss of the ambitious v4_2ph run on the conservative (q35) data mix.

v4_2ph trains on the option8_1 v8ml mix (opt81, agent tokenizer); q35 (and the first ambitious run) on
flag-decon-v1 (base tokenizer). The two training losses are not comparable: opt81 holds 20.9 % reasoning
/ SFT data (loss ~0.7) where flag-v1 holds 5.4 % reasoning + 11.4 % full mixture-vitae. This script
re-weights v4's per-category losses on the fixed held-out validation sets (W&B `lm loss validation-<set>`,
every 1000 steps; docs/datamix_v8/README.md) to flag-v1's category shares:

    L_flag(t) = sum_c w_flag[c] * L_c(t) + w_flag[mixture-vitae] * L_all(t) + r_q

    c in web_en, multilingual, code, math, reasoning (the fixed sets partition the v8ml mix exactly);
    full mixture-vitae (web + instruct, no category of its own) is modelled by the 'all' set;
    r_q = q35's own residual: its training loss minus the same re-weighting of ITS fixed-set losses,
    -0.0494 at step 4000 (67BT) and -0.0528 at step 59600 (1T) -> -0.051 (8k training stream vs 4k sets,
    within-category differences, mixture-vitae vs 'all').

Checks: the 5-category re-weighting with v8ml weights reproduces v4's own 'all' set to +0.003..+0.011;
re-weighting to opt81 reproduces v4's training loss to +0.017..-0.017. The first ambitious run (flag-v1,
16k) gives r = -0.071 at both steps 1000 and 2500 (16k context ~ 0.02 lower than 8k). Modelling
mixture-vitae as 'reasoning' instead of 'all' (residual +0.027) lowers L_flag by 0.009-0.015.

Output (--out): recal_eval_steps.csv (one row per evaluation), recal_per_step.csv (the training loss with
the offset delta(t) = L_flag(t) - train(t) interpolated in log(step) between evaluations, held flat
outside), recal_vs_q35.png.

    python scripts/korbi/production/recalibrated_loss.py --out <dir> [--mix-dir <dir with the datamix files>]
"""

from __future__ import annotations

import argparse
import collections
import os

import numpy as np
import pandas as pd

ENTITY_PROJECT = "openeurollm-project/flag_hybrid"
CATS = ["web_en", "multilingual", "code", "math", "reasoning"]

# Category shares (fractions) of the training mixes, from the datamix files with the v8ml report's
# categories (--mix-dir recomputes them). opt81: datamix_v8ml_opt81_mn5.txt (stage-1 and stage-2 files
# identical shares); flag_v1: flag_decon_v1_datamix_mn5.txt (= the _users_ file q35 trains on).
WEIGHTS = {
    "v8ml": {"web_en": 0.30979, "multilingual": 0.20000, "code": 0.16720, "math": 0.11423,
             "reasoning": 0.20878, "mixture-vitae": 0.0},
    "opt81": {"web_en": 0.25120, "multilingual": 0.20000, "code": 0.22580, "math": 0.11423,
              "reasoning": 0.20878, "mixture-vitae": 0.0},
    "flag_v1": {"web_en": 0.28934, "multilingual": 0.19958, "code": 0.22027, "math": 0.12279,
                "reasoning": 0.05360, "mixture-vitae": 0.11444},
}

# q35 fixed-set losses (basetok sets; $R/fixedvalid_eval/q35c1t_basetok/s59600, q35c4k_basetok/s4000 +
# web/multilingual/code/math from q35c4k/s4000 -- no agent delimiters in those) and its training loss
# (W&B mean over +-50 steps)
Q35_CALIB = {
    4000: ({"web_en": 2.347623, "multilingual": 2.261750, "code": 0.9823838, "math": 1.184634,
            "reasoning": 0.9785464, "all": 1.670638}, 1.6867),
    59600: ({"web_en": 2.092081, "multilingual": 1.926137, "code": 0.8811576, "math": 0.9729916,
             "reasoning": 0.7617094, "all": 1.440920}, 1.4562),
}


def flag_mix(L, w=WEIGHTS["flag_v1"]):
    return sum(w[c] * L[c] for c in CATS) + w["mixture-vitae"] * L["all"]


def residual_q35():
    return float(np.mean([train - flag_mix(L) for L, train in Q35_CALIB.values()]))


def weights_from_mix_dir(d):
    """Recompute WEIGHTS from datamix_v8ml_mn5_report.tsv + the datamix files."""
    rep = []
    for line in open(os.path.join(d, "datamix_v8ml_mn5_report.tsv")):
        f = line.rstrip("\n").split("\t")
        if f[0] != "category":
            rep.append((f[6].rstrip("/") + "/", f[0]))
    rep.sort(key=lambda r: -len(r[0]))
    root = "/gpfs/projects/ehpc533/users/korbip/data/"

    def v8cat(p):
        for pre in ("v8ml-stages/s1/", "v8ml-stages/s2/", "v8ml-stages/s3/", "v8ml-stages-8k64k/s1/",
                    "v8ml-stages-8k64k/s2/"):
            p = p.replace(root + pre, root)
        c = next(c for dd, c in rep if p.startswith(dd))
        return {"web": "web_en", "mv1-instruct": "reasoning", "other": "reasoning"}.get(c, c)

    def flagcat(p):
        s = p.split("flag-decon-v1/")[1]
        src = s.split("/")[0]
        if src == "mixture-vitae-1.0":
            return "mixture-vitae"
        if src in ("nemotron-cc-1.0", "dclm-1.0", "finephrase-0.0.0"):
            return "web_en"
        if src in ("hplt-4.0", "finepdfs-edu-1.0.0"):
            return "web_en" if "/eng_Latn" in s else "multilingual"
        if src in ("nemotron-cc-tower+-0.1", "finepdfs-1.0.0", "nemotron-cc-opus-1.1", "fineopus-filtered-0.4",
                   "finewiki-0.0.0", "dochplt-3.1"):
            return "multilingual"
        if src in ("finemath-0.0.0", "megamath-0.0.0", "swallow-math-2.0", "openwebmath-0.0.0",
                   "nemotron-mind-0.0"):
            return "math"
        if src in ("swallow-code-2.0", "starcoder-0.0.0", "common-pile-stackv2-0.1",
                   "common-pile-stackv2-edu-0.1"):
            return "code"
        if src in ("nemotron-pretraining-specialized-1.0", "nemotron-pretraining-specialized-1.1",
                   "agenttrove-0.0"):
            return "reasoning"
        if src == "dolmino-mix-100b-1125":
            return "code" if any(k in s for k in ("cranecode", "stack_edu", "program_verifiable")) else "math"
        raise KeyError(p)

    def shares(name, fn):
        acc, tot = collections.Counter(), 0.0
        for line in open(os.path.join(d, name)):
            line = line.split("#")[0].strip()
            if line:
                w, p = line.split()[:2]
                acc[fn(p)] += float(w)
                tot += float(w)
        return {c: acc.get(c, 0.0) / tot for c in CATS + ["mixture-vitae"]}

    return {"v8ml": shares("datamix_v8ml_mn5.txt", v8cat), "opt81": shares("datamix_v8ml_opt81_mn5.txt", v8cat),
            "flag_v1": shares("flag_decon_v1_datamix_mn5.txt", flagcat)}


def fetch(run_id, keys, cache_dir):
    path = os.path.join(cache_dir, f"{run_id}.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)
    import wandb

    run = wandb.Api(timeout=300).run(f"{ENTITY_PROJECT}/{run_id}")
    cols = []
    for k in keys:
        rows = [(x["_step"], x[k]) for x in run.scan_history(keys=["_step", k], page_size=20000)]
        cols.append(pd.DataFrame(rows, columns=["step", k]).drop_duplicates("step", keep="last").set_index("step"))
    df = pd.concat(cols, axis=1).sort_index()
    df.to_csv(path)
    return df


def window_mean(s, step, half=50):
    return s[(s.index > step - half) & (s.index <= step + half)].mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mix-dir", help="local copies of the datamix files + v8ml report (recompute the shares)")
    ap.add_argument("--v4", default="ambitious_v4_2ph_main")
    ap.add_argument("--q35", default="conservative_q35_main_gqa")
    ap.add_argument("--v4-tokens-per-step", type=float, default=8192 * 8192)   # phase 1 (8k x 8192)
    ap.add_argument("--q35-tokens-per-step", type=float, default=2048 * 8192)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.mix_dir:
        WEIGHTS.update(weights_from_mix_dir(a.mix_dir))
    w_o, r_q = WEIGHTS["opt81"], residual_q35()

    sets = CATS + ["all"]
    v = fetch(a.v4, ["lm loss"] + [f"lm loss validation-{s}" for s in sets], a.out)
    q = fetch(a.q35, ["lm loss", "lm loss validation"], a.out)
    vt, qt = v["lm loss"].dropna(), q["lm loss"].dropna()

    def q35_at_tokens(tokens, rel=0.03):
        s = tokens / a.q35_tokens_per_step
        x = qt[(qt.index > s * (1 - rel)) & (qt.index <= s * (1 + rel))]
        return x.mean() if len(x) and s <= qt.index.max() else np.nan

    rows = []
    for step in v[f"lm loss validation-{sets[0]}"].dropna().index:
        L = {s: v.loc[step, f"lm loss validation-{s}"] for s in sets}
        train = window_mean(vt, step)
        heldout_opt81 = sum(w_o[c] * L[c] for c in CATS)
        heldout_flag = flag_mix(L)
        recal = heldout_flag + r_q
        q35 = q35_at_tokens(step * a.v4_tokens_per_step)
        rows.append({"step": int(step), "tokens_B": step * a.v4_tokens_per_step / 1e9, "train": train,
                     "heldout_opt81": heldout_opt81, "heldout_flag": heldout_flag,
                     "mix_offset_heldout": heldout_flag - heldout_opt81, "recal_train": recal,
                     "delta": recal - train, "ratio": recal / train, "q35_train_same_tokens": q35,
                     "recal_minus_q35": recal - q35, "raw_minus_q35": train - q35})
    ev = pd.DataFrame(rows).set_index("step")
    ev.to_csv(os.path.join(a.out, "recal_eval_steps.csv"), float_format="%.5f")

    # per step: train + delta(t), delta linear in log(step) between evaluations, flat outside
    steps = vt.index.values.astype(float)
    delta = np.interp(np.log(np.maximum(steps, 1)), np.log(ev.index.values.astype(float)), ev["delta"].values)
    per = pd.DataFrame({"tokens_B": steps * a.v4_tokens_per_step / 1e9, "train": vt.values, "delta": delta,
                        "recal_train": vt.values + delta}, index=vt.index)
    per.to_csv(os.path.join(a.out, "recal_per_step.csv"), float_format="%.5f")

    pd.set_option("display.width", 200)
    print(f"r_q = {r_q:+.4f}; weights opt81 / flag_v1:")
    print(pd.DataFrame({k: WEIGHTS[k] for k in ("opt81", "flag_v1")}).T.round(4))
    print(ev[["tokens_B", "train", "mix_offset_heldout", "delta", "ratio", "recal_train", "q35_train_same_tokens",
              "recal_minus_q35", "raw_minus_q35"]].round(4))
    plot(per, qt, ev, a)


def plot(per, qt, ev, a):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#1a1a19", "#6b6a63", "#e4e3dc"
    blue, orange, aqua = "#2a78d6", "#eb6834", "#1baf7a"
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
    smooth = lambda s: s.rolling(100, min_periods=20, center=True).mean()   # noqa: E731
    q = smooth(qt)
    ax.plot(q.index * a.q35_tokens_per_step / 1e9, q.values, color=orange, lw=2, label="q35 conservative (flag-v1)")
    ax.plot(per.tokens_B, smooth(per.train).values, color=blue, lw=1.5, alpha=0.45, ls="--",
            label="v4_2ph raw (opt81 mix)")
    ax.plot(per.tokens_B, smooth(per.recal_train).values, color=blue, lw=2, label="v4_2ph recalibrated to flag-v1")
    ax.scatter(ev.tokens_B, ev.recal_train, s=22, color=blue, zorder=3, edgecolor="white", linewidth=1.5)
    ax.set_xscale("log")
    ax.set_xlim(20, 1300)
    ax.set_ylim(1.15, 2.0)
    ax.set_xlabel("training tokens (B)", color=muted)
    ax.set_ylabel("training loss (8k)", color=muted)
    ax.set_title("v4_2ph training loss re-weighted to the q35 data mix", color=ink, loc="left", fontsize=12)
    ax.grid(True, which="both", color=grid, lw=0.6)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=muted)
    ax.legend(frameon=False, labelcolor=ink)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "recal_vs_q35.png"))


if __name__ == "__main__":
    main()
