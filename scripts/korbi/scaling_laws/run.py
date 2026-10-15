#!/usr/bin/env python3
"""Scaling-law pipeline for the multilingual architecture sweep.

Extract eval val losses -> annotate with training FLOPs and GPU-hours -> fit and
plot per-architecture compute scaling curves L(C) = E + A*C^-alpha.

Usage:
  # from the pre-extracted metrics CSV:
  python scripts/korbi/scaling_laws/run.py --input dump/eval_val_metrics.csv --out dump/scaling_laws
  # or directly from the eval job dirs on the cluster / a synced copy:
  python scripts/korbi/scaling_laws/run.py --input /path/to/..._main_7to1_eval --out dump/scaling_laws

Outputs (in --out): scaling_data.csv, fits.csv, chinchilla.json,
scaling_flops.png, scaling_gpu_hours.png.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chinchilla  # noqa: E402
import compare  # noqa: E402
import extract  # noqa: E402
import fit  # noqa: E402
import hpselect  # noqa: E402  (NOT `select.py` — that shadows the stdlib module)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input",
        required=True,
        help="metrics CSV (variant,size,gbs,lr,type,budgetBT,val_loss,val_ppl,iter) "
        "or an eval output dir of qwen3_*_eval*/slurm-*.log",
    )
    ap.add_argument("--out", default="dump/scaling_laws", help="output directory")
    ap.add_argument(
        "--type",
        choices=["cooldown", "stable", "both"],
        default="cooldown",
        help="which checkpoints to fit (default: cooldown finals)",
    )
    ap.add_argument(
        "--flops-mode",
        choices=["6nd", "full"],
        default="6nd",
        help="'6nd' Chinchilla, or 'full' adds self-attention FLOPs",
    )
    ap.add_argument(
        "--n-mode",
        choices=["measured", "nominal"],
        default="measured",
        help="scaling-law N: each arch's MEASURED non-embedding params "
        "(default), or the shared dense-geometry 'nominal' count "
        "(the pre-2026-09-29 accounting; credits heavier hybrids "
        "with too few params)",
    )
    ap.add_argument(
        "--hp-select",
        choices=["per-arch", "shared", "none"],
        default="per-arch",
        help="lr/bs selection before fitting (default: per-arch envelope). "
        "The head-to-head comparison always uses matched 'shared' cells.",
    )
    ap.add_argument(
        "--keep-incomplete",
        action="store_true",
        help="keep evals whose checkpoint iteration disagrees with the budget",
    )
    ap.add_argument(
        "--bootstrap",
        type=int,
        default=200,
        help="bootstrap resamples for the Chinchilla exponent CIs (0 = skip)",
    )
    ap.add_argument(
        "--no-chinchilla",
        action="store_true",
        help="skip the L(N,D) fit and compute-optimal frontier",
    )
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    df = extract.load(
        args.input,
        flops_mode=args.flops_mode,
        keep_incomplete=args.keep_incomplete,
        n_mode=args.n_mode,
    )
    if df.empty:
        print("No metrics found in input.")
        return
    type_map = {"evalcd": "cooldown", "evalstable": "stable"}
    df["kind"] = df["type"].map(type_map).fillna(df["type"])
    if args.type != "both":
        df = df[df["kind"] == args.type]

    # ---- step 1: pick the best (gbs, lr) per cell BEFORE anything is fitted ----
    hpselect.report_coverage(df)
    hpselect.report_tier_switches(df)
    matched = hpselect.select_best(df, mode="shared", verbose=False)
    df = hpselect.select_best(df, mode=args.hp_select)

    df = df.sort_values(["arch", "size", "budget_bt"]).reset_index(drop=True)
    data_csv = os.path.join(args.out, "scaling_data.csv")
    df.to_csv(data_csv, index=False)
    print(f"[data] {len(df)} points ({args.type}) -> {data_csv}")

    # ---- step 2: head-to-head vs full attention on matched cells ----
    _compare(matched, args.out)
    if args.n_mode == "measured":
        _compare_param_adjusted(matched, args.out, n_boot=args.bootstrap)

    # per-arch power-law fits for both cost axes
    fits = []
    for xcol, xlabel, png in [
        ("flops", f"training FLOPs ({args.flops_mode})", "scaling_flops.png"),
        ("gpu_hours", "training GPU-hours", "scaling_gpu_hours.png"),
    ]:
        f = fit.fit_all(df, xcol)
        fits.append(f)
        fit.plot(df, f, xcol, xlabel, os.path.join(args.out, png))
        print(f"[plot] {os.path.join(args.out, png)}")

    import pandas as pd

    fits_df = pd.concat(fits, ignore_index=True)
    fits_csv = os.path.join(args.out, "fits.csv")
    fits_df.to_csv(fits_csv, index=False)
    print(f"[fits] -> {fits_csv}")
    print(fits_df.to_string(index=False))

    # pooled Chinchilla L(N,D) fit (bonus)
    ch = fit.fit_chinchilla(df)
    if ch:
        with open(os.path.join(args.out, "chinchilla.json"), "w") as fh:
            json.dump(ch, fh, indent=2)
        print(f"[chinchilla] L(N,D)=E+A/N^a+B/D^b : {ch}")

    # how the architecture term enters L(N,D): offset vs effective size/data
    hyp = fit.fit_hypotheses(df)
    if not hyp.empty:
        hyp.to_csv(os.path.join(args.out, "hypotheses.csv"), index=False)
        print("\n[hypotheses] how the architecture advantage enters L(N,D)")
        print(hyp.to_string(index=False))

    # ---- step 4: Chinchilla-style L(N,D) + compute-optimal frontier ----
    if not args.no_chinchilla:
        chinchilla.isoflop_feasibility(df)
        res = chinchilla.analyse(df, n_boot=args.bootstrap, n_mode=args.n_mode)
        if not res.empty:
            res.to_csv(os.path.join(args.out, "chinchilla_fits.csv"), index=False)
            print("\n[chinchilla] L(N,D) = E + A*N^-alpha + B*D^-beta, per architecture")
            cols = [
                "arch",
                "n",
                "rmse_mn",
                "E",
                "A",
                "alpha",
                "B",
                "beta",
                "N_exp_flops",
                "N_exp_gpuh",
                "tok_per_param_ref",
                "L_opt_ref",
            ]
            print(res[cols].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
            print(
                f"  L_opt_ref = compute-optimal loss at the SHARED budget "
                f"C={res['C_ref'].iloc[0]:.3g} FLOPs (top of the range all archs cover)."
            )
            print(
                "  (A and B are only comparable across archs at equal alpha/beta — "
                "rank with L_opt_ref or the pairwise table, never with A/B directly)"
            )
            print("\n  bootstrap 90% CIs (size-stratified):")
            ci = [
                "arch",
                "alpha_lo",
                "alpha_hi",
                "beta_lo",
                "beta_hi",
                "n_exp_lo",
                "n_exp_hi",
                "n_boot_ok",
            ]
            print(res[[c for c in ci if c in res]].round(4).to_string(index=False))
            chinchilla.plot_frontier(df, res, os.path.join(args.out, "chinchilla_frontier.png"))
            print(f"[plot] {os.path.join(args.out, 'chinchilla_frontier.png')}")
            for xcol, xlabel, png in [
                ("gpu_hours", "training GPU-hours", "loglog_gpu_hours.png"),
                ("flops", "training FLOPs (6ND)", "loglog_flops.png"),
            ]:
                chinchilla.plot_loglog(df, res, os.path.join(args.out, png), xcol, xlabel)
                print(f"[plot] {os.path.join(args.out, png)}")


def _compare(matched, out):
    """Matched-pair head-to-head vs full attention + the two comparison
    plots."""
    pairs = compare.matched_pairs(matched)
    if pairs.empty:
        print("[compare] no matched fullattn pairs — skipping head-to-head")
        return
    pairs.to_csv(os.path.join(out, "hybrid_vs_fullattn.csv"), index=False)
    print(
        f"\n[compare] {len(pairs)} matched pairs vs fullattn "
        f"-> {os.path.join(out, 'hybrid_vs_fullattn.csv')}"
    )
    print("  loss RATIO vs fullattn (%, negative = hybrid better):")
    print(
        pairs.pivot_table(index=["size", "budget_bt"], columns="arch", values="ratio_pct")
        .round(3)
        .to_string()
    )

    trend = compare.trend_test(pairs)
    trend.to_csv(os.path.join(out, "hybrid_vs_fullattn_trend.csv"), index=False)
    print("\n  is the RATIO constant? (a flat ratio extrapolates to arbitrary scale)")
    rcols = [
        "arch",
        "n",
        "mean_ratio_pct",
        "se_ratio",
        "dRatio_dlnN",
        "se_rlnN",
        "dRatio_dlnD",
        "se_rlnD",
        "resid_ratio_pct",
    ]
    print(trend[[c for c in rcols if c in trend]].round(4).to_string(index=False))
    print("\n  same test on the absolute difference, and which invariant the data prefers:")
    dcols = [
        "arch",
        "mean_delta_mn",
        "dDelta_dlnN",
        "se_dlnN",
        "dDelta_dlnD",
        "se_dlnD",
        "t_delta_lnD",
        "t_ratio_lnD",
        "better_invariant",
    ]
    print(trend[[c for c in dcols if c in trend]].round(3).to_string(index=False))

    pw = compare.pairwise(matched)
    if not pw.empty:
        pw.to_csv(os.path.join(out, "pairwise_ranking.csv"), index=False)
        print("\n  WHICH IS BEST — every arch vs every other, paired on shared cells")
        print("  (mean_pct < 0 => arch_a better; |t| > 2 => resolved)")
        print(pw.round(3).to_string(index=False))

    by_tier = compare.ranking_by_tier(matched)
    if not by_tier.empty:
        by_tier.to_csv(os.path.join(out, "ranking_by_tier.csv"), index=False)
        print("\n  is that ranking stable? loss ratio vs gdn7_nope per (gbs,lr) tier (%):")
        print(
            by_tier.pivot_table(index="tier", columns="arch", values="mean_pct")
            .round(3)
            .to_string()
        )
        print(
            "  a margin that swings between tiers is hyperparameter sensitivity, "
            "not architecture — do not average over it."
        )

    mult = compare.iso_loss_multiplier(matched)
    if not mult.empty:
        mult.to_csv(os.path.join(out, "iso_loss_multiplier.csv"), index=False)
        print("\n  iso-loss GPU-hour multiplier (>1 = hybrid reaches the loss cheaper):")
        print(
            mult.pivot_table(index=["size", "budget_bt"], columns="arch", values="gpuh_mult")
            .round(2)
            .to_string()
        )
        compare.plot_multipliers(mult, os.path.join(out, "compute_multiplier.png"))
    compare.plot_ratios(pairs, os.path.join(out, "ratio_vs_fullattn.png"))
    compare.plot_ratio_vs_compute(
        pairs, os.path.join(out, "ratio_vs_gpu_hours.png"), xcol="gpu_hours"
    )
    compare.plot_ratio_vs_compute(pairs, os.path.join(out, "ratio_vs_flops.png"), xcol="flops")
    compare.plot_deltas(pairs, os.path.join(out, "delta_vs_fullattn.png"))


def _compare_param_adjusted(matched, out, n_boot=200):
    """Step 2b: the same head-to-head with each arch's extra parameters priced out.

    Raw per-cell deltas mix two things — architecture and model size — because
    the hybrids are 9-24% bigger in N_ne at the same size label. Every loss is
    moved to fullattn's N of its size along fullattn's fitted N-term (see
    `compare.param_adjusted`); what remains is the architecture effect under the
    null that a mixer parameter is worth what a dense parameter is.
    """
    adj, fp, samples = compare.param_adjusted(matched, n_boot=n_boot)
    if adj.empty:
        print("[compare-adj] fullattn L(N,D) fit failed — skipping param adjustment")
        return
    adj.to_csv(os.path.join(out, "param_adjusted_losses.csv"), index=False)
    print(
        f"\n[compare-adj] fullattn N-term used for the adjustment: "
        f"A={fp['A']:.4g}, alpha={fp['alpha']:.4f} ({len(samples)} bootstrap draws)"
    )
    credit = adj.groupby(["arch", "size"])[
        [c for c in ("N", "n_credit_mn", "n_credit_lo_mn", "n_credit_hi_mn") if c in adj]
    ].first()
    credit["N_ne_vs_fullattn_pct"] = 100.0 * (
        credit["N"] / adj.groupby(["arch", "size"])["N_ref"].first() - 1.0
    )
    print("  loss credited to extra parameters alone (milli-nats, 90% band):")
    print(credit.round(2).to_string())

    pairs = compare.matched_pairs(compare.with_adjusted_loss(adj))
    pairs.to_csv(os.path.join(out, "hybrid_vs_fullattn_nadj.csv"), index=False)
    print("\n  PARAM-ADJUSTED loss ratio vs fullattn (%, negative = hybrid better):")
    print(
        pairs.pivot_table(index=["size", "budget_bt"], columns="arch", values="ratio_pct")
        .round(3)
        .to_string()
    )
    trend = compare.trend_test(pairs)
    trend.to_csv(os.path.join(out, "hybrid_vs_fullattn_nadj_trend.csv"), index=False)
    print(
        trend[
            [
                "arch",
                "n",
                "mean_delta_mn",
                "se_mean",
                "mean_ratio_pct",
                "se_ratio",
                "dRatio_dlnD",
                "se_rlnD",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    pw = compare.pairwise_adjusted(adj, samples)
    if not pw.empty:
        pw.to_csv(os.path.join(out, "pairwise_ranking_nadj.csv"), index=False)
        print(
            "\n  PARAM-ADJUSTED pairwise (mean_pct < 0 => arch_a better; "
            "fit_lo/hi = 90% band from the fullattn-fit bootstrap):"
        )
        print(pw.round(3).to_string(index=False))

    png = os.path.join(out, "param_adjusted_vs_fullattn.png")
    compare.plot_param_adjusted(adj, png)
    print(f"[plot] {png}")

    by_tier = compare.ranking_by_tier(compare.with_adjusted_loss(adj))
    if not by_tier.empty:
        by_tier.to_csv(os.path.join(out, "ranking_by_tier_nadj.csv"), index=False)
        print("\n  PARAM-ADJUSTED loss ratio vs gdn7_nope per tier (%):")
        print(
            by_tier.pivot_table(index="tier", columns="arch", values="mean_pct")
            .round(3)
            .to_string()
        )


if __name__ == "__main__":
    main()
