"""End-to-end first test of the procedure on the data already in this repo.

    OMP_NUM_THREADS=1 python -m optimization_procedure.run_first_test [--out DIR] [--quick]

Steps (all numbers land in DIR/report.md + CSVs; the ledger in DIR/ledger/):

  1. load the architecture sweep (cooled evals) + the dense (lr x batch) baseline grid
  2. hierarchical robust fit (tags -> shared effects, EB shrinkage, LOO hold-out)
  3. leave-one-scale-out validation -> calibration inflation factor
  4. compute multipliers at the target regime (3B, D/N 100..1000) -> drop/live/promote
  5. next batch by information gain per GPU-hour; predictions registered in the ledger
  6. sequential-vs-parallel launch decisions for that batch at several wall-clock prices
  7. curve QA (spike / bump / stall) on the longest locally exported val curves

Single-threaded BLAS matters: the matrices are small and multi-threaded BLAS makes
them ~100x slower on a busy node.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import curves, data, decision, design, scheduler, validate  # noqa: E402
from .forms import get_form  # noqa: E402
from .ledger import Ledger, fingerprint  # noqa: E402
from .model import HierarchicalModel  # noqa: E402

SIZES = ("0.1B", "0.2B", "0.4B", "0.9B")


def md_table(df: pd.DataFrame, floatfmt: str = ".4g") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [format(v, floatfmt) if isinstance(v, (float, np.floating)) else str(v) for v in r]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def curve_qa_demo(max_curves: int = 6) -> pd.DataFrame:
    """QA on the exported dense-baseline val curves (stable trunks: constant lr after
    warmup). These are sparse val curves (<=200 points) -- a smoke test of the
    detectors on real data, not a substitute for the dense train-loss logs."""
    cdir = data.REPO / "dump" / "scaling_experiments_multilingual" / "curves"
    # one W&B file per restart segment: stitch segments of the same run first
    runs: dict[str, list[pd.DataFrame]] = {}
    for f in cdir.glob("*stable*.csv"):
        runs.setdefault(f.stem.split("__")[0], []).append(pd.read_csv(f).dropna())
    stitched = {k: pd.concat(v).drop_duplicates("_step", keep="last").sort_values("_step")
                for k, v in runs.items()}
    names = sorted(stitched, key=lambda k: -len(stitched[k]))[:max_curves]
    rows = []
    for name in names:
        c = stitched[name]
        steps = c["_step"].values.astype(int)
        loss = c.iloc[:, 1].values
        eta = curves.wsd_schedule(int(steps.max()), warmup=2000, stable_until=int(steps.max()))
        qa = curves.assess_run(steps, loss, eta)
        rows.append({"run": name, "segments": len(runs[name]), "points": len(steps),
                     "weight": qa.weight,
                     "flags": ",".join(qa.flags) or "-",
                     "events": "; ".join(f"{e.kind}@{e.start}({e.size:.3g})" for e in qa.events[:4])})
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(data.REPO / "dump" / "optimization_procedure"))
    ap.add_argument("--quick", action="store_true", help="fewer fantasies / validations")
    ap.add_argument("--batch", type=int, default=6)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(out / "ledger")
    rep: list[str] = ["# Optimization procedure: first test on repo data", "",
                      f"generated {time.strftime('%Y-%m-%d %H:%M')}", ""]
    T0 = time.time()

    # 1) data ----------------------------------------------------------------
    df = data.load_all()
    ledger.add_observations(df)
    rep += ["## 1. Data", "", md_table(df.groupby(["study", "arch"]).size().rename("n").reset_index()), ""]

    # 2) fit -----------------------------------------------------------------
    model = HierarchicalModel(get_form("chinchilla_bowl"), data.ARCH_TAGS,
                              ["arch_sweep", "dense_baseline"], "arch_sweep")
    # NB: ledger.load_priors() is for the *next* study (new arms / new data); loading
    # it here would count this data twice.
    fit = model.fit(df)
    print(f"[fit] {time.time() - T0:.1f}s sigma={fit.sigma} taus={fit.taus}")
    pt = fit.param_table()
    eff = fit.arch_effects()
    rep += ["## 2. Hierarchical fit", "",
            f"noise sd (log loss) per study: {({k: round(float(v), 5) for k, v in fit.sigma.items()})}; "
            f"learned shrinkage scales tau: {fit.taus}", "",
            "### Held-out observations", "",
            md_table(pd.DataFrame([h.__dict__ for h in fit.held_out])) if fit.held_out else "none", "",
            "### Per-architecture LOO residuals (structural misfit = the form does not describe the arm)", "",
            md_table(fit.arch_misfit), "",
            "### Population (base) parameters", "",
            md_table(pt[pt.level == "base"][["param", "mean", "sd"]]), "",
            "### Architecture effects (tag + arch deltas; arms without data = tag prior only)", "",
            md_table(eff.pivot(index="arch", columns="param", values="effect").reset_index()), ""]
    sim = fit.arch_similarity(N_grid=[1e8, 1e9, 3e9], D_grid=[3e10, 3e11, 1e12])
    sim.to_csv(out / "arch_similarity.csv")
    rep += ["### Functional distance between architectures (RMS posterior z of the envelope "
            "difference over an (N, D) grid; small = indistinguishable)", "",
            md_table(sim.reset_index().rename(columns={"index": "arch"}), ".2f"), ""]

    # 3) validation / calibration -----------------------------------------------
    sweep_sizes = sorted(df[df.study == "arch_sweep"].N.unique(), reverse=True)
    val_rows = []
    for N in sweep_sizes[: (1 if args.quick else 2)]:
        mask = ((df.N == N) & (df.study == "arch_sweep")).values
        v = validate.holdout_validation(model, df, mask, taus=fit.taus)
        v.to_csv(out / f"holdout_N{N:.3g}.csv", index=False)
        s = validate.summarize(v)
        per_arch = v.groupby("arch").rel_err_pct.mean().round(2).to_dict()
        val_rows.append({"held_out_N": f"{N:.3g}", **{k: round(x, 3) for k, x in s.items()},
                         "mean_err_pct_by_arch": per_arch})
    val = pd.DataFrame(val_rows)
    infl = float(max(1.0, val.rms_z.max()))
    calib = ledger.calibration()
    infl = max(infl, calib.get("inflation", 1.0))
    fit_c = fit.inflate(infl)
    print(f"[validate] {time.time() - T0:.1f}s inflation={infl:.2f}")
    rep += ["## 3. Leave-one-scale-out validation (refit without the size, predict it)", "",
            md_table(val), "",
            f"Calibration inflation used downstream: **{infl:.2f}** (max RMS z; ledger calibration "
            f"n={calib.get('n', 0)}).", ""]

    # 4) decisions ----------------------------------------------------------------
    regime = decision.Regime(N_ref=(3e9,), ratios=(100.0, 300.0, 1000.0))
    mult = decision.compute_multipliers(fit_c, regime=regime)
    flags = decision.stability_flags(fit)
    status = decision.eliminate(mult, flags=flags)
    mult.drop(columns="logM_samples").to_csv(out / "multipliers.csv", index=False)
    status.to_csv(out / "arm_status.csv", index=False)
    rep += ["## 4. Compute multipliers vs fullattn at 3B, D/N in {100, 300, 1000} (GPU-hours, "
            "incl. HP regret; calibrated)", "",
            md_table(mult.drop(columns=["logM_samples", "N_ref"])), "",
            md_table(status), ""]
    ledger.log_decision("arm_status", {"model": fingerprint(fit), "status": status.to_dict("records")})

    # 5) design -------------------------------------------------------------------
    live = status.loc[status.status.isin(["live", "review", "promote"]), "arch"].tolist()
    arms = ["fullattn"] + live
    sizes = {s: data.n_ne(s) for s in SIZES}
    cand = design.candidate_grid(fit_c, arms, sizes)
    Q, qlab = design.make_Q(fit_c, arms, N_T=3e9, ratios=regime.ratios)
    batch = design.select_batch(fit_c, cand, Q, k=args.batch)
    mu, sd = fit_c.predict(batch, noise=True)
    ledger.register_predictions(batch.assign(study="arch_sweep"), mu, sd, fingerprint(fit),
                                note="first_test proposal")
    batch["pred_loss"] = np.exp(mu)
    batch["pred_sd_pct"] = 100 * sd
    batch.to_csv(out / "proposed_batch.csv", index=False)
    print(f"[design] {time.time() - T0:.1f}s")
    rep += ["## 5. Proposed next runs (greedy information gain per GPU-hour about the "
            f"{len(qlab)} decision quantities)", "",
            md_table(batch[["arch", "size", "D", "lr", "gbs", "cost", "IG", "acq", "pred_loss",
                            "pred_sd_pct"]]), ""]

    # 6) sequential vs parallel ---------------------------------------------------
    k_p = max(1, args.batch // 2)
    pending = batch.iloc[:k_p].drop(columns=["cand_index"])
    gpus = 16  # 4 nodes x 4 GPUs, the sweep's typical allocation
    dt = float(pending.cost.max() / gpus)

    def mk_cand(f, archs):
        return design.candidate_grid(f, archs, sizes)

    def mk_Q(f, archs):
        return design.make_Q(f, archs, N_T=3e9, ratios=regime.ratios)

    sched_rows = []
    for lam in (0.0, 10.0, 100.0):
        dec = scheduler.launch_decisions(fit_c, mk_cand, mk_Q, pending, k=args.batch - k_p,
                                         K=6 if args.quick else 12,
                                         clock=scheduler.Clock(lam=lam, dt_hours=dt))
        dec.insert(0, "lambda_gpuh_per_h", lam)
        sched_rows.append(dec)
    sched = pd.concat(sched_rows, ignore_index=True)
    sched.to_csv(out / "launch_decisions.csv", index=False)
    print(f"[schedule] {time.time() - T0:.1f}s")
    rep += ["## 6. Launch now or wait? (the first "
            f"{k_p} proposals are pending, results in ~{dt:.1f} h on {gpus} GPUs)", "",
            md_table(sched[["lambda_gpuh_per_h", "arch", "size", "D", "lr", "cost",
                            "P_wanted_after_pending", "waste_gpuh", "wallclock_gain_gpuh",
                            "launch_now"]]), ""]

    # 7) curve QA -----------------------------------------------------------------
    qa = curve_qa_demo()
    qa.to_csv(out / "curve_qa.csv", index=False)
    rep += ["## 7. Curve QA on exported val curves", "", md_table(qa), ""]

    ledger.save_priors(fit.to_prior())
    rep += [f"_total runtime {time.time() - T0:.0f}s_", ""]
    (out / "report.md").write_text("\n".join(rep))
    print(f"[done] {time.time() - T0:.1f}s -> {out / 'report.md'}")


if __name__ == "__main__":
    main()
