"""Learning-curve fits and health indicators for every run of the 3B campaign
(docs/hybrid_moe_3b_campaign.md 2.3), from the Megatron logs, with losspred.

    uv run --project ../loss_prediction python scripts/korbi/campaign/curves.py \
        <suite dir> [<suite dir> ...] [--predict-tokens 6e9 2e10] [--csv out.csv]

Per run (newest log of ``<suite>/<arm>``, as analyze.py finds it):

- **Fit**: the stable part of the training loss after the warmup (from max(warmup, 200)), with losspred's
  family model, time = cumulative relative LR ("LR area", so warmup and stable phase are one
  curve), anchored at the initial loss: ``L = E + A (tau + t0)^-alpha``. Parameters with
  Gauss-Newton standard errors and the trend RMSE (log-loss) of the fit.
- **Prediction**: the fitted trend's training loss when the same stable phase continues to each
  ``--predict-tokens`` budget.
- **Backtest**: the same fit on the first ``--backtest`` fraction of the stable phase,
  predicting the last logged point: relative error of the extrapolation (how far the run's own
  early part predicts its end).
- **Data-order correction** (``--data-correction``): runs with the same seed / gbs / length see
  the same batches, whose difficulty drifts against the fixed validation mix (round 1: train -
  val +0.08 early -> 0 at 6k steps, identical across arms). That drift is shared, so it is
  estimated from all such runs at the validation points and subtracted before fitting.
- **Health**:
  - spikes: losspred's hysteresis detector over the whole run from step 20 (warmup included);
    count, largest height, summed excess (instability penalty);
  - growth: the monotonicity detector's growth events (the loss rising above its running
    minimum for longer than a spike);
  - flattening: mean residual of the last 25 % of the stable phase against the backtest fit.
    Positive = the curve flattens faster than the power law predicts;
  - noise: the per-step noise of the log-loss around the trend.

Flags: ``healthy`` (no growth event, no spike above ``--max-spike``) and ``extrapolable``
(backtest error within ``--max-backtest``) -- a short run can be healthy but not extrapolable.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import LP, detector, parse_log, run_logs  # noqa: E402
from predict import schedule_from_log  # noqa: E402

sys.path.insert(0, str(LP / "src"))
from losspred.fitting.blocks import block_curve  # noqa: E402
from losspred.fitting.family import FamilyFitConfig, fit_family, predict_log  # noqa: E402
from losspred.schedule import WSDScheduleConfig  # noqa: E402

# no decay in a stable-only fit: the drop terms are unidentifiable and held fixed
FIXED = {"drop": 1e-5, "drop_exponent": 0.0, "shape": 1.0}


ANCHOR = True


def fit_stable(t, y, warmup, total, block):
    sched = WSDScheduleConfig(total=int(total), warmup=int(warmup), decay=0)
    # from the end of the warmup (at least step 200): a single power law in LR-area time misfits
    # the warmup's fast initial drop by 2-4 % (losspred findings), which would dominate short runs
    curve = block_curve("run", t, y, block=block, start=max(float(warmup), 200.0), max_steps=200.0,
                        rel_width=0.05)
    cfg = FamilyFitConfig(time="lr_area", fixed=dict(FIXED), anchor_init=ANCHOR)
    fit = fit_family({"run": curve}, {"run": sched}, cfg, init_loss=float(y[0]) if ANCHOR else None)
    p = dict(fit.params)
    pred = predict_log(p, sched, curve.t, p["drop_ref"], fit.time)
    rmse = float(np.sqrt(np.mean((pred - curve.z) ** 2)))
    return fit, sched, rmse


FORM = "pl"  # pl: losspred's family fit; pl_exp / pl_pl: the two-component forms


def _losspred_forms():
    import importlib.util

    spec = importlib.util.spec_from_file_location("compare_stable_forms",
                                                  LP / "scripts" / "compare_stable_forms.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._forms


def fit_form(t, y, warmup, total, block, form):
    """One of losspred's stable-phase forms (scripts/compare_stable_forms.py: power law, + exp,
    2 power laws; LR-area time, anchored at the initial loss) fitted to a run's blocks from
    max(warmup, 200). Returns a predictor of the loss at steps t."""
    from scipy.optimize import least_squares

    sched = WSDScheduleConfig(total=int(total), warmup=int(warmup), decay=0)
    curve = block_curve("run", t, y, block=block, start=max(float(warmup), 200.0), max_steps=200.0,
                        rel_width=0.05)
    name = {"pl": "power law", "pl_exp": "power law + exp", "pl_pl": "2 power laws"}[form]
    f, starts, (lo, hi) = _losspred_forms()(sched, float(y[0]), float(curve.z.min()))[name]
    best = None
    for x0 in starts:
        x0 = np.clip(x0, np.array(lo) + 1e-6, np.array(hi) - 1e-6)
        r = least_squares(lambda x: (f(x, curve.t) - curve.z) / np.sqrt(curve.var), x0, bounds=(lo, hi),
                          loss="soft_l1", f_scale=3, max_nfev=3000)
        if best is None or r.cost < best.cost:
            best = r
    return lambda tt: np.exp(f(best.x, np.asarray(tt, float)))


def trend(fit, warmup, total, t):
    sched = WSDScheduleConfig(total=int(total), warmup=int(warmup), decay=0)
    p = dict(fit.params)
    return np.exp(predict_log(p, sched, np.asarray(t, float), p["drop_ref"], fit.time))


def data_key(d, meta, text):
    """Runs with the same key see the same batches in the same order (Megatron's shuffle is
    seeded and sized by the number of samples): seed, gbs, total iterations."""
    import re

    m = re.search(r"^\S*\s*seed \.+ (\d+)", text, re.M)
    return (m.group(1) if m else "?", meta["gbs"], int(d["total"]))


def gap_points(d, window=250):
    """(iteration, train - validation) at each validation point after the warmup region: the
    training loss averaged over the ``window`` iterations before it."""
    t, y = d["it"], d["loss"]
    out = []
    for it, v in d["val"]:
        sel = (t > it - window) & (t <= it)
        if sel.sum() >= 5 and it >= 500:
            out.append((it, float(np.mean(y[sel])) - v))
    return out


def analyze(arm, log, det, a, delta=None):
    text = log.read_text(errors="replace")
    d = parse_log(log)
    if not d or len(d["it"]) < 50:
        return None
    sched, meta = schedule_from_log(text, int(d["total"]))
    t, y = d["it"].astype(float), d["loss"]
    ok = np.isfinite(y) & (y > 0)
    t, y = t[ok], y[ok]
    if delta is not None:  # the data-order correction: the shared train - validation gap
        y = y - delta(t)
    stable_end = sched.decay_start if sched.decay > 0 else sched.total
    st = t <= stable_end
    ts, ys = t[st], y[st]
    if len(ts) < 50:
        return None
    row = {"arm": arm, "iters": int(t[-1]), "total": sched.total, "warmup": sched.warmup,
           "gbs": meta["gbs"], "tokens_seen": int(t[-1]) * meta["gbs"] * meta["seq"],
           "stable_end_loss": float(np.mean(ys[-max(len(ys) // 50, 5):]))}
    # health over the whole run
    h = t >= 20
    sp, rg = det(t[h], y[h])
    ev = [e for e in sp.events if e.end < h.sum()]
    row.update(spikes=len(ev), max_spike=max((e.height for e in ev), default=0.0),
               instability=float(sum(e.height * max(e.end - e.start, 1) for e in ev)),
               growth=sum(e.kind == "growth" for e in rg.events))
    try:
        fit, _, rmse = fit_stable(ts, ys, sched.warmup, stable_end, a.block)
        p, se = fit.params, fit.stderr
        row.update(E=p["E"], E_se=se.get("E", np.nan), A=p["A"], alpha=p["alpha"],
                   alpha_se=se.get("alpha", np.nan), t0=p.get("t0", np.nan),
                   lr_power=p.get("lr_power", np.nan), fit_rmse=rmse,
                   noise=float(np.std(np.log(ys) - np.log(trend(fit, sched.warmup, stable_end, ts)))))
        for tok in a.predict_tokens:
            steps = int(tok) // (meta["gbs"] * meta["seq"])
            row[f"pred_{tok:.0e}"] = float(trend(fit, sched.warmup, max(steps, stable_end), [steps])[0])
    except Exception as e:  # noqa: BLE001
        row["fit_error"] = str(e)[:80]
    # backtest: fit the first part, predict the end of the stable phase
    cut = ts[0] + a.backtest * (ts[-1] - ts[0])
    early = ts <= cut
    if early.sum() >= 50 and cut > sched.warmup:
        try:
            if FORM == "pl":
                bfit, _, _ = fit_stable(ts[early], ys[early], sched.warmup, stable_end, a.block)
                pred = lambda tt: trend(bfit, sched.warmup, stable_end, tt)  # noqa: E731
            else:
                pred = fit_form(ts[early], ys[early], sched.warmup, stable_end, a.block, FORM)
            end = float(np.mean(ys[-max(len(ys) // 50, 5):]))
            row["backtest_err"] = float(pred([ts[-1]])[0] / end - 1)
            late = ts >= ts[0] + 0.75 * (ts[-1] - ts[0])
            row["flattening"] = float(np.mean(np.log(ys[late]) - np.log(pred(ts[late]))))
        except Exception:  # noqa: BLE001
            pass
    # training health (spikes above --max-spike, growth) and fit quality are separate flags: a
    # short run can be perfectly healthy and still too short to extrapolate
    row["healthy"] = bool(row["growth"] == 0 and row["max_spike"] <= a.max_spike)
    bt = row.get("backtest_err")
    row["extrapolable"] = bool(bt is not None and abs(bt) <= a.max_backtest)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("suites", nargs="+", type=Path)
    ap.add_argument("--predict-tokens", nargs="*", type=float, default=[6e9, 2e10])
    ap.add_argument("--backtest", type=float, default=0.5)
    ap.add_argument("--max-backtest", type=float, default=0.01)
    ap.add_argument("--max-spike", type=float, default=0.05, help="spike height (log-loss) that makes a run unhealthy")
    ap.add_argument("--block", type=int, default=20)
    ap.add_argument("--csv", type=Path)
    ap.add_argument("--data-correction", action="store_true",
                    help="fit train - delta(t), delta = the train - validation gap shared by all runs "
                         "with the same data order (seed, gbs, iterations): removes the data-order "
                         "drift that biases extrapolation")
    ap.add_argument("--form", choices=["pl", "pl_exp", "pl_pl"], default="pl",
                    help="backtest form: losspred's family power law, or the two-component forms of "
                         "losspred's compare_stable_forms.py (+ exp, 2 power laws)")
    ap.add_argument("--no-anchor", action="store_true",
                    help="fit t0 freely instead of anchoring the curve at the initial loss")
    a = ap.parse_args()
    global ANCHOR, FORM
    ANCHOR = not a.no_anchor
    FORM = a.form
    det = detector()
    logs = [(suite, arm, log) for suite in a.suites for arm, log in sorted(run_logs(suite).items())]
    deltas = {}
    if a.data_correction:
        # delta(t) per data order: the train - validation gap at the validation points, averaged
        # over every run with that order (the order, not the model, sets it: identical across
        # arms and sizes, e.g. +0.087 at 1500 and -0.016 at 4000 for attn / h_kda_xsa / s2),
        # linear in between, constant beyond the outermost points
        pts = {}
        for suite, arm, log in logs:
            d = parse_log(log)
            if not d or not d["val"]:
                continue
            text = log.read_text(errors="replace")
            _, meta = schedule_from_log(text, int(d["total"]))
            for it, g in gap_points(d):
                pts.setdefault(data_key(d, meta, text), {}).setdefault(it, []).append(g)
        for key, by_it in pts.items():
            its = np.array(sorted(by_it), float)
            gaps = np.array([np.mean(by_it[i]) for i in sorted(by_it)])
            deltas[key] = (lambda x, its=its, gaps=gaps: np.interp(x, its, gaps)) if len(its) >= 3 else None
    rows = []
    for suite, arm, log in logs:
        delta = None
        if a.data_correction:
            d = parse_log(log)
            if d:
                text = log.read_text(errors="replace")
                _, meta = schedule_from_log(text, int(d["total"]))
                delta = deltas.get(data_key(d, meta, text))
        r = analyze(arm, log, det, a, delta)
        if r:
            r["suite"] = suite.name
            r["data_corrected"] = delta is not None
            rows.append(r)
    cols = ["suite", "arm", "iters", "warmup", "stable_end_loss", "E", "alpha", "fit_rmse",
            "backtest_err", "flattening", "spikes", "max_spike", "growth", "healthy", "extrapolable"] + \
        [f"pred_{t:.0e}" for t in a.predict_tokens]
    fmt = {"E": ".3f", "alpha": ".3f", "fit_rmse": ".4f", "backtest_err": "+.4f", "flattening": "+.4f",
           "max_spike": ".3f", "stable_end_loss": ".4f"}
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                cells.append(format(v, fmt.get(c, ".4f")))
            else:
                cells.append(str(v))
        print("| " + " | ".join(cells) + " |")
    if a.csv:
        keys = sorted({k for r in rows for k in r})
        with a.csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
