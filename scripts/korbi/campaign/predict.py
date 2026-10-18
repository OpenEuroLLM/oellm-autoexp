"""Loss prediction for the 3B campaign (docs/hybrid_moe_3b_campaign.md 2.3): fit a run's training
curve with losspred's WSD family model and predict the cooled loss of a longer WSD run.

    uv run --project ../loss_prediction python scripts/korbi/campaign/predict.py \
        <suite dir> [<suite dir> ...] --target-tokens 6e9 [--check <suite dir> ...]

Per run (the newest log of ``<suite>/<arm>``, as analyze.py finds it): the schedule is read from
the log (warmup / decay iterations, total, min-lr ratio) and the curve (lm loss per logged
iteration) is fitted with ``fit_family`` (time = LR area, anchored at the initial loss). A
single run has one decay start, so the drop's dependence on the decay start
(``drop_exponent``) is not identifiable and is held at ``--drop-exponent`` (losspred findings:
-0.02 to 0.25 across families). The prediction is the trend's training loss at the end of the
same schedule stretched to ``--target-tokens`` (the same warmup, decay fraction, min-lr ratio).
``--check`` suites hold runs at the target length with the same arm names; their measured
end-of-run loss (mean of the last ``--tail`` fraction of the cooldown, despiked by nothing)
is printed next to the prediction.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import LP, parse_log, run_logs  # noqa: E402

sys.path.insert(0, str(LP / "src"))
from losspred.fitting.blocks import block_curve  # noqa: E402
from losspred.fitting.family import FamilyFitConfig, fit_family, predict_log  # noqa: E402
from losspred.schedule import WSDScheduleConfig  # noqa: E402

ARG = {
    "warmup": re.compile(r"lr_warmup_iters \.+ (\d+)"),
    "decay": re.compile(r"lr_wsd_decay_iters \.+ (\d+|None)"),
    "lr": re.compile(r"^\s*lr \.+ ([0-9.eE+-]+)", re.M),
    "min_lr": re.compile(r"min_lr \.+ ([0-9.eE+-]+)"),
    "gbs": re.compile(r"global_batch_size \.+ (\d+)"),
    "seq": re.compile(r"seq_length \.+ (\d+)"),
    "style": re.compile(r"lr_decay_style \.+ (\S+)"),
}


def schedule_from_log(text: str, total: int) -> tuple[WSDScheduleConfig, dict]:
    """Megatron's argument dump -> the run's WSD schedule (decay 0 for a constant LR)."""
    got = {}
    for k, rx in ARG.items():
        m = rx.search(text)
        got[k] = m.group(1) if m else None
    warmup = int(got["warmup"] or 0)
    decay = int(got["decay"]) if got["style"] == "WSD" and got["decay"] not in (None, "None") else 0
    lr, min_lr = float(got["lr"] or 1.0), float(got["min_lr"] or 0.0)
    sched = WSDScheduleConfig(total=total, warmup=warmup, decay=decay,
                              min_lr_ratio=min_lr / lr if lr else 0.0)
    return sched, {"gbs": int(got["gbs"] or 0), "seq": int(got["seq"] or 4096)}


def fit_run(log: Path, drop_exponent: float, block: int):
    text = log.read_text(errors="replace")
    d = parse_log(log)
    t, y = d["it"].astype(float), d["loss"]
    sched, meta = schedule_from_log(text, int(d["total"]))
    curve = block_curve(log.parent.name, t, y, block=block, max_steps=200.0, rel_width=0.05)
    cfg = FamilyFitConfig(time="lr_area", fixed={"drop_exponent": drop_exponent})
    fit = fit_family({"run": curve}, {"run": sched}, cfg, init_loss=float(y[np.isfinite(y)][0]))
    return fit, sched, meta, d


def stretched(sched: WSDScheduleConfig, total: int) -> WSDScheduleConfig:
    """The same schedule shape at another length: warmup kept, decay fraction kept."""
    frac = sched.decay / sched.total if sched.total else 0.0
    return WSDScheduleConfig(total=total, warmup=sched.warmup, decay=int(round(frac * total)),
                             min_lr_ratio=sched.min_lr_ratio)


def end_loss(d: dict, tail: float) -> float:
    y = d["loss"][np.isfinite(d["loss"])]
    return float(np.mean(y[-max(int(tail * len(y)), 1):]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("suites", nargs="+", type=Path)
    ap.add_argument("--target-tokens", type=float, required=True)
    ap.add_argument("--check", nargs="*", type=Path, default=[])
    ap.add_argument("--drop-exponent", type=float, default=0.1)
    ap.add_argument("--tail", type=float, default=0.01)
    ap.add_argument("--block", type=int, default=20)
    a = ap.parse_args()
    checks = {}
    for suite in a.check:
        for arm, log in run_logs(suite).items():
            d = parse_log(log)
            if d and d["total"] and d["it"][-1] >= 0.99 * d["total"]:
                checks[arm] = end_loss(d, a.tail)
    print("| suite | arm | fit end (measured) | fit end (trend) | predicted @ target | measured @ target | error |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for suite in a.suites:
        for arm, log in sorted(run_logs(suite).items()):
            try:
                fit, sched, meta, d = fit_run(log, a.drop_exponent, a.block)
            except Exception as e:  # noqa: BLE001 -- report and go on
                print(f"| {suite.name} | {arm} | fit failed: {e} | | | | |")
                continue
            p = dict(fit.params)
            p["t0"] = p.get("t0", 0.0)
            own = float(np.exp(predict_log(p, sched, np.array([sched.total]), p["drop_ref"], fit.time))[0])
            target_total = int(a.target_tokens) // (meta["gbs"] * meta["seq"])
            tgt = stretched(sched, target_total)
            pred = float(np.exp(predict_log(p, tgt, np.array([target_total]), p["drop_ref"], fit.time))[0])
            meas = checks.get(arm)
            err = f"{100 * (pred / meas - 1):+.2f} %" if meas else ""
            print(f"| {suite.name} | {arm} | {end_loss(d, a.tail):.4f} | {own:.4f} | {pred:.4f} | "
                  f"{meas:.4f} | {err} |" if meas else
                  f"| {suite.name} | {arm} | {end_loss(d, a.tail):.4f} | {own:.4f} | {pred:.4f} | | |")


if __name__ == "__main__":
    main()
