"""Health and results of the 3B campaign's small-scale runs (docs/hybrid_moe_3b_campaign.md 2.3).

    uv run --project ../loss_prediction python scripts/korbi/campaign/analyze.py \
        <suite dir> [<suite dir> ...] [--tail 0.1] [--csv out.csv]

A suite dir holds one directory per arm (``<suite>/<arm>/slurm-*.log``, the newest log is used),
as autoexp writes them (rsync them from MN5 first). Per run: the training curve (iteration, lm
loss, lr, grad norm, tok/s/GPU) and the validation losses are parsed from the Megatron log; the
health metrics come from losspred (HysteresisSpikeDetector on the training loss, the
MonotonicityRegimeDetector on the despiked curve): spike count, the largest spike, the summed
excess loss of the spikes (the "instability penalty"), growth / drop regimes after warmup.
Reported loss: the last validation loss and the mean training loss over the last ``--tail``
fraction of the run (despiked).
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

LP = Path(__file__).resolve().parents[4] / "loss_prediction"
sys.path.insert(0, str(LP / "src"))

from compoconf import parse_config  # noqa: E402

from losspred.config import compose_dict  # noqa: E402
from losspred.detection.evaluate import DetectionConfig, DetectionPipeline  # noqa: E402

IT = re.compile(r"iteration\s+(\d+)/\s*(\d+) \|(.*)")
FIELD = {
    "loss": re.compile(r"lm loss: ([0-9.Ee+-]+)"),
    "lr": re.compile(r"learning rate: ([0-9.Ee+-]+)"),
    "gnorm": re.compile(r"grad norm: ([0-9.Ee+-]+)"),
    "tok_s": re.compile(r"Tok/s/GPU\): ([0-9.]+)"),
    "gbs": re.compile(r"global batch size:\s+(\d+)"),
}
VAL = re.compile(r"validation loss at iteration\s+(\d+).*?lm loss value: ([0-9.Ee+-]+)")


def parse_log(path: Path) -> dict:
    rows, val, total = [], [], None
    for line in path.read_text(errors="replace").splitlines():
        m = IT.search(line)
        if m and "lm loss:" in m.group(3):
            it, total = int(m.group(1)), int(m.group(2))
            rec = {"it": it}
            for k, rx in FIELD.items():
                f = rx.search(m.group(3))
                rec[k] = float(f.group(1)) if f else np.nan
            rows.append(rec)
            continue
        v = VAL.search(line)
        if v:
            val.append((int(v.group(1)), float(v.group(2))))
    if not rows:
        return {}
    # restarts log iterations again: keep the last record per iteration
    by_it = {r["it"]: r for r in rows}
    its = np.array(sorted(by_it))
    get = lambda k: np.array([by_it[i][k] for i in its])  # noqa: E731
    return dict(it=its, total=total, loss=get("loss"), lr=get("lr"), gnorm=get("gnorm"),
                tok_s=get("tok_s"), gbs=get("gbs"), val=sorted(dict(val).items()))


def run_logs(suite: Path) -> dict[str, Path]:
    """{arm: newest log}: per-arm dirs (single jobs) or the array dir <suite>/<suite>
    (slurm-<job>_<task>.log, the arm named by task_<task>.sh's output path)."""
    out = {}
    arr = suite / suite.name
    if arr.is_dir():
        for task in arr.glob("task_*.sh"):
            i = task.stem.split("_")[1]
            m = re.search(rf"{re.escape(suite.name)}/([A-Za-z0-9_.]+)/", task.read_text())
            logs = sorted(arr.glob(f"slurm-*_{i}.log"), key=lambda p: p.stat().st_mtime)
            if m and logs:
                out[m.group(1)] = logs[-1]
    for arm_dir in (p for p in suite.iterdir() if p.is_dir() and p.name != suite.name):
        logs = sorted(arm_dir.glob("slurm-*.log"), key=lambda p: p.stat().st_mtime)
        if logs and arm_dir.name not in out:
            out[arm_dir.name] = logs[-1]
    return out


def detector() -> DetectionPipeline:
    raw = compose_dict("detection", [], LP / "config")
    return DetectionPipeline(parse_config(DetectionConfig, raw["detection"]))


def analyze_run(arm: str, log: Path, det: DetectionPipeline, tail: float) -> dict | None:
    d = parse_log(log)
    if not d or len(d["it"]) < 20:
        return None
    t, y = d["it"].astype(float), d["loss"]
    ok = np.isfinite(y)
    t, y = t[ok], y[ok]
    lr = d["lr"][ok]
    peak = np.nanmax(lr) if np.isfinite(lr).any() else np.nan
    warm_end = t[np.argmax(lr >= 0.999 * peak)] if np.isfinite(peak) else 0.0
    n_tail = max(int(tail * len(t)), 1)
    # detection after the warmup (its fast initial drop is not a regime the detectors model) and
    # only on curves long enough for the rolling baselines (edge effects otherwise)
    post = t > max(2 * warm_end, 100.0)
    spikes, max_spike, excess, growth, drops, sigma = 0, 0.0, 0.0, 0, 0, np.nan
    tail_loss = float(np.mean(y[-n_tail:]))
    if post.sum() >= 200:
        tp, yp = t[post], y[post]
        sp, rg = det(tp, yp)
        # spikes that run into the end of the curve are unresolved: no despiking there
        events = [e for e in sp.events if e.end < len(tp)]
        spikes = len(events)
        max_spike = max((e.height for e in events), default=0.0)
        m = np.zeros(len(tp), bool)
        for e in events:
            m[e.start:e.end] = True
        excess = float(np.sum((yp[m] - sp.baseline[m]) / sp.baseline[m])) if m.any() else 0.0
        clean = np.where(m, sp.baseline, yp)
        tail_loss = float(np.mean(clean[-n_tail:]))
        growth = sum(e.kind == "growth" for e in rg.events)
        drops = sum(e.kind == "drop" for e in rg.events)
        sigma = float(sp.sigma)
    return {
        "arm": arm,
        "iters": int(t[-1]),
        "total": d["total"],
        "done": bool(d["total"] and t[-1] >= 0.99 * d["total"]),
        "train_tail": tail_loss,
        "val_last": d["val"][-1][1] if d["val"] else np.nan,
        "val_it": d["val"][-1][0] if d["val"] else 0,
        "tok_s_gpu": float(np.nanmedian(d["tok_s"][len(d["tok_s"]) // 3:])),
        "gnorm_tail": float(np.nanmedian(d["gnorm"][-n_tail:])),
        "spikes": spikes,
        "max_spike": max_spike,
        "instability": excess,
        "growth_after_warmup": growth,
        "drops_after_warmup": drops,
        "noise_sigma": sigma,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("suites", nargs="+", type=Path)
    ap.add_argument("--tail", type=float, default=0.1)
    ap.add_argument("--csv", type=Path)
    a = ap.parse_args()
    det = detector()
    rows = []
    for suite in a.suites:
        for arm, log in run_logs(suite).items():
            r = analyze_run(arm, log, det, a.tail)
            if r:
                r["suite"] = suite.name
                rows.append(r)
    if not rows:
        sys.exit("no runs with training logs")
    cols = ["suite", "arm", "iters", "done", "val_last", "train_tail", "tok_s_gpu", "gnorm_tail",
            "spikes", "max_spike", "instability", "growth_after_warmup"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in sorted(rows, key=lambda r: (r["suite"], r["val_last"] if np.isfinite(r["val_last"]) else 9)):
        print("| " + " | ".join(
            f"{r[c]:.4f}" if isinstance(r[c], float) and c in ("val_last", "train_tail", "max_spike",
                                                              "instability", "gnorm_tail")
            else f"{r[c]:,.0f}" if c == "tok_s_gpu" else str(r[c]) for c in cols) + " |")
    if a.csv:
        with a.csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
