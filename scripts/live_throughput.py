#!/usr/bin/env python3
"""Live throughput vs. config estimate for the multilingual scaling sweep.

For every actively-running job it parses the Megatron log for steady-state
iteration time, derives measured Tok/s/GPU, compares it to the config `tok_s`
table (the halved-mbs smoke measurements that drive aux.true_time), and turns
the measured step time into a LIVE runtime estimate for the job (its full run
and remaining ETA) — held against the config's slurm reservation.

Run on JUPITER (needs the output dir + job.sbatch + slurm logs):
    uv run python scripts/live_throughput.py
"""
from __future__ import annotations

import glob
import os
import re
import time
from statistics import median

BASE = os.environ.get(
    "SWEEP_DIR",
    "/e/scratch/e-sta-openeurollm/poeppel1/output/"
    "architecture_scaling_variants_multilingual_main_7to1",
)
SEQ = 4096
GPUS_PER_NODE = 4
RECENT_LOG_S = 900  # a log touched within 15 min counts as "running"
STEADY_TAIL = 30    # median over the last N iteration samples (skip warm-up)

# Measured Tok/s/GPU per (variant, size) — mirrors aux.tok_s in main_scaling_all.
TOK_S = {
    "fullattn":      {"0.1B": 193000, "0.2B": 115000, "0.4B": 59000, "0.9B": 36000},
    "swa7_rope100k": {"0.1B": 206000, "0.2B": 124000, "0.4B": 67000, "0.9B": 38000},
    "gdn7_nope":     {"0.1B": 165000, "0.2B": 101000, "0.4B": 52000, "0.9B": 34000},
    "mlstm7_nope":   {"0.1B": 188000, "0.2B": 115000, "0.4B": 56000, "0.9B": 34000},
    "mamba2_nope":   {"0.1B": 172000, "0.2B": 94000,  "0.4B": 55000, "0.9B": 32000},
    # 3:1 arms (thr_hmbs indices 5/6, measured 2026-08-13). Without these the
    # live comparison silently falls back to "no config reference" (TOK_S.get)
    # for every gdn3 run in the sweep.
    "gdn3_nope":     {"0.1B": 169000, "0.2B": 100000, "0.4B": 53000, "0.9B": 33000},
    "gdn3_qwen35":   {"0.1B": 168000, "0.2B": 97000,  "0.4B": 50000, "0.9B": 33000},
}

NAME_RE = re.compile(
    r"qwen3_(?P<variant>.+?)_(?P<size>\d\.\dB)_ne_lr(?P<lr>[\d.]+)_gbsz(?P<gbs>\d+)_(?P<stage>.+)"
)
ITER_RE = re.compile(
    r"iteration\s+(\d+)/\s*(\d+).*?elapsed time per iteration \(ms\):\s*([\d.]+)"
)


def _parse_slurm_time(t: str) -> float:
    t = t.strip()
    days = 0
    if "-" in t:
        d, t = t.split("-", 1)
        days = int(d)
    p = [int(x) for x in t.split(":")]
    h, m, s = (p + [0, 0, 0])[:3] if len(p) >= 3 else ([0] + p + [0])[:3]
    return days * 24 + h + m / 60 + s / 3600


def _sbatch_field(sbatch_text: str, pat: str) -> str | None:
    m = re.search(pat, sbatch_text)
    return m.group(1) if m else None


def collect() -> list[dict]:
    rows = []
    now = time.time()
    for d in sorted(glob.glob(os.path.join(BASE, "qwen3_*"))):
        m = NAME_RE.match(os.path.basename(d))
        if not m:
            continue
        logs = glob.glob(os.path.join(d, "slurm-*.log"))
        if not logs:
            continue
        log = max(logs, key=os.path.getmtime)
        if now - os.path.getmtime(log) > RECENT_LOG_S:
            continue  # stale -> finished / not currently running
        try:
            txt = open(log, errors="ignore").read()
        except OSError:
            continue
        its = ITER_RE.findall(txt)
        if len(its) < 3:
            continue  # not enough steady-state samples yet
        cur, total = int(its[-1][0]), int(its[-1][1])
        first = int(its[0][0])
        ms = median(float(x[2]) for x in its[-STEADY_TAIL:])
        step_s = ms / 1000.0

        sb = ""
        sbf = os.path.join(d, "job.sbatch")
        if os.path.exists(sbf):
            sb = open(sbf, errors="ignore").read()
        nodes = int(_sbatch_field(sb, r"#SBATCH --nodes=(\d+)") or 1)
        res_t = _sbatch_field(sb, r"#SBATCH --time=([0-9:-]+)")
        gbs = int(m.group("gbs"))
        variant, size, stage = m.group("variant"), m.group("size"), m.group("stage")
        gpus = nodes * GPUS_PER_NODE

        meas = gbs * SEQ / (step_s * gpus)                 # measured Tok/s/GPU
        cfg = TOK_S.get(variant, {}).get(size)             # config Tok/s/GPU
        run_iters = total - first                          # iters THIS run performs
        proj_full_h = run_iters * step_s / 3600.0          # live full-run estimate
        eta_h = (total - cur) * step_s / 3600.0            # remaining to train_iters
        res_h = _parse_slurm_time(res_t) if res_t else float("nan")
        rows.append(dict(
            name=os.path.basename(d), variant=variant, size=size, stage=stage,
            nodes=nodes, gbs=gbs, cur=cur, total=total, ms=ms,
            meas=meas, cfg=cfg, proj_full_h=proj_full_h, eta_h=eta_h, res_h=res_h,
        ))
    return rows


def main() -> None:
    rows = collect()
    if not rows:
        print("No actively-running jobs with steady-state logs found under\n  " + BASE)
        return
    rows.sort(key=lambda r: (r["size"], r["variant"], r["stage"]))
    print(f"Live throughput vs config estimate — {len(rows)} running job(s)\n")
    hdr = (f"{'size':4} {'variant':13} {'stage':10} {'nod':>3} {'iter':>13} "
           f"{'ms/it':>7} {'meas kT/s':>9} {'cfg':>5} {'Δ%':>5} "
           f"{'liveFull_h':>10} {'resv_h':>7} {'ETA_h':>7}")
    print(hdr)
    print("-" * len(hdr))
    ratios = []
    for r in rows:
        d = (r["meas"] / r["cfg"] - 1) * 100 if r["cfg"] else float("nan")
        if r["cfg"]:
            ratios.append(r["meas"] / r["cfg"])
        stage = r["stage"][:10]
        fits = "" if r["proj_full_h"] != r["proj_full_h"] else (
            "!" if (r["res_h"] == r["res_h"] and r["proj_full_h"] > r["res_h"] + 0.05) else "")
        print(f"{r['size']:4} {r['variant']:13} {stage:10} {r['nodes']:>3} "
              f"{r['cur']:>6}/{r['total']:<6} {r['ms']:>7.0f} {r['meas']/1000:>9.1f} "
              f"{(r['cfg'] or 0)/1000:>5.0f} {d:>+5.0f} "
              f"{r['proj_full_h']:>10.1f}{fits:1} {r['res_h']:>6.1f} {r['eta_h']:>7.1f}")
    if ratios:
        mr = median(ratios)
        print("\nSummary:")
        print(f"  median measured/config throughput ratio: {mr:.3f} "
              f"({'+' if mr>=1 else ''}{(mr-1)*100:.1f}% vs estimate)")
        print(f"  => live sweep GPU-h projection ~ 29,824 / {mr:.3f} = "
              f"{29824/mr:,.0f} GPU-h (config tok_s implied 29,824)")
        print("  '!' after liveFull_h = projected run exceeds its slurm reservation "
              "(will TIMEOUT and restart-chain).")


if __name__ == "__main__":
    main()
