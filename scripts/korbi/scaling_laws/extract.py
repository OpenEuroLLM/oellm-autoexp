"""Extract eval val losses into a tidy, cost-annotated table.

Two input sources:
  - a metrics CSV (columns variant,size,gbs,lr,type,budgetBT,val_loss,val_ppl,iter),
    e.g. dump/eval_val_metrics.csv produced from the eval logs; or
  - a directory of eval job dirs (…_eval/qwen3_*/slurm-*.log) parsed directly.

Output columns: arch, size, gbs, lr, type, budget_bt, iter, D (tokens),
N (non-emb params; measured per arch unless n_mode="nominal"), d_over_n, tier, flops, gpu_hours, val_loss, val_ppl.

Rows whose checkpoint iteration does not match the budget are DROPPED: an eval
that picked up the `firstcd` seed checkpoint (start_iter+1, i.e. one step into
the decay) instead of the completed `contcd` reports a loss that is not the
cooldown endpoint at all. `--keep-incomplete` on run.py disables the filter.
"""

from __future__ import annotations

import glob
import math
import os
import re

import numpy as np
import pandas as pd

from costs import DECAY_FRACTION, SEQ_LEN, gpu_hours, n_ne, tokens_from_iter, total_flops

DIR_RE = re.compile(
    r"qwen3_(?P<arch>[a-z0-9_]+?)_(?P<size>0\.\d+B)_ne_lr(?P<lr>[0-9.]+)_gbsz(?P<gbs>\d+)_"
    r"(?P<type>evalcd|evalstable)_(?:decay|branch)(?P<budget>\d+)BT"
)
VAL_RE = re.compile(
    r"validation loss at iteration (?P<iter>\d+) on validation set \| "
    r"lm loss value: (?P<loss>[0-9.E+]+) \| lm loss PPL: (?P<ppl>[0-9.E+]+)"
)


def _row_costs(
    arch: str, size: str, gbs: int, iteration: int, flops_mode: str, n_mode: str = "measured"
) -> dict:
    d = tokens_from_iter(iteration, gbs)
    n = n_ne(size, arch, n_mode)
    return {
        "D": d,
        "N": n,
        "d_over_n": d / n,
        "flops": total_flops(size, arch, d, mode=flops_mode, n_mode=n_mode),
        "gpu_hours": gpu_hours(d, arch, size),
    }


def expected_iter(budget_bt: float, gbs: int, kind: str) -> int:
    """Checkpoint iteration a completed eval of this (budget, gbs, kind) must
    sit at.

    Cooldown finals end at the full budget; `stable`/branch checkpoints are taken
    at (1 - decay_fraction) of it — the point the WSD cooldowns branch from.
    """
    end = math.ceil(budget_bt * 1e9 / (SEQ_LEN * gbs))
    return end if kind in ("evalcd", "cooldown") else int(end * (1 - DECAY_FRACTION))


def drop_incomplete(df: pd.DataFrame, tol: float = 0.01, verbose: bool = True) -> pd.DataFrame:
    """Drop evals whose checkpoint iteration disagrees with their nominal
    budget."""
    want = [expected_iter(b, g, t) for b, g, t in zip(df["budget_bt"], df["gbs"], df["type"])]
    ok = (df["iter"] / pd.Series(want, index=df.index) - 1.0).abs() <= tol
    if verbose and (~ok).any():
        bad = df[~ok].assign(iter_expected=pd.Series(want, index=df.index)[~ok])
        print(f"[extract] dropping {len(bad)} eval(s) not at the budget checkpoint:")
        print(
            bad[
                [
                    "arch",
                    "size",
                    "gbs",
                    "lr",
                    "type",
                    "budget_bt",
                    "iter",
                    "iter_expected",
                    "val_loss",
                ]
            ].to_string(index=False)
        )
    return df[ok].reset_index(drop=True)


def from_csv(path: str, flops_mode: str = "6nd", n_mode: str = "measured") -> pd.DataFrame:
    """Load the pre-extracted metrics CSV and annotate with costs."""
    df = pd.read_csv(path)
    df = df.rename(columns={"variant": "arch", "budgetBT": "budget_bt", "val_ppl": "val_ppl"})
    rows = []
    for _, r in df.iterrows():
        arch, size, gbs, it = r["arch"], r["size"], int(r["gbs"]), int(r["iter"])
        rows.append({**r.to_dict(), **_row_costs(arch, size, gbs, it, flops_mode, n_mode)})
    return pd.DataFrame(rows)


def from_logs(
    eval_output_dir: str, flops_mode: str = "6nd", n_mode: str = "measured"
) -> pd.DataFrame:
    """Parse eval job dirs directly
    (…_main_7to1_eval/qwen3_*_eval*/slurm-*.log)."""
    rows = []
    for d in sorted(glob.glob(os.path.join(eval_output_dir, "qwen3_*eval*"))):
        m = DIR_RE.search(os.path.basename(d))
        if not m:
            continue
        logs = sorted(glob.glob(os.path.join(d, "slurm-*.log")), key=os.path.getmtime)
        if not logs:
            continue
        txt = open(logs[-1], errors="ignore").read()
        vm = None
        for vm in VAL_RE.finditer(txt):
            pass  # keep the last validation line
        if vm is None:
            continue
        arch, size = m["arch"], m["size"]
        gbs, it = int(m["gbs"]), int(vm["iter"])
        rows.append(
            {
                "arch": arch,
                "size": size,
                "gbs": gbs,
                "lr": float(m["lr"]),
                "type": m["type"],
                "budget_bt": int(m["budget"]),
                "iter": it,
                "val_loss": float(vm["loss"]),
                "val_ppl": float(vm["ppl"]),
                **_row_costs(arch, size, gbs, it, flops_mode, n_mode),
            }
        )
    return pd.DataFrame(rows)


def load(
    source: str, flops_mode: str = "6nd", keep_incomplete: bool = False, n_mode: str = "measured"
) -> pd.DataFrame:
    """Dispatch on whether `source` is a CSV file or an eval-output
    directory."""
    df = (
        from_logs(source, flops_mode, n_mode)
        if os.path.isdir(source)
        else from_csv(source, flops_mode, n_mode)
    )
    if df.empty:
        return df
    # Drop diverged/broken evals. Megatron writes "lm loss value: NAN" when the
    # loaded model produces a non-finite loss (the companion PPL is the overflow
    # value e^20 ~ 4.85e8, which looks like a number and would otherwise poison
    # every mean/argmin downstream). These are NOT missing data points to be
    # imputed -- they are evidence that a specific checkpoint is bad, so name
    # them loudly rather than filtering silently.
    bad = df[~np.isfinite(df["val_loss"])]
    if len(bad):
        print(
            f"[extract] WARNING: dropping {len(bad)} eval(s) with non-finite "
            f"val_loss (diverged or corrupt checkpoint) -- investigate these:"
        )
        cols = [
            c
            for c in ("arch", "size", "gbs", "lr", "type", "budget_bt", "iter", "val_ppl")
            if c in bad
        ]
        print(bad[cols].to_string(index=False))
        df = df[np.isfinite(df["val_loss"])].reset_index(drop=True)
    if df.empty:
        return df
    # The (gbs, lr) "tier": within a size the sweep switches tier as the budget
    # grows, so it is the hyperparameter cell any fit must not silently mix.
    df["tier"] = (
        df["size"].astype(str) + " gbs" + df["gbs"].astype(str) + " lr" + df["lr"].astype(str)
    )
    if not keep_incomplete:
        df = drop_incomplete(df)
    return df
