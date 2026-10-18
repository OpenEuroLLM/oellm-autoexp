"""Observation table + loaders for the data this repo already has.

The canonical observation schema (one row = one measured loss):

  obs_id  unique, stable id (used by the ledger and the hold-out record)
  arch    architecture key (must be in ARCH_TAGS)
  study   which validation set / campaign produced the number
  N       non-embedding parameters     D   tokens     lr   peak lr     B   batch (tokens)
  y       log of the (cooled) validation loss
  weight  run-quality weight in [0, 1] (from curves.assess_run), default 1
  cost    GPU-hours of the run (optional)

Architectures are described by *tags*. Tags are the unit of transfer: whatever the
model learns about "gdn" is shared by every architecture tagged "gdn". Add new
architectures here (or pass your own dict) -- a new arm with no data at all still
gets a sensible prior prediction from its tags.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SEQ_LEN = 4096

ARCH_TAGS: dict[str, tuple[str, ...]] = {
    "fullattn": (),
    "swa7_rope100k": ("local", "ratio7"),
    "gdn7_nope": ("linear", "gdn", "nope", "ratio7"),
    "gdn3_nope": ("linear", "gdn", "nope", "ratio3"),
    "gdn3_qwen35": ("linear", "gdn", "ratio3", "outgate"),
    "mlstm7_nope": ("linear", "mlstm", "nope", "ratio7"),
    "mamba2_nope": ("linear", "mamba", "nope", "ratio7"),
    # arms without (local) results yet: predicted purely from their tags
    "pda7_nope": ("linear", "pda", "nope", "ratio7"),
    "skda7_nope": ("linear", "kda", "nope", "ratio7"),
}


def _costs_module():
    """scripts/scaling_laws/costs.py (not a package import: it uses flat imports)."""
    path = REPO / "scripts" / "scaling_laws" / "costs.py"
    spec = importlib.util.spec_from_file_location("_scaling_costs", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


costs = _costs_module()


def n_ne(size: str) -> float:
    return float(costs.n_ne(size))


def throughput(arch: str, fallback: str = "fullattn") -> tuple[float, float]:
    """(s0, sigma) of tok/s/GPU = s0 * N^-sigma (measured; fallback for unmeasured arms)."""
    return costs.throughput_powerlaw(arch if arch in costs.TOK_S else fallback)


def gpu_hours(arch: str, N, D) -> np.ndarray:
    s0, sig = throughput(arch)
    return np.asarray(D, float) / (s0 * np.asarray(N, float) ** (-sig) * 3600.0)


def load_arch_sweep(path: str | Path = REPO / "dump" / "eval_val_metrics.csv",
                    kind: str = "evalcd") -> pd.DataFrame:
    """Held-out val losses of the multilingual architecture sweep (cooled endpoints).

    ``kind="evalstable"`` gives the pre-decay branch losses instead (a different
    quantity: do not mix the two in one fit without a cooldown-gap term)."""
    d = pd.read_csv(path)
    d = d[d["type"] == kind].copy()
    out = pd.DataFrame({
        "arch": d["variant"],
        "study": "arch_sweep",
        "size": d["size"],
        "N": d["size"].map(n_ne),
        "D": d["budgetBT"].astype(float) * 1e9,
        "lr": d["lr"].astype(float),
        "B": d["gbs"].astype(float) * SEQ_LEN,
        "y": np.log(d["val_loss"].astype(float)),
    })
    # A missing loss with a huge perplexity is a diverged run (e.g. mamba2_nope 0.9B
    # 300BT: ppl 4.9e8). Keep it as a flagged observation with weight 0 -- it is
    # stability evidence for the arm, not a data point for the loss law.
    diverged = ~np.isfinite(out["y"]) & np.isfinite(d["val_ppl"].astype(float))
    out["y"] = out["y"].where(~diverged, np.log(np.log(d["val_ppl"].astype(float))))
    out["weight"] = np.where(diverged, 0.0, 1.0)
    out["flag"] = np.where(diverged, "diverged", "")
    out["obs_id"] = ("arch_sweep/" + out.arch + "/" + out["size"] + "/gbs" + d["gbs"].astype(str)
                     + "/lr" + d["lr"].astype(str) + "/" + d["budgetBT"].astype(str) + "BT")
    out["cost"] = gpu_hours_vec(out)
    return out.reset_index(drop=True)


def load_dense_baseline(
    path: str | Path = REPO / "dump" / "scaling_experiments_multilingual" / "all_losses_multilingual.csv",
) -> pd.DataFrame:
    """Dense (fullattn) baseline (lr x batch) grid: the only data with real HP spread.

    Different validation split (1% train-mix slice, other clusters), hence its own
    ``study`` -- the model gives it a learned loss offset and shares everything else.
    """
    d = pd.read_csv(path)
    out = pd.DataFrame({
        "arch": "fullattn",
        "study": "dense_baseline",
        "size": d["model_size"],
        "N": d["model_size"].map(n_ne),
        "D": d["budget_bt"].astype(float) * 1e9,
        "lr": d["lr"].astype(float),
        "B": d["global_batch_size"].astype(float) * SEQ_LEN,
        "y": np.log(d["lm_loss_validation"].astype(float)),
    })
    out["obs_id"] = ("dense_baseline/" + out["size"] + "/gbs" + d["global_batch_size"].astype(str)
                     + "/lr" + d["lr"].astype(str) + "/" + d["budget_bt"].astype(str) + "BT")
    out["cost"] = gpu_hours_vec(out)
    out["weight"] = 1.0
    out["flag"] = ""
    return out.dropna(subset=["y"]).reset_index(drop=True)


def gpu_hours_vec(df: pd.DataFrame) -> np.ndarray:
    return np.array([float(gpu_hours(a, n, d)) for a, n, d in zip(df.arch, df.N, df.D)])


def load_all() -> pd.DataFrame:
    return pd.concat([load_arch_sweep(), load_dense_baseline()], ignore_index=True)
