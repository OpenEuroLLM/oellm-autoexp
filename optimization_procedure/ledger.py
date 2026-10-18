"""Knowledge store: the procedure learns from every experiment.

Append-only JSONL files in one directory:

  observations.jsonl  every measured loss (the canonical schema of data.py) + QA flags
  predictions.jsonl   predictions registered *before* the result exists
                      (obs key, mean, sd, model fingerprint, time)
  decisions.jsonl     launches / drops / promotions with the evidence at that time
  priors.json         posterior of base + tag effects of the last fit (warm start)

Two feedback loops close here:

  1. calibration: registered predictions are matched to later observations; the
     z-scores give an inflation factor for predictive sd (used by design and
     decision) and a coverage report -- the planner is graded on its own forecasts;
  2. prior transfer: ``save_priors`` / ``load_priors`` carry what was learned about
     shared (tag-level) effects into the next study, widened by an inflation factor.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


def _key(arch, study, N, D, lr, B) -> str:
    return f"{arch}|{study}|N={float(N):.4g}|D={float(D):.4g}|lr={float(lr):.4g}|B={float(B):.4g}"


class Ledger:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------- io helpers
    def _append(self, name: str, records: list[dict]):
        with open(self.root / name, "a") as f:
            for r in records:
                f.write(json.dumps(r, default=float) + "\n")

    def _read(self, name: str) -> pd.DataFrame:
        p = self.root / name
        if not p.exists():
            return pd.DataFrame()
        return pd.read_json(p, lines=True)

    # --------------------------------------------------------- observations
    def add_observations(self, df: pd.DataFrame) -> int:
        """Append new observations (dedup by obs_id). Returns number added."""
        old = self._read("observations.jsonl")
        seen = set(old["obs_id"]) if len(old) else set()
        new = df[~df["obs_id"].isin(seen)].copy()
        new["key"] = [_key(*r) for r in new[["arch", "study", "N", "D", "lr", "B"]].values]
        new["logged_at"] = time.time()
        self._append("observations.jsonl", new.to_dict("records"))
        return len(new)

    def observations(self) -> pd.DataFrame:
        return self._read("observations.jsonl")

    # ---------------------------------------------------------- predictions
    def register_predictions(self, cand: pd.DataFrame, mean, sd, fingerprint: str,
                             note: str = "") -> None:
        recs = []
        for (_, r), m, s in zip(cand.iterrows(), np.atleast_1d(mean), np.atleast_1d(sd)):
            recs.append({"key": _key(r.arch, r.study, r.N, r.D, r.lr, r.B), "arch": r.arch,
                         "mean": float(m), "sd": float(s), "model": fingerprint,
                         "registered_at": time.time(), "note": note})
        self._append("predictions.jsonl", recs)

    def calibration(self) -> dict:
        """Match predictions to observations; z = (y - mean) / sd."""
        pred, obs = self._read("predictions.jsonl"), self._read("observations.jsonl")
        if not len(pred) or not len(obs):
            return {"n": 0, "inflation": 1.0}
        m = pred.merge(obs[["key", "y", "weight"]], on="key")
        m = m[m.weight > 0]
        if not len(m):
            return {"n": 0, "inflation": 1.0}
        z = (m["y"] - m["mean"]) / m["sd"]
        # robust RMS of z -> sd inflation (never deflate below 1: stay conservative)
        infl = max(1.0, float(1.4826 * np.median(np.abs(z))))
        return {"n": int(len(z)), "inflation": infl,
                "coverage_90": float(np.mean(np.abs(z) < 1.645)),
                "mean_z": float(z.mean()), "table": m.assign(z=z)}

    # ------------------------------------------------------------ decisions
    def log_decision(self, kind: str, payload: dict):
        self._append("decisions.jsonl", [{"kind": kind, "at": time.time(), **payload}])

    # --------------------------------------------------------------- priors
    def save_priors(self, priors: dict[str, tuple[float, float]]):
        (self.root / "priors.json").write_text(json.dumps(priors, indent=1))

    def load_priors(self) -> dict[str, tuple[float, float]] | None:
        p = self.root / "priors.json"
        return {k: tuple(v) for k, v in json.loads(p.read_text()).items()} if p.exists() else None


def fingerprint(fit) -> str:
    h = hashlib.sha1(np.round(fit.theta, 6).tobytes()).hexdigest()[:10]
    return f"{fit.layout.form.name}:{h}"
