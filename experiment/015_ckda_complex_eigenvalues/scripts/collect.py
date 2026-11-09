"""Collect the ckda_spectra outputs of several checkpoints into one JSON for the report.

    python experiment/015_ckda_complex_eigenvalues/scripts/collect.py <res root> <out.json> [--main ambitious_v4_2ph_hd256_fp8_21000]

<res root>/<export>/ holds each export's ckda_spectra output with summary.json (summarize.py already run).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def step_of(name: str) -> int:
    return int(name.rsplit("_", 1)[-1])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--main", default="ambitious_v4_2ph_hd256_fp8_21000")
    p.add_argument("--plane-layers", default="0,7,20,38")
    a = p.parse_args()
    runs = sorted((d for d in a.root.iterdir() if (d / "summary.json").exists()), key=lambda d: step_of(d.name))
    traj = []
    for d in runs:
        S = json.loads((d / "summary.json").read_text())
        ab = S.get("ablations", {})
        traj.append({
            "step": step_of(d.name), "export": d.name, "overall": S["overall"], "by_set": S["overall_by_set"],
            "base_nll": S.get("base_nll"), "ablations": ab,
            "ablations_mean": {k: float(np.mean(list(v.values()))) for k, v in ab.items()},
            "layer_cplx": S["layer"]["cplx"], "layer_beta_gt1": S["layer"]["beta_gt1"],
            "verify": S.get("verify"),
        })
    m = a.root / a.main
    S = json.loads((m / "summary.json").read_text())
    meta = json.loads((m / "meta.json").read_text())
    sets = list(meta["sets"])
    st = {s: dict(np.load(m / f"stats_{s}.npz")) for s in sets}
    hist = sum(st[s]["hist"] for s in sets)
    planes = {}
    for layer in map(int, a.plane_layers.split(",")):
        h = hist[layer]
        planes[layer] = {"density": np.round(np.log1p(h) / max(np.log1p(h).max(), 1e-9), 3).T.tolist(),
                         "cplx": S["layer"]["cplx"][layer]}
    pb = np.array(S["posbin_cplx"], dtype=float)
    groups = [(0, 14), (14, 28), (28, 42)]
    pos = {"bins": S["posbin_count"], "groups": {f"{g0}-{g1 - 1}": np.nanmean(pb[g0:g1], 0).tolist() for g0, g1 in groups}}
    out = {
        "main": a.main, "main_step": step_of(a.main), "sets": sets,
        "docs": {s: len(v) for s, v in meta["sets"].items()},
        "tokens": {s: int(sum(x["len"] for x in v)) for s, v in meta["sets"].items()},
        "ckda_block": S["meta"]["ckda_block"],
        "traj": traj,
        "layer": S["layer"], "layer_head_cplx": S["layer_head_cplx"], "layer_head_beta_gt1": S["layer_head_beta_gt1"],
        "by_set": S["overall_by_set"], "overall": S["overall"],
        "pos": pos, "planes": planes,
        "tokens_top": S["tokens_top"][:20], "tokens_bottom": S["tokens_bottom"][:20],
        "tokens_frequent": S["tokens_frequent"][:20],
        "ablations": S.get("ablations"), "base_nll": S.get("base_nll"), "verify": S.get("verify"),
        "self_check": json.loads((m / "self_check.json").read_text()),
    }

    def clean(x):
        if isinstance(x, float):
            if not np.isfinite(x):
                return None
            return round(x, 5) if abs(x) >= 1e-3 or x == 0 else float(f"{x:.3g}")
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, list):
            return [clean(v) for v in x]
        return x

    a.out.write_text(json.dumps(clean(out)))
    print(a.out, a.out.stat().st_size, "bytes;", [t["step"] for t in traj])


if __name__ == "__main__":
    main()
