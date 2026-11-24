#!/usr/bin/env python3
"""Offline fixed-query rescore of the Engram gate diagnosis arrays (no model forward).

Reads the per-token ``.npz`` files of ``engram_gate_diag.py`` and recomputes the gate under the proposed
scale-invariant query for a sweep of ``logit_scale`` values::

    x_fixed = x * logit_scale / |h~|,   gate = sigmoid(signed_sqrt(x_fixed))

The current gate is the same expression with ``logit_scale = |h~|`` per token, so the sweep shows how far
the gate opens while the rank correlation with the layer's per-token utility ``dNLL = zero - base`` says
whether the trained keys carry selection information. The other Engram layers keep
``logit_scale = mean(|h~|)``, their function-preserving init.

    python scripts/rluukkon/engram_gate_rescore.py \
        --diag-dir /gpfs/scratch/ehpc533/users/rluukkon/engram_gate_diag/ambitious_v4b_learned_fp8_53000 \
        --logit-scales 0.025,0.1,0.3,1 --out report_fixed_query.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from oellm_autoexp.hf_export.engram_gate import (  # noqa: E402
    engram_layers,
    rescore,
    rescore_report,
)


def _scales(text: str) -> list[float]:
    return [float(part) for part in text.replace(" ", "").split(",") if part]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diag-dir", type=Path, help="directory with the diagnosis .npz files")
    ap.add_argument("--npz", type=Path, nargs="+", help="explicit .npz files (instead of --diag-dir)")
    ap.add_argument("--logit-scales", default="0.025,0.1,0.3,1.0", help="comma-separated absolute logit_scale values")
    ap.add_argument("--sweep-layer", type=int, default=None, help="layer whose logit_scale is swept (default: the first)")
    ap.add_argument("--out", type=Path, default=None, help="write the markdown here (default: stdout)")
    a = ap.parse_args()

    if not a.npz and not a.diag_dir:
        ap.error("give --diag-dir or --npz")
    files = list(a.npz) if a.npz else sorted(a.diag_dir.glob("*.npz"))
    if not files:
        print(f"no .npz under {a.diag_dir}", file=sys.stderr)
        return 1
    scales = _scales(a.logit_scales)

    blocks = []
    summary: dict[int, dict[str, list[float]]] = {}
    for path in files:
        with np.load(path) as z:
            layers = engram_layers(z)
            sweep = a.sweep_layer if a.sweep_layer is not None else (layers[0] if layers else None)
            result = rescore(z, scales, sweep_layer=sweep)
        blocks.append(rescore_report(result, source=f"{path.parent.name}/{path.name}"))
        for i, d in result["detail"].items():
            per_scale = summary.setdefault(i, {"current": [], **{s: [] for s in scales}})
            per_scale["current"].append(d["current"]["rho"])
            for scale, s in d["fixed"].items():
                per_scale[scale].append(s["rho"])

    lines = ["# Engram gate: fixed-query rescore\n"]
    lines.append(
        "`x_fixed = x * logit_scale / |h~|`; the current gate is `logit_scale = |h~|` per token. "
        "The non-swept Engram layers keep `logit_scale = mean(|h~|)`.\n"
    )
    lines.append("## Summary: mean rho(gate, dNLL) over sets\n")
    header = "| layer | current | " + " | ".join(f"{s:g}" for s in scales) + " |"
    lines.append(header)
    lines.append("|---:|---:|" + "---:|" * len(scales))
    for i in sorted(summary):
        cur = np.nanmean(np.asarray(summary[i]["current"], dtype=np.float64))
        cells = []
        for scale in scales:
            vals = np.asarray(summary[i].get(scale, []), dtype=np.float64)
            cells.append(f"{np.nanmean(vals):+.3f}" if vals.size else "-")
        lines.append(f"| {i} | {cur:+.3f} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.extend(blocks)
    text = "\n".join(lines) + "\n"
    if a.out:
        a.out.write_text(text)
        print(f"wrote {a.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
