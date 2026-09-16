#!/usr/bin/env python3
"""4b.

The open-sci-0.01 results (from table_hf.py) as a colour-coded table,
same layout as plot_reasoning.py. python3 plot_hf.py [out.png] (default:
downstream_evals.png)
"""

import os
import runpy
import sys

import numpy as np

D = os.environ.get(
    "EVAL_WORK_ROOT", os.path.dirname(os.path.realpath(__file__))
)  # exports, caches and runs
HERE = os.path.dirname(os.path.realpath(__file__))  # the scripts themselves
sys.path.insert(0, HERE)
from table_plot import draw  # noqa: E402 (needs sys.path above)

T = runpy.run_path(f"{HERE}/table_hf.py")
res, models, tasks, full, info = T["res"], T["models"], T["rows"], T["full"], T["info"]
out = next((a for a in sys.argv[1:] if a.endswith(".png")), "downstream_evals.png")

rows = []
for t in tasks:
    vals = {m: res[t, m][0] for m in models if (t, m) in res}
    rows.append(
        (
            f"{t}  {info[t][3]}-shot  {info[t][2]}",
            vals,
            np.mean([res[t, m][1] for m in vals]),
            False,
        )
    )
if full:
    means = {m: np.mean([res[t, m][0] for t in full]) for m in models}
    se = np.mean([np.sqrt(sum(res[t, m][1] ** 2 for t in full)) / len(full) for m in models])
    rows.append((f"mean ({len(full)} tasks)", means, se, False))

draw(
    rows,
    models,
    out,
    "32B dense: downstream evals (open-sci-0.01, lm-eval, bf16)",
    sep_before=len(tasks) if full else None,
)
