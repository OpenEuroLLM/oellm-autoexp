#!/usr/bin/env python3
"""6. Plot the math/code/reasoning evals (from table_reasoning.py) as a colour-coded table (table_plot.py).
python3 plot_reasoning.py [out.png]      (default: reasoning_evals.png)"""

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

T = runpy.run_path(f"{HERE}/table_reasoning.py")
res, models = T["res"], T["models"]
out = next((a for a in sys.argv[1:] if a.endswith(".png")), "reasoning_evals.png")

# (label in 5_reasoning_table, row title, 0-shot instruction-style = format-sensitive)
ROWS = [
    ("gsm8k strict", "GSM8K  4-shot  (strict)", False),
    ("mbpp pass@1", "MBPP  3-shot  pass@1", False),
    ("MATH500", "MATH500  0-shot †", True),
    ("HumanEval py", "HumanEval Python  0-shot †", True),
    ("GPQA-Diamond", "GPQA-Diamond  0-shot  (chance 0.25)", False),
]
rows = []
for label, title, fmt in ROWS:
    vals = {m: res[label, m][0] for m in models if (label, m) in res}
    if vals:
        rows.append((title, vals, np.mean([res[label, m][1] for m in vals]), fmt))

draw(
    rows,
    models,
    out,
    "32B dense: math & code evals (vLLM DP4, oellm-eval feat/vllm-data-parallel)",
    footnote="† 0-shot, instruction-style prompt: scores mostly track whether a checkpoint follows the answer format "
    "(fork_86k often returns nothing).\n   Counting only answers with a code block, HumanEval is 0.61–0.72 "
    "for every checkpoint. The few-shot rows (GSM8K, MBPP) are the fair comparison.",
)
