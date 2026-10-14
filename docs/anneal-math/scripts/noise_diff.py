#!/usr/bin/env python3
"""Same checkpoint, two eval runs: diff on the suite metric of every task both
runs have.

python3 noise_diff.py ORIGINAL.csv RERUN.csv
"""

import sys
import pandas as pd

ds = pd.read_csv("../evals-438.datasets.csv").fillna({"suite_filter": ""})
met = {(r.task, r.n_shot): (r.suite_metric, r.suite_filter) for r in ds.itertuples()}


def load(p):
    d = pd.read_csv(p).fillna({"filter": ""})
    keep = [
        (r.task, r.n_shot) in met and (r.metric, r.filter) == met[(r.task, r.n_shot)]
        for r in d.itertuples()
    ]
    return d[keep].set_index(["task", "n_shot"])[["metric", "value", "stderr", "n_samples"]]


a, b = load(sys.argv[1]), load(sys.argv[2])
j = a.join(b, lsuffix="_orig", rsuffix="_rerun", how="inner")
j["delta"] = j.value_rerun - j.value_orig
j["|delta|/stderr"] = (j.delta.abs() / j.stderr_orig).round(1)
j = j.reset_index()[
    [
        "task",
        "n_shot",
        "metric_orig",
        "n_samples_orig",
        "value_orig",
        "value_rerun",
        "delta",
        "stderr_orig",
        "|delta|/stderr",
    ]
]
j = j.sort_values("delta", key=abs, ascending=False)
pd.set_option("display.width", 200)
print(j.round(4).to_string(index=False))
