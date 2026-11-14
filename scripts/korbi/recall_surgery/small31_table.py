#!/usr/bin/env python3
"""Table of the small-scale testbed's Phase-1 readouts (docs/recall_surgery_ambitious.md §14): per arm and checkpoint
the critical rows (CoQA, Winogrande, SQuAD v2 re-scored with the uniform abstain rule of squad_uniform.py: HasAns F1,
F1 when answering, abstain % on answerable / unanswerable and their difference, the abstention discrimination), the recall rows (SWDE / FDA / squad_completion), the RULER subset at 4096 and the v8ml-valid
loss. Reads <root>/evals/small31_<arm>_<step>/ (small31_eval.py's layout); missing cells print as '-'.

    small31_table.py [--root /e/scratch/e-sta-openeurollm/poeppel1/recall_surgery/small31] [--arms b7 r3 r3p]
"""

import argparse
import collections
import glob
import json
import os
import re
import string

TOKENS_PER_STEP = 64 * 8192


def _norm(s):
    s = "".join(ch for ch in s.lower() if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _f1(p, g):
    p, g = _norm(p).split(), _norm(g).split()
    if not p or not g:
        return float(p == g)
    c = sum((collections.Counter(p) & collections.Counter(g)).values())
    if c == 0:
        return 0.0
    pr, rc = c / len(p), c / len(g)
    return 2 * pr * rc / (pr + rc)


def _resp(x):
    r = x.get("filtered_resps") or x.get("resps")
    while isinstance(r, list) and r:
        r = r[0]
    return r if isinstance(r, str) else ""


def _done(d, key):
    f = os.path.join(d, "done", f"{key}.ok")
    return json.load(open(f)) if os.path.exists(f) else None


def _metric(res, task, names):
    """The first metric of ``names`` present in lm-eval's results[task] (keys 'metric,filter')."""
    r = (res or {}).get(task, {})
    for n in names:
        for k, v in r.items():
            if k.split(",")[0] == n and isinstance(v, (int, float)):
                return v
    return None


def squad_uniform(d):
    """SQuAD v2 re-scored from the logged samples with the uniform abstain rule of squad_uniform.py (empty output or
    containing "unanswerable"): HasAns F1 (abstain = 0), abstain % on answerable, F1 when answering, abstain % on
    unanswerable, and the abstention discrimination = abstain % on unanswerable - abstain % on answerable."""
    fs = glob.glob(os.path.join(d, "results", "squadv2_10", "*", "samples_*.jsonl"))
    if not fs:
        return {}
    ab = lambda g: g.strip() == "" or "unanswerable" in g.lower()  # noqa: E731
    has, answered, n_abst, unans = [], [], 0, []
    for f in sorted(fs):
        for line in open(f):
            x = json.loads(line)
            g, golds = _resp(x), x["doc"].get("answers", {}).get("text", [])
            if golds:
                v = 0.0 if ab(g) else max(_f1(g, gg) for gg in golds)
                has.append(v)
                if ab(g):
                    n_abst += 1
                else:
                    answered.append(v)
            else:
                unans.append(1.0 if ab(g) else 0.0)
    if not has:
        return {}
    mean = lambda v: 100 * sum(v) / max(1, len(v))  # noqa: E731
    a_ans, a_unans = 100 * n_abst / len(has), mean(unans)
    return {"squad_hasans_u": mean(has), "squad_abst": a_ans, "squad_f1_answered": mean(answered),
            "squad_abst_unans": a_unans, "squad_discrim": a_unans - a_ans}


def row(d):
    out = {}
    res = _done(d, "coqa_0")
    out["coqa_f1"] = _metric(res, "coqa", ["f1"])
    out["wino"] = _metric(_done(d, "winogrande_0"), "winogrande", ["acc"])
    res = _done(d, "squadv2_10")
    out["squad_f1"] = _metric(res, "squadv2", ["f1"])
    out["squad_hasans"] = _metric(res, "squadv2", ["HasAns_f1"])
    out.update({k: None for k in ("squad_hasans_u", "squad_abst", "squad_f1_answered", "squad_abst_unans",
                                  "squad_discrim")})
    out.update(squad_uniform(d))
    for t in ("swde", "fda", "squad_completion"):
        out[t] = _metric(_done(d, f"{t}_0"), t, ["contains", "acc"])
    res = _done(d, "ruler_4096")
    scores = [v for t, r in (res or {}).items() for k, v in r.items()
              if k.split(",")[0] in ("4096",) and isinstance(v, (int, float))]
    out["ruler"] = 100 * sum(scores) / len(scores) if scores else None
    f = os.path.join(d, "valid_loss.json")
    out["valid"] = json.load(open(f))["mean"] if os.path.exists(f) else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="/e/scratch/e-sta-openeurollm/poeppel1/recall_surgery/small31")
    ap.add_argument("--arms", nargs="*", default=["b7", "r3", "r3p", "a0", "E", "Ep", "C", "L", "Lp", "M", "Mp"])
    a = ap.parse_args()
    cols = [("coqa_f1", "CoQA F1", 100), ("wino", "Wino", 100), ("squad_f1", "SQuAD F1", 1),
            ("squad_hasans_u", "HasAnsU", 1), ("squad_f1_answered", "F1|ans", 1), ("squad_abst", "abst@ans", 1),
            ("squad_abst_unans", "abst@un", 1), ("squad_discrim", "discrim", 1),
            ("swde", "SWDE", 100), ("fda", "FDA", 100), ("squad_completion", "SQcomp", 100), ("ruler", "RULER4k", 1),
            ("valid", "valid", 1)]
    print(f"{'arm':5} {'step':>6} {'BT':>5} | " + " | ".join(f"{c[1]:>8}" for c in cols))
    for arm in a.arms:
        dirs = glob.glob(os.path.join(a.root, "evals", f"small31_{arm}_[0-9]*"))
        steps = sorted(int(re.search(r"_(\d+)$", d).group(1)) for d in dirs if re.search(r"_(\d+)$", d))
        for step in steps:
            r = row(os.path.join(a.root, "evals", f"small31_{arm}_{step}"))
            cells = []
            for key, _, scale in cols:
                v = r[key]
                cells.append(f"{'-':>8}" if v is None else (f"{v:8.4f}" if key == "valid" else f"{v * scale:8.2f}"))
            print(f"{arm:5} {step:6d} {step * TOKENS_PER_STEP / 1e9:5.2f} | " + " | ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
