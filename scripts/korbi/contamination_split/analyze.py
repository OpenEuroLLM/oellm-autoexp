#!/usr/bin/env python3
"""Paired per-item comparison, split by contamination in the vanilla flag mix (docs/contamination_split_eval.md).

    python analyze.py --items items.jsonl.gz --samples <dir with <model>/<task>.jsonl.gz> --ref b1consTW_300k \
        --ours main_23500 [--extra main_24000 oldprod_60k b1cons_90k] [--out table.json]

Item labels (item_labels.py): every reference segment of the selected benchmark sources with its longest verbatim
run in vanilla flag and the number of sequences passing the flag-decon-v2 drop rule (frac 0.7 / abs 200 / min 20).
An eval item (compact_samples.py) is aligned to segments by normalized text of its passage / question fields:
  seen     >= 1 of its segments passes the drop rule (in the 32B's data; dropped from ours)
  clean    it has a matchable segment and none passes (the longest run below the rule: both models saw at most
           fragments)
  unknown  no matchable segment (only short fields, < 20 word tokens / 80 chars: the matcher cannot tell)
Per group: n, the paired mean difference ref - ours with a bootstrap 95 % CI, and the same for the extra models.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import re
import statistics

TASK_SRC = {
    "squadv2": r"^(lighteval|rajpurkar)/squad_v2$", "coqa": r"^EleutherAI/coqa$", "boolq": r"^aps/super_glue$",
    "winogrande": r"^allenai/winogrande$", "wsc273": r"^winograd_wsc$", "belebele": r"^facebook/belebele$",
    "hellaswag": r"^Rowan/hellaswag$", "piqa": r"^baber/piqa$", "social_iqa": r"^allenai/social_i_qa$",
    "openbookqa": r"^allenai/openbookqa$",
}


def tnorm(s) -> str:
    return " ".join(re.sub(r"[^0-9a-z]+", " ", str(s).lower()).split())


def load_segments(path):
    """task -> {normalized segment text: seg info} (for that task's sources)."""
    rx = {t: re.compile(p) for t, p in TASK_SRC.items()}
    out = {t: {} for t in TASK_SRC}
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            for t, x in rx.items():
                if x.match(r["src"]):
                    for s in r["seg_info"]:
                        k = tnorm(s["text"])
                        if k:
                            out[t][k] = s
    return out


def label(rec, segs):
    found, flagged, cover = [], False, 0.0
    for field, txt in rec["align"].items():
        s = segs.get(txt)
        if s is None:
            # SQuAD-style passages: the reference may hold the passage once per item with tiny formatting drift
            continue
        if s["matchable"]:
            found.append(field)
            flagged |= s["nflag"] > 0
            if s["n_tokens"]:
                cover = max(cover, s["maxrun"] / s["n_tokens"])
    if flagged:
        return "seen", cover
    return ("clean" if found else "unknown"), cover


def boot(diffs, n=2000, seed=0):
    if not diffs:
        return float("nan"), float("nan"), float("nan")
    rnd = random.Random(seed)
    m = statistics.fmean(diffs)
    k = len(diffs)
    bs = sorted(statistics.fmean(rnd.choices(diffs, k=k)) for _ in range(n))
    return m, bs[int(0.025 * n)], bs[int(0.975 * n) - 1]


def load_scores(path, metric):
    out = {}
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            out[r["key"]] = (r.get(metric, r["score"]), r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--ref", default="b1consTW_300k")
    ap.add_argument("--ours", default="main_23500")
    ap.add_argument("--extra", nargs="*", default=["main_24000", "oldprod_60k", "b1cons_90k"])
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    segs = load_segments(a.items)
    table = {}
    for task in TASK_SRC:
        metric = "f1" if task == "coqa" else "score"
        fo = os.path.join(a.samples, a.ours, f"{task}.jsonl.gz")
        fr = os.path.join(a.samples, a.ref, f"{task}.jsonl.gz")
        if not (os.path.exists(fo) and os.path.exists(fr)):
            continue
        ours, ref = load_scores(fo, metric), load_scores(fr, metric)
        extras = {e: load_scores(os.path.join(a.samples, e, f"{task}.jsonl.gz"), metric)
                  for e in a.extra if os.path.exists(os.path.join(a.samples, e, f"{task}.jsonl.gz"))}
        keys = sorted(set(ours) & set(ref))
        groups = {"all": [], "seen": [], "clean": [], "unknown": []}
        covbins = {"<0.3": [], "0.3-0.7": [], ">=0.7 (not flagged)": []}
        for k in keys:
            lab, cov = label(ours[k][1], segs[task])
            groups["all"].append(k)
            groups[lab].append(k)
            if lab == "clean":
                covbins["<0.3" if cov < 0.3 else "0.3-0.7" if cov < 0.7 else ">=0.7 (not flagged)"].append(k)
        res = {"n_segments_task": len(segs[task])}
        for g, ks in list(groups.items()) + [("clean cov " + b, v) for b, v in covbins.items()]:
            if not ks:
                continue
            row = {"n": len(ks), "ours": 100 * statistics.fmean(ours[k][0] for k in ks),
                   "ref": 100 * statistics.fmean(ref[k][0] for k in ks)}
            m, lo, hi = boot([ref[k][0] - ours[k][0] for k in ks])
            row.update(diff=100 * m, lo=100 * lo, hi=100 * hi)
            for e, sc in extras.items():
                kk = [k for k in ks if k in sc]
                if kk:
                    row[f"{e}-ours"] = 100 * statistics.fmean(sc[k][0] - ours[k][0] for k in kk)
            res[g] = row
        table[task] = res
        print(f"\n== {task} ({metric}); reference segments for alignment: {len(segs[task])}")
        for g, row in res.items():
            if not isinstance(row, dict):
                continue
            ex = "  ".join(f"{k} {v:+.2f}" for k, v in row.items() if k.endswith("-ours"))
            print(f"  {g:28s} n={row['n']:5d}  ours {row['ours']:6.2f}  32B {row['ref']:6.2f}  "
                  f"32B-ours {row['diff']:+6.2f} [{row['lo']:+.2f},{row['hi']:+.2f}]  {ex}")
    if a.out:
        json.dump(table, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
