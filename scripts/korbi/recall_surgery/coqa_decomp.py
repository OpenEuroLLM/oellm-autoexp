#!/usr/bin/env python3
"""CoQA (lm-eval coqa, last turn of 500 stories) F1 by gold kind, rationale position, turn count, story length and
source, plus behaviour (unknown on answerable, yes/no accuracy, answer length); docs/recall_surgery_ambitious.md §15.

    python3 coqa_decomp.py <flag_evals_oellm root> <export> [...]
"""
import glob, json, sys, statistics, collections
F = sys.argv[1]
def resp(x):
    r = x.get("filtered_resps") or x.get("resps")
    while isinstance(r, list) and r:
        r = r[0]
    return (r if isinstance(r, str) else "").strip()
def load(m):
    f = sorted(glob.glob(f"{F}/{m}/results/coqa_0/*/samples_coqa*.jsonl"))[-1]
    out = {}
    for l in open(f):
        x = json.loads(l)
        d = x["doc"]
        last = len(d["questions"]["input_text"]) - 1
        gold = d["answers"]["input_text"][last]
        g = gold.strip().lower()
        kind = "unknown" if g == "unknown" else "yes" if g == "yes" else "no" if g == "no" else "span"
        story = d["story"]
        ss = d["answers"]["span_start"][last]
        out[d["id"]] = dict(f1=x["f1"], kind=kind, turn=last + 1, pos=(ss / max(1, len(story))) if ss is not None and ss >= 0 else None,
                            slen=len(story), src=d["source"], r=resp(x).lower(), glen=len(gold.split()), rlen=len(resp(x).split()))
    return out
def mean(v):
    v = list(v)
    return 100 * sum(v) / len(v) if v else float("nan")
models = sys.argv[2:]
D = {m: load(m) for m in models}
ids = set.intersection(*(set(d) for d in D.values()))
def row(title, pred):
    cells = []
    for m in models:
        sel = [D[m][i] for i in ids if pred(D[m][i])]
        cells.append(f"{mean(s['f1'] for s in sel):5.1f}")
    n = sum(1 for i in ids if pred(D[models[0]][i]))
    print(f"{title:34s} n={n:4d} | " + " ".join(cells))
print("F1 by slice | models: " + " / ".join(m.replace("ambitious_v4_2ph_hd256_fp8_", "amb").replace("ext_", "") for m in models))
row("all", lambda s: True)
for k in ("span", "yes", "no", "unknown"):
    row(f"gold kind = {k}", lambda s, k=k: s["kind"] == k)
for lo, hi in ((0, .25), (.25, .5), (.5, .75), (.75, 1.01)):
    row(f"span answers, rationale pos {lo:.2f}-{hi:.2f}", lambda s, lo=lo, hi=hi: s["kind"] == "span" and s["pos"] is not None and lo <= s["pos"] < hi)
turns = sorted(s["turn"] for s in D[models[0]].values()); t1, t2 = turns[len(turns)//3], turns[2*len(turns)//3]
row(f"turn <= {t1}", lambda s: s["turn"] <= t1); row(f"turn {t1+1}-{t2}", lambda s: t1 < s["turn"] <= t2); row(f"turn > {t2}", lambda s: s["turn"] > t2)
lens = sorted(s["slen"] for s in D[models[0]].values()); l1, l2 = lens[len(lens)//3], lens[2*len(lens)//3]
row(f"story < {l1} chars", lambda s: s["slen"] < l1); row(f"story {l1}-{l2}", lambda s: l1 <= s["slen"] < l2); row(f"story >= {l2}", lambda s: s["slen"] >= l2)
for src in sorted({s["src"] for s in D[models[0]].values()}):
    row(f"source {src}", lambda s, src=src: s["src"] == src)
print()
print("behaviour (answerable = gold not 'unknown'):")
for m in models:
    v = [D[m][i] for i in ids]
    ans = [s for s in v if s["kind"] != "unknown"]
    unk = mean(("unknown" in s["r"]) for s in ans)
    yn = [s for s in v if s["kind"] in ("yes", "no")]
    yn_ok = mean((s["r"].split()[:1] == [s["kind"]]) for s in yn) if yn else float("nan")
    sp = [s for s in v if s["kind"] == "span"]
    ratio = statistics.median([s["rlen"] / max(1, s["glen"]) for s in sp])
    print(f"  {m:36s} says 'unknown' on answerable {unk:4.1f}% | yes/no correct {yn_ok:4.1f}% | span: median answer length / gold length {ratio:.2f}")
