#!/usr/bin/env python3
"""Re-score logged lm-eval samples (docs/recall_surgery_ambitious.md §15): FDA on short targets (<= 4 words,
normalized containment) and SQuAD v2 answerable questions split into abstention rate and F1 when answering.

    python3 rescore_fda_squad.py <flag_evals_oellm root> <export> [...]
"""
import glob, json, re, string, sys, collections
F = sys.argv[1]
models = sys.argv[2:]
def norm(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())
def resp(x):
    r = x.get("filtered_resps") or x.get("resps")
    while isinstance(r, list) and r:
        r = r[0]
    return r if isinstance(r, str) else ""
def f1(p, g):
    p, g = norm(p).split(), norm(g).split()
    if not p or not g:
        return float(p == g)
    c = sum((collections.Counter(p) & collections.Counter(g)).values())
    if c == 0:
        return 0.0
    pr, rc = c / len(p), c / len(g)
    return 2 * pr * rc / (pr + rc)
print(f"{'model':36s} | FDA all / short(<=4 words) n | SQuADv2 answerable: abstain% , F1 when answered , HasAns F1")
for m in models:
    out = [m.ljust(36)]
    fs = glob.glob(f"{F}/{m}/results/fda_0/*/samples_*.jsonl")
    if fs:
        rows = [json.loads(l) for l in open(sorted(fs)[-1])]
        allc = sum(norm(x["target"]) in norm(resp(x)) for x in rows) / len(rows)
        sh = [x for x in rows if len(str(x["target"]).split()) <= 4]
        shc = sum(norm(x["target"]) in norm(resp(x)) for x in sh) / max(1, len(sh))
        out.append(f"FDA {allc*100:5.1f} / {shc*100:5.1f} n={len(sh)}")
    else:
        out.append("FDA -")
    fs = glob.glob(f"{F}/{m}/results/squadv2_10/samples_squadv2*.jsonl") + glob.glob(f"{F}/{m}/results/squadv2_10/*/samples_*.jsonl")
    if fs:
        rows = []
        for f in sorted(set(fs)):
            rows += [json.loads(l) for l in open(f)]
        ans = []
        for x in rows:
            golds = x["doc"].get("answers", {}).get("text", [])
            if golds:
                ans.append((resp(x), golds))
        abst = [r for r, g in ans if norm(r) in ("", "unanswerable", "no answer") or "unanswerable" in r.lower()]
        answered = [(r, g) for r, g in ans if not (norm(r) in ("", "unanswerable", "no answer") or "unanswerable" in r.lower())]
        f1a = sum(max(f1(r, gg) for gg in g) for r, g in answered) / max(1, len(answered))
        f1h = sum(max(f1(r, gg) for gg in g) for r, g in ans) / max(1, len(ans))
        out.append(f"SQuADv2 n={len(ans)} abstain {len(abst)/max(1,len(ans))*100:5.1f}%  F1|answered {f1a*100:5.1f}  HasAns {f1h*100:5.1f}")
    else:
        out.append("SQuADv2 -")
    print(" | ".join(out))
