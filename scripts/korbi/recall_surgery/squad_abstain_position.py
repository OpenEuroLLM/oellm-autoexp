#!/usr/bin/env python3
"""SQuAD v2 abstention on answerable questions by answer position and paragraph length (§15).

    python3 squad_abstain_position.py <flag_evals_oellm root> <export> [...]
"""
import glob, json, sys
F = sys.argv[1]
def resp(x):
    r = x.get("filtered_resps") or x.get("resps")
    while isinstance(r, list) and r:
        r = r[0]
    return r if isinstance(r, str) else ""
def abst(r):
    return r.strip() == "" or "unanswerable" in r.lower()
print("abstention on ANSWERABLE questions, by answer position in the paragraph (quartile of answer_start/len) and by paragraph length (chars)")
for m in sys.argv[2:]:
    rows = []
    for f in sorted(glob.glob(f"{F}/{m}/results/squadv2_10/samples_squadv2*.jsonl")):
        rows += [json.loads(l) for l in open(f)]
    data = []
    for x in rows:
        a = x["doc"].get("answers", {})
        if not a.get("text"):
            continue
        ctx = x["doc"].get("context", "")
        pos = a["answer_start"][0] / max(1, len(ctx))
        data.append((pos, len(ctx), abst(resp(x))))
    def rate(sel):
        return 100 * sum(s[2] for s in sel) / max(1, len(sel))
    pq = [rate([d for d in data if lo <= d[0] < hi]) for lo, hi in ((0, .25), (.25, .5), (.5, .75), (.75, 1.01))]
    lens = sorted(d[1] for d in data); t1, t2 = lens[len(lens)//3], lens[2*len(lens)//3]
    lq = [rate([d for d in data if d[1] < t1]), rate([d for d in data if t1 <= d[1] < t2]), rate([d for d in data if d[1] >= t2])]
    print(f"{m:36s} pos Q1..Q4: " + " ".join(f"{v:5.1f}" for v in pq) + f" | len short/mid/long ({t1}/{t2}): " + " ".join(f"{v:5.1f}" for v in lq))
