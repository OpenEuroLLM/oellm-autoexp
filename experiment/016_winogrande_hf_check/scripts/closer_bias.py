"""Closer-mention bias on Winogrande from lm-eval samples files: how often the model picks the option
mentioned last before the blank, and accuracy split by whether the answer is the closer or the farther
entity.  python closer_bias.py <samples_winogrande_*.jsonl> ..."""
import json, glob, statistics as st, sys, re
for p in sys.argv[1:]:
    docs = [json.loads(l) for l in open(p)]
    rows = []
    for d in docs:
        s = d["doc"]["sentence"]; pre = s[:s.index("_")]
        p1, p2 = pre.rfind(d["doc"]["option1"]), pre.rfind(d["doc"]["option2"])
        if p1 < 0 or p2 < 0: continue
        closer = 0 if p1 > p2 else 1
        ll = [float(r[0][0]) for r in d["resps"]]; pick = 0 if ll[0] > ll[1] else 1
        rows.append((closer, pick, int(d["doc"]["answer"]) - 1))
    name = re.sub(r".*/(flag_evals[^/]*)/([^/]+)/results.*", r"\2 [\1]", p) if "flag_evals" in p else p.split("/")[-2] + "/" + p.split("/")[-1][:20]
    acc = st.mean(p == g for c, p, g in rows)
    print(f"{name:58s} acc {acc:.3f}  picks closer {st.mean(p == c for c, p, g in rows):.3f}  acc gold=closer {st.mean(p == g for c, p, g in rows if g == c):.3f}  gold=farther {st.mean(p == g for c, p, g in rows if g != c):.3f}")
