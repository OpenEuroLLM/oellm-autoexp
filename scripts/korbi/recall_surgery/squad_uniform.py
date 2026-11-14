#!/usr/bin/env python3
"""SQuAD v2 re-scored uniformly from logged lm-eval samples (docs/recall_surgery_ambitious.md §15): the official
aggregates differ across lm-eval versions (the 32B's 0.4.12 run gives a literal "unanswerable" no NoAns credit).
HasAns F1 (abstain = 0), abstention rate on answerable, F1 when answering, abstention rate on unanswerable
(NoAnsAcc), total F1. Abstain = empty output or containing "unanswerable".

    python3 squad_uniform.py <root with <model>/results/squadv2_10/samples_squadv2*.jsonl> <model> [...]
"""
F = sys.argv[1]
def norm(s):
    s = s.lower(); s = "".join(c for c in s if c not in set(string.punctuation)); s = re.sub(r"\b(a|an|the)\b", " ", s); return " ".join(s.split())
def f1(p, g):
    p, g = norm(p).split(), norm(g).split()
    if not p or not g: return float(p == g)
    c = sum((collections.Counter(p) & collections.Counter(g)).values())
    if c == 0: return 0.0
    pr, rc = c / len(p), c / len(g); return 2 * pr * rc / (pr + rc)
def gen(x):
    r = x["filtered_resps"]; g = r[0] if isinstance(r, list) else r
    while isinstance(g, list) and g: g = g[0]
    return g if isinstance(g, str) else str(g)
ab = lambda g: g.strip() == "" or "unanswerable" in g.lower()
print("%-36s %8s %8s %9s %8s %9s " % ("model", "HasAnsF1", "abst@ans", "F1|answ", "NoAnsAcc", "totalF1"))
for m in sys.argv[2:]:
    fs = sorted(glob.glob(f"{F}/{m}/results/squadv2_10/samples_squadv2*.jsonl"))
    H, A, AN, NA, T = [], 0, [], [], []
    off = []
    for f in fs:
        for l in open(f):
            x = json.loads(l); g = gen(x); golds = x["doc"]["answers"]["text"]
            if golds:
                v = 0.0 if ab(g) else max(f1(g, gg) for gg in golds)
                H.append(v); T.append(v)
                if ab(g): A += 1
                else: AN.append(v)
            else:
                v = 1.0 if ab(g) else 0.0
                NA.append(v); T.append(v)
    mean = lambda v: 100 * sum(v) / max(1, len(v))
    print("%-36s %8.1f %7.1f%% %9.1f %7.1f%% %9.1f " % (m, mean(H), 100 * A / max(1, len(H)), mean(AN), mean(NA), mean(T)))
