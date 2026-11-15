import csv, re, sys, collections
G = sys.argv[1]
PREF = ["acc_norm", "acc", "exact_match", "f1", "em", "contains", "accuracy", "accuracy_avg", "pass_at_1", "python_pass@1", "chrf", "bleu", "prompt_level_strict_acc", "inst_level_strict_acc"]
def cat(t):
    t = t.lower()
    if "belebele" in t: return "multilingual-RC"
    if any(s in t for s in ["coqa", "squad", "drop", "race", "boolq", "triviaqa", "naturalq", "nq_open", "quac", "swde", "fda"]): return "RC/QA"
    if any(s in t for s in ["winogrande", "wsc", "xwino", "winogender"]): return "coreference"
    if re.search(r"_mt_|xcopa|xstory|xnli|pawsx|global_piqa|m_mmlu|mmmlu|include|okapi|_eu$|_de$|_fr$", t): return "multilingual"
    if any(s in t for s in ["flores", "wmt", "trans", "ntrex"]): return "translation"
    if any(s in t for s in ["hellaswag", "piqa", "siqa", "social", "openbookqa", "copa", "commonsense", "storycloze", "arc_"]): return "commonsense/science"
    if any(s in t for s in ["mmlu", "agieval", "gpqa", "truthful", "bbh", "musr"]): return "knowledge/reasoning"
    if any(s in t for s in ["gsm", "math", "aime", "amc", "jee", "polymath", "minerva"]): return "math"
    if any(s in t for s in ["humaneval", "mbpp", "code", "lcb", "livecode"]): return "code"
    if any(s in t for s in ["ruler", "niah", "longbench", "babilong"]): return "long-context"
    if "ifeval" in t: return "instruction"
    if any(s in t for s in ["lambada", "wikitext", "perplex"]): return "LM"
    return "other"
def load_o32(path):
    d = {}
    for r in csv.DictReader(open(path)):
        if r.get("is_group") in ("True", "true", "1"): continue
        try: v = float(r["value"])
        except: continue
        d[(r["task"], str(r["n_shot"]), r["metric"], r["filter"] or "none")] = v
    return d
def load_ours(path, ckpt):
    d = {}
    for r in csv.DictReader(open(path)):
        if r["checkpoint"] != ckpt: continue
        m = re.match(r"(.+)_(\d+)$", r["rowdir"])
        if not m: continue
        rowtask, ns = m.groups()
        try: v = float(r["value"])
        except: continue
        d[(r["task"], ns, r["metric"], r["filter"] or "none")] = v
    return d
def scale(a, b):
    return (a * 100, b * 100) if a <= 1.0 and b <= 1.0 else (a, b)
def pick(A, B):
    """(task, nshot) -> (metric, a, b) with the preferred metric both have."""
    by = collections.defaultdict(dict)
    for k, v in A.items():
        if k in B: by[(k[0], k[1])][k[2] + "|" + k[3]] = (v, B[k])
    out = {}
    for tk, ms in by.items():
        for p in PREF:
            c = [m for m in ms if m.split("|")[0] == p]
            if c:
                m = sorted(c)[0]; out[tk] = (m,) + ms[m]; break
    return out
amb = sys.argv[2]; refs = sys.argv[3:]
A = load_ours(f"{G}/amb_rows.csv", amb)
rows = {}
for ref in refs:
    B = load_o32(f"{G}/o32_{ref}.csv") if ref.startswith(("b1", "old")) else load_ours(f"{G}/amb_rows.csv", ref)
    for tk, (m, a, b) in pick(A, B).items():
        a2, b2 = scale(a, b)
        rows.setdefault(tk, {"metric": m, "amb": a2})[ref] = b2
w = csv.writer(open(f"{G}/gap_{amb.split('_')[-1]}.csv", "w"))
w.writerow(["category", "task", "n_shot", "metric", "ambitious"] + refs + [f"gap_{r}" for r in refs])
for (t, ns), r in sorted(rows.items(), key=lambda kv: (cat(kv[0][0]), kv[0][0])):
    w.writerow([cat(t), t, ns, r["metric"], round(r["amb"], 2)] + [round(r.get(x, float("nan")), 2) for x in refs] + [round(r.get(x, float("nan")) - r["amb"], 2) for x in refs])
print(len(rows), "rows")
