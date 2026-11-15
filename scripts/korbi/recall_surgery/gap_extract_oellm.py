"""Rows (task, n_shot, metric, value) from oellm-eval result trees (flag_evals_oellm/<export>/results/<task>_<shot>/...)."""
import csv, glob, json, os, re, sys
root, out = sys.argv[1], sys.argv[2]
exports = sys.argv[3:]
w = csv.writer(open(out, "w")); w.writerow(["checkpoint", "rowdir", "task", "metric", "filter", "value", "n_samples"])
for ex in exports:
    seen = set()
    for f in sorted(glob.glob(f"{root}/{ex}/results/*/**/results_*.json", recursive=True)):
        rowdir = os.path.relpath(f, f"{root}/{ex}/results").split("/")[0]
        try:
            d = json.load(open(f))
        except Exception:
            continue
        ns = d.get("n-samples", {}) or {}
        for task, m in (d.get("results") or {}).items():
            for k, v in m.items():
                if not isinstance(v, (int, float)) or "stderr" in k or k == "alias":
                    continue
                met, _, flt = k.partition(",")
                key = (rowdir, task, met, flt)
                if key in seen:
                    continue
                seen.add(key)
                n = ns.get(task, {}).get("effective") if isinstance(ns.get(task), dict) else ""
                w.writerow([ex, rowdir, task, met, flt, v, n])
print("done")
