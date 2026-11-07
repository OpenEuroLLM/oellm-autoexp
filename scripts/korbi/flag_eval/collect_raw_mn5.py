"""Per-task raw results of some checkpoints, straight from oellm-eval run directories.

MN5 copy of the JUPITER collector (/e/project1/e-sta-openeurollm/vanosch1/v1-vs-v2/anneal/scripts/
collect_raw.py, 2026-11-07), the input of summarize-evals (github.com/spyysalo/summarize-evals).
One addition, --lmeval-object: our flag_lmeval.py passes lm-eval a model object, so its result files
carry no ``model_args`` and ``config.model`` None (single engine) or NodeEPVLLM (DP adapter); with exactly
one checkpoint, such lm-eval files (they have ``configs``) are taken as that checkpoint's. lighteval's
``<suite>:_average`` rows (its mean over the tasks of one grouped call) are dropped. MN5 layout: an export's <out>/<export> holds results/
(lm-eval), evalchemy/ and lighteval/; pass a directory whose results/ links all three
(flag_summary_mn5.sh).

One row per (checkpoint, task, n_shot, metric, filter): every task and subtask a harness wrote, value and
stderr as written (0-1 for accuracies; BLEU and chrF 0-100). No primary-metric choice, no aggregation beyond
what the harness reports, no comparison. Harnesses:

  lm-eval    results_*.json / <hash>_<date>.json with `configs`
  Evalchemy  results json without `configs`
  lighteval  results_*.json with `config_general`. flores' chrF++ and chrF carry filter "rescored" (the
             corrected corpus score). Unpatched lighteval scores only the first sentence against every
             reference, so they come from the run's chrf_rescored.csv (lighteval_rescore_chrf.py). Runs from
             the patched image (oellm-eval feat/flag-evals, patches/lighteval-chrf.patch) compute the same
             score themselves: pass --chrf-patched to take lighteval's own. BLEU is always lighteval's.

             lighteval reports no sample count. With --datasets, a run whose sbatch has LIMIT="" (full split)
             gets n_samples from that CSV's n_samples; otherwise it comes from chrf_rescored.csv or stays empty.
             The same CSV fills Evalchemy rows without num_total; lm-eval groups sum their leaves' counts.

A result in more than one results file (reruns) is taken from the newest file; n_files says how many existed.
Only vLLM (lm-eval, Evalchemy) and lighteval's hf/accelerate results are read. Dropped: sib200 acc_norm, which
always picks the longest label (" science/technology") and so equals that label's share of the test set.

    python3 collect_raw.py --checkpoints v2zloss_86k --runs <run dir> ... [--chrf-patched] [--datasets <csv>] -o out.csv
"""
import argparse
import csv
import glob
import json
import os
import re
from pathlib import Path

COLS = ["checkpoint", "task", "n_shot", "harness", "backend", "is_group", "metric", "filter", "value", "stderr",
        "n_samples", "results_file", "result_time", "n_files"]


def lmeval_model(d):
    a = (d.get("config") or {}).get("model_args")
    p = a.get("pretrained") if isinstance(a, dict) else (re.search(r"pretrained=([^,]+)", str(a)) or [None, None])[1]
    return p.rstrip("/").split("/")[-1] if p else None


def lmeval_rows(d, f):
    lm = "configs" in d                       # lm-eval writes task configs; Evalchemy does not
    gs = d.get("group_subtasks", {})
    for task, res in (d.get("results") or {}).items():
        shot = d.get("n-shot", {}).get(task)
        if shot is None:                      # groups record no n-shot: take a leaf's
            leaf = task
            while leaf in gs and gs[leaf]:
                leaf = gs[leaf][0]
            shot = d.get("configs", {}).get(leaf, {}).get("num_fewshot", 0) if lm else 0
        n = d.get("n-samples", {}).get(task, {}).get("effective", "") if lm else res.get("num_total", "")
        if lm and n == "" and task in gs:     # groups record no count: sum their leaves'
            leaves, todo = [], list(gs[task])
            while todo:
                t = todo.pop()
                todo.extend(gs[t]) if gs.get(t) else leaves.append(t)
            ns = [d.get("n-samples", {}).get(t, {}).get("effective") for t in leaves]
            n = sum(ns) if ns and None not in ns else ""
        for key, val in res.items():
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                continue
            if lm:
                if "_stderr" in key or key in ("alias", "sample_len"):
                    continue
                metric, _, filt = key.partition(",")
                se = res.get(f"{metric}_stderr,{filt}") if filt else None
            else:
                if key.endswith("_std_err") or key in ("num_total", "num_repeat"):
                    continue
                metric, filt = key, ""
                se = res.get(key.replace("_avg", "_std_err")) if key.endswith("_avg") else None
            yield {"task": task, "n_shot": shot, "harness": "lm-eval" if lm else "Evalchemy",
                   "backend": "vllm" if (d.get("config") or {}).get("model") in (None, "NodeEPVLLM", "VLLM")
                   else d["config"]["model"], "is_group": "yes" if task in gs else "",
                   "metric": metric, "filter": filt, "value": val,
                   "stderr": se if isinstance(se, (int, float)) else "", "n_samples": n,
                   "results_file": f, "result_time": d.get("date") or os.path.getmtime(f)}


def lighteval_rows(d, f, rescored, sizes):
    when = os.path.getmtime(f)                # lighteval's start/end times are not wall-clock
    for name, res in d["results"].items():
        if name == "all":
            continue
        task, shot = name.rsplit("|", 1)
        if task.endswith(":_average"):        # MN5 groups: lighteval's mean over the tasks of one call
            continue
        n = rescored[name]["n"] if name in rescored else sizes.get((task, shot), "")
        base = {"task": task, "n_shot": int(shot), "harness": "lighteval", "backend": "hf/accelerate",
                "is_group": "", "results_file": f, "result_time": when, "n_samples": n}
        if task.startswith("flores200:"):
            for m in ("chrf++", "chrf"):
                if rescored:
                    val, se = float(rescored[name][m]), ""
                elif m in res:                    # patched image: its own score is the corrected one (chrF++ only)
                    val, se = res[m], res.get(f"{m}_stderr", "")
                else:
                    continue
                yield {**base, "metric": m, "filter": "rescored", "value": val, "stderr": se}
            keep = {"bleu", "bleu_1", "bleu_4"}
        else:
            keep = {k for k in res if not k.endswith("_stderr")}
        for m in sorted(keep & set(res)):
            yield {**base, "metric": m, "filter": "none", "value": res[m], "stderr": res.get(f"{m}_stderr", "")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True, help="model dir basenames")
    ap.add_argument("--runs", nargs="+", required=True, help="oellm-eval run directories")
    ap.add_argument("--chrf-patched", action="store_true",
                    help="lighteval runs without chrf_rescored.csv are from the chrF-patched image")
    ap.add_argument("--datasets", help="datasets CSV (task, n_shot, n_samples): lighteval sample counts")
    ap.add_argument("--lmeval-object", action="store_true",
                    help="lm-eval files without config.model (a model object) belong to the one checkpoint")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    sizes = {(r["task"], r["n_shot"]): r["n_samples"] for r in csv.DictReader(open(a.datasets))} if a.datasets else {}
    best, files = {}, {}
    for run in a.runs:
        rp = Path(run) / "chrf_rescored.csv"
        rescored = {r["task"]: r for r in csv.DictReader(open(rp))} if rp.exists() else {}
        sb = Path(run) / "submit_evals.sbatch"
        full = sb.exists() and re.search(r'^LIMIT=""$', sb.read_text(), re.M)
        for f in sorted({str(Path(p).resolve()) for p in glob.glob(f"{run}/results/**/*.json", recursive=True)}):
            if "_artifacts/" in f or Path(f).name.startswith("samples_"):
                continue
            try:
                d = json.loads(Path(f).read_text())
            except (OSError, ValueError):
                continue
            if not isinstance(d, dict) or "results" not in d:
                continue
            if "config_general" in d:
                m = os.path.basename(d["config_general"]["model_config"]["model_name"].rstrip("/"))
                if m not in a.checkpoints:
                    continue
                if any(k.startswith("flores200:") for k in d["results"]) and not (rescored or a.chrf_patched):
                    raise SystemExit(f"{run}: flores results but no chrf_rescored.csv (lighteval_rescore_chrf.py); "
                                     "--chrf-patched if the run used the patched lighteval image")
                rows = lighteval_rows(d, f, rescored, sizes if full else {})
            elif (d.get("config") or {}).get("model") == "vllm" or (
                    a.lmeval_object and "configs" in d and (d.get("config") or {}).get("model") in (None, "NodeEPVLLM", "VLLM")):
                m = lmeval_model(d) or (a.checkpoints[0] if a.lmeval_object and len(a.checkpoints) == 1 else None)
                if m not in a.checkpoints:
                    continue
                rows = ({**r, "n_samples": r["n_samples"] if r["n_samples"] != "" or not full
                         else sizes.get((r["task"], str(r["n_shot"])), "")} for r in lmeval_rows(d, f))
            else:
                continue
            for row in rows:
                if row["task"].startswith("sib200_") and row["metric"] == "acc_norm":
                    continue
                k = (m, row["task"], str(row["n_shot"]), row["metric"], row["filter"])
                files.setdefault(k[:3], set()).add(f)
                if k not in best or float(row["result_time"]) >= float(best[k]["result_time"]):
                    best[k] = {"checkpoint": m, **row}
    rows = sorted(best.values(), key=lambda r: (r["checkpoint"], r["task"], int(r["n_shot"]), r["metric"], r["filter"]))
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow({**r, "n_files": len(files[(r["checkpoint"], r["task"], str(r["n_shot"]))])})
    print(f"{a.out}: {len(rows)} rows, checkpoints {sorted({r['checkpoint'] for r in rows})}")


if __name__ == "__main__":
    main()
