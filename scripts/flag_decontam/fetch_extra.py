"""Download the benchmarks the FLAG prefetch does not cover, for the decontamination references (JUPITER login node).

    apptainer exec --cleanenv -B /e --env HF_HOME=$D/hf_extra --env HF_TOKEN_PATH=$HOME/.cache/huggingface/token \
        $VLLM_SIF python fetch_extra.py --oellm-tasks <oellm-eval>/oellm/resources/custom_lm_eval_tasks

Two sources (docs/flag_post_decontamination.md, 2026-10-24):
  1. oellm-eval task groups outside the FLAG suite (pawsx-eu, xnli-eu, mmlu-prox-eu, mmmlu-eu,
     generic-multilingual: xwinograd / xstorycloze / xcopa, the v0.02 reference corrections) and
     the benchmarks of another group's contamination table that we do not evaluate yet (RACE,
     MathQA, TriviaQA, NQ, MMLU-Pro, MMLU-Redux, PubMedQA, SciQ, SVAMP, AQuA-RAT, AlpacaEval,
     AdvBench, MedMCQA, WikiAnn, Tatoeba, MultiLoKo, LeetCode, SciRIFF, NorIdiom, the Spanish /
     Catalan / French / Basque bench groups) -- as lm-eval task names where lm-eval has them, so
     the data is fetched exactly as the harness reads it;
  2. the rest as plain Hugging Face dataset ids (all configs, all splits; extract_refs.py applies
     the split policy); script-based ones from the parquet branch; lm-eval language-bench groups
     subtask by subtask; DeepSeek-Coder's LeetCode contest file from GitHub into $HF_HOME/files.
  3. (v6, 2026-10-29) `--eval-set docs/decontamination_eval_set.csv`: every row with fetch=hf, only its
     eval splits (the dataset viewer's parquet files; raw data files for unconverted datasets),
     <= 4 GiB per dataset, model answers / judgments skipped; the GitHub-only tau2-bench tasks, BrowseComp (decrypted) and the DeepMind
     Mathematics test sets (from the tarball the script-based HF dataset downloads).
     `--only-eval-set` skips 1. and 2. (already in the cache).
Every task / dataset is tried on its own; failures (gated, renamed, missing) are listed at the end
and in fetch_extra.report.json, never fatal.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback

LM_EVAL_TASKS = [
    # oellm-eval groups outside FLAG (task names as in oellm/resources/task-groups.yaml)
    "pawsx_deu_Latn", "pawsx_eng_Latn", "pawsx_spa_Latn", "pawsx_fra_Latn",
    "xnli_bul_Cyrl", "xnli_deu_Latn", "xnli_ell_Grek", "xnli_eng_Latn", "xnli_spa_Latn", "xnli_fra_Latn",
    "mmlu_prox_cs", "mmlu_prox_de", "mmlu_prox_en", "mmlu_prox_es", "mmlu_prox_fr", "mmlu_prox_hu",
    "mmlu_prox_it", "mmlu_prox_pt", "mmlu_prox_sr", "mmlu_prox_uk",
    "mmmlu_de_de", "mmmlu_es_la", "mmmlu_fr_fr", "mmmlu_it_it", "mmmlu_pt_br",
    "xwinograd", "xstorycloze", "xcopa",
    "gsm8k_cot_numeric_v1", "squadv2_abstention_v1", "gpqa_diamond_cot0_v2",
    # the external table, where lm-eval has a task
    "race", "mathqa", "triviaqa", "nq_open", "mmlu_pro", "mmlu_redux", "pubmedqa", "sciq", "medmcqa",
    "spanish_bench", "catalan_bench", "french_bench", "basque_bench", "noridiom", "tatoeba_eng_nor",
]

HF_DATASETS = [  # (repo id, config or None) -- the external table's benchmarks lm-eval may not have
    ("ChilleD/SVAMP", None), ("deepmind/aqua_rat", "raw"), ("tatsu-lab/alpaca_eval", "alpaca_eval"),
    ("walledai/AdvBench", None), ("unimelb-nlp/wikiann", None), ("facebook/multiloko", None),
    ("allenai/SciRIFF", None), ("TIGER-Lab/MMLU-Pro", None), ("edinburgh-dawg/mmlu-redux-2.0", None),
    ("ehovy/race", "all"), ("allenai/math_qa", None), ("google-research-datasets/nq_open", None),
    ("mandarjoshi/trivia_qa", "rc.nocontext"), ("qiaojin/PubMedQA", "pqa_labeled"), ("allenai/sciq", None),
    ("openlifescienceai/medmcqa", None),
    ("google-research-datasets/paws-x", None), ("facebook/xnli", None), ("li-lab/MMLU-ProX", None),
    ("openai/MMMLU", None), ("Muennighoff/xwinograd", None), ("juletxara/xstory_cloze", None),
    ("cambridgeltl/xcopa", None), ("rajpurkar/squad_v2", None), ("jbross-ibm-research/mgsm", None),
    ("mimir-project/noridiom", None), ("Sprakbanken/Norwegian_idioms", None),   # NorIdiom (NorEval) and its source
]
# script-based datasets the current `datasets` refuses: Hugging Face's auto-converted parquet branch
PARQUET_FALLBACK = [("allenai/math_qa", None), ("csebuetnlp/xlsum", "data_dir=spanish"), ("PlanTL-GOB-ES/wnli-es", None),
                    ("GEM/OrangeSum", None), ("asi/wikitext_fr", None),   # "data_dir=x": a subdirectory, not a config
                    ("Helsinki-NLP/tatoeba_mt", "eng-nob"), ("Helsinki-NLP/tatoeba_mt", "eng-nno"),
                    ("Helsinki-NLP/tatoeba_mt", "eng-nor"), ("Helsinki-NLP/tatoeba_mt", "data_dir=eng-nob"),
                    ("Helsinki-NLP/tatoeba_mt", "data_dir=eng-nor")]   # Tatoeba Challenge test sets (gated)
# single data files of a dataset repo
HF_FILES = [("tatsu-lab/alpaca_eval", "alpaca_eval.json")]
# lm-eval groups whose subtasks are fetched one by one (one script-based or gated subtask fails a whole group)
LM_EVAL_GROUPS = ["spanish_bench", "catalan_bench", "french_bench", "basque_bench"]
# files outside the Hub: DeepSeek-Coder's LeetCode contest benchmark; tau2-bench tasks (v6)
URLS = ["https://raw.githubusercontent.com/deepseek-ai/DeepSeek-Coder/main/Evaluation/LeetCode/data/20240121-Jul.jsonl",
        "https://raw.githubusercontent.com/sierra-research/tau2-bench/main/data/tau2/domains/airline/tasks.json",
        "https://raw.githubusercontent.com/sierra-research/tau2-bench/main/data/tau2/domains/retail/tasks.json",
        "https://raw.githubusercontent.com/sierra-research/tau2-bench/main/data/tau2/domains/telecom/tasks.json",
        "https://s3.amazonaws.com/my89public/quac/val_v0.2.json",                    # QuAC (script-based on HF)
        "https://raw.githubusercontent.com/wilburOne/cosmosqa/master/data/valid.csv"]  # CosmosQA (script-based)
URL_NAMES = {"https://s3.amazonaws.com/my89public/quac/val_v0.2.json": "quac_val_v0.2.json",
             "https://raw.githubusercontent.com/wilburOne/cosmosqa/master/data/valid.csv": "cosmosqa_valid.csv"}
# OpenAI simple-evals BrowseComp: problem / answer XOR-encrypted with the row's canary (v6)
BROWSECOMP = "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"
EVAL_SET_MAX_BYTES = 4 * 2**30        # per dataset (largest eval split: OneRuler 3.2 GiB); larger ones reported, not fetched
# raw repo files that are model outputs, not benchmark items (Arena-Hard-Auto: 3.1 GiB of answers / judgments)
MODEL_OUTPUTS = re.compile(r"model_answer|model_judg", re.I)


def eval_set_rows(path):
    """Rows of docs/decontamination_eval_set.csv to fetch: fetch == 'hf' (not yet indexed, or a mirror /
    partial whose missing part is fetched)."""
    import csv
    return [r for r in csv.DictReader(open(path)) if r.get("fetch", "") == "hf" and r["hf_id"]]


def _wanted_configs(row, available):
    cfg = [c for c in row["config"].split(";") if c and " " not in c and c != "all"]
    hit = [c for c in cfg if c in available]
    return hit or sorted(available)


def _wanted_splits(row, splits):
    """Every non-train split, train when it is the only one; the CSV's eval_split when it names real splits."""
    named = [s for s in row["eval_split"].split(";") if s in splits]
    if named:
        return named
    other = [s for s in splits if s != "train"]
    return other or list(splits)


def fetch_eval_set(path, report, token=None):
    """Eval splits of the CSV's datasets into the hub cache, as the dataset viewer's parquet files
    (config/split/xxxx.parquet: extract_refs.py reads the split from the path); script-based or
    unconverted datasets: the repo's data files whose path names an eval split."""
    import urllib.parse
    import urllib.request
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=token)
    fdir = os.path.join(os.environ["HF_HOME"], "files")
    hdr = {"Authorization": f"Bearer {token}"} if token else {}
    for row in eval_set_rows(path):
        repo = row["hf_id"]
        try:
            req = urllib.request.Request(f"https://datasets-server.huggingface.co/parquet?dataset={repo}", headers=hdr)
            files = json.load(urllib.request.urlopen(req, timeout=60)).get("parquet_files", [])
        except Exception:
            files = []
        try:
            if files:
                by_cfg = {}
                for f in files:
                    by_cfg.setdefault(f["config"], {}).setdefault(f["split"], []).append(f)
                pick = [f for c in _wanted_configs(row, by_cfg) for s in _wanted_splits(row, by_cfg[c])
                        for f in by_cfg[c][s]]
                size = sum(f["size"] for f in pick)
                if size > EVAL_SET_MAX_BYTES:
                    raise RuntimeError(f"too large: {size / 2**30:.1f} GiB")
                for f in pick:
                    rel = urllib.parse.unquote(f["url"].split("/resolve/", 1)[1]).split("refs/convert/parquet/", 1)[1]
                    hf_hub_download(repo, rel, repo_type="dataset", revision="refs/convert/parquet", token=token)
                report["hf_ok"].append(f"eval_set {repo} ({len(pick)} parquet, {size / 2**20:.0f} MiB)")
            else:                                         # raw data files of the main branch
                info = api.dataset_info(repo, files_metadata=True)
                data = [s for s in info.siblings if s.rfilename.lower().endswith(DATA_EXT_FETCH + ARCHIVE_EXT)
                        and not MODEL_OUTPUTS.search(s.rfilename)]
                splits = {}
                for s in data:
                    splits.setdefault(_split_word(s.rfilename), []).append(s)
                pick = [s for sp in _wanted_splits(row, splits) for s in splits[sp]]
                size = sum(s.size or 0 for s in pick)
                if not pick:
                    raise RuntimeError("no data files")
                if size > EVAL_SET_MAX_BYTES:
                    raise RuntimeError(f"too large: {size / 2**30:.1f} GiB")
                for s in pick:
                    p = hf_hub_download(repo, s.rfilename, repo_type="dataset", token=token)
                    if s.rfilename.lower().endswith(ARCHIVE_EXT):
                        unpack(p, fdir, repo.replace("/", "__") + "__" + s.rfilename.replace("/", "__"))
                report["hf_ok"].append(f"eval_set {repo} ({len(pick)} raw files, {size / 2**20:.0f} MiB)")
            print(f"ok   eval_set {repo}", flush=True)
        except Exception as e:
            report["hf_failed"][f"eval_set {repo}"] = f"{type(e).__name__}: {str(e)[:200]}"
            print(f"FAIL eval_set {repo}: {type(e).__name__}: {str(e)[:160]}", flush=True)


ARCHIVE_EXT = (".zip", ".tar.gz", ".tgz")
# zip archives outside the Hub (script-based HF datasets download these): MLQA dev / test
ARCHIVE_URLS = ["https://dl.fbaipublicfiles.com/MLQA/MLQA_V1.zip"]


def unpack(path, fdir, prefix):
    """The eval-split data files of an archive into $HF_HOME/files, flattened under `prefix`
    (train members dropped when the archive has other splits)."""
    import tarfile
    import zipfile
    if path.lower().endswith(".zip"):
        z = zipfile.ZipFile(path)
        members = [(m, lambda m=m: z.read(m)) for m in z.namelist() if m.lower().endswith(DATA_EXT_FETCH)]
    else:
        t = tarfile.open(path)
        members = [(m.name, lambda m=m: t.extractfile(m).read()) for m in t.getmembers()
                   if m.isfile() and m.name.lower().endswith(DATA_EXT_FETCH)]
    splits = {_split_word(m) for m, _ in members}
    n = 0
    for m, read in members:
        if _split_word(m) == "train" and splits - {"train"}:
            continue
        with open(os.path.join(fdir, prefix.rsplit(".", 1)[0] + "__" + m.replace("/", "__")), "wb") as f:
            f.write(read())
        n += 1
    return n


DATA_EXT_FETCH = (".parquet", ".arrow", ".json", ".jsonl", ".csv", ".tsv", ".json.gz", ".jsonl.gz")


def _split_word(path):
    """The split a raw file belongs to (same rule as extract_refs.split_of)."""
    import re
    parts = re.split(r"[-_/.\s]+", path.lower())
    for word, split in (("devtest", "test"), ("validation", "validation"), ("valid", "validation"),
                        ("test", "test"), ("dev", "validation"), ("val", "validation"), ("train", "train")):
        if word in parts:
            return split
    return "unknown"


DM_MATH = "https://storage.googleapis.com/mathematics-dataset/mathematics_dataset-v1.0.tar.gz"


def fetch_deepmind_math(fdir, report):
    """DeepMind Mathematics test sets (deepmind/math_dataset is script-based and downloads this tarball):
    streamed, only interpolate/ and extrapolate/ kept (question / answer line pairs), one jsonl per split."""
    import tarfile
    import urllib.request
    n = {}
    try:
        outs = {s: open(os.path.join(fdir, f"deepmind_math_{s}.jsonl"), "w") for s in ("interpolate", "extrapolate")}
        with urllib.request.urlopen(DM_MATH, timeout=600) as resp, tarfile.open(fileobj=resp, mode="r|gz") as tar:
            for m in tar:
                parts = m.name.split("/")
                if not m.isfile() or len(parts) < 3 or parts[-2] not in outs:
                    continue
                lines = tar.extractfile(m).read().decode().splitlines()
                for q, a in zip(lines[0::2], lines[1::2]):
                    outs[parts[-2]].write(json.dumps({"module": parts[-1].removesuffix(".txt"),
                                                      "question": q, "answer": a}) + "\n")
                n[parts[-2]] = n.get(parts[-2], 0) + len(lines) // 2
        for f in outs.values():
            f.close()
        report["hf_ok"].append(f"deepmind_math {n}")
        print(f"ok   deepmind_math {n}", flush=True)
    except Exception as e:
        report["hf_failed"]["deepmind_math"] = f"{type(e).__name__}: {str(e)[:200]}"
        print(f"FAIL deepmind_math: {e}", flush=True)


def fetch_browsecomp(fdir, report):
    """BrowseComp test set, decrypted (simple-evals: key = sha256(canary) repeated, XOR, base64)."""
    import base64
    import csv
    import hashlib
    import io
    import urllib.request

    def dec(ct, pw):
        b = base64.b64decode(ct)
        k = hashlib.sha256(pw.encode()).digest()
        k = k * (len(b) // len(k)) + k[: len(b) % len(k)]
        return bytes(x ^ y for x, y in zip(b, k)).decode()
    try:
        text = urllib.request.urlopen(BROWSECOMP, timeout=120).read().decode()
        rows = list(csv.DictReader(io.StringIO(text)))
        with open(os.path.join(fdir, "browse_comp_test.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps({"problem": dec(r["problem"], r["canary"]),
                                    "answer": dec(r["answer"], r["canary"])}) + "\n")
        report["hf_ok"].append(f"browsecomp ({len(rows)} rows)")
        print(f"ok   browsecomp {len(rows)}", flush=True)
    except Exception as e:
        report["hf_failed"]["browsecomp"] = f"{type(e).__name__}: {str(e)[:200]}"
        print(f"FAIL browsecomp: {e}", flush=True)


def group_subtasks(tm, group):
    """Leaf task names of an lm-eval language-bench group: every yaml in the group's directory of the
    lm-eval package that defines a `task` (and is not itself a group)."""
    import pathlib
    import yaml
    import lm_eval.tasks

    class _L(yaml.SafeLoader):
        pass
    _L.add_multi_constructor("!", lambda loader, suffix, node: None)
    root = pathlib.Path(lm_eval.tasks.__file__).parent / group
    out = []
    for y in sorted(root.rglob("*.yaml")):
        try:
            cfg = yaml.load(open(y), Loader=_L) or {}
        except Exception:
            continue
        if isinstance(cfg.get("task"), str) and "group" not in cfg:
            out.append(cfg["task"])
    return out or [group]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oellm-tasks", required=True, help="oellm-eval custom lm-eval task dir (include path)")
    ap.add_argument("--report", default="fetch_extra.report.json")
    ap.add_argument("--eval-set", default="", help="docs/decontamination_eval_set.csv: also fetch its fetch=hf rows")
    ap.add_argument("--only-eval-set", action="store_true",
                    help="skip the lm-eval tasks / fixed lists (fetched before, in the cache) and fetch only the CSV's rows")
    args = ap.parse_args()
    os.environ["HF_DATASETS_OFFLINE"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "0"
    report = {"lm_eval_ok": [], "lm_eval_failed": {}, "hf_ok": [], "hf_failed": {}}

    if args.only_eval_set:
        return fetch_new(args, report)
    from lm_eval.tasks import TaskManager, get_task_dict
    tm = TaskManager(include_path=args.oellm_tasks)
    for t in LM_EVAL_TASKS:
        try:
            get_task_dict([t], tm)                       # downloads exactly what the task reads
            report["lm_eval_ok"].append(t)
            print(f"ok   lm-eval {t}", flush=True)
        except Exception as e:
            report["lm_eval_failed"][t] = f"{type(e).__name__}: {str(e)[:200]}"
            print(f"FAIL lm-eval {t}: {type(e).__name__}: {str(e)[:160]}", flush=True)

    for g in LM_EVAL_GROUPS:
        for t in group_subtasks(tm, g):
            try:
                get_task_dict([t], tm)
                report["lm_eval_ok"].append(f"{g}/{t}")
                print(f"ok   lm-eval {g}/{t}", flush=True)
            except Exception as e:
                report["lm_eval_failed"][f"{g}/{t}"] = f"{type(e).__name__}: {str(e)[:200]}"
                print(f"FAIL lm-eval {g}/{t}: {type(e).__name__}: {str(e)[:160]}", flush=True)

    from datasets import get_dataset_config_names, load_dataset
    for repo, cfg in PARQUET_FALLBACK:
        try:
            if cfg and cfg.startswith("data_dir="):
                load_dataset(repo, data_dir=cfg.split("=", 1)[1], revision="refs/convert/parquet")
            elif cfg:
                load_dataset(repo, cfg, revision="refs/convert/parquet")
            else:
                load_dataset(repo, revision="refs/convert/parquet")
            report["hf_ok"].append(f"{repo}@parquet")
            print(f"ok   hf {repo}@parquet", flush=True)
        except Exception as e:
            report["hf_failed"][f"{repo}@parquet"] = f"{type(e).__name__}: {str(e)[:200]}"
            print(f"FAIL hf {repo}@parquet: {type(e).__name__}: {str(e)[:160]}", flush=True)
    from huggingface_hub import hf_hub_download
    import shutil
    for repo, fn in HF_FILES:
        try:
            p = hf_hub_download(repo, fn, repo_type="dataset")
            os.makedirs(os.path.join(os.environ["HF_HOME"], "files"), exist_ok=True)
            shutil.copy(p, os.path.join(os.environ["HF_HOME"], "files", repo.replace("/", "__") + "__" + fn))
            report["hf_ok"].append(f"{repo}/{fn}")
            print(f"ok   hf file {repo}/{fn}", flush=True)
        except Exception as e:
            report["hf_failed"][f"{repo}/{fn}"] = f"{type(e).__name__}: {str(e)[:200]}"
            print(f"FAIL hf file {repo}/{fn}: {type(e).__name__}: {str(e)[:160]}", flush=True)
    import urllib.request
    fdir = os.path.join(os.environ["HF_HOME"], "files")
    os.makedirs(fdir, exist_ok=True)
    for u in URLS:
        try:
            urllib.request.urlretrieve(u, os.path.join(fdir, url_name(u)))
            report["hf_ok"].append(u)
            print(f"ok   url {u}", flush=True)
        except Exception as e:
            report["hf_failed"][u] = f"{type(e).__name__}: {str(e)[:200]}"
            print(f"FAIL url {u}: {e}", flush=True)
    for repo, cfg in HF_DATASETS:
        try:
            cfgs = [cfg] if cfg else get_dataset_config_names(repo)
        except Exception:
            cfgs = [None]
        for c in cfgs:
            name = f"{repo}:{c}"
            try:
                load_dataset(repo, c) if c else load_dataset(repo)
                report["hf_ok"].append(name)
                print(f"ok   hf {name}", flush=True)
            except Exception as e:
                report["hf_failed"][name] = f"{type(e).__name__}: {str(e)[:200]}"
                print(f"FAIL hf {name}: {type(e).__name__}: {str(e)[:160]}", flush=True)
    if args.eval_set:
        fetch_new(args, report)
        return
    with open(args.report, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: len(v) for k, v in report.items()}))


def url_name(u):
    """Local name of a fetched URL; names shared by several URLs (tau2's tasks.json) get the parent dir."""
    base = os.path.basename(u)
    if u in URL_NAMES:
        return URL_NAMES[u]
    if sum(os.path.basename(x) == base for x in URLS) > 1:
        return u.split("/")[-2] + "__" + base
    return base


def fetch_new(args, report):
    """The v6 additions: the CSV's datasets, the URLs (cheap to refetch) and BrowseComp."""
    import urllib.request
    tp = os.environ.get("HF_TOKEN_PATH", os.path.expanduser("~/.cache/huggingface/token"))
    token = open(tp).read().strip() if os.path.exists(tp) else None
    fdir = os.path.join(os.environ["HF_HOME"], "files")
    os.makedirs(fdir, exist_ok=True)
    for u in URLS:
        try:
            urllib.request.urlretrieve(u, os.path.join(fdir, url_name(u)))
            report["hf_ok"].append(u)
        except Exception as e:
            report["hf_failed"][u] = f"{type(e).__name__}: {str(e)[:200]}"
    for u in ARCHIVE_URLS:
        try:
            p = os.path.join(fdir, "_" + os.path.basename(u))
            urllib.request.urlretrieve(u, p)
            k = unpack(p, fdir, os.path.basename(u))
            os.remove(p)
            report["hf_ok"].append(f"{u} ({k} files)")
        except Exception as e:
            report["hf_failed"][u] = f"{type(e).__name__}: {str(e)[:200]}"
    fetch_browsecomp(fdir, report)
    if not os.path.exists(os.path.join(fdir, "deepmind_math_extrapolate.jsonl")):
        fetch_deepmind_math(fdir, report)
    if args.eval_set:
        fetch_eval_set(args.eval_set, report, token)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: len(v) for k, v in report.items()}))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
