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
     the split policy).
Every task / dataset is tried on its own; failures (gated, renamed, missing) are listed at the end
and in fetch_extra.report.json, never fatal.
"""

from __future__ import annotations

import argparse
import json
import os
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
    ("openlifescienceai/medmcqa", None), ("greengerong/leetcode", None),
    ("google-research-datasets/paws-x", None), ("facebook/xnli", None), ("li-lab/MMLU-ProX", None),
    ("openai/MMMLU", None), ("Muennighoff/xwinograd", None), ("juletxara/xstory_cloze", None),
    ("cambridgeltl/xcopa", None), ("rajpurkar/squad_v2", None), ("jbross-ibm-research/mgsm", None),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oellm-tasks", required=True, help="oellm-eval custom lm-eval task dir (include path)")
    ap.add_argument("--report", default="fetch_extra.report.json")
    args = ap.parse_args()
    os.environ["HF_DATASETS_OFFLINE"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "0"
    report = {"lm_eval_ok": [], "lm_eval_failed": {}, "hf_ok": [], "hf_failed": {}}

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

    from datasets import get_dataset_config_names, load_dataset
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
    with open(args.report, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: len(v) for k, v in report.items()}))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
