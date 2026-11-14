#!/bin/bash
# Cache everything the Phase-1 readout (small31_readout.sh) downloads, on the JUPITER LOGIN node (compute nodes are
# offline): the lm-eval datasets of every row (a 2-document run with lm-eval's dummy model), RULER's haystack
# (baber/paul_graham_essays) and its QA sets, nltk punkt, and `wonderwords` (not in the image) into a library dir.
#   ssh jupiter 'cd ~/work/Projects/oellm-autoexp-hybrid && bash scripts/korbi/recall_surgery/small31_precache.sh'
set -euo pipefail
SIF=${SIF:-/e/project1/e-sta-openeurollm/poeppel1/container_cache/MegatronTraining-JUPITER_pt2512_aarch64_202602231451.sif}
HF=${HF_CACHE:-/e/scratch/e-sta-openeurollm/poeppel1/cache/hf}
NLTK=${NLTK_CACHE:-/e/scratch/e-sta-openeurollm/poeppel1/cache/nltk_data}
LIBS=${RULER_LIBS:-/e/project1/e-sta-openeurollm/poeppel1/pylibs_ruler}
TOK=${TOK:-/e/data1/datasets/playground/mmlaion/oellm_testing/tokenizers/tokenizer-256k-agent}
mkdir -p "$HF" "$NLTK" "$LIBS"
[ -d "$LIBS/wonderwords" ] || cp -r ~/work/pylibs/wonderwords ~/work/pylibs/wonderwords-*.dist-info "$LIBS/"
apptainer exec --bind /e --env HF_HOME="$HF" --env NLTK_DATA="$NLTK" --env PYTHONPATH="$LIBS" "$SIF" python - "$TOK" <<'PY'
import sys
import nltk
for r in ("punkt", "punkt_tab"):
    nltk.download(r, download_dir=__import__("os").environ["NLTK_DATA"], quiet=True)
import lm_eval
sys.path.insert(0, "scripts/korbi/recall_surgery")
from small31_eval import task_objects  # the SQuAD2 one-argument task-API patch
tok = sys.argv[1]
rows = [("coqa", 0), ("winogrande", 0), ("squadv2", 10), ("swde", 0), ("fda", 0), ("squad_completion", 0)]
for task, n in rows:
    r = lm_eval.simple_evaluate(model="dummy", tasks=task_objects([task]), num_fewshot=n, limit=2)
    print("cached", task, list(r["results"]))
ruler = ["niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multivalue", "niah_multiquery",
         "ruler_vt", "ruler_qa_squad"]
r = lm_eval.simple_evaluate(model="dummy", tasks=ruler, limit=2,
                            metadata={"max_seq_lengths": [4096], "pretrained": tok})
print("cached ruler", list(r["results"]))
PY
echo "[precache] done: HF_HOME=$HF NLTK_DATA=$NLTK libs=$LIBS"
