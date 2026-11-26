#!/bin/bash
# Fork checkpoint of the main run for the Engram gate trial (docs/engram_main_run_trial_plan.md), one node:
#   1. HF export of the main run's pinned save <MAIN>/ckpt_eval/iter_<IT> (export_run.sh, fp8 experts)
#   2. the gate diag on the validation sets (engram_gate_diag.py --fixed-query --sweep-layer 2)
#   3. layer-wise mean |h~| (the query RMS) over those tokens -> <DIAG>/query_rms.json; the early layer
#      (EARLY, 2) is set to EARLY_SCALE (0.3, gate-opening), every other Engram layer keeps its mean
#   4. add_engram_branch_gates.py --query-mix --logit-scale -> <FORK>/ckpt/iter_<IT> + tracker (the fork waits on it)
#   5. the rest of the diag (GSM8K, FLORES) for the offline record; the fork does not wait for it
# The main tree is only read; every step skips work that is already done.
set -euo pipefail
cd "$(dirname "$0")/../../.."

MAIN=${MAIN:?set MAIN to the main run dir}
IT=${IT:?set IT}
EXPORT=${EXPORT:?set EXPORT to the HF export dir}
DIAG=${DIAG:?set DIAG to the diag output dir}
FORK=${FORK:?set FORK to the fork checkpoint root}
EARLY=${EARLY:-2}
EARLY_SCALE=${EARLY_SCALE:-0.3}
LOGIT_SCALES=${LOGIT_SCALES:-0.025,0.1,0.3,1.0}
it=iter_$(printf %07d "$IT")

echo "== [1/5] export $MAIN $it -> $EXPORT"
EXPERT_DTYPE=fp8 bash scripts/korbi/hf_export/export_run.sh "$MAIN" "$IT" "$EXPORT"

echo "== [2/5] diag (validation sets) -> $DIAG"
python3 scripts/rluukkon/engram_gate_diag.py --model-dir "$EXPORT" --out "$DIAG" --docs 16 --flores 0 --gsm8k 0 \
    --fixed-query --logit-scales "$LOGIT_SCALES" --sweep-layer "$EARLY"

echo "== [3/5] query RMS"
scale=$(python3 - "$DIAG" "$EARLY" "$EARLY_SCALE" <<'PY'
import glob, json, sys
import numpy as np
diag, early, early_scale = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
layers = json.load(open(f"{diag}/runs.json"))["check"]["engram_layers"]
files = sorted(glob.glob(f"{diag}/valid_*.npz"))
assert len(files) == 5, files
rms = {i: float(np.concatenate([np.load(f)[f"hn_L{i}"] for f in files]).mean()) for i in layers}
assert early in rms, (early, layers)
init = {i: (early_scale if i == early else rms[i]) for i in layers}
json.dump({"sets": files, "mean_query_rms": rms, "logit_scale_init": init}, open(f"{diag}/query_rms.json", "w"), indent=1)
print(f"mean query RMS {rms}", file=sys.stderr)
print(",".join(f"{i}:{init[i]:.6g}" for i in layers))
PY
)
echo "logit_scale init: $scale"

echo "== [4/5] surgery -> $FORK/ckpt/$it"
if [ -f "$FORK/ckpt/latest_checkpointed_iteration.txt" ]; then
    echo "fork checkpoint exists: skipped"
else
    PYTHONPATH="scripts/korbi/production${PYTHONPATH:+:$PYTHONPATH}" \
        python3 scripts/korbi/production/add_engram_branch_gates.py \
        --src "$MAIN/ckpt_eval/$it" --dst "$FORK/ckpt/$it" --query-mix --logit-scale "$scale"
    echo "$scale" > "$FORK/logit_scale_init.txt"
    echo "$IT" > "$FORK/ckpt/latest_checkpointed_iteration.txt"
fi
echo "fork checkpoint ready: $FORK/ckpt/$it"

echo "== [5/5] diag (GSM8K, FLORES)"
python3 scripts/rluukkon/engram_gate_diag.py --model-dir "$EXPORT" --out "$DIAG" --docs 16 --flores 16 --gsm8k 64 \
    --fixed-query --logit-scales "$LOGIT_SCALES" --sweep-layer "$EARLY" \
    || echo "diag (GSM8K, FLORES) failed: the fork checkpoint is unaffected"
