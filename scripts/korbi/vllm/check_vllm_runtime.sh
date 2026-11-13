#!/bin/bash
# One-node check of the shared vLLM runtime (setup_vllm_shared.sh: base install + fork overlay on GPFS)
# and of node-local DP with unequal response caches (oellm-eval db648b5): a winogrande slice with
# --dp 4 and the response cache, then again after removing one rank's cache -- ranks 0, 2, 3 fully
# cached, rank 1 not, which hung the engines (30 min gloo timeout) before db648b5.
#
#   bash scripts/korbi/vllm/check_vllm_runtime.sh <hf export> <out dir>
set -euo pipefail
model=$1 out=$2
cd "$(dirname "$0")/../../.."
t0=$(date +%s)
source scripts/korbi/vllm/setup_vllm_shared.sh
echo "[check] setup $(($(date +%s) - t0)) s: INFER_PY=$INFER_PY OELLM_VLLM_SRC=$OELLM_VLLM_SRC"
export PYTHONPATH="$PWD/submodules/oellm-eval:$PYTHONPATH"
"$INFER_PY" - <<'EOF'
import importlib

import torch
import vllm
import vllm._C  # noqa: F401  the stock compiled extension, linked into the overlay

m = importlib.import_module("vllm.model_executor.models.oellm_ambitious")
print("[check] vllm", vllm.__version__, vllm.__file__, "| torch", torch.__version__, "| model", m.__file__, flush=True)
assert "/src_" in vllm.__file__ and "/src_" in m.__file__, "vLLM must come from the fork overlay"
EOF
root=$out/run_${SLURM_JOB_ID:-local}
command rm -rf "$root"
run() {
  local t=$(date +%s) rc=0
  timeout 1500 "$INFER_PY" -m oellm.runner.lm_eval_queue --model "$model" --root "$root" --names vllm/winogrande/0 \
    --tp 1 --dp 4 --ep 1 --gpu-mem 0.85 --max-num-seqs 256 --max-num-batched-tokens 32768 \
    --request-batch-size 256 --response-cache --limit 200 || rc=$?
  echo "[check] $1: rc $rc in $(($(date +%s) - t)) s"
  [ "$rc" = 0 ] || { echo "[check] FAIL ($1)"; exit 1; }
}
run "first run (fills the response cache)"
ls "$root"/response_cache/*/
command rm -f "$root/done/winogrande_0.ok" "$root"/response_cache/*/*_rank1.db*
run "rerun, rank 1 without its cache"
grep -h winogrande "$root"/summary_shard0.tsv
echo "[check] PASS"
