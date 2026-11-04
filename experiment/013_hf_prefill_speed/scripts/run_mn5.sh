#!/bin/bash
# 013: HF prefill on one MN5 node: the reduced-model kernel checks / layer timings on GPU 0, then
# the real export (v4_2ph fp8_6000) fast vs experiment 008's HF path, with logprob parity.
set -uo pipefail
source scripts/korbi/vllm/setup_vllm_uv.sh
export CC=${CC:-gcc}
D=experiment/013_hf_prefill_speed
OUT=$PWD/$D/results/${RESULT_TAG:-mn5_run1}
mkdir -p "$OUT"
R=/gpfs/projects/ehpc533/users/korbip/hybrid_v019
BASE=${BASE:-/gpfs/scratch/ehpc533/hybrid_nucleus/hf_exports/ambitious_v4_2ph_fp8_6000}
VIEW=${BASE}_hf013
PROMPTS=$R/autoexp_infer008/experiment/008_ambitious_inference_speed/results/ep4/prompts.json
{ hostname; nvidia-smi -L; free -g; "$INFER_PY" -c "import torch, triton, transformers; print(torch.__version__, triton.__version__, transformers.__version__)"; } > "$OUT/node.txt" 2>&1
cat "$OUT/node.txt"
status=0
STAGES=${STAGES:-reduced,full}

if [ "${STAGES#*reduced}" != "$STAGES" ]; then
  export CUDA_VISIBLE_DEVICES=0
  "$INFER_PY" $D/scripts/check_truth.py > "$OUT/check_truth.log" 2>&1 || { echo "check_truth failed"; status=1; }
  grep -E '^(attention|moe|engram) ' "$OUT/check_truth.log"
  "$INFER_PY" $D/scripts/check_fast.py --shape 4x512 --out "$OUT/check_fast.json" > "$OUT/check_fast.log" 2>&1 || { echo "check_fast failed"; status=1; }
  grep -E '^(logits|nll|host_syncs|engram)' "$OUT/check_fast.log"
  SHAPES="1x2048 8x512 32x256 4x2048 1x8192"
  "$INFER_PY" $D/scripts/bench_layers.py --shapes $SHAPES --out "$OUT/layers_fast.json" \
      --profile "$OUT/prof_fast_8x512.txt" > "$OUT/layers_fast.log" 2>&1 || { echo "layers fast failed"; status=1; }
  grep -A1 '"shape"' "$OUT/layers_fast.log"
  OELLM_HF_IMPL=reference OELLM_HF_MOE=grouped OELLM_MHC_IMPL=triton \
    "$INFER_PY" $D/scripts/bench_layers.py --shapes $SHAPES --out "$OUT/layers_hf008.json" \
      > "$OUT/layers_hf008.log" 2>&1 || { echo "layers hf008 failed"; status=1; }
  grep -A1 '"shape"' "$OUT/layers_hf008.log"
  unset CUDA_VISIBLE_DEVICES
fi

if [ "${STAGES#*gemm}" != "$STAGES" ]; then
  CUDA_VISIBLE_DEVICES=0 "$INFER_PY" $D/scripts/bench_gemm.py --out "$OUT/gemm.json" > "$OUT/gemm.log" 2>&1 || { echo "gemm failed"; status=1; }
  grep tokens "$OUT/gemm.log"
fi

if [ "${STAGES#*full}" != "$STAGES" ]; then
  # an export view: hard links to the immutable weights; this checkout's model code replaces the
  # code files atomically (new inodes: the original export's files are never written)
  [ -d "$VIEW" ] || cp -al "$BASE" "$VIEW"
  for f in oellm_ambitious/configuration_oellm_ambitious.py oellm_ambitious/modeling_oellm_ambitious.py \
           oellm_ambitious/mhc_kernels.py oellm_ambitious/oellm_kernels.py \
           oellm_hybrid/configuration_oellm_hybrid.py oellm_hybrid/modeling_oellm_hybrid.py; do
    cp "oellm_autoexp/hf_export/$f" "$VIEW/$(basename $f).new" && mv "$VIEW/$(basename $f).new" "$VIEW/$(basename $f)"
  done
  VARIANTS=${VARIANTS:-fast,hf008}
  for v in ${VARIANTS//,/ }; do
    timeout ${CASE_TIMEOUT:-1800} "$INFER_PY" $D/scripts/bench_full.py --model "$VIEW" --variant "$v" \
      --prompts "$PROMPTS" --out "$OUT/full" > "$OUT/full_$v.log" 2>&1 || { echo "full $v failed rc=$?"; status=1; }
    grep -E '"(case|variant)"' "$OUT/full_$v.log" || tail -30 "$OUT/full_$v.log"
  done
fi
echo "BENCH013_DONE status=$status"
exit $status
