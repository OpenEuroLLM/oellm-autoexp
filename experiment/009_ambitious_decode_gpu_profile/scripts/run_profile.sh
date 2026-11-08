#!/bin/bash
set -euo pipefail
source scripts/korbi/vllm/setup_vllm_uv.sh
export OELLM_MHC_IMPL=triton OELLM_LOCAL_EXPERT_LOAD=1 OELLM_ENGRAM_DECODE_IMPL=batched
export OELLM_ENGRAM_TABLES=ram OELLM_ENGRAM_LOOKUP=leader OELLM_ENGRAM_MADVISE=normal
export VLLM_LOGGING_LEVEL=INFO
OUT=$PWD/experiment/009_ambitious_decode_gpu_profile/results
mkdir -p "$OUT"
nsys --version > "$OUT/nsys_version.txt"
nvidia-smi -q > "$OUT/nvidia_start.txt"
nvidia-smi topo -m > "$OUT/topology.txt"
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
 --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true \
 -o "$OUT/smoke" "$INFER_PY" experiment/009_ambitious_decode_gpu_profile/scripts/smoke_cuda.py \
 > "$OUT/smoke.log" 2>&1
nsys export --type=sqlite --force-overwrite=true -o "$OUT/smoke.sqlite" "$OUT/smoke.nsys-rep"
"$INFER_PY" experiment/009_ambitious_decode_gpu_profile/scripts/check_smoke.py "$OUT/smoke.sqlite"
for layout in ep4 tp4; do
 mkdir -p "$OUT/$layout"
 nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=repeat:3 --force-overwrite=true \
  -o "$OUT/$layout/capture" \
  "$INFER_PY" experiment/009_ambitious_decode_gpu_profile/scripts/profile_decode.py \
  --layout "$layout" --out "$OUT/$layout" \
  --model /gpfs/scratch/ehpc533/hybrid_nucleus/hf_exports/ambitious_v4_2ph_fp8_3700 \
  --prompts /gpfs/projects/ehpc533/users/korbip/hybrid_v019/autoexp_infer008_batch/experiment/008_ambitious_inference_speed/results/batching/ep4/prompts.json \
  > "$OUT/$layout/run.log" 2>&1
 for trace in "$OUT/$layout/"*.nsys-rep; do
  nsys export --type=sqlite --force-overwrite=true -o "${trace%.nsys-rep}.sqlite" "$trace"
 done
done
nvidia-smi -q > "$OUT/nvidia_end.txt"
