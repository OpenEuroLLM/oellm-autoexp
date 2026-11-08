#!/bin/bash
set -euo pipefail
RESULT=$PWD/experiment/008_ambitious_inference_speed/results
MODEL=/gpfs/scratch/ehpc533/hybrid_nucleus/hf_exports/ambitious_v4_2ph_fp8_3700
mkdir -p "$RESULT"
date -Is | tee "$RESULT/start.txt"
df -Th /tmp | tee "$RESULT/local_storage.txt"
nvidia-smi --query-gpu=name,memory.total --format=csv | tee "$RESULT/gpus.txt"
source experiment/008_ambitious_inference_speed/scripts/setup_runtime.sh
"$INFER_PY" - <<'PY' | tee "$RESULT/runtime.txt"
import torch,vllm
print('torch',torch.__version__,'vllm',vllm.__version__,'GPUs',torch.cuda.device_count())
print('grouped_mm',getattr(torch,'_grouped_mm',None))
PY
"$INFER_PY" experiment/008_ambitious_inference_speed/scripts/bench_hf_moe.py --model "$MODEL" --out "$RESULT/hf_moe.json" 2>&1 | tee "$RESULT/hf_moe.log"
for layout in ep4 tp4; do
  "$INFER_PY" experiment/008_ambitious_inference_speed/scripts/bench_vllm.py \
    --model "$MODEL" --layout "$layout" --out "$RESULT/$layout" \
    2>&1 | tee "$RESULT/$layout.log"
done
