#!/bin/bash
set -euo pipefail
export HF_EXPORT_READER=fast HF_EXPORT_READ_THREADS=8 EXPERT_DTYPE=fp8
export PYTHONPATH="${PYTHONPATH:-}:/gpfs/projects/ehpc533/users/korbip/hybrid_v019/pylibs_hfx"
BASE=/gpfs/scratch/ehpc533/hybrid_nucleus/hf_exports/ambitious_v4_2ph_fp8_3700
CROOT=/gpfs/scratch/ehpc533/hybrid_nucleus/hf_exports_fast007parallel
RESULT=$PWD/experiment/007_fast_hf_export/results/parallel
mkdir -p "$CROOT" "$RESULT"
[ -e "$CROOT/paired" ] || ln -s /gpfs/scratch/ehpc533/hybrid_nucleus/hf_exports/paired "$CROOT/paired"
# Check dependencies and two-GPU capacity before doing another full export.
CUDA_VISIBLE_DEVICES=0,1 python - "$BASE" <<'PY'
import sys
import accelerate
import torch
from transformers import AutoConfig
from oellm_autoexp.hf_export.score_on_flag import layer_device_map
assert torch.cuda.device_count() == 2
layer_device_map(AutoConfig.from_pretrained(sys.argv[1], trust_remote_code=True), 54, True)
print('Two-GPU validation capacity preflight passed', flush=True)
PY
python experiment/007_fast_hf_export/scripts/time_command.py "$RESULT/pipeline_time.json" \
  bash experiment/007_fast_hf_export/scripts/export_main_run_v2.sh ambitious_v4 3700 "$CROOT" \
  2>&1 | tee "$RESULT/pipeline.log"
python experiment/007_fast_hf_export/scripts/compare_exports.py "$BASE" "$CROOT/ambitious_v4_2ph_fp8_3700" \
  --out "$RESULT/value_comparison.json" 2>&1 | tee "$RESULT/value_comparison.log"
echo FAST_EXPORT007_PARALLEL_VERIFIED
