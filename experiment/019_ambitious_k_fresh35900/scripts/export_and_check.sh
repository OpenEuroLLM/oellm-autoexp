#!/bin/bash
set -euo pipefail
run=${1:?run directory}
step=${2:?step}
out=${3:?export directory}
export HF_EXPORTER=parallel HF_EXPORT_WORKERS=32 HF_EXPORT_READ_THREADS=8 EXPERT_DTYPE=fp8
bash scripts/korbi/hf_export/export_run.sh "$run" "$step" "$out"
python3 -m oellm_autoexp.hf_export.check_export "$out"
view=/gpfs/scratch/ehpc533/hybrid_nucleus/views/identity/$(basename "$out")
if [ ! -d "$view" ]; then
  python3 submodules/oellm-eval/containers/prepare_base_model_view.py --source "$out" --out "$view" --template identity
fi
bash scripts/korbi/hf_export/refresh_code.sh "$out" "$view"
source scripts/korbi/vllm/setup_vllm_uv.sh
"$INFER_PY" experiment/019_ambitious_k_fresh35900/scripts/check_export_parity.py "$out"
touch "$out/.downstream_ready"
echo "[K-readout] step $step passed export and parity checks"
