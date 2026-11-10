#!/bin/bash
# End-to-end check of a ckda_branch arm's export (user 2026-11-10: CKLSTM downstream evals) on one node:
#   1. the HF export of <run dir>'s checkpoint <step> (export_run.sh; a branch past its hand-over folds
#      into attention / kronlstm blocks)
#   2. check_export (names / shapes against the HF model)
#   3. paired_valid_check: the HF model's loss on the 64 fixed-validation samples of v8ml-valid-sub64 --
#      compare with Megatron's own loss on them (fixedvalid_eval_ckdabsub64_mn5)
#   4. the vLLM port against the HF export (run_vllm_parity.sh export, tp 1)
#
#   validate_branch_export.sh <run dir> <step> <out dir>
set -euo pipefail
run=${1:?run dir}
step=${2:?step}
out=${3:?out dir}
cd "$(dirname "$0")/../../.."
mkdir -p "$out"
export_dir=$out/export
bash scripts/korbi/hf_export/export_run.sh "$run" "$step" "$export_dir"
python -m oellm_autoexp.hf_export.check_export "$export_dir"
config=$(readlink -f "$(ls -t "$run"/current_*.yaml | head -1)")
python -m oellm_autoexp.hf_export.paired_valid_check --model-dir "$export_dir" --config "$config" \
  --prefix /gpfs/scratch/ehpc533/hybrid_nucleus/hf_exports/paired/v8ml-valid-sub64/all_text_document \
  --seq-len 4096 --out "$out/paired_hf.json"
bash scripts/korbi/vllm/run_vllm_parity.sh export "$export_dir" 1 | tee "$out/vllm_parity.txt"
echo "[validate_branch_export] done: $out"
