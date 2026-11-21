#!/bin/bash
# HF export of a t16 arm (t16_s3_sparse_fixed_mn5) with its sparse attention in the model: export_ambitious at the
# arm's step (default 3000; config.sparse_attention + the trained selector weights, oellm_ambitious/
# sparse_attention.py), the current model code, check_export. One MN5 node:
#
#   t16_hf_export_mn5.sh <arm> [step]
#
# EXPORT_SUITE (default t16_s3_sparse_fixed) / EXPORT_PREFIX (default t16_s3): another suite's arms (production/
# <suite>/<arm>) into hf_exports/<prefix>_<arm>_<step> (the prefix t16_long_evals_mn5.sh reads as EVAL_EXPORT_PREFIX).
set -euo pipefail
arm=$1; step=${2:-3000}
R=${HYBRID_OUTPUT_ROOT:-/gpfs/scratch/ehpc533/hybrid_nucleus}
run=$R/production/${EXPORT_SUITE:-t16_s3_sparse_fixed}/$arm
ckpt=$run/ckpt/iter_$(printf %07d "$step")
[ -d "$ckpt" ] || { echo "no step-$step checkpoint under $run" >&2; exit 1; }
out=$R/hf_exports/${EXPORT_PREFIX:-t16_s3}_${arm}_${step}
config=$(readlink -f "$(ls -t "$run"/current_*.yaml | head -1)")
tok=$(python - "$config" <<'PY'
import sys
from oellm_autoexp.hf_export.export_q35 import megatron_args
print(megatron_args(sys.argv[1])["tokenizer_model"])
PY
)
echo "[t16_hf_export] $arm: $ckpt ($config, $tok) -> $out"
if [ ! -f "$out/.export_done" ]; then
  rm -rf "$out"
  python -m oellm_autoexp.hf_export.export_ambitious --ckpt-dir "$ckpt" --config "$config" --out-dir "$out" \
    --tokenizer "$tok"
  touch "$out/.export_done"
fi
python -m oellm_autoexp.hf_export.refresh_export "$out"
python - "$out" <<'PY'
import json, sys
cfg = json.load(open(f"{sys.argv[1]}/config.json"))
print("[t16_hf_export] sparse_attention:", json.dumps(cfg.get("sparse_attention")))
PY
[ -f "$out/.check_done" ] || { python -m oellm_autoexp.hf_export.check_export "$out" && touch "$out/.check_done"; }
echo "[t16_hf_export] $arm done: $out"
