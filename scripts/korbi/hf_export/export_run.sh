#!/bin/bash
# HF export (oellm_ambitious) of a checkpoint of ANY training-run directory -- export only (no check /
# score), for runs export_main_run_v2.sh does not name, e.g. the ckda_branch post-fit arms
# (production/ckda_branch/<arm>; a branch past its hand-over exports as a standard attention layer,
# export_ambitious.branch_fold). The eval launch's export job calls it for an export with export.run_dir.
#
#   export_run.sh <run dir> <step> <out dir>
#   EXPERT_DTYPE=fp8 (default) | bf16; HF_EXPORTER=parallel (default) | sequential;
#   OELLM_EXPORT_MAX_POSITIONS (export_ambitious.max_positions)
#
# - checkpoint: <run>/ckpt_eval/iter_N, else ckpt/, else ckpt_rolling/
# - architecture: the run's newest current_*.yaml (its resolved config)
# - tokenizer: the config's tokenizer_model
# Writes <out dir> and <out dir>/.export_done; a finished export is kept.
set -euo pipefail
run=${1:?run dir}
step=${2:?step}
out=${3:?out dir}
expert_dtype=${EXPERT_DTYPE:-fp8}
exporter=export_ambitious
[ "${HF_EXPORTER:-parallel}" = parallel ] && exporter=export_parallel
it=$(printf 'iter_%07d' "$step")
ckpt=""
for sub in ckpt_eval ckpt ckpt_rolling; do
  [ -d "$run/$sub/$it" ] && { ckpt=$run/$sub/$it; break; }
done
[ -n "$ckpt" ] || { echo "no checkpoint $it under $run" >&2; exit 1; }
config=$(readlink -f "$(ls -t "$run"/current_*.yaml | head -1)")
tok=$(python - "$config" <<'PY'
import sys
from oellm_autoexp.hf_export.export_q35 import megatron_args
print(megatron_args(sys.argv[1])["tokenizer_model"])
PY
)
echo "[export_run] $run step $step: $ckpt, config $config, tokenizer $tok -> $out"
if [ -f "$out/.export_done" ]; then
  echo "[export_run] $out exists (.export_done)"
  exit 0
fi
rm -rf "$out"
args=()
[ "$expert_dtype" = fp8 ] && args=(--expert-dtype fp8)
python -m "oellm_autoexp.hf_export.$exporter" --ckpt-dir "$ckpt" --config "$config" --out-dir "$out" \
  --tokenizer "$tok" "${args[@]}"
touch "$out/.export_done"
echo "[export_run] done: $out"
