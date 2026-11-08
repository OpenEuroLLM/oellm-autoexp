#!/bin/bash
# v2: same export/check inputs, with concurrent validation on disjoint GPU pairs.
# HF export of a main-run checkpoint, then the conversion checks: the convert stage of an
# export -> eval chain (config/experiments/korbi/production/hf_export_mn5.yaml runs it).
#
#   export_main_run.sh <ambitious|ambitious_v4|ambitious_v4_ldm|conservative> <step> [<export root>]
#   EXPERT_DTYPE=fp8: the ambitious runs' routed experts as the run's fsdp_fp8_param_gather tiles (E4M3,
#   1 x 128 fp32 scales), the export named <name>_<step>_fp8
#
# - checkpoint: <run>/ckpt_eval/iter_N, else ckpt/, else ckpt_rolling/ (link it into ckpt_eval
#   first via scripts/korbi/eval_marks/marks.tsv when it is a rolling one)
# - architecture: the run's current resolved config (the architecture is fixed within a run)
# - tokenizer: the config's tokenizer_model
# - check_export (names / shapes), then score_on_flag on 64 documents of the fixed validation set
#   "all" in the run's own tokenization (v8ml-valid, agent tokenizer; v8ml-valid-basetok, base)
# - paired_valid_check against Megatron's own fixed-validation loss on a 64-sample subset, if built
# Writes <export root>/<name>_<step>/ with .export_done / .check_done, <name>_<step>.score.json / .paired.json;
# finished parts are skipped on a rerun.
set -euo pipefail
model=$1; step=$2
R=${HYBRID_OUTPUT_ROOT:-/gpfs/scratch/ehpc533/hybrid_nucleus}
D=/gpfs/projects/ehpc533/users/korbip/data
E=${3:-$R/hf_exports}
expert_dtype=${EXPERT_DTYPE:-bf16}
# the ambitious runs' Engram tables (80 GB each in bf16) stay in host memory when scoring
amb_score="--gpu-mem-gib 54 --engram-tables-cpu"
case $model in
  ambitious)        run=$R/production/main_ldm/ambitious_e360_eng80_v3; name=ambitious_v3
                    exporter=export_ambitious; score_args=$amb_score ;;
  ambitious_v4)     run=$R/production/main_mhc4/ambitious_e360_eng80_v4_2ph; name=ambitious_v4_2ph
                    exporter=export_ambitious; score_args=$amb_score ;;
  ambitious_v4_ldm) run=$R/production/main_ldm/ambitious_e360_eng80_v4_ldm; name=ambitious_v4_ldm
                    exporter=export_ambitious; score_args=$amb_score ;;
  conservative)     run=$R/q35_conservative/main_gqa/main; name=q35c_gqa
                    exporter=export_q35; score_args="" ;;
  *) echo "unknown model $model" >&2; exit 2 ;;
esac
export_args=""
if [ "$expert_dtype" = fp8 ]; then
  [ "$exporter" = export_ambitious ] || { echo "fp8 experts: the ambitious runs only" >&2; exit 2; }
  export_args="--expert-dtype fp8"; name=${name}_fp8
fi
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
# the fixed validation sets in the run's own tokenization
case $(basename "$tok") in
  tokenizer-256k-agent) valid=$D/v8ml-valid ;;
  tokenizer-256k)       valid=$D/v8ml-valid-basetok ;;
  *) echo "no validation set in the tokenization of $tok" >&2; exit 2 ;;
esac
out=$E/${name}_${step}
echo "[export_main_run] $model step $step: $ckpt, config $config, tokenizer $tok -> $out"
if [ ! -f "$out/.export_done" ]; then
  rm -rf "$out"
  # shellcheck disable=SC2086
  python -m oellm_autoexp.hf_export.$exporter --ckpt-dir "$ckpt" --config "$config" --out-dir "$out" --tokenizer "$tok" $export_args
  touch "$out/.export_done"
fi
# the model code from this checkout (an export from an earlier run of this stage keeps its weights)
python -m oellm_autoexp.hf_export.refresh_export "$out"
if [ ! -f "$out/.check_done" ]; then
  python -m oellm_autoexp.hf_export.check_export "$out"
  touch "$out/.check_done"
fi
score_export() {
  if [ ! -f "$E/${name}_${step}.score.json" ]; then
    mix=$E/datamix_$(basename "$valid")_all.txt
    echo "1.0 $valid/all_text_document" > "$mix"
    # shellcheck disable=SC2086
    python experiment/007_fast_hf_export/scripts/time_command.py "$E/${name}_${step}.score.time.json" \
      python -m oellm_autoexp.hf_export.score_on_flag --model-dir "$out" --datamix "$mix" --num-docs 64 \
      --seq-len 4096 $score_args --out "$E/${name}_${step}.score.json"
  fi
}
paired_export() {
  sub=$E/paired/$(basename "$valid")-sub64/all_text_document
  if [ -f "$sub.idx" ] && [ ! -f "$E/${name}_${step}.paired.json" ]; then
    # shellcheck disable=SC2086
    python experiment/007_fast_hf_export/scripts/time_command.py "$E/${name}_${step}.paired.time.json" \
      python -m oellm_autoexp.hf_export.paired_valid_check --model-dir "$out" --config "$config" --prefix "$sub" \
      --seq-len 4096 $score_args --out "$E/${name}_${step}.paired.json"
  fi
}
# This experimental v2 is scoped to the architecture tested below. Both
# processes use the same tensors and arithmetic as v1; only GPU placement changes.
[ "$model" = ambitious_v4 ] || { echo "v2 currently tested for ambitious_v4 only" >&2; exit 2; }
python -c 'import torch; assert torch.cuda.device_count() == 4, "v2 requires exactly four visible GPUs"'
IFS=',' read -r -a validation_gpus <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
[ "${#validation_gpus[@]}" = 4 ] || { echo "expected four CUDA_VISIBLE_DEVICES entries" >&2; exit 2; }
CUDA_VISIBLE_DEVICES="${validation_gpus[0]},${validation_gpus[1]}" score_export &
score_pid=$!
CUDA_VISIBLE_DEVICES="${validation_gpus[2]},${validation_gpus[3]}" paired_export &
paired_pid=$!
status=0
wait "$score_pid" || status=$?
wait "$paired_pid" || status=$?
[ "$status" = 0 ] || exit "$status"
echo "[export_main_run_v2] done: $out"
