#!/bin/bash
# HF export of a main-run checkpoint, then the conversion checks: the convert stage of an
# export -> eval chain (config/experiments/korbi/production/hf_export_mn5.yaml runs it).
#
#   export_main_run.sh <ambitious|conservative> <step> [<export root>]
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
case $model in
  ambitious)    run=$R/production/main_ldm/ambitious_e360_eng80_v3; name=ambitious_v3
                exporter=export_ambitious; valid=$D/v8ml-valid
                # v3's Engram tables (80 GB each in bf16) stay in host memory
                score_args="--gpu-mem-gib 54 --engram-tables-cpu" ;;
  conservative) run=$R/q35_conservative/main_gqa/main; name=q35c_gqa
                exporter=export_q35; valid=$D/v8ml-valid-basetok; score_args="" ;;
  *) echo "unknown model $model" >&2; exit 2 ;;
esac
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
out=$E/${name}_${step}
echo "[export_main_run] $model step $step: $ckpt, config $config, tokenizer $tok -> $out"
if [ ! -f "$out/.export_done" ]; then
  rm -rf "$out"
  python -m oellm_autoexp.hf_export.$exporter --ckpt-dir "$ckpt" --config "$config" --out-dir "$out" --tokenizer "$tok"
  touch "$out/.export_done"
fi
if [ ! -f "$out/.check_done" ]; then
  python -m oellm_autoexp.hf_export.check_export "$out"
  touch "$out/.check_done"
fi
if [ ! -f "$E/${name}_${step}.score.json" ]; then
  mix=$E/datamix_$(basename "$valid")_all.txt
  echo "1.0 $valid/all_text_document" > "$mix"
  # shellcheck disable=SC2086
  python -m oellm_autoexp.hf_export.score_on_flag --model-dir "$out" --datamix "$mix" --num-docs 64 \
    --seq-len 4096 $score_args --out "$E/${name}_${step}.score.json"
fi
# paired check (paired_valid_check): the very samples of Megatron's fixed-validation eval on the 64-sample subset
# of "all" (make_valid_subset; Megatron side: fixedvalid_eval_paired_*), when that subset exists
sub=$E/paired/$(basename "$valid")-sub64/all_text_document
if [ -f "$sub.idx" ] && [ ! -f "$E/${name}_${step}.paired.json" ]; then
  # shellcheck disable=SC2086
  python -m oellm_autoexp.hf_export.paired_valid_check --model-dir "$out" --config "$config" --prefix "$sub" \
    --seq-len 4096 $score_args --out "$E/${name}_${step}.paired.json"
fi
echo "[export_main_run] done: $out"
