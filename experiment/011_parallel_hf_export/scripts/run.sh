#!/bin/bash
# 011: parallel HF export (oellm_autoexp.hf_export.export_parallel) of ambitious v4_2ph.
#   A: checkpoint 5800 -> a fresh root; byte comparison against the existing (v2, bit-verified) 5800 export
#   B: checkpoint 5900 (ckpt_rolling) -> the production export root, check_export
# Each export timed end to end (python start, metadata, conversion, index, model code).
set -euo pipefail
R=/gpfs/scratch/ehpc533/hybrid_nucleus
run=$R/production/main_mhc4/ambitious_e360_eng80_v4_2ph
config=$(readlink -f "$(ls -t "$run"/current_*.yaml | head -1)")
tok=/gpfs/projects/ehpc533/users/korbip/data/tokenizers/tokenizer-256k-agent
RES=$PWD/experiment/011_parallel_hf_export/results/${RESULT_TAG:-run1}
CROOT=$R/hf_exports_par011
mkdir -p "$RES" "$CROOT"
export PYTHONUNBUFFERED=1
echo "host $(hostname) cpus $(nproc) $(free -g | awk '/Mem/{print $2" GB"}')" | tee "$RES/node.txt"
WORKERS=${WORKERS:-32} THREADS=${THREADS:-4}
export_one() {  # step ckpt out tag
  rm -rf "$3"
  python scripts/korbi/hf_export/time_command.py "$RES/export_$4.time.json" \
    python -m oellm_autoexp.hf_export.export_parallel --ckpt-dir "$2" --config "$config" --out-dir "$3" \
      --tokenizer "$tok" --expert-dtype fp8 --workers "$WORKERS" --read-threads "$THREADS" 2>&1 | tee "$RES/export_$4.log"
}
if [ "${SKIP_A:-0}" != 1 ]; then
  export_one 5800 "$run/ckpt_eval/iter_0005800" "$CROOT/ambitious_v4_2ph_fp8_5800" 5800
  python experiment/011_parallel_hf_export/scripts/compare_raw.py "$R/hf_exports/ambitious_v4_2ph_fp8_5800" \
    "$CROOT/ambitious_v4_2ph_fp8_5800" --out "$RES/compare_5800.json" 2>&1 | tee "$RES/compare_5800.log"
fi
out=$R/hf_exports/ambitious_v4_2ph_fp8_5900
if [ "${SKIP_B:-0}" != 1 ] && [ ! -f "$out/.export_done" ]; then
  export_one 5900 "$run/ckpt_rolling/iter_0005900" "$out" 5900
  touch "$out/.export_done"
  python -m oellm_autoexp.hf_export.check_export "$out" 2>&1 | tail -3 | tee "$RES/check_5900.log"
  touch "$out/.check_done"
fi
echo PAR011_DONE
