#!/bin/bash
set -euo pipefail
export EXPERT_DTYPE=fp8
bash scripts/korbi/hf_export/export_main_run.sh ambitious_v4 3700
out=/gpfs/scratch/ehpc533/hybrid_nucleus/hf_exports/ambitious_v4_2ph_fp8_3700
# Gate evaluation on the complete export/check/validation-score pipeline.
test -f "$out/.export_done"
test -f "$out/.check_done"
test -s "$out.score.json"
touch "$out/.downstream_ready"
