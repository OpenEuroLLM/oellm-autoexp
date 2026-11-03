#!/bin/bash
set -euo pipefail
# Existing v2 conversion/check pipeline; inference optimizations are independent.
bash scripts/korbi/hf_export/export_main_run_v2.sh ambitious_v4 5000
out=/gpfs/scratch/ehpc533/hybrid_nucleus/hf_exports/ambitious_v4_2ph_fp8_5000
test -f "$out/.export_done"
test -f "$out/.check_done"
test -s "$out.score.json"
# Same downstream gate as the prior 3700 export workflow.
touch "$out/.downstream_ready"
