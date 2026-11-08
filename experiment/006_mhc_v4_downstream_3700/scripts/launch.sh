#!/bin/bash
set -euo pipefail
R=/gpfs/projects/ehpc533/users/korbip/hybrid_v019
cd "$R/autoexp_mhc_eval3700"
P=/gpfs/scratch/ehpc533/hybrid_nucleus/pools/mn5_pool_p132
common=(--yes --no-monitor --monitor-state-dir "$R/pool_state" --pool "$P")
scripts/korbi/mn5_autoexp.sh scripts/run_autoexp.py "${common[@]}" \
  --config-name experiments/korbi/production/hf_export_mn5 \
  aux.model=ambitious_v4 aux.step=3700 aux.expert_dtype=fp8 \
  'aux.bench="bash experiment/006_mhc_v4_downstream_3700/scripts/export.sh"'
scripts/korbi/mn5_autoexp.sh scripts/run_autoexp.py "${common[@]}" \
  --config-name experiments/korbi/production/flag_eval_mn5 \
  aux.export=ambitious_v4_2ph_fp8_3700 \
  'job.start_condition={class_name:FileExistsCondition,path:/gpfs/scratch/ehpc533/hybrid_nucleus/hf_exports/ambitious_v4_2ph_fp8_3700/.downstream_ready,persistent_pass:true}'
