#!/bin/bash
set -euo pipefail
R=/gpfs/projects/ehpc533/users/korbip/hybrid_v019
cd "$R/autoexp_fusion010_v2"
# This marker is written only after numerical, state, batch256 memory, and lm-eval checks.
test -s experiment/010_ambitious_fusion_graphs/results/validated.json
export_ready=/gpfs/scratch/ehpc533/hybrid_nucleus/hf_exports/ambitious_v4_2ph_fp8_5000/.downstream_ready
test -f "$export_ready"
scripts/korbi/mn5_autoexp.sh scripts/run_autoexp.py --yes --no-monitor \
 --monitor-state-dir "$R/pool_state" \
 --pool /gpfs/scratch/ehpc533/hybrid_nucleus/pools/mn5_pool_p129 \
 --config-name experiments/korbi/production/flag_eval_fast_mn5 \
 aux.export=ambitious_v4_2ph_fp8_5000 slurm.sbatch.account=cns144 \
 "job.start_condition={class_name:FileExistsCondition,path:$export_ready,persistent_pass:true}"
