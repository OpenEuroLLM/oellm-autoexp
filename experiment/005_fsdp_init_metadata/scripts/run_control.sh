#!/bin/bash
set -euo pipefail
R=/gpfs/projects/ehpc533/users/korbip/hybrid_v019
cd "$R/autoexp_rowmeta2"
scripts/korbi/mn5_autoexp.sh scripts/run_autoexp.py --yes --no-monitor \
  --monitor-state-dir "$R/pool_state" \
  --pool /gpfs/scratch/ehpc533/hybrid_nucleus/pools/mn5_pool_o134e \
  --config-name experiments/korbi/production/hybrid_ambitious_resume_test_mn5_pool \
  aux.phase=resume aux.suite=fsdp_meta005 'aux.only=[cont]' \
  job.name=fsdp_meta005_cont_baseline \
  job.base_output_dir=/gpfs/scratch/ehpc533/hybrid_nucleus/resume_tests/fsdp_meta005/cont_baseline \
  backend.megatron.exit_interval=null backend.megatron.train_iters=30
