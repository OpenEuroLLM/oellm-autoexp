#!/bin/bash
set -euo pipefail
COORD=/gpfs/projects/ehpc533/users/korbip/hybrid_v019/autoexp_infer008_batch/experiment/008_ambitious_inference_speed/results
while [ -e "$COORD/candidate_active" ] && [ ! -e "$COORD/candidate_done" ]; do
 if [ "${SLURM_JOB_END_TIME:-0}" -gt 0 ] && [ "$(date +%s)" -ge "$((SLURM_JOB_END_TIME - 30))" ]; then
  echo 'Allocation deadline approaching; stop waiting for candidate'
  break
 fi
 sleep 5
done
