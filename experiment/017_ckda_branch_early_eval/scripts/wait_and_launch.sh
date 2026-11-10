#!/usr/bin/env bash
# Add only the 23450 eval; never change existing pool runs or Slurm jobs.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec 9>"$ROOT/results/launcher.lock"
flock -n 9 || exit 0
[[ ! -f "$ROOT/results/submitted" ]] || exit 0

echo "$(date -Is) waiting for bgqar checkpoint 23450"
while true; do
  if ssh -T -o BatchMode=yes -o ConnectTimeout=20 marenostrum_new 'python3 -' <<'PY'
from pathlib import Path
import sys
base = Path('/gpfs/scratch/ehpc533/hybrid_nucleus/production/ckda_branch/bgqar/ckpt')
latest = base / 'latest_checkpointed_iteration.txt'
ready = (base / 'iter_0023450/.metadata').is_file()
ready = ready and latest.is_file() and int(latest.read_text().strip()) >= 23450
if ready:
    sys.exit(0)
rc = Path('/gpfs/scratch/ehpc533/hybrid_nucleus/pools/mn5_pool_o260c/runs/mn5-pool-o260c.0041/rc')
if rc.exists():
    print('Training ended without a committed 23450 checkpoint; inspect before launching.')
    sys.exit(20)
sys.exit(10)
PY
  then
    break
  else
    status=$?
    if [[ "$status" == 20 ]]; then exit 20; fi
    if [[ "$status" != 10 ]]; then
      echo "$(date -Is) checkpoint probe returned $status; retrying"
    fi
  fi
  sleep 60
done

echo "$(date -Is) checkpoint committed; launching export and focused eval"
# Do not retry a failed submission automatically: inspect for partial enqueues first.
ssh -T -o BatchMode=yes -o ConnectTimeout=20 marenostrum_new '
  R=/gpfs/projects/ehpc533/users/korbip/hybrid_v019
  cd "$R/autoexp_ckdab27"
  "$R/autoexp_env/bin/python" scripts/evals/launch.py config/evals/ckdab_fast_mn5.yaml \
    --step 23450 --export ckdab_bgqar_fp8_23450 \
    --set export.run_dir=/gpfs/scratch/ehpc533/hybrid_nucleus/production/ckda_branch/bgqar
'
date -Is > "$ROOT/results/submitted"
echo "$(date -Is) submission finished"
