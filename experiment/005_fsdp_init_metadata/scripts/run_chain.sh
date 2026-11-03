#!/bin/bash
set -euo pipefail
R=/gpfs/projects/ehpc533/users/korbip/hybrid_v019
BASE=$R/autoexp_rowmeta2
CAND=$R/autoexp_fsdpmeta005
PD=/gpfs/scratch/ehpc533/hybrid_nucleus/pools/mn5_pool_o134e
B=/gpfs/scratch/ehpc533/hybrid_nucleus/resume_tests/fsdp_meta005
RESULT=$CAND/experiment/005_fsdp_init_metadata/results
mkdir -p "$RESULT"
# A reused destination would invalidate the restore comparison.
[ ! -e "$B" ] || { echo 'Refusing reused fsdp_meta005 output'; exit 1; }
wait_job() {
    "$R/autoexp_env/bin/python" - "$PD" "$1" <<'PY'
import pathlib,json,sys,time
p=pathlib.Path(sys.argv[1]);name=sys.argv[2]
for _ in range(240):
    for r in (p/'runs').glob('*'):
        f=r/'run.json'
        if not f.exists():continue
        if json.loads(f.read_text()).get('name')!=name:continue
        if (r/'rc').exists():
            rc=(r/'rc').read_text().strip();print(name,r.name,rc,flush=True)
            if rc!='0 completed':sys.exit(1)
            sys.exit(0)
    time.sleep(15)
raise SystemExit('Timed out waiting for '+name)
PY
}
cd "$CAND"
scripts/korbi/mn5_autoexp.sh scripts/run_autoexp.py --yes --no-monitor \
  --monitor-state-dir "$R/pool_state" --pool "$PD" \
  --config-name experiments/korbi/hybrid_speed/kernel_tests_mn5 \
  aux.name=fsdpmeta005 aux.tests='' \
  'aux.bench="bash experiment/005_fsdp_init_metadata/scripts/run_unit.sh"' \
  slurm.sbatch.qos=acc_xehpc
wait_job kernel_tests_fsdpmeta005
for phase in save resume; do
    if [ "$phase" = save ]; then cd "$BASE"; arm=ref; else cd "$CAND"; arm=cont; fi
    scripts/korbi/mn5_autoexp.sh scripts/run_autoexp.py --yes --no-monitor \
      --monitor-state-dir "$R/pool_state" --pool "$PD" \
      --config-name experiments/korbi/production/hybrid_ambitious_resume_test_mn5_pool \
      aux.phase="$phase" aux.suite=fsdp_meta005 "aux.only=[$arm]" \
      backend.megatron.exit_interval=null backend.megatron.train_iters=30
    wait_job "fsdp_meta005_$arm"
    [ "$(cat "$B/ref/ckpt_rolling/latest_checkpointed_iteration.txt")" = 20 ]
done
cd "$CAND"
"$R/autoexp_env/bin/python" experiment/005_fsdp_init_metadata/scripts/compare.py "$B" | tee "$RESULT/comparison.log"
cp "$B/comparison.json" "$RESULT/comparison.json"
echo 'FSDP_META005_VALIDATION_PASSED'
