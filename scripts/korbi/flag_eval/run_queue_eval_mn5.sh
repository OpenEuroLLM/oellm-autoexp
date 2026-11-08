#!/bin/bash
# Fill an lm-eval work queue (flag_queue.py) for one export and start <workers> queue workers in a
# pool (config flag_eval_queue_mn5): one engine per node, items until the queue is empty. Run on
# the MN5 login node from a synced snapshot.
#
#   bash scripts/korbi/flag_eval/run_queue_eval_mn5.sh <export> <workers> <short|lm-eval> [hydra overrides]
#
#   short    the 21 rows of the default FLAG subset (flag_eval_mn5 aux.flag_names) -> flag_evals_fast
#   lm-eval  all lm-eval rows of the 438-row suite                                  -> flag_evals_oellm
#
# Rows longer than SPLIT_S (default 300 s of engine time, config/eval_tasks/
# flag_evals_engine_times.json) become document parts; done rows are not queued, so a rerun
# queues only what is missing. The workers wait for <export>/.export_done. Extra overrides go to
# run_autoexp, e.g. job.metadata.pool_priority=65 +job.metadata.pool_use_excluded_nodes=true.
set -euo pipefail
export_name=$1 workers=$2 set=$3; shift 3
R=${R:-/gpfs/projects/ehpc533/users/korbip/hybrid_v019}
POOL=${POOL:-/gpfs/scratch/ehpc533/hybrid_nucleus/pools/mn5_pool_p134c}
EXPORTS=${EXPORTS:-/gpfs/scratch/ehpc533/hybrid_nucleus/hf_exports}
PY=${PY:-$R/autoexp_env/bin/python}
cd "$(dirname "$0")/../../.."
case "$set" in
  short) out=${OUT:-/gpfs/scratch/ehpc533/hybrid_nucleus/flag_evals_fast} ;;
  lm-eval) out=${OUT:-/gpfs/scratch/ehpc533/hybrid_nucleus/flag_evals_oellm} ;;
  *) echo "set must be short or lm-eval, got $set" >&2; exit 2 ;;
esac
mapfile -t names < <("$PY" - "$set" <<'PY' | grep '^vllm/'
import sys, yaml
if sys.argv[1] == "short":
    cfg = yaml.safe_load(open("config/experiments/korbi/production/flag_eval_mn5.yaml"))
    print("\n".join(cfg["aux"]["flag_names"].split()))
else:
    rows = yaml.safe_load(open("config/eval_tasks/flag_evals.yaml"))["tasks"]
    print("\n".join(r["name"] for r in rows if r["suite"] == "lm-eval-harness"))
PY
)
[ "${#names[@]}" -gt 0 ] || { echo "no rows for set $set" >&2; exit 2; }
queue=$out/$export_name/queue_lm
PYTHONPATH=. "$PY" scripts/korbi/flag_eval/flag_queue.py init --queue "$queue" \
  --done "$out/$export_name/done" --split-s "${SPLIT_S:-300}" --max-parts "${MAX_PARTS:-8}" --names "${names[@]}"
shards=$(seq -s, 0 $((workers - 1)))
scripts/korbi/mn5_autoexp.sh scripts/run_autoexp.py --no-monitor --monitor-state-dir "$R/pool_state" --pool "$POOL" \
  --config-name experiments/korbi/production/flag_eval_queue_mn5 \
  "aux.export=$export_name" "aux.out=$out" "aux.shards=[$shards]" slurm.sbatch.account=cns144 \
  "job.start_condition={class_name:FileExistsCondition,path:$EXPORTS/$export_name/.export_done,persistent_pass:true}" \
  "$@" --yes
