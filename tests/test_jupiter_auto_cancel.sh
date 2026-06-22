#!/bin/bash
# Test that the `job=auto_cancel` config treats a SLURM TIMEOUT as "finished":
# such a job must be removed from monitoring (the dependency chain handles the
# restart), instead of being restarted by the monitor itself.
#
# It submits two chained autoexp runs with a short 2 min wall-clock limit so the
# jobs hit the SLURM TIMEOUT state. The second run (repeat 2) depends on the
# first run's SLURM job via `afterany`, exercising the dependency chaining.
#
# NOTE: run_autoexp.py logs to STDERR (logging.basicConfig), so we must redirect
# both stdout and stderr (`> file 2>&1`) to capture the submitted SLURM job id.
set -uo pipefail

cd "$(dirname "$0")/.."

COMMON_ARGS=(
  --config-name experiments/megatron_jupiter_speed_test
  slurm.sbatch.time=00:02:00
  "container.image=$CONTAINER_CACHE_DIR/MegatronTraining-JUPITER_pt2512_aarch64_202602231451.sif"
  slurm.sbatch.nodes=1
  backend.megatron.data_path.0=/e/data1/datasets/products/openeurollm/pretokenized/neox/copy_nemotron_downsampled_15p/high-all
  job=auto_cancel
)

# First chain (repeat 1). Capture stdout+stderr so the submitted SLURM job id is logged.
LOGLEVEL=INFO PYTHONPATH=. python3 scripts/run_autoexp.py --repeat 1 --chain \
  "${COMMON_ARGS[@]}" '++slurm.name=${oc.timestring:}' \
  > test_jupiter_auto_cancel1.log 2>&1 &
RUN1_PID=$!

# Wait for the first run to submit its SLURM job, then grab the job id from its log.
# The orchestrator logs: "chain_submit_jobs: submitted '<job>' as Slurm job <ID> (...)".
SLURMID=""
for _ in $(seq 1 60); do
  SLURMID=$(grep -oE "as Slurm job [0-9]+" test_jupiter_auto_cancel1.log 2>/dev/null \
    | grep -oE "[0-9]+" | head -1)
  [ -n "$SLURMID" ] && break
  # Bail out early if the first run died before submitting.
  kill -0 "$RUN1_PID" 2>/dev/null || break
  sleep 2
done

if [ -z "$SLURMID" ]; then
  echo "ERROR: could not determine SLURM job id from test_jupiter_auto_cancel1.log" >&2
  echo "----- test_jupiter_auto_cancel1.log -----" >&2
  cat test_jupiter_auto_cancel1.log >&2
  kill "$RUN1_PID" 2>/dev/null
  exit 1
fi
echo "First chain submitted as SLURM job $SLURMID"

# Second chain (repeat 2); its first job depends on the first chain's job.
LOGLEVEL=INFO PYTHONPATH=. python3 scripts/run_autoexp.py --repeat 2 --chain \
  "${COMMON_ARGS[@]}" '++slurm.name=${oc.timestring:}' \
  ++slurm.sbatch.dependency="afterany:$SLURMID" \
  > test_jupiter_auto_cancel2.log 2>&1 &

wait
