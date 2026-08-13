#!/usr/bin/env bash
# Resilience recovery test for the Megatron in-process / warm-reserve setup.
#
# Submits the resilient experiment (which injects a simulated node failure via
# FT_SIM_FAULT_DESC), waits for it, then parses the SLURM log to VERIFY that
# recovery actually happened: a checkpoint was saved, the fault fired, and a
# restart RELOADED the checkpoint (resume from disk) rather than starting from
# random. Prints a checklist and exits 0 (PASS) / 1 (FAIL).
#
# Usage (RUN_TAG is an optional positional arg that namespaces everything so
# multiple invocations can run IN PARALLEL without clobbering each other's job
# name / output dir / checkpoint dirs / monitor-state):
#   bash scripts/check_resilience_recovery.sh [RUN_TAG]
#   ssh jupiter "cd ~/work/Projects/oellm-autoexp && bash scripts/check_resilience_recovery.sh r1"
#
# Knobs (env vars):
#   CONFIG          experiment config name (default: experiments/korbi/megatron_moe_resilient_jupiter)
#   TRAIN_ITERS     short run length for the test          (default: 15)
#   SAVE_INTERVAL   non-persistent local ckpt interval     (default: 2)
#   MAX_RESTARTS    inprocess_max_iterations               (default: 5)
#   SEQ_LENGTH      seq length for the test (memory headroom; recovery mechanics
#                   are seq-length-independent; 4096 OOMs at the GH200 edge) (default: 2048)
#   SLURM_TIME      sbatch time limit                      (default: 00:20:00)
#   POLL_SECS / MAX_POLLS   log poll interval / max polls  (default: 45 / 80)
#   MAX_SUBMITS     resubmit attempts on pre-training infra flake (default: 3)
#   OUTPUT_BASE     base output dir            (default: /e/scratch/projectnucleus/poeppel1/output)
#   CKPT_ROOT       base ckpt dir              (default: $OUTPUT_BASE/resilient_demo)
#   MONITOR_BASE    base monitor-state dir (on scratch to dodge /e/project1 quota)
#                                              (default: /e/scratch/projectnucleus/poeppel1/monitor_state)
#   CONTAINER_IMAGE optional apptainer .sif override (else config default)
set -uo pipefail

RUN_TAG="${1:-${RUN_TAG:-}}"

CONFIG="${CONFIG:-experiments/korbi/megatron_moe_resilient_jupiter}"
TRAIN_ITERS="${TRAIN_ITERS:-15}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2}"
MAX_RESTARTS="${MAX_RESTARTS:-5}"
SEQ_LENGTH="${SEQ_LENGTH:-2048}"
SLURM_TIME="${SLURM_TIME:-00:20:00}"
POLL_SECS="${POLL_SECS:-45}"
MAX_POLLS="${MAX_POLLS:-80}"
MAX_SUBMITS="${MAX_SUBMITS:-3}"
OUTPUT_BASE="${OUTPUT_BASE:-/e/scratch/projectnucleus/poeppel1/output}"
CKPT_ROOT="${CKPT_ROOT:-/e/scratch/projectnucleus/poeppel1/output/resilient_demo}"
MONITOR_BASE="${MONITOR_BASE:-/e/scratch/projectnucleus/poeppel1/monitor_state}"

# Per-run namespacing (so parallel runs are isolated).
BASE_NAME="$(basename "${CONFIG}")"
JOB_NAME="${BASE_NAME}${RUN_TAG:+_${RUN_TAG}}"
RUN_CKPT_DIR="${CKPT_ROOT}/${JOB_NAME}/ckpt"
RUN_LOCAL_CKPT_DIR="${CKPT_ROOT}/${JOB_NAME}/local_ckpt"
MONITOR_STATE_DIR="${MONITOR_BASE}/${JOB_NAME}"
TAG="[${RUN_TAG:-default}]"

cd "$(dirname "$0")/.." || { echo "cannot cd to repo root"; exit 2; }
[ -d ~/work/venv ] && source ~/work/venv/bin/activate 2>/dev/null

submit_once() {
  # Submit via run_autoexp with short-test + per-run-namespace overrides; echo the job id.
  local overrides=(
    "job.name=${JOB_NAME}"
    "backend.megatron.train_iters=${TRAIN_ITERS}"
    "backend.megatron.non_persistent_save_interval=${SAVE_INTERVAL}"
    "backend.megatron.inprocess_max_iterations=${MAX_RESTARTS}"
    "backend.megatron.seq_length=${SEQ_LENGTH}"
    "backend.megatron.save=${RUN_CKPT_DIR}"
    "backend.megatron.load=${RUN_CKPT_DIR}"
    "backend.megatron.non_persistent_local_ckpt_dir=${RUN_LOCAL_CKPT_DIR}"
    "slurm.sbatch.time=${SLURM_TIME}"
  )
  [ -n "${CONTAINER_IMAGE:-}" ] && overrides+=("container.image=${CONTAINER_IMAGE}")
  mkdir -p "${MONITOR_STATE_DIR}" 2>/dev/null
  PYTHONPATH=. python scripts/run_autoexp.py \
      --config-name "${CONFIG}" --submit-and-exit \
      --monitor-state-dir "${MONITOR_STATE_DIR}" "${overrides[@]}" 2>&1 \
    | grep -oE "Submitted job \([0-9]+\)" | grep -oE "[0-9]+" | head -1
}

# Clear this run's checkpoint dirs so each test is a clean cold start.
clear_ckpts() {
  for d in "${RUN_CKPT_DIR}" "${RUN_LOCAL_CKPT_DIR}"; do
    case "${d}" in /e/scratch/*|*/output/*)   # safety: only scratch/output paths
      rm -rf "${d}" && mkdir -p "${d}" && echo "${TAG}   cleared ckpt dir: ${d}" ;;
    esac
  done
}

log_path_for() { echo "${OUTPUT_BASE}/${JOB_NAME}/slurm-$1.log"; }

# --- Recovery analysis: prints a checklist; sets global PASS=0/1. ------------
analyze() {  # $1 = log file
  local LOG="$1"
  [ -f "$LOG" ] || { echo "${TAG}   log not found: $LOG"; PASS=1; return; }
  local faults saves coldstarts loads nccl oom finished
  faults=$(grep -cE "FT: Simulating fault" "$LOG")
  saves=$(grep -cE "Successfully saved local checkpoint" "$LOG")
  coldstarts=$(grep -cE "will not load any checkpoints and will start from random" "$LOG")
  loads=$(grep -cE "sharded_state_dict metadata loaded from the checkpoint|loaded local checkpoint" "$LOG")
  nccl=$(grep -cE "NCCL communicator was aborted|ncclInternalError" "$LOG")
  oom=$(grep -cE "out of memory|OutOfMemory" "$LOG")
  finished=$(grep -cE "after training is done" "$LOG")
  echo "${TAG}   fault_injected=${faults} saves=${saves} coldstarts=${coldstarts} reloads=${loads} finished=${finished} nccl_aborts=${nccl} oom=${oom}"
  if [ "${faults}" -ge 1 ] && [ "${saves}" -ge 1 ] && [ "${loads}" -ge 1 ]; then PASS=0; else PASS=1; fi
}

flaked_before_training() {  # $1 = log
  local LOG="$1"
  [ -f "$LOG" ] || return 1
  grep -qE "DistStoreError|clients joined" "$LOG" && ! grep -qE "iteration +1/" "$LOG"
}

PASS=1
for attempt in $(seq 1 "${MAX_SUBMITS}"); do
  echo "${TAG} === submit ${attempt}/${MAX_SUBMITS} (job_name=${JOB_NAME}, iters=${TRAIN_ITERS}, seq=${SEQ_LENGTH}) ==="
  clear_ckpts
  JOBID="$(submit_once)"
  if [ -z "${JOBID}" ]; then echo "${TAG} FAILED to submit (no job id)"; exit 2; fi
  LOG="$(log_path_for "${JOBID}")"
  echo "${TAG}   job ${JOBID}  log ${LOG}"
  for i in $(seq 1 "${MAX_POLLS}"); do
    state="$(squeue -j "${JOBID}" -h -o %T 2>/dev/null | tr -d '[:space:]')"
    [ -z "${state}" ] && { echo "${TAG}   [poll $i] job ${JOBID} finished"; break; }
    sleep "${POLL_SECS}"
  done
  analyze "${LOG}"
  if [ "${PASS}" -eq 0 ]; then
    echo "${TAG} RESULT: PASS (job ${JOBID}) — fault injected and training resumed from a reloaded checkpoint."
    exit 0
  fi
  if flaked_before_training "${LOG}"; then
    echo "${TAG}   transient infra flake (rendezvous/store timeout before training) — retrying."
    continue
  fi
  echo "${TAG} RESULT: FAIL (job ${JOBID}) — recovery criteria not met."
  exit 1
done
echo "${TAG} RESULT: FAIL — exhausted ${MAX_SUBMITS} submit attempts (repeated infra flakes)."
exit 1
