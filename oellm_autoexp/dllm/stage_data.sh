#!/usr/bin/env bash
# Stage the tokenized dataset on the node-local disk for one training job, run the command on the
# local copy, and delete the copy when the job ends (D-049: GPFS random reads limit throughput).
# Used as the first part of backend.aux.dist_prefix, for example
#   stage_data.sh ./oellm_autoexp/dllm/train_then_eval.sh torchrun <flags> <launcher> <flags>
# Every argument that is a Megatron dataset prefix (an absolute path with <arg>.bin and <arg>.idx)
# is copied to <local dir>/dllm-data and replaced by the local prefix; data cache paths and all
# other arguments stay. The local directory is the job's own node-local scratch directory:
# DLLM_LOCAL_SCRATCH when set, else /scratch-local/<user>.<job id> or a TMPDIR below /scratch-local.
# Without one, or without free space for the data plus 10 percent, the command runs on the
# original paths. Each node of a multi-node job stages its own copy. The local paths name the job,
# so Megatron's dataset index cache (keyed by the paths) is built again by every staged job.
# `stage_data.sh --check` prints the local directory and its free space, and runs nothing.
set -euo pipefail

local_scratch() {
  if [ -n "${DLLM_LOCAL_SCRATCH:-}" ]; then
    [ -d "$DLLM_LOCAL_SCRATCH" ] && [ -w "$DLLM_LOCAL_SCRATCH" ] || return 1
    echo "$DLLM_LOCAL_SCRATCH"
    return 0
  fi
  local candidate
  for candidate in "/scratch-local/${USER:-unknown}.${SLURM_JOB_ID:-unknown}" "${TMPDIR:-}"; do
    # On nodes without a local disk, /scratch-local lives on GPFS; copying there gains nothing.
    if [[ "$candidate" == /scratch-local/?* ]] && [ -d "$candidate" ] && [ -w "$candidate" ] \
      && [ "$(stat -f -c %T "$candidate")" != "gpfs" ]; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

if [ "${1:-}" = "--check" ]; then
  echo "[stage_data] job ${SLURM_JOB_ID:-none} on $(hostname), TMPDIR=${TMPDIR:-unset}"
  if [ -n "${TMPDIR:-}" ] && [ -d "$TMPDIR" ]; then
    echo "[stage_data] TMPDIR file system: $(stat -f -c %T "$TMPDIR")"
  fi
  if base="$(local_scratch)"; then
    echo "[stage_data] local scratch: $base"
    df -h "$base" | tail -n 1
  else
    echo "[stage_data] local scratch: none found (missing, or on GPFS)"
  fi
  exit 0
fi

if [ "$#" -eq 0 ]; then
  echo "usage: stage_data.sh <command> [args...] | stage_data.sh --check" >&2
  exit 64
fi
ARGS=("$@")
declare -A LOCAL_OF=()
for arg in "${ARGS[@]}"; do
  if [[ "$arg" == /* ]] && [ -f "$arg.bin" ] && [ -f "$arg.idx" ]; then
    LOCAL_OF["$arg"]=""
  fi
done
if [ "${#LOCAL_OF[@]}" -eq 0 ]; then
  echo "[stage_data] no dataset prefix in the command: nothing to stage"
  exec "${ARGS[@]}"
fi
if ! base="$(local_scratch)"; then
  candidate="/scratch-local/${USER:-unknown}.${SLURM_JOB_ID:-unknown}"
  if [ -d "$candidate" ]; then
    detail="$candidate is on $(stat -f -c %T "$candidate")"
  else
    detail="$candidate does not exist"
  fi
  echo "[stage_data] no node-local scratch ($detail; TMPDIR=${TMPDIR:-unset}):" \
    "reading the original paths"
  exec "${ARGS[@]}"
fi

need=0
for prefix in "${!LOCAL_OF[@]}"; do
  need=$(( need + $(stat -L -c %s "$prefix.bin") + $(stat -L -c %s "$prefix.idx") ))
done
avail=$(df --output=avail -B1 "$base" | tail -n 1)
if [ "$avail" -lt $(( need + need / 10 )) ]; then
  echo "[stage_data] $base has $avail bytes free, the data needs $need: reading the original paths"
  exec "${ARGS[@]}"
fi

LOCAL_DIR="$base/dllm-data"
cleanup() {
  rm -rf -- "$LOCAL_DIR"
  echo "[stage_data] removed $LOCAL_DIR"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
mkdir -p "$LOCAL_DIR"

start=$SECONDS
n=0
PIDS=()
for prefix in "${!LOCAL_OF[@]}"; do
  n=$(( n + 1 ))
  target="$LOCAL_DIR/${n}_$(basename "$prefix")"
  LOCAL_OF["$prefix"]="$target"
  cp "$prefix.idx" "$target.idx" &
  PIDS+=($!)
  cp "$prefix.bin" "$target.bin" &
  PIDS+=($!)
done
for pid in "${PIDS[@]}"; do
  wait "$pid"
done
for prefix in "${!LOCAL_OF[@]}"; do
  for ext in bin idx; do
    if [ "$(stat -L -c %s "$prefix.$ext")" != "$(stat -c %s "${LOCAL_OF[$prefix]}.$ext")" ]; then
      echo "[stage_data] size mismatch for $prefix.$ext" >&2
      exit 1
    fi
  done
done
echo "[stage_data] copied ${#LOCAL_OF[@]} prefixes ($need bytes) to $LOCAL_DIR in" \
  "$(( SECONDS - start )) s"

for i in "${!ARGS[@]}"; do
  arg="${ARGS[$i]}"
  if [ -n "$arg" ] && [ -n "${LOCAL_OF[$arg]+set}" ]; then
    ARGS[i]="${LOCAL_OF[$arg]}"
  fi
done
"${ARGS[@]}"
