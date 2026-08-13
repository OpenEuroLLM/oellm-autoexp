#!/bin/bash
# Stage a Megatron IndexedDataset prefix (.bin + .idx) to node-local storage.
#
# Runs ONCE PER NODE (launch with `srun --ntasks-per-node=1`). Reads config from
# the environment so it stays brace-free callers and can be driven from the SLURM
# env block. Idempotent: skips files already present at the destination with a
# matching byte size, so re-runs / requeues are cheap.
#
# Required env:
#   STAGE_SRC_PREFIX   GPFS path prefix WITHOUT extension; expects
#                      ${STAGE_SRC_PREFIX}.bin and ${STAGE_SRC_PREFIX}.idx
#   STAGE_LOCAL_DIR    node-local destination directory (must also be bound into
#                      the training container)
#
# Optional env:
#   STAGE_MODE         "whole" (default) copies the full prefix. "window" is
#                      reserved for the Phase-2 windowed controller (not yet
#                      implemented here).
#
# Phase 1 only handles STAGE_MODE=whole. Use it with a SMALL prefix that fits in
# node-local storage to validate the local-read path end to end.

set -euo pipefail

: "${STAGE_SRC_PREFIX:?set STAGE_SRC_PREFIX to the GPFS dataset prefix (no extension)}"
: "${STAGE_LOCAL_DIR:?set STAGE_LOCAL_DIR to the node-local destination directory}"
STAGE_MODE="${STAGE_MODE:-whole}"

host="$(hostname)"
log() { echo "[stage_to_local][${host}] $*"; }

if [[ "${STAGE_MODE}" != "whole" ]]; then
  log "ERROR: STAGE_MODE=${STAGE_MODE} not implemented yet (Phase 1 supports 'whole')."
  exit 2
fi

name="$(basename "${STAGE_SRC_PREFIX}")"
mkdir -p "${STAGE_LOCAL_DIR}"

# Report available space up front so a too-small node fails loudly, not silently.
avail_kb="$(df -Pk "${STAGE_LOCAL_DIR}" | awk 'NR==2 {print $4}')"
log "destination ${STAGE_LOCAL_DIR} has $(( avail_kb / 1024 / 1024 )) GiB free"

copy_one() {
  local ext="$1"
  local src="${STAGE_SRC_PREFIX}.${ext}"
  local dst="${STAGE_LOCAL_DIR}/${name}.${ext}"

  if [[ ! -f "${src}" ]]; then
    log "ERROR: source not found: ${src}"
    exit 1
  fi
  local src_bytes
  src_bytes="$(stat -c %s "${src}")"

  if [[ -f "${dst}" ]]; then
    local dst_bytes
    dst_bytes="$(stat -c %s "${dst}")"
    if [[ "${dst_bytes}" == "${src_bytes}" ]]; then
      log "skip ${name}.${ext} (already staged, $(( src_bytes / 1024 / 1024 )) MiB)"
      return 0
    fi
    log "size mismatch for ${name}.${ext} (have ${dst_bytes}, want ${src_bytes}); re-copying"
  fi

  log "copy ${name}.${ext}: $(( src_bytes / 1024 / 1024 )) MiB  ${src} -> ${dst}"
  local t0 t1 secs
  t0="$(date +%s)"
  # Big sequential read from GPFS, large block size; this is the access pattern
  # GPFS handles well (unlike the random 4 KB page faults of the live mmap).
  cp --preserve=timestamps "${src}" "${dst}.partial"
  mv -f "${dst}.partial" "${dst}"
  t1="$(date +%s)"
  secs=$(( t1 - t0 ))
  [[ "${secs}" -eq 0 ]] && secs=1
  log "done ${name}.${ext} in ${secs}s ($(( src_bytes / 1024 / 1024 / secs )) MiB/s)"

  local dst_bytes
  dst_bytes="$(stat -c %s "${dst}")"
  if [[ "${dst_bytes}" != "${src_bytes}" ]]; then
    log "ERROR: post-copy size mismatch for ${name}.${ext} (${dst_bytes} != ${src_bytes})"
    exit 1
  fi
}

copy_one idx
copy_one bin

log "staged prefix available at ${STAGE_LOCAL_DIR}/${name}"
