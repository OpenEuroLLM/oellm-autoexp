#!/bin/bash
# End-to-end test of the MULTI-FILE (blend) node-local prefetch on MareNostrum.
#
# Proves: a Megatron blend (data_path = list of prefixes) consumed with the windowed shuffle
# is served entirely from per-file node-local mirrors -- each sub-dataset gets its own mirror +
# lane prefetcher, discovery + blend weights resolve, and training reads 100% locally (0 GPFS).
#
# Run ON MareNostrum, from the repo root, with the venv active:
#   source ~/work/venv/bin/activate
#   bash scripts/test_blend_prefetch.sh                 # set up data, submit, wait, validate
#   WAIT=0 bash scripts/test_blend_prefetch.sh          # submit only, validate later
#   JOB=<id> bash scripts/test_blend_prefetch.sh --check-only   # re-validate a finished job's log
#
# Knobs (env): ITERS BUDGET QOS TIME WAIT  SRC_PREFIX  OUTPUT_BASE
# Exit code: 0 = PASS, 1 = FAIL (so it can gate CI / serve as a regression check).
set -uo pipefail
cd "$(dirname "$0")/.."

# Paths are FIXED here because they must match the YAML config (data_path / PREFETCH_PREFIX
# cannot be passed as hydra CLI overrides -- commas/lists trip sweep-override validation).
CONFIG_NAME="experiments/megatron_marenostrum_blend_test"
BLEND_DIR="/gpfs/scratch/ehpc390/data/blend_test"          # blendA + blendB live here
SRC_PREFIX="${SRC_PREFIX:-/gpfs/scratch/ehpc390/data/cerebrase-SlimPajama-627B/train/small}"
OUTPUT_BASE="${OUTPUT_BASE:-/gpfs/projects/ehpc390/outputs}"
ITERS="${ITERS:-1500}"; BUDGET="${BUDGET:-50}"
QOS="${QOS:-acc_debug}"; TIME="${TIME:-00:30:00}"; WAIT="${WAIT:-1}"
MODE="${1:-}"

red() { printf '\033[31m%s\033[0m\n' "$*"; }
grn() { printf '\033[32m%s\033[0m\n' "$*"; }

# Assert the expected validation signals in a finished run's log; sets exit status via $fail.
validate_log() {
  local LOG="$1" fail=0 DESC
  [ -f "$LOG" ] || { red "[test_blend] FAIL: log $LOG not found"; return 1; }
  echo "[test_blend] validating $LOG"
  check() { if "$@" >/dev/null 2>&1; then grn "  PASS: $DESC"; else red "  FAIL: $DESC"; fail=1; fi; }

  DESC="both sub-datasets get a node-local mirror (blendA.bin + blendB.bin)"
  check bash -c "grep -a 'mirror read-through active' '$LOG' | grep -aq blendA.bin && grep -a 'mirror read-through active' '$LOG' | grep -aq blendB.bin"
  DESC="prefetcher discovered 2 blend files (lanes: files=2)"
  check grep -aq "lanes: files=2" "$LOG"
  DESC="100% node-local mirror hits, 0 GPFS source reads"
  check bash -c "grep -a 'mirror-reader' '$LOG' | grep -aq '100.0% local) source_reads=0'"
  DESC="no source (GPFS) reads in any mirror-reader sample"
  check bash -c "! grep -a 'mirror-reader' '$LOG' | grep -aqE 'source_reads=[1-9]'"
  DESC="training reached 'after training is done'"
  check grep -aq "after training is done" "$LOG"
  DESC="no malformed-.idx assertion / discovery crash"
  check bash -c "! grep -aqE 'AssertionError|no GPTDataset .* description matches' '$LOG'"

  echo
  if [ "$fail" = "0" ]; then grn "[test_blend] ===== PASS ====="; else red "[test_blend] ===== FAIL -- see $LOG ====="; fi
  return $fail
}

# --- check-only: re-validate an existing job's log and exit ----------------------------------
if [ "$MODE" = "--check-only" ]; then
  : "${JOB:?set JOB=<slurm id> for --check-only}"
  validate_log "$OUTPUT_BASE/megatron_marenostrum_blend_test/slurm-$JOB.log"; exit $?
fi

# --- 1. ensure two VALID blend files exist (copies of a known-good single-file prefix) -------
echo "[test_blend] ensuring blend files in $BLEND_DIR (from $SRC_PREFIX)"
if [ ! -f "$SRC_PREFIX.idx" ] || [ ! -f "$SRC_PREFIX.bin" ]; then
  red "[test_blend] FAIL: source prefix $SRC_PREFIX(.bin/.idx) not found"; exit 1
fi
# fail fast if the source .idx is malformed (Megatron's _IndexReader asserts seq_count==doc_idx[-1];
# e.g. chunk_0000.idx in this dataset is broken and crashes the dataset build).
python3 - "$SRC_PREFIX.idx" <<'PY' || exit 1
import struct, sys, numpy as np
p = sys.argv[1]
with open(p, "rb") as f:
    assert f.read(9) == b"MMIDIDX\x00\x00", "bad .idx header"
    struct.unpack("<Q", f.read(8)); code, = struct.unpack("<B", f.read(1))
    sc, = struct.unpack("<Q", f.read(8)); dc, = struct.unpack("<Q", f.read(8)); he = f.tell()
isz = {1:1,2:1,3:2,4:4,5:8,6:8,7:4,8:2}[code]
buf = np.memmap(p, mode="r")
sl = np.frombuffer(buf, dtype=np.int32, count=sc, offset=he)
di = np.frombuffer(buf, dtype=np.int64, count=dc, offset=he+sl.nbytes+sc*8)
assert sl.shape[0] == di[-1], f"malformed .idx: seq_count={sc} != document_indices[-1]={int(di[-1])}"
print(f"[test_blend] source .idx OK (seq_count={sc})")
PY
mkdir -p "$BLEND_DIR"
for F in blendA blendB; do
  if [ ! -f "$BLEND_DIR/$F.idx" ] || [ ! -f "$BLEND_DIR/$F.bin" ]; then
    echo "[test_blend] creating $BLEND_DIR/$F"
    cp -f "$SRC_PREFIX.bin" "$BLEND_DIR/$F.bin"
    cp -f "$SRC_PREFIX.idx" "$BLEND_DIR/$F.idx"
  fi
done

# --- 2. submit the blend prefetch run -------------------------------------------------------
# Small files -> few lanes; OELLM_SHUFFLE_BLOCK must stay << per-lane samples (Me/K).
# OELLM_MIRROR_LOG_EVERY low so the reader emits a hit-rate line within this short run.
echo "[test_blend] submitting (iters=$ITERS budget=${BUDGET}G qos=$QOS)"
SUB=$(CONFIG_NAME="$CONFIG_NAME" \
  OELLM_MIRROR_LOG_EVERY=1000 OELLM_SHUFFLE_LANES=8 OELLM_SHUFFLE_BLOCK=1024 PREFETCH_LANE_BLOCK=4096 \
  bash scripts/run_mirror_test.sh - "$ITERS" "$BUDGET" "$QOS" "$TIME" 2>&1)
JOB=$(echo "$SUB" | grep -aoE "Submitted job \(([0-9]+)\)" | grep -aoE "[0-9]+" | head -1)
[ -n "$JOB" ] || { red "[test_blend] FAIL: submission produced no job id"; echo "$SUB" | tail -20; exit 1; }
LOG="$OUTPUT_BASE/megatron_marenostrum_blend_test/slurm-$JOB.log"
echo "[test_blend] job=$JOB log=$LOG"
if [ "$WAIT" = "0" ]; then
  grn "[test_blend] submitted (WAIT=0). Validate when it finishes with:"
  echo "    JOB=$JOB bash scripts/test_blend_prefetch.sh --check-only"
  exit 0
fi

# --- 3. wait for the job (cache build + ~1500 tiny iters; ~10-12 min wall) -------------------
echo "[test_blend] waiting for job $JOB ..."
for i in $(seq 1 110); do   # ~18 min cap
  ST=$(squeue -j "$JOB" -h -o "%T" 2>/dev/null)
  [ -z "$ST" ] && { echo "[test_blend] job left the queue"; break; }
  grep -aq "after training is done" "$LOG" 2>/dev/null && { echo "[test_blend] training done"; break; }
  grep -aqE "AssertionError|no GPTDataset .* description matches" "$LOG" 2>/dev/null && { echo "[test_blend] early failure detected"; break; }
  sleep 10
done
sleep 3

# --- 4. validate ----------------------------------------------------------------------------
validate_log "$LOG"; exit $?
