#!/bin/bash
# =============================================================================
# End-to-end test: automatic recovery from an unloadable checkpoint.
# =============================================================================
# Reproduces the 2026-08-19 incident in miniature and asserts that the monitor
# now handles it without human intervention.
#
# WHAT IT PROVES
#   phase 1  train 20 iters on 1 node, saving at 10 and 20
#   phase 2  delete ONE shard from iter_0000020 -> `.metadata` now promises a
#            file that does not exist, exactly like the real iter_0003000 which
#            had a valid .metadata and 927 missing shards
#   phase 3  resubmit with /job: auto_restart_ckptreset and a LIVE monitor
#   phase 4  assert: iter_0000020 was renamed to failed_iter_0000020, and the
#            run resumed from iteration 10 instead of dying in a restart loop
#
# The monitor must be running for the actions to fire, so phase 3 deliberately
# does NOT use --submit-and-exit.
#
# USAGE
#   bash scripts/tests/test_checkpoint_recovery.sh            # full test
#   KEEP_OUTPUT=1 bash scripts/tests/test_checkpoint_recovery.sh
#   SKIP_PHASE1=1 bash scripts/tests/test_checkpoint_recovery.sh   # reuse ckpts
#
# Cost: ~2 x 1 node x <20 min. Safe to run on a login node (it only submits).
# =============================================================================
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CONFIG="experiments/tests/ckpt_recovery_1node"
RUN_NAME="ckpt_recovery_test"
CKPT_DIR="${OUTPUT_DIR:?OUTPUT_DIR must be set}/${RUN_NAME}/checkpoints"
TRACKER="${CKPT_DIR}/latest_checkpointed_iteration.txt"
STATE_DIR="$(mktemp -d)/monitor_state"
LOGDIR="${REPO_ROOT}/dump/ckpt_recovery_test"
mkdir -p "$LOGDIR"

PASS=0
FAIL=0
check() { # check <description> <condition-result>
  if [ "$2" = "0" ]; then echo "   PASS: $1"; PASS=$((PASS + 1));
  else echo "   FAIL: $1"; FAIL=$((FAIL + 1)); fi
}
banner() { echo; echo "=============================================="; echo "$1"; echo "=============================================="; }

show_ckpts() {
  echo "   checkpoint tree:"
  ls -d "$CKPT_DIR"/*iter_* 2>/dev/null | while read -r d; do
    printf "     %-26s .metadata=%-3s shards=%s\n" "$(basename "$d")" \
      "$([ -f "$d/.metadata" ] && echo YES || echo NO)" \
      "$(find "$d" -name '*.distcp' 2>/dev/null | wc -l)"
  done
  echo "   tracker: $(cat "$TRACKER" 2>/dev/null || echo '<none>')"
}

# -----------------------------------------------------------------------------
banner "PHASE 1 — train 20 iterations, save at 10 and 20"
# -----------------------------------------------------------------------------
if [ "${SKIP_PHASE1:-0}" = "1" ]; then
  echo "   SKIP_PHASE1=1, reusing existing checkpoints"
else
  rm -rf "${OUTPUT_DIR}/${RUN_NAME}"
  echo "   submitting (this blocks until the job finishes)..."
  timeout 2400 env PYTHONPATH=. python scripts/run_autoexp.py \
    --config-name "$CONFIG" --monitor-state-dir "$STATE_DIR" \
    > "$LOGDIR/phase1.log" 2>&1
  echo "   exit=$? (see $LOGDIR/phase1.log)"
fi
show_ckpts

[ -d "$CKPT_DIR/iter_0000010" ]; check "iter_0000010 exists" $?
[ -d "$CKPT_DIR/iter_0000020" ]; check "iter_0000020 exists" $?
[ "$(cat "$TRACKER" 2>/dev/null)" = "20" ]; check "tracker points at 20" $?

if [ ! -d "$CKPT_DIR/iter_0000020" ]; then
  echo; echo "ABORT: phase 1 produced no iter_0000020 — nothing to corrupt."
  echo "Check $LOGDIR/phase1.log"; exit 1
fi

# -----------------------------------------------------------------------------
banner "PHASE 2 — corrupt iter_0000020 (delete one shard)"
# -----------------------------------------------------------------------------
VICTIM="$(find "$CKPT_DIR/iter_0000020" -name '*.distcp' | sort | head -1)"
echo "   deleting: $(basename "$VICTIM")"
rm -f "$VICTIM"
SHARDS_LEFT="$(find "$CKPT_DIR/iter_0000020" -name '*.distcp' | wc -l)"
echo "   shards remaining in iter_0000020: $SHARDS_LEFT"
[ -f "$CKPT_DIR/iter_0000020/.metadata" ]; check ".metadata still present (the trap this test is about)" $?

# -----------------------------------------------------------------------------
banner "PHASE 3 — resubmit with auto_restart_ckptreset + live monitor"
# -----------------------------------------------------------------------------
# train_iters is raised so the run has work left after rolling back to 10;
# at 20 it would resume and immediately declare itself finished.
echo "   submitting with job=auto_restart_ckptreset, train_iters=30 ..."
timeout 2400 env PYTHONPATH=. python scripts/run_autoexp.py \
  --config-name "$CONFIG" --monitor-state-dir "$STATE_DIR" \
  job=auto_restart_ckptreset \
  backend.megatron.train_iters=30 \
  > "$LOGDIR/phase3.log" 2>&1
echo "   exit=$? (see $LOGDIR/phase3.log)"
show_ckpts

# -----------------------------------------------------------------------------
banner "PHASE 4 — assertions"
# -----------------------------------------------------------------------------
[ -d "$CKPT_DIR/failed_iter_0000020" ]; check "iter_0000020 was renamed to failed_iter_0000020" $?
[ ! -d "$CKPT_DIR/iter_0000020" ] || [ -d "$CKPT_DIR/failed_iter_0000020" ]
check "no live iter_0000020 left behind" $?

grep -q "checkpoint recovery in" "$LOGDIR/phase3.log" 2>/dev/null
check "monitor logged a checkpoint-recovery action" $?

# The job log is the ground truth for what was actually resumed. NB this config
# does not set job.log_path, so the log lands directly in base_output_dir, not
# in a logs/ subdirectory the way the flagship configs arrange it — search both.
JOBLOG="$(ls -t "${OUTPUT_DIR}/${RUN_NAME}"/logs/slurm-*.log \
                "${OUTPUT_DIR}/${RUN_NAME}"/slurm-*.log 2>/dev/null | head -1)"
if [ -n "$JOBLOG" ]; then
  echo "   newest job log: $(basename "$JOBLOG")"
  grep -qE "successfully loaded checkpoint.*iteration +10|loading checkpoint.*iter_0000010" "$JOBLOG"
  check "run resumed from iteration 10" $?
else
  check "job log found" 1
fi

# Progression is asserted on the checkpoint TREE, not on the log. Two reasons:
# recovery normally takes more than one restart (the generic "Exited with exit
# code 1" pattern can win the race and relaunch before the specific checkpoint
# pattern is flushed to the shared-FS log), so the NEWEST log is often a final
# no-op job that resumed at 30 with nothing left to do; and log_interval has to
# cooperate for iteration lines to exist at all. iter_0000030 existing is direct
# evidence that training ran past the quarantined checkpoint.
[ -d "$CKPT_DIR/iter_0000030" ] && [ -f "$CKPT_DIR/iter_0000030/.metadata" ]
check "training progressed past the quarantined checkpoint (iter_0000030 written)" $?

# Belt and braces: some job in the chain should show iterations beyond 20.
grep -qhE "iteration +(2[1-9]|30)/" "${OUTPUT_DIR}/${RUN_NAME}"/slurm-*.log \
       "${OUTPUT_DIR}/${RUN_NAME}"/logs/slurm-*.log 2>/dev/null
check "some job logged iterations past 20" $?

banner "RESULT: $PASS passed, $FAIL failed"
if [ "${KEEP_OUTPUT:-0}" != "1" ] && [ "$FAIL" = "0" ]; then
  echo "cleaning ${OUTPUT_DIR}/${RUN_NAME} (KEEP_OUTPUT=1 to retain)"
  rm -rf "${OUTPUT_DIR}/${RUN_NAME}"
else
  echo "artifacts kept: ${OUTPUT_DIR}/${RUN_NAME}, logs in $LOGDIR"
fi
[ "$FAIL" = "0" ]
