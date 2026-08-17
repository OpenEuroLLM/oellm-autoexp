#!/usr/bin/env bash
# =============================================================================
# Shared entry point for the OELLM 32B dense flagship runs.
# =============================================================================
# Two jobs:
#
# 1. cd to the repo root before calling run_autoexp.py. SLURM's SLURM_SUBMIT_DIR
#    is the CWD at sbatch time, so `PYTHONPATH=.` only resolves if we submit from
#    the root — and several people have more than one clone of this repo.
#
# 2. Pin the monitor state to ONE SHARED directory instead of the per-clone
#    default `./monitor_state`. The monitor keeps its job records under
#    <state-dir>/<session-id>/ (orchestrator.py:197). With the default, two
#    people babysitting the same chain each get a private session, both poll the
#    same SLURM job, and both act on its log events — so a handover means two
#    monitors racing to submit the next link in the chain. With a shared dir the
#    person taking over ATTACHES to the existing session instead:
#
#      # who is running what
#      ls -lt "$OELLM_MONITOR_STATE_DIR"
#      # take over an existing session (after the previous monitor is stopped)
#      PYTHONPATH=. python scripts/monitor_autoexp.py \
#          --monitor-state-dir "$OELLM_MONITOR_STATE_DIR" --session <session-id>
#
#    umask 002 so group members can write each other's session files.
#
# USAGE
#   scripts/oellm_32b.sh --config-name experiments/oellm_32b_dense/speed_test
#   scripts/oellm_32b.sh --config-name experiments/oellm_32b_dense/warmup_lr_only \
#       -o 'backend.aux.warmup_iters_equiv=1000'
#   scripts/oellm_32b.sh --config-name ... --dry-run
#   scripts/oellm_32b.sh --config-name ... --submit-and-exit   # fire and forget
#
# Override the shared directory with OELLM_MONITOR_STATE_DIR if needed.
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

: "${OELLM_MONITOR_STATE_DIR:=/e/project1/e-sta-openeurollm/production_training/_monitor_state}"

umask 002
mkdir -p "$OELLM_MONITOR_STATE_DIR"

echo "[oellm_32b] repo         : $REPO_ROOT"
echo "[oellm_32b] monitor state: $OELLM_MONITOR_STATE_DIR"

exec env PYTHONPATH=. python scripts/run_autoexp.py \
    --monitor-state-dir "$OELLM_MONITOR_STATE_DIR" \
    "$@"
