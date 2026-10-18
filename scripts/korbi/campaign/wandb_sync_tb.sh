#!/bin/bash
# Upload the 3B campaign's runs to wandb from a machine with internet (MN5 compute nodes have
# none):
#
#   scripts/korbi/campaign/wandb_sync_tb.sh PROJECT [ENTITY] [suite-glob ...]
#
#   e.g. scripts/korbi/campaign/wandb_sync_tb.sh hybrid_baby_arch_exp openeurollm-project 'screen2*'
#
# 1. rsyncs each run's wandb/ (offline wandb runs, written since base_train_mn5.yaml enabled
#    wandb in WANDB_MODE=offline) and tb/ (TensorBoard) from MN5 into $CAMPAIGN_TB
#    (default ~/.cache/campaign_3b_tb);
# 2. per run: the offline wandb run(s) if there are any (`wandb sync -p PROJECT [-e ENTITY]
#    <offline-run-*>`: all of Megatron's wandb metrics, the config, the run name / group set at
#    training time); else the TensorBoard logs (`wandb sync --sync-tensorboard`, run id
#    "<suite>__<arm>", job_type = suite) -- for the runs from before wandb was enabled.
# Sync finished suites. Re-syncing a run that already exists in wandb (same id) needs wandb's
# --append (APPEND=1 here; untested) -- or delete the wandb run and sync it again.
# DRY_RUN=1 prints the commands without uploading.
set -euo pipefail
PROJECT=${1:?usage: wandb_sync_tb.sh PROJECT [ENTITY] [suite-glob ...]}
ENTITY=${2:-}
shift $(( $# >= 2 ? 2 : 1 ))
REMOTE=${REMOTE:-marenostrum_new:/gpfs/scratch/ehpc533/users/korbip/hybrid_v019_runs/autoexp/campaign_3b}
TB=${CAMPAIGN_TB:-$HOME/.cache/campaign_3b_tb}
mkdir -p "$TB"
rsync -a --include="*/" --include="tb/***" --include="wandb/***" --exclude="*" --prune-empty-dirs \
    "$REMOTE/" "$TB/" || echo "warning: rsync from $REMOTE failed; syncing the cached logs" >&2
globs=("$@")
[ ${#globs[@]} -eq 0 ] && globs=("*")
ent=()
[ -n "$ENTITY" ] && ent=(-e "$ENTITY")
# wandb <= 0.29 still has --sync-tensorboard (removed in 0.31); the local 0.25 crashes in its
# server-settings introspection. Default: wandb 0.29.0 via uvx (needs tensorboard alongside).
read -r -a WANDB <<< "${WANDB_CMD:-uvx --from wandb==0.29.0 --with tensorboard wandb}"
run() { if [ "${DRY_RUN:-0}" = 1 ]; then echo "$*"; else "$@"; fi; }
for g in "${globs[@]}"; do
    for rundir in "$TB"/$g/*/; do
        [ -d "$rundir" ] || continue
        rundir=${rundir%/}
        arm=$(basename "$rundir")
        suite=$(basename "$(dirname "$rundir")")
        mapfile -t offline < <(find "$rundir" -maxdepth 3 -type d -name "offline-run-*" 2>/dev/null | sort)
        extra=()
        [ "${APPEND:-0}" = 1 ] && extra+=(--append)
        if [ ${#offline[@]} -gt 0 ]; then
            run "${WANDB[@]}" sync -p "$PROJECT" "${ent[@]}" "${extra[@]}" "${offline[@]}"
        elif [ -d "$rundir/tb" ]; then
            run "${WANDB[@]}" sync --sync-tensorboard -p "$PROJECT" "${ent[@]}" --id "${suite}__${arm}" \
                --job_type "$suite" "${extra[@]}" "$rundir/tb"
        fi
    done
done
