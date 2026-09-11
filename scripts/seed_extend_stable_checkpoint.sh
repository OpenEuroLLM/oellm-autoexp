#!/bin/sh
# Seeds a brand-new stable job's checkpoint dir from an older, shorter stable
# run's checkpoints, so training resumes from that run's furthest point
# instead of iteration 0 -- for combos where hp_grid/v2's
# checkpoint_status=extend_stable (a stable run exists but didn't reach the
# new target token budget yet).
#
# Symlinks in EVERY iter_* checkpoint the source has, not just the highest:
# some may still be needed as decay branch points below the resume iter (a
# decay whose own budget is smaller than what the old run already reached
# loads from this same job's checkpoints dir too, at its own lower ckpt_step
# -- if only the resume checkpoint were seeded, that decay would wait forever
# for a checkpoint this job's own training will never produce again, having
# jumped straight past it). The tracker file is set to the MAX iter found, so
# the stable stage itself still resumes forward from the old run's true
# endpoint rather than one of the earlier branch points.
#
# No-op if $1 is empty (non-extend_stable combo) or if $2 already has content
# (already bootstrapped, or has since made its own real training progress --
# never overwrite that). Safe to invoke on every launch attempt, including
# every auto_restart resubmission: idempotent, never rolls back real progress.
#
# Usage: seed_extend_stable_checkpoint.sh <source_checkpoints_dir> <target_checkpoints_dir>
#
# The caller renders this command through an unquoted shell interpolation
# (see the ShellCommandCondition in qwen3_dense_0.9B_ne_lumi_train_v2.yaml --
# quoting isn't an option there since the value round-trips through Hydra's
# CLI override grammar first). When source is the empty string, the shell
# drops that argument entirely instead of passing an empty positional param,
# so only the target dir arrives, in $1. Detect that by argument count rather
# than requiring the caller to pass two tokens.

set -e

if [ "$#" -eq 1 ]; then
    src=""
    ckpt_dir="$1"
else
    src="$1"
    ckpt_dir="$2"
fi

mkdir -p "$ckpt_dir"

if [ -z "$src" ] || [ -n "$(ls -A "$ckpt_dir")" ]; then
    exit 0
fi

max_iter=0
for d in "$src"/iter_*; do
    [ -d "$d" ] || continue
    name=$(basename "$d")
    ln -s "$d" "$ckpt_dir/$name"
    n=$(echo "$name" | sed "s/^iter_0*//")
    [ -z "$n" ] && n=0
    if [ "$n" -gt "$max_iter" ]; then
        max_iter=$n
    fi
done

echo "$max_iter" > "$ckpt_dir/latest_checkpointed_iteration.txt"
