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

set -e

src="$1"
ckpt_dir="$2"

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
