#!/bin/bash
# Keep the equal-token checkpoints of the main runs (marks.tsv): once a run's checkpoint at a mark
# step is complete (its tracker has reached the step), hard-link it into <run dir>/ckpt_eval/iter_N
# -- no extra space while the original exists, and it survives the rolling checkpoint's deletion.
# One pass per call; sync_loop.sh calls it. By default it runs on the workstation and acts on MN5
# over ssh; LOCAL=1 runs the pass in place (the machine that holds the run dirs), for the loop that
# runs on MN5 itself. ONLY=<glob> restricts the pass to matching marks names (e.g. 'ambitious_v4_2ph*').
#
# Megatron writes latest_checkpointed_iteration.txt from the async-save finalize callback, i.e.
# after the shards are on disk, so tracker >= step means the save of that step is complete. A
# persistent save stays in ckpt/ until the next one (save_interval = 1000 steps, ~8 h at 28 s/step);
# a rolling save stays in ckpt_rolling/ for the last two (non_persistent_save_interval = 100 steps),
# so poll well inside those windows.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
MARKS="${MARKS:-$HERE/marks.tsv}"
HOST="${HOST:-marenostrum_new}"
LOCAL="${LOCAL:-0}"
ONLY="${ONLY:-}"

# The linking pass itself; args: name dir steps.
BODY="$(mktemp)"
trap 'rm -f "$BODY"' EXIT
cat > "$BODY" <<'REMOTE'
set -u
name=$1 dir=$2 steps=$3 out="$dir/ckpt_eval"
for s in $(echo "$steps" | tr ',' ' '); do
    it=$(printf 'iter_%07d' "$s")
    [ -e "$out/$it" ] && continue
    for sub in ckpt ckpt_rolling; do
        src="$dir/$sub/$it"; tracker="$dir/$sub/latest_checkpointed_iteration.txt"
        [ -d "$src" ] && [ -f "$tracker" ] || continue
        latest=$(cat "$tracker" 2>/dev/null | tr -dc 0-9)
        [ -n "$latest" ] && [ "$latest" -ge "$s" ] || continue   # the save of step s is complete
        mkdir -p "$out"
        if cp -al "$src" "$out/$it.tmp" && mv "$out/$it.tmp" "$out/$it"; then
            echo "$name: kept step $s ($sub) -> $out/$it"
        else
            rm -rf "$out/$it.tmp"; echo "$name: FAILED to link step $s from $sub"
        fi
        break
    done
done
REMOTE

grep -v '^\s*#' "$MARKS" | while IFS=$'\t' read -r name dir steps; do
    [ -z "${name:-}" ] && continue
    if [ -n "$ONLY" ]; then
        case "$name" in $ONLY) ;; *) continue ;; esac
    fi
    if [ "$LOCAL" = 1 ]; then
        bash "$BODY" "$name" "$dir" "$steps"
    else
        ssh "$HOST" "bash -s -- $(printf '%q %q %q' "$name" "$dir" "$steps")" < "$BODY" \
            2>&1 | grep -v "^load\|CPLUS\|CUDA\|CUDNN"
    fi
done
