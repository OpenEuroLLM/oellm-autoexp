#!/bin/bash
# Mirror the offline W&B runs of the MN5 production runs and upload them to
# openeurollm-project/flag_hybrid, every $INTERVAL seconds (default 15 min), after keeping the equal-token checkpoints (eval_marks/) (runs on a machine with internet and a
# W&B login in ~/.netrc). Re-uploading a growing run is safe: Megatron logs with explicit steps and
# W&B keeps only steps beyond the ones it has (checked 2026-10-24 with two fragments, one id).
#
#   nohup setsid scripts/korbi/wandb_sync/sync_loop.sh >> ~/wandb_sync_mn5.log 2>&1 &
#   stop it with: kill "$(cat ~/wandb_sync_mn5.pid)"   (never pkill -f: it matches the calling shell)
#   ONCE=1 scripts/korbi/wandb_sync/sync_loop.sh        # a single pass
set -u
[ -z "${ONCE:-}" ] && echo $$ > "${PIDFILE:-$HOME/wandb_sync_mn5.pid}"
HERE="$(cd "$(dirname "$0")" && pwd)"
RUNS="${RUNS:-$HERE/runs.tsv}"
MIRROR="${MIRROR:-$HOME/wandb_mirror_mn5}"
ENTITY="${ENTITY:-openeurollm-project}"
PROJECT="${PROJECT:-flag_hybrid}"
INTERVAL="${INTERVAL:-900}"
HOST="${HOST:-marenostrum_new}"
WANDB="${WANDB:-$HOME/.miniconda3/envs/oellm/bin/wandb}"
while true; do
    echo "=== $(date -Is) sync pass"
    # keep the equal-token checkpoints first (the rolling ones are deleted ~2 saves later)
    "$HERE/../eval_marks/link_marks.sh"
    grep -v '^\s*#' "$RUNS" | while IFS=$'\t' read -r id dir; do
        [ -z "${id:-}" ] && continue
        dest="$MIRROR/$id"
        mkdir -p "$dest"
        # the offline runs only: *.wandb (history, summary, config, console) + files/
        rsync -a -e ssh --include='*/' --include='offline-run-*/**' --exclude='*' \
            "$HOST:$dir/wandb/" "$dest/" </dev/null 2>&1 | grep -v "^load\|CPLUS\|CUDA\|CUDNN\|^$" | tail -3
        # the fragments' .wandb files, given explicitly (beta sync does not search below a run's
        # wandb/wandb/ nesting); it uploads them in order of start time
        mapfile -t files < <(find "$dest" -name 'run-*.wandb' -path '*offline-run-*' 2>/dev/null | sort)
        if [ "${#files[@]}" -eq 0 ]; then echo "$id: no offline runs yet"; continue; fi
        echo "$id: ${#files[@]} fragment(s) -> $ENTITY/$PROJECT/$id"
        # one call per fragment, oldest first (the names sort by start time): a fragment cut off by a
        # crash ends mid-record, and in one multi-file call that error stops the later fragments
        for f in "${files[@]}"; do
            "$WANDB" beta sync --no-skip-synced -e "$ENTITY" -p "$PROJECT" --id "$id" "$f" </dev/null 2>&1 \
                | grep -E "uploading history|Finished|rror|failed" | tail -2 | sed "s#^#  $(basename "$(dirname "$f")"): #"
        done
    done
    [ -n "${ONCE:-}" ] && break
    sleep "$INTERVAL"
done
