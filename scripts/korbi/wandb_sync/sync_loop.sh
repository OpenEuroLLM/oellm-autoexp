#!/bin/bash
# Mirror the offline W&B runs of the MN5 production runs and upload them to
# openeurollm-project/flag_hybrid, every $INTERVAL seconds (runs on a machine with internet and a
# W&B login in ~/.netrc). Re-uploading a growing run is safe: Megatron logs with explicit steps and
# W&B keeps only steps beyond the ones it has (checked 2026-10-24 with two fragments, one id).
#
#   nohup setsid scripts/korbi/wandb_sync/sync_loop.sh >> ~/wandb_sync_mn5.log 2>&1 &
#   ONCE=1 scripts/korbi/wandb_sync/sync_loop.sh        # a single pass
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
RUNS="${RUNS:-$HERE/runs.tsv}"
MIRROR="${MIRROR:-$HOME/wandb_mirror_mn5}"
ENTITY="${ENTITY:-openeurollm-project}"
PROJECT="${PROJECT:-flag_hybrid}"
INTERVAL="${INTERVAL:-1800}"
HOST="${HOST:-marenostrum_new}"
WANDB="${WANDB:-$HOME/.miniconda3/envs/oellm/bin/wandb}"
while true; do
    echo "=== $(date -Is) sync pass"
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
        "$WANDB" beta sync --no-skip-synced -e "$ENTITY" -p "$PROJECT" --id "$id" "${files[@]}" </dev/null 2>&1 \
            | grep -E "Finished|uploading|rror|failed|No runs" | tail -4
    done
    [ -n "${ONCE:-}" ] && break
    sleep "$INTERVAL"
done
