#!/bin/bash
# Poll link_marks.sh so every marked checkpoint is pinned into <run dir>/ckpt_eval before the
# training rotation deletes it (see the timing note in link_marks.sh). Meant to run on MN5, where
# the run dirs are, under setsid/nohup -- the same place the pool monitor runs.
#
#   setsid nohup scripts/korbi/eval_marks/sync_loop.sh >/dev/null 2>&1 &
#   ONLY='ambitious_v4_2ph*' INTERVAL=300 scripts/korbi/eval_marks/sync_loop.sh   # foreground
#   kill "$(cat scripts/korbi/eval_marks/sync_loop.pid)"                          # stop
#
# ONLY is a glob over the marks.tsv names (empty = every entry), INTERVAL the seconds between
# passes (a marked save lives for hours, 5 min is plenty). One pidfile/log per marks.tsv.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
export LOCAL="${LOCAL:-1}"
export ONLY="${ONLY:-}"
INTERVAL="${INTERVAL:-300}"
LOG="${LOG:-$HERE/sync_loop.log}"
PIDF="${PIDF:-$HERE/sync_loop.pid}"

if [ -s "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
    echo "sync_loop already running (pid $(cat "$PIDF")); nothing to do"
    exit 0
fi
echo $$ > "$PIDF"
trap 'rm -f "$PIDF"' EXIT

echo "[$(date -Is)] sync_loop start pid $$ interval ${INTERVAL}s only='${ONLY:-all}'" >> "$LOG"
last_beat=$(date +%s)
while true; do
    out=$("$HERE/link_marks.sh" 2>&1)
    now=$(date +%s)
    if [ -n "$out" ]; then
        printf '[%s] pass\n%s\n' "$(date -Is)" "$out" >> "$LOG"
        last_beat=$now
    elif [ $((now - last_beat)) -ge 3600 ]; then
        printf '[%s] idle, no new marked checkpoints\n' "$(date -Is)" >> "$LOG"
        last_beat=$now
    fi
    sleep "$INTERVAL"
done
