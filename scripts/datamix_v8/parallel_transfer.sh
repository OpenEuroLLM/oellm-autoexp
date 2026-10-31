#!/bin/bash
# Run several hpctransfer sessions side by side (throughput scales ~linearly with sessions).
#   parallel_transfer.sh LIST SESSIONS [hpctransfer options...]
# LIST: one "SRC DST" pair per line (HOST:PATH HOST:PATH); each pair is one hpctransfer session.
# Per-pair logs go to $LOGDIR (default: <LIST>.logs/); a pair is skipped when its log ends in "done:".
set -u
LIST=$1 K=$2
shift 2
HPCT=${HPCT:-$(dirname "$0")/hpctransfer/hpctransfer.py}
LOGDIR=${LOGDIR:-$LIST.logs}
mkdir -p "$LOGDIR"
export HPCT LOGDIR
OPTS="${*:---jobs 32 --split-size 1G --mkpath}"
export OPTS
one() {
  local src=$1 dst=$2 log
  log="$LOGDIR/$(printf '%s' "$src" | tr '/:' '__').log"
  if grep -q "^done:" "$log" 2>/dev/null; then echo "skip $src"; return 0; fi
  echo "$(date +%T) start $src"
  # shellcheck disable=SC2086
  python3 "$HPCT" $OPTS "$src" "$dst" >"$log" 2>&1
  local rc=$?
  echo "$(date +%T) rc=$rc $(grep -E '^done:|FAILED' "$log" | tail -1) $src"
}
export -f one
grep -v '^\s*$' "$LIST" | xargs -P "$K" -L 1 bash -c 'one "$0" "$1"'
echo "ALL DONE $(date +%T)"
