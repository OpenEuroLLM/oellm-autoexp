#!/usr/bin/env bash
# Alert once, loudly, when the v4b main line fails, restarts or stops making progress.
#
# Runs in the foreground (meant for a harness background task that notifies on exit), so the exit
# code IS the alert: 0 = budget spent quietly, 4 = illegal access / MoonEP trap, 5 = run failed,
# 6 = no live run, 7 = tracker stalled, 8 = repeated restarts.
#
# Usage: bash watch_line.sh [--interval 30] [--budget 43200] [--stall-min 40] [--max-restarts 3]
set -uo pipefail

POOL=/gpfs/scratch/ehpc533/hybrid_nucleus/pools/mn5_pool_p264
RUN_DIR=/gpfs/scratch/ehpc533/hybrid_nucleus/production/kvshare_k_fresh35900_20261116/runs/v4b_learned/main
NAME_PREFIX=ambitious_v4b_learned_
TRACKER=$RUN_DIR/ckpt/latest_checkpointed_iteration.txt

INTERVAL=30
BUDGET=43200
STALL_MIN=40
MAX_RESTARTS=3
AUTO_EXCLUDE=0
EXCLUDE_FILE=/gpfs/projects/ehpc533/users/korbip/hybrid_v019/mn5_exclude_nodes.txt

while [ $# -gt 0 ]; do
  case "$1" in
    --interval) INTERVAL=$2; shift 2;;
    --budget) BUDGET=$2; shift 2;;
    --stall-min) STALL_MIN=$2; shift 2;;
    --max-restarts) MAX_RESTARTS=$2; shift 2;;
    --auto-exclude) AUTO_EXCLUDE=1; shift;;
    -h|--help) sed -n '2,12p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 9;;
  esac
done

say() { echo "[$(date +%H:%M:%S)] $*"; }

failing_node() {  # $1 = log; prints "node device iteration signature" of the first named fault
  local log=$1
  local m
  m=$(grep -m1 -oE "Rank [0-9]+, node [a-z0-9]+, device [0-9]+, iteration [0-9]+" "$log" 2>/dev/null)
  if [ -n "$m" ]; then
    echo "$m" | sed -E "s/Rank [0-9]+, node ([a-z0-9]+), device ([0-9]+), iteration ([0-9]+)/\1 \2 \3 NaN/"
    return 0
  fi
  m=$(grep -A4 "Root Cause" "$log" 2>/dev/null | grep -oE "host +: [a-z0-9]+\.bsc\.mn" | head -1 | grep -oE "as[0-9]{2}r[0-9]b[0-9]{2}")
  if [ -n "$m" ]; then
    echo "$m - - root-cause-host"
    return 0
  fi
  return 1
}

exclude_node() {  # $1 node, $2 device, $3 iteration, $4 signature
  local node=$1 dev=$2 it=$3 sig=$4
  [ -n "$node" ] || return 1
  if grep -qx "$node" "$EXCLUDE_FILE" 2>/dev/null; then
    say "  $node already excluded"
    return 1
  fi
  printf "# %s %s (device %s, iteration %s): excluded automatically by watch_line.sh after a single-device fault\n%s\n" \
    "$(date +%Y-%m-%d)" "$sig" "$dev" "$it" "$node" >> "$EXCLUDE_FILE"
  say "  EXCLUDED $node (device $dev, iteration $it, $sig) -> the next re-render drops it"
  return 0
}

probe() {  # newest run of the prefix: "<rid> <name> <rc|-> <started:0|1> <tracker> <last_iter>"
  /usr/bin/env python3 - "$POOL" "$RUN_DIR" "$NAME_PREFIX" <<'PY'
import glob, json, os, re, sys
pool, run_dir, prefix = sys.argv[1], sys.argv[2], sys.argv[3]
best = None
for d in glob.glob(pool + "/runs/mn5-pool-p264.*"):
    try:
        m = json.load(open(os.path.join(d, "run.json")))
    except Exception:
        continue
    if not str(m.get("name") or "").startswith(prefix):
        continue
    rid = int(d.rsplit(".", 1)[-1])
    if best is None or rid > best[0]:
        rc = open(os.path.join(d, "rc")).read().strip() if os.path.exists(os.path.join(d, "rc")) else "-"
        started = os.path.exists(os.path.join(d, "started"))
        best = (rid, m.get("name"), rc, started, d)
if best is None:
    print("- - - 0 - -")
    raise SystemExit
rid, name, rc, started, d = best
tracker = "-"
tp = os.path.join(run_dir, "ckpt", "latest_checkpointed_iteration.txt")
if os.path.exists(tp):
    tracker = open(tp).read().strip()
last = "-"
log = os.path.join(run_dir, "slurm-mn5-pool-p264.%d.log" % rid)
if os.path.exists(log):
    it = None
    with open(log, errors="ignore") as fh:
        for line in fh:
            m = re.search(r"iteration +(\d+)/.*elapsed time per iteration", line)
            if m:
                it = m.group(1)
    last = it or "-"
print(rid, name, rc, int(started), tracker, last)
PY
}

START=$(date +%s)
prev_rid=""
prev_rc=""
baseline_rid=""
baseline_rc=""
first_probe=1
restarts=0
last_progress=$(date +%s)
last_tracker=""
say "watching ${NAME_PREFIX}* (interval ${INTERVAL}s, budget ${BUDGET}s, stall ${STALL_MIN}min)"

while :; do
  now=$(date +%s)
  if [ $((now - START)) -ge "$BUDGET" ]; then
    say "budget spent: line still under watch"
    exit 0
  fi

  read -r rid name rc started tracker last <<<"$(probe)"

  if [ "$rid" = "-" ]; then
    if [ $((now - last_progress)) -ge $((STALL_MIN * 60)) ]; then
      say "ALERT no live ${NAME_PREFIX}* run for $(( (now - last_progress) / 60 )) min"
      exit 6
    fi
    sleep "$INTERVAL"
    continue
  fi

  if [ "$first_probe" = 1 ]; then
    # The run that is already there when the watch starts is history, not an event: its rc must not
    # fire an alert (2026-11-24: the watcher exited 5 on the cancelled run .2185 seconds after start).
    baseline_rid=$rid
    baseline_rc=$rc
    first_probe=0
    say "baseline .$rid rc=$rc tracker=$tracker last_iter=$last"
  fi

  if [ "$rid" != "$prev_rid" ]; then
    if [ -n "$prev_rid" ]; then
      restarts=$((restarts + 1))
      say "RESTART .$prev_rid -> .$rid (restart #$restarts) last_iter=${last}"
      if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
        say "ALERT $restarts restarts; latest .$rid name=$name"
        exit 8
      fi
    else
      say "live .$rid $name started=$started tracker=$tracker last_iter=$last"
    fi
    prev_rid=$rid
    last_progress=$now
  fi

  if [ "$rc" = "-" ]; then :; elif [ "$rid" = "$baseline_rid" ] && [ "$rc" = "$baseline_rc" ]; then :;
  elif [ "$rc" != "$prev_rc" ]; then
    log=$RUN_DIR/slurm-mn5-pool-p264.$rid.log
    # grep -c prints 0 AND exits 1 on no match: "|| echo 0" would append a second 0 and break the
    # integer test (2026-11-24, first run of this script). Take the count only.
    hit=$(grep -cE "cudaErrorIllegalAddress|illegal memory access" "$log" 2>/dev/null | head -1)
    trap=$(grep -c "disagrees with topk" "$log" 2>/dev/null | head -1)
    hit=${hit:-0}; trap=${trap:-0}
    say "RUN ENDED .$rid rc=$rc illegal=$hit trap=$trap last_iter=$last"
    if [ "$AUTO_EXCLUDE" = 1 ] && [ "$rc" != "0" ]; then
      # Fast path: the node named by the failure goes into the exclude list NOW, so the restart
      # the pool is about to make already carries it (2026-11-24: a NaN on as05r4b24 sat
      # undiagnosed for ~20 min while the line reloaded on the same node).
      read -r fnode fdev fit fsig <<<"$(failing_node "$log" || true)"
      if [ -n "${fnode:-}" ]; then
        say "  fault names $fnode (device ${fdev:-?}, iteration ${fit:-?}, ${fsig:-?})"
        exclude_node "$fnode" "$fdev" "$fit" "$fsig"
      else
        say "  no single node named by the failure: nothing excluded automatically"
      fi
    fi
    if [ "${hit:-0}" -gt 0 ] || [ "${trap:-0}" -gt 0 ]; then exit 4; fi
    if [ "$rc" != "0" ]; then exit 5; fi
    prev_rc=$rc
  fi

  if [ "$tracker" != "-" ] && [ "$tracker" != "$last_tracker" ]; then
    last_tracker=$tracker
    last_progress=$now
  fi
  if [ "$last" != "-" ]; then last_progress=$now; fi

  if [ $((now - last_progress)) -ge $((STALL_MIN * 60)) ]; then
    say "ALERT no progress for $(( (now - last_progress) / 60 )) min (live .$rid, tracker=$tracker, last_iter=$last)"
    exit 7
  fi

  sleep "$INTERVAL"
done
