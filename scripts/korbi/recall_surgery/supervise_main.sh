#!/usr/bin/env bash
# Keep the v4b main line training: detect a failure, get the restart placed fast, and preempt the
# evals that stand in its way.
#
# Why: the pool restarts the job record itself, but the new run is only *assigned* -- it gets its
# `started` marker only when 256 usable nodes are free on host010. Evals (priority 55/60) refill
# freed nodes within seconds, so without an evictor the main run sits assigned forever.
#
# What one round does (every 15 s):
#   * a run of the main name that ended non-zero  -> exclude the node it names, note the failure;
#   * nothing live and nothing pending           -> submit the fixed full-recipe run;
#   * live but not started for > GRACE_S          -> preempt the runs on usable nodes until it fits;
#   * live and stepping                           -> report a heartbeat now and then.
#
# Usage (harness background task): bash supervise_main.sh [--interval 15] [--budget 43200]
set -uo pipefail

R=/gpfs/projects/ehpc533/users/korbip/hybrid_v019
P=/gpfs/scratch/ehpc533/hybrid_nucleus/pools/mn5_pool_p264
H=$P/hosts/host010
STATE=$R/pool_state/pool_mn5-pool-p264
SNAP=$R/autoexp_v4bfix2
PY=$R/autoexp_env/bin/python
EXCLUDE_FILE=$R/mn5_exclude_nodes.txt
RUN_DIR=/gpfs/scratch/ehpc533/hybrid_nucleus/production/kvshare_k_fresh35900_20261116/runs/v4b_learned/main
PREFIX=ambitious_v4b_learned_

INTERVAL=15
BUDGET=43200
GRACE_S=90            # a live-but-not-started main run gets this long before evals are evicted
HEARTBEAT_S=1800
while [ $# -gt 0 ]; do
  case "$1" in
    --interval) INTERVAL=$2; shift 2;;
    --budget) BUDGET=$2; shift 2;;
    --grace) GRACE_S=$2; shift 2;;
    -h|--help) sed -n '2,20p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

say() { echo "[$(date +%H:%M:%S)] $*"; }

# newest main run: "<rid> <name> <rc|-> <started:0|1>" (evals are named ev_<prefix>..., skipped)
probe() {
  "$PY" - "$P" "$PREFIX" <<'PY'
import glob, json, os, sys
pool, prefix = sys.argv[1], sys.argv[2]
best = None
for d in glob.glob(pool + "/runs/mn5-pool-p264.*"):
    try:
        m = json.load(open(os.path.join(d, "run.json")))
    except Exception:
        continue
    name = str(m.get("name") or "")
    if not name.startswith(prefix) or name.startswith("ev_"):
        continue
    rid = int(d.rsplit(".", 1)[-1])
    if best is None or rid > best[0]:
        rc = open(os.path.join(d, "rc")).read().strip() if os.path.exists(os.path.join(d, "rc")) else "-"
        best = (rid, name, rc, os.path.exists(os.path.join(d, "started")))
print("%d %s %s %d" % best if best else "- - - 0")
PY
}

live_main() {  # rid of a live main run, empty if none
  "$PY" - "$P" "$PREFIX" <<'PY'
import glob, json, os, sys
pool, prefix = sys.argv[1], sys.argv[2]
out = []
for d in glob.glob(pool + "/runs/mn5-pool-p264.*"):
    if os.path.exists(os.path.join(d, "rc")):
        continue
    try:
        m = json.load(open(os.path.join(d, "run.json")))
    except Exception:
        continue
    name = str(m.get("name") or "")
    if name.startswith(prefix) and not name.startswith("ev_"):
        out.append(int(d.rsplit(".", 1)[-1]))
print(max(out) if out else "")
PY
}

pending() {
  "$PY" - "$STATE" <<'PY'
import glob, json, sys
n = 0
for p in glob.glob(sys.argv[1] + "/*learned_fix_256*.job.json"):
    try:
        d = json.load(open(p))
    except Exception:
        continue
    rt = d.get("runtime") or {}
    if rt.get("submitted") and rt.get("final_state") is None:
        n += 1
print(n)
PY
}

blockers() {  # runs holding a node that is NOT in the host exclude list
  "$PY" - "$H" <<'PY'
import glob, os, sys
H = sys.argv[1]
excl = set(x.strip() for x in open(os.path.join(H, "exclude_nodes")).read().split() if x.strip())
out = []
for f in glob.glob(os.path.join(H, "inuse", "*")):
    nodes = [n.strip() for n in open(f).read().replace(",", "\n").split() if n.strip()]
    if any(n not in excl for n in nodes):
        out.append(os.path.basename(f))
print(" ".join(sorted(out)))
PY
}

failing_node() {  # node/device/iteration named by the failure, if any
  local log=$1 m
  m=$(grep -m1 -oE "Rank [0-9]+, node [a-z0-9]+, device [0-9]+, iteration [0-9]+" "$log" 2>/dev/null)
  if [ -n "$m" ]; then
    echo "$m" | sed -E "s/Rank [0-9]+, node ([a-z0-9]+), device ([0-9]+), iteration ([0-9]+)/\1 \2 \3 NaN/"
    return 0
  fi
  m=$(grep -A4 "Root Cause" "$log" 2>/dev/null | grep -oE "host +: [a-z0-9]+\.bsc\.mn" | head -1 | grep -oE "as[0-9]{2}r[0-9]b[0-9]{2}")
  [ -n "$m" ] && { echo "$m - - root-cause-host"; return 0; }
  return 1
}

exclude_node() {
  local node=$1 dev=$2 it=$3 sig=$4
  [ -n "${node:-}" ] || return 1
  if grep -qx "$node" "$EXCLUDE_FILE" 2>/dev/null; then say "  $node already excluded"; return 1; fi
  printf "# %s %s (device %s, iteration %s): excluded by supervise_main.sh after a single-node fault\n%s\n" \
    "$(date +%Y-%m-%d)" "$sig" "$dev" "$it" "$node" >> "$EXCLUDE_FILE"
  say "  EXCLUDED $node (device $dev, iteration $it, $sig)"
}

submit_main() {
  say "submitting the fixed full-recipe run"
  ( cd "$SNAP" && PYTHONPATH=. "$PY" scripts/run_autoexp.py \
      --config-name experiments/korbi/recall_surgery/v4b_learned_fix_mn5 \
      --no-monitor --monitor-state-dir "$R/pool_state" --pool "$P" -y 2>&1 \
    | grep -E "Enqueued|ERROR|Traceback" | head -3 | sed 's/^/    /' )
}

START=$(date +%s)
last_rid=""
last_rc=""
last_hb=0
assigned_since=0
say "supervising ${PREFIX}* (interval ${INTERVAL}s, grace ${GRACE_S}s, budget ${BUDGET}s)"

while :; do
  now=$(date +%s)
  if [ $((now - START)) -ge "$BUDGET" ]; then say "budget spent"; exit 0; fi

  read -r rid name rc started <<<"$(probe)"

  # --- a failure of the newest main run -----------------------------------
  if [ "$rid" != "-" ] && [ "$rc" != "-" ] && [ "$rc" != "$last_rc" ]; then
    log=$RUN_DIR/slurm-mn5-pool-p264.$rid.log
    if [ "$rc" != "0" ]; then
      hit=$(grep -cE "cudaErrorIllegalAddress|illegal memory access" "$log" 2>/dev/null | head -1); hit=${hit:-0}
      trap=$(grep -c "disagrees with topk" "$log" 2>/dev/null | head -1); trap=${trap:-0}
      nan=$(grep -c "found NaN" "$log" 2>/dev/null | head -1); nan=${nan:-0}
      say "FAILURE .$rid rc=$rc illegal=$hit trap=$trap nan=$nan"
      read -r fnode fdev fit fsig <<<"$(failing_node "$log" || true)"
      if [ -n "${fnode:-}" ]; then
        say "  fault names $fnode (device ${fdev:-?}, iteration ${fit:-?})"
        exclude_node "$fnode" "$fdev" "$fit" "$fsig"
      else
        say "  no single node named: nothing excluded"
      fi
    else
      say "run .$rid ended rc=0 (preempt/rotation)"
    fi
    last_rc=$rc
    assigned_since=0
  fi

  # --- make sure a successor exists and gets placed ------------------------
  live=$(live_main)
  if [ -z "$live" ]; then
    if [ "$(pending)" = "0" ]; then submit_main; fi
    sleep "$INTERVAL"
    continue
  fi

  if [ -f "$P/runs/mn5-pool-p264.$live/started" ]; then
    assigned_since=0
    if [ $((now - last_hb)) -ge "$HEARTBEAT_S" ]; then
      last_hb=$now
      it=$(grep -E "iteration +[0-9]+/" "$RUN_DIR/slurm-mn5-pool-p264.$live.log" 2>/dev/null | tail -1 | grep -oE "iteration +[0-9]+/")
      say "live .$live ${it:-loading} inuse=$(ls -1 $H/inuse 2>/dev/null | tr '\n' ' ')"
    fi
  else
    # assigned but not started: evict evals until it fits
    if [ "$assigned_since" = 0 ]; then
      assigned_since=$now
      say ".$live assigned but not started; waiting ${GRACE_S}s before evicting"
    elif [ $((now - assigned_since)) -ge "$GRACE_S" ]; then
      b=$(blockers)
      if [ -n "$b" ]; then
        say ".$live still not started: preempting usable-node holders: $b"
        for r in $b; do
          [ "$r" = "mn5-pool-p264.$live" ] && continue
          ( cd "$R/autoexp_v4bfix2" && PYTHONPATH=. "$PY" scripts/pool_ctl.py "$P" preempt "$r" 2>&1 | tail -1 | sed 's/^/    /' )
        done
      else
        say ".$live not started and no usable-node holders left (capacity or a stuck supervisor)"
      fi
      assigned_since=$now
    fi
  fi

  sleep "$INTERVAL"
done
