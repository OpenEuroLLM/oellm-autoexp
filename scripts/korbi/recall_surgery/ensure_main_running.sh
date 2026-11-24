#!/usr/bin/env bash
# Get the main run launched by clearing evals off the host's USABLE nodes, keeping the ones on the
# excluded (slow / bad-interconnect) nodes where the user wants them.
#
# Host010: 264 nodes, some excluded. The main run needs 256 usable nodes. Evals are 1-node runs with
# USE_EXCLUDED_NODES=1, so they can sit on the excluded nodes for free -- but when they spill onto
# usable nodes the main run is short and never gets its `started` marker.
#
# Usage: bash ensure_main_running.sh [--rid 2226] [--rounds 8]
set -uo pipefail

R=/gpfs/projects/ehpc533/users/korbip/hybrid_v019
P=/gpfs/scratch/ehpc533/hybrid_nucleus/pools/mn5_pool_p264
H=$P/hosts/host010
PY=$R/autoexp_env/bin/python
CTL_DIR=$R/autoexp_v4bfix2

RID=2226
ROUNDS=8
while [ $# -gt 0 ]; do
  case "$1" in
    --rid) RID=$2; shift 2;;
    --rounds) ROUNDS=$2; shift 2;;
    -h|--help) sed -n '2,12p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

say() { echo "[$(date +%H:%M:%S)] $*"; }

# runs holding a node that is NOT in the host's exclude list: they block the main run
blockers() {
  "$PY" - "$H" <<'PY'
import glob, os, sys
H = sys.argv[1]
excl = set(x.strip() for x in open(os.path.join(H, "exclude_nodes")).read().split() if x.strip())
out = []
for f in glob.glob(os.path.join(H, "inuse", "*")):
    rid = os.path.basename(f)
    nodes = [n.strip() for n in open(f).read().replace(",", "\n").split() if n.strip()]
    if any(n not in excl for n in nodes):
        out.append(rid)
print(" ".join(sorted(out)))
PY
}

for round in $(seq 1 "$ROUNDS"); do
  if [ -f "$P/runs/mn5-pool-p264.$RID/started" ]; then
    say "main run .$RID STARTED"
    say "  inuse: $(ls -1 $H/inuse 2>/dev/null | tr '\n' ' ')"
    say "  state: $(cat $H/state)"
    exit 0
  fi
  b=$(blockers)
  if [ -z "$b" ]; then
    say "round $round: no usable-node blockers left, waiting for .$RID"
    sleep 30
    continue
  fi
  say "round $round: .$RID not started; blockers on usable nodes: $b"
  for rid in $b; do
    say "  preempt $rid"
    ( cd "$CTL_DIR" && PYTHONPATH=. "$PY" scripts/pool_ctl.py "$P" preempt "$rid" 2>&1 | tail -1 | sed 's/^/    /' )
  done
  sleep 45
done

if [ -f "$P/runs/mn5-pool-p264.$RID/started" ]; then
  say "main run .$RID STARTED"
  exit 0
fi
say "ALERT .$RID still not started after $ROUNDS rounds; blockers: $(blockers)"
say "  inuse: $(ls -1 $H/inuse 2>/dev/null | tr '\n' ' ')"
exit 5
