#!/usr/bin/env bash
# Keep every run's meta.env to ONE line per key.
#
# Why: the pool writes meta.env more than once (submit, then the restart hook's exclusion refresh),
# and the writer appends instead of replacing. The supervisor sources meta.env with `.`, so the LAST
# assignment wins -- and the last EXCLUDE had grown to ~292 entries, which made `usable_nodes` report
# 2 usable nodes and every 256-node run die with rc=-1 insufficient_nodes.
#
# This loop rewrites any meta.env that carries duplicate keys, keeping the FIRST occurrence (the run's
# own list, 35 entries), so the supervisor computes the real capacity.
#
# Usage: bash dedupe_meta.sh [--pool <dir>] [--interval 3] [--budget 43200]
set -uo pipefail

P=/gpfs/scratch/ehpc533/hybrid_nucleus/pools/mn5_pool_p264
INTERVAL=3
BUDGET=43200
while [ $# -gt 0 ]; do
  case "$1" in
    --pool) P=$2; shift 2;;
    --interval) INTERVAL=$2; shift 2;;
    --budget) BUDGET=$2; shift 2;;
    -h|--help) sed -n '2,14p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

START=$(date +%s)
echo "[$(date +%H:%M:%S)] deduping meta.env under $P/runs (every ${INTERVAL}s)"
while :; do
  now=$(date +%s)
  [ $((now - START)) -ge "$BUDGET" ] && { echo "[$(date +%H:%M:%S)] budget spent"; exit 0; }
  /usr/bin/env python3 - "$P" <<'PY'
import glob, os, sys
pool = sys.argv[1]
fixed = 0
for d in glob.glob(os.path.join(pool, "runs", "*")):
    m = os.path.join(d, "meta.env")
    if not os.path.exists(m):
        continue
    try:
        lines = open(m).read().splitlines()
    except OSError:
        continue
    seen, out, dup = set(), [], 0
    for line in lines:
        if "=" not in line:
            out.append(line)
            continue
        key = line.split("=", 1)[0]
        if key in seen:
            dup += 1
            continue
        seen.add(key)
        out.append(line)
    if dup:
        tmp = m + ".dedupe.tmp"
        open(tmp, "w").write("\n".join(out) + "\n")
        os.replace(tmp, m)
        fixed += 1
        print("[%s] deduped %s (dropped %d duplicate key(s))" % (
            __import__("time").strftime("%H:%M:%S"), os.path.basename(d), dup), flush=True)
if fixed:
    print("fixed %d file(s)" % fixed, flush=True)
PY
  sleep "$INTERVAL"
done
