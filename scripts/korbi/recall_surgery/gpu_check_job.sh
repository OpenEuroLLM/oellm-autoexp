#!/usr/bin/env bash
# Probe every node of the running pool allocation for its real GPU count.
#
# Why: sinfo / scontrol show nodes are denied for this account ("slurm_load_node: Access/permission
# denied"), so the pool cannot see node state or GRES. The only trustworthy view is an srun step
# inside the allocation that asks each node itself: nvidia-smi -L | wc -l.
#
# Steps are built in chunks, because one 261-node step is refused outright when Slurm considers any
# node unavailable ("Requested node configuration is not available") and then tells us nothing about
# the others.
#
# Usage: bash gpu_check_job.sh [--job 48867560] [--chunk 64] [--timeout 180]
set -uo pipefail

JOB=$(squeue -h -u "$USER" -t R -o "%i %D" 2>/dev/null | awk '$2==261 || $2==264 {print $1; exit}')
CHUNK=64
STEP_TIMEOUT=180
while [ $# -gt 0 ]; do
  case "$1" in
    --job) JOB=$2; shift 2;;
    --chunk) CHUNK=$2; shift 2;;
    --timeout) STEP_TIMEOUT=$2; shift 2;;
    -h|--help) sed -n '2,14p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
[ -n "${JOB:-}" ] || { echo "no running pool job found"; exit 2; }

WORK=$(mktemp -d)
echo "[$(date +%H:%M:%S)] job $JOB: probing GPU counts in chunks of $CHUNK"
scontrol show hostnames "$(squeue -h -j "$JOB" -o "%N")" 2>/dev/null | sort > "$WORK/nodes.txt"
total=$(wc -l < "$WORK/nodes.txt")
echo "  nodes in the allocation: $total"

split -l "$CHUNK" "$WORK/nodes.txt" "$WORK/chunk."
for f in "$WORK"/chunk.*; do
  nl=$(paste -sd, - < "$f")
  n=$(wc -l < "$f")
  echo "[$(date +%H:%M:%S)] chunk $(basename "$f"): $n node(s)"
  timeout "$STEP_TIMEOUT" srun --overlap --mem=0 --jobid="$JOB" --nodelist="$nl" --nodes="$n" \
    --ntasks="$n" --ntasks-per-node=1 --cpus-per-task=1 --job-name=gpucheck \
    bash -c 'echo "GPU $(hostname -s) $(nvidia-smi -L 2>/dev/null | wc -l) $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 | tr -d " ")"' \
    >> "$WORK/out.txt" 2>> "$WORK/err.txt"
  echo "    -> $(grep -c . "$WORK/out.txt" 2>/dev/null) node(s) reported so far"
done

echo "==================== GPU CHECK ===================="
/usr/bin/env python3 - "$WORK" <<'PY'
import collections, os, sys
work = sys.argv[1]
want = [l.strip() for l in open(os.path.join(work, "nodes.txt")) if l.strip()]
got = {}
for line in open(os.path.join(work, "out.txt"), errors="ignore"):
    p = line.split()
    if len(p) >= 4 and p[0] == "GPU":
        got[p[1]] = (int(p[2]) if p[2].isdigit() else -1, p[3])
missing = [n for n in want if n not in got]
bad = [(n, v) for n, v in got.items() if v[0] != 4]
print("  allocation nodes : %d" % len(want))
print("  answered         : %d" % len(got))
print("  no answer        : %d" % len(missing))
if missing:
    print("    " + " ".join(missing[:20]) + (" ..." if len(missing) > 20 else ""))
print("  GPU count != 4   : %d" % len(bad))
for n, v in bad[:20]:
    print("    %s: %d GPU(s), %s" % (n, v[0], v[1]))
mem = collections.Counter(v[1] for v in got.values())
print("  GPU memory seen  : %s" % dict(mem))
eff = len(got) - len(bad)
print("  EFFECTIVELY AVAILABLE (answered and 4 GPUs): %d node(s) = %d GPUs" % (eff, eff * 4))
print("  the run needs    : 256 node(s) = 1024 GPUs")
PY
echo "  raw: $WORK/out.txt  errors: $WORK/err.txt"
grep -m3 . "$WORK/err.txt" 2>/dev/null | sed 's/^/    /'
