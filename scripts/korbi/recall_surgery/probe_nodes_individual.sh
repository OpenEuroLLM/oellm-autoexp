#!/usr/bin/env bash
# Probe every node of the allocation INDIVIDUALLY: 1-node srun steps with --gres=gpu:4.
#
# Why individually: a large step fails as a whole when a few nodes cannot satisfy their GRES, and
# the failure names no node. One node at a time makes the GRES-limited node itself the failure, so
# it can be excluded. Chunked probes hide exactly the nodes that matter.
#
# Per node the step asks:
#   * can Slurm give this node 4 GPUs (--gres=gpu:4) and how many does nvidia-smi see;
#   * GPU memory (a lowered GRES often shows as a smaller MIG/partitioned device);
#   * InfiniBand port states (/sys/class/infiniband/*/ports/*/state: 4 = ACTIVE, 1 = DOWN).
#
# Usage: bash probe_nodes_individual.sh [--job 48867560] [--parallel 8] [--timeout 120] [--apply]
#   --apply  append the failing nodes to the global exclude list (default: report only)
set -uo pipefail

JOB=$(squeue -h -u "$USER" -t R -o "%i %D" 2>/dev/null | awk '$2==261 || $2==264 || $2==260 {print $1; exit}')
PAR=8
STEP_TIMEOUT=120
APPLY=0
EXCLUDE_FILE=/gpfs/projects/ehpc533/users/korbip/hybrid_v019/mn5_exclude_nodes.txt
while [ $# -gt 0 ]; do
  case "$1" in
    --job) JOB=$2; shift 2;;
    --parallel) PAR=$2; shift 2;;
    --timeout) STEP_TIMEOUT=$2; shift 2;;
    --apply) APPLY=1; shift;;
    -h|--help) sed -n '2,20p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
[ -n "${JOB:-}" ] || { echo "no running pool job found"; exit 2; }

WORK=$(mktemp -d)
echo "[$(date +%H:%M:%S)] job $JOB: 1-node probes, $PAR in parallel"
scontrol show hostnames "$(squeue -h -j "$JOB" -o "%N")" 2>/dev/null | sort > "$WORK/nodes.txt"
echo "  nodes: $(wc -l < "$WORK/nodes.txt")"

probe_one() {
  local node=$1
  local out rc
  out=$(timeout "$STEP_TIMEOUT" srun --overlap --mem=0 --jobid="$JOB" --nodelist="$node" \
          --nodes=1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=1 --gres=gpu:4 \
          --job-name=nodecheck \
          bash -c 'g=$(nvidia-smi -L 2>/dev/null | wc -l); m=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 | tr -d " "); ib=""; for d in /sys/class/infiniband/*/ports/*/state; do [ -e "$d" ] || continue; dev=$(echo "$d" | cut -d/ -f5); port=$(echo "$d" | cut -d/ -f7); st=$(cat "$d" 2>/dev/null | awk "{print \$2}"); ib="$ib$dev:$port=$st,"; done; echo "OK gpus=$g mem=$m ib=$ib"' 2>&1)
  rc=$?
  echo "NODE $node rc=$rc ${out//$'\n'/ | }"
}
export -f probe_one
export JOB STEP_TIMEOUT

# shellcheck disable=SC2016
xargs -P "$PAR" -I{} bash -c 'probe_one "$@"' _ {} < "$WORK/nodes.txt" > "$WORK/results.txt" 2>&1
echo "[$(date +%H:%M:%S)] probes done"

/usr/bin/env python3 - "$WORK/results.txt" "$WORK/nodes.txt" <<'PY'
import re, sys
from collections import Counter

res, nodes = sys.argv[1], sys.argv[2]
want = [l.strip() for l in open(nodes) if l.strip()]
rows = {}
for line in open(res, errors="ignore"):
    m = re.match(r"NODE (\S+) rc=(\d+) ?(.*)", line.strip())
    if m:
        rows[m.group(1)] = (int(m.group(2)), m.group(3))

gres_fail, bad_gpu, bad_mem, bad_ib, other = [], [], [], [], []
active_counts = {}
for node in want:
    if node not in rows:
        other.append((node, -1, "no result"))
        continue
    rc, rest = rows[node]
    if rc != 0 or "OK gpus=" not in rest:
        low = rest.lower()
        if "gres" in low or "configuration is not available" in low or "invalid generic resource" in low:
            gres_fail.append((node, rest[:110]))
        else:
            other.append((node, rc, rest[:110]))
        continue
    g = re.search(r"gpus=(\d+)", rest)
    mem = re.search(r"mem=([0-9]+)MiB", rest)
    gpus = int(g.group(1)) if g else -1
    if gpus != 4:
        bad_gpu.append((node, gpus, mem.group(1) if mem else "?"))
    memv = int(mem.group(1)) if mem else -1
    ib = rest.split("ib=")[-1] if "ib=" in rest else ""
    act = ib.count("=ACTIVE")
    active_counts[node] = act
    down = [x for x in ib.split(",") if "=DOWN" in x]
    if memv != 65247:
        bad_mem.append((node, memv))
    if down:
        bad_ib.append((node, act, " ".join(down)[:90]))

# A port being DOWN is only suspicious if it departs from the fleet norm: mlx5_3 is DOWN
# everywhere on MN5, so the norm is "5 of 6 ACTIVE" and a node below that is the anomaly.
norm = Counter(active_counts.values()).most_common(1)[0][0] if active_counts else 0
odd_ib = [(n, a, d) for (n, a, d) in bad_ib if a < norm]
print("==================== PER-NODE CHECK (1-node steps) ====================")
print("  probed            : %d / %d" % (len(rows), len(want)))
print("  ACTIVE IB norm    : %d of 6 ports" % norm)
print("  GRES refused      : %d" % len(gres_fail))
for n, r in gres_fail[:20]:
    print("     %s  %s" % (n, r))
print("  GPU count != 4    : %d" % len(bad_gpu))
for n, g, m in bad_gpu[:20]:
    print("     %s  %d GPU(s) %s" % (n, g, m))
print("  GPU memory odd    : %d" % len(bad_mem))
for n, m in bad_mem[:10]:
    print("     %s  %d MiB" % (n, m))
print("  IB below the norm : %d" % len(odd_ib))
for n, a, d in odd_ib[:20]:
    print("     %s  %d ACTIVE  %s" % (n, a, d))
print("  other failures    : %d" % len(other))
for n, rc, r in other[:10]:
    print("     %s  rc=%d %s" % (n, rc, r))
suspect = sorted({n for n, _ in gres_fail} | {n for n, _, _ in bad_gpu} |
                 {n for n, _ in bad_mem} | {n for n, _, _ in odd_ib})
print("  SUGGESTED EXCLUSIONS (%d): %s" % (len(suspect), " ".join(suspect)))
open("/tmp/probe_suspects.txt", "w").write("\n".join(suspect) + ("\n" if suspect else ""))
PY


if [ "$APPLY" = 1 ] && [ -s /tmp/probe_suspects.txt ]; then
  echo "[$(date +%H:%M:%S)] applying exclusions"
  while read -r n; do
    [ -n "$n" ] || continue
    if grep -qx "$n" "$EXCLUDE_FILE"; then echo "  $n already excluded"; else
      printf "# %s probe_nodes_individual.sh: 1-node GRES/GPU/IB check failed on job %s\n%s\n" \
        "$(date +%Y-%m-%d)" "$JOB" "$n" >> "$EXCLUDE_FILE"
      echo "  excluded $n"
    fi
  done < /tmp/probe_suspects.txt
  echo "  exclude list now: $(grep -cvE '^#|^$' "$EXCLUDE_FILE") node(s)"
fi
