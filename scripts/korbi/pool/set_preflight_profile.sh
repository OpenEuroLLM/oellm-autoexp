#!/bin/bash
# Switch a pool's preflight node check between the light and the thorough profile
# (oellm_autoexp/node_health/bench.py PROFILES; experiments/korbi/pool/node_health[_thorough]_mn5).
#
#   scripts/korbi/pool/set_preflight_profile.sh <pool dir> light|thorough
#
# Run from an autoexp snapshot on the MareNostrum login node (as run_autoexp). It
#   1. records the check config in the pool's settings (pool_ctl set preflight_check=...), so a
#      later run_autoexp on the pool re-enqueues the same profile, and
#   2. enqueues that check as the pool's preflight template (+job.metadata.pool_preflight_template),
#      with the pool hosts' account / partition / qos (as orchestrator.enqueue_preflight_check).
# Only NEW hosts run the preflight; a running host is not re-checked (enqueue the check config
# as an ordinary run for that: experiments/korbi/pool/node_health_thorough_mn5 slurm.sbatch.nodes=N).
# No SLURM job is submitted, changed or cancelled.
set -euo pipefail
POOL=${1:?usage: $0 <pool dir> light|thorough}
PROFILE=${2:?usage: $0 <pool dir> light|thorough}
case "$PROFILE" in
  light) CHECK=experiments/korbi/pool/node_health_mn5 ;;
  thorough) CHECK=experiments/korbi/pool/node_health_thorough_mn5 ;;
  *) echo "profile must be light or thorough" >&2; exit 2 ;;
esac
HERE="$(cd "$(dirname "$0")/../../.." && pwd)"
R=${HYBRID_ENV:-/gpfs/projects/ehpc533/users/korbip/hybrid_v019}
PY=${AUTOEXP_PYTHON:-$R/autoexp_env/bin/python}
[ -f "$POOL/pool.json" ] || { echo "$POOL is not a pool (no pool.json)" >&2; exit 2; }

# the pool hosts' account / partition / qos (the check runs inside a host)
mapfile -t SAME < <("$PY" - "$POOL" <<'PYEOF'
import json, sys
d = json.load(open(f"{sys.argv[1]}/host_definition.json"))
sb = (d.get("slurm") or d).get("sbatch") or {}
for k in ("account", "partition", "qos"):
    if sb.get(k):
        print(f"++slurm.sbatch.{k}={sb[k]}")
PYEOF
)
echo "pool $POOL: preflight -> $PROFILE ($CHECK) ${SAME[*]}"
"$PY" "$HERE/scripts/pool_ctl.py" "$POOL" set "preflight_check=$CHECK"
cd "$HERE"
scripts/korbi/mn5_autoexp.sh scripts/run_autoexp.py --no-monitor --monitor-state-dir "$R/pool_state" \
  --pool "$POOL" --config-name "$CHECK" "${SAME[@]}" "+job.metadata.pool_preflight_template=true" --yes
