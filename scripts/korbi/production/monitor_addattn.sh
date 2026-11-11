#!/bin/bash
# Status of the ckda_addattn campaign (docs/recall_surgery_ambitious.md) into $R/monitor_addattn.log every
# 2 minutes: pool state, which arm holds which nodes, each arm's checkpoints, its last `iteration N/M` line and
# the last `[ckda_branch]` line (alpha / m / gamma / rms_ckda / rms_branch, i.e. whether the branch is really
# on), and the surgery outputs. Run detached on the login node:
#   setsid nohup scripts/korbi/production/monitor_addattn.sh >/dev/null 2>&1 </dev/null &
# It is a read-only observer: nothing here touches a run or a SLURM job.
set -u
P=${POOL:-/gpfs/scratch/ehpc533/hybrid_nucleus/pools/mn5_pool_o134c}
R=${R:-/gpfs/projects/ehpc533/users/korbip/hybrid_v019}
D=${ADDATTN_DIR:-/gpfs/scratch/ehpc533/hybrid_nucleus/production/ckda_addattn}
LOG=$R/monitor_addattn.log
PY=$R/autoexp_env/bin/python
SAY=${SAY:-}
INTERVAL=${INTERVAL:-120}

while true; do
  {
    echo "=== $(date -Iseconds) state=$(cat "$P/hosts/host002/state" 2>/dev/null) assigned=$(ls "$P/hosts/host002/assign" 2>/dev/null | wc -l)"
    "$PY" "$R/autoexp_poolshrink1/scripts/pool_ctl.py" "$P" status 2>/dev/null |
      grep -E 'ckda_addattn|ckda_branch_surgery|host002' | head -14
    for a in ctrl add6 add6_rope add3 add6b rope; do
      ck=$D/$a/ckpt
      [ -d "$ck" ] && echo "  $a ckpt: $(ls "$ck" 2>/dev/null | tr '\n' ' ')"
      f=$(ls -t "$D/$a"/slurm-*.log 2>/dev/null | head -1)
      if [ -n "${f:-}" ]; then
        echo "  $a $(basename "$f"): $(grep -oE 'iteration +[0-9]+/ *[0-9]+' "$f" 2>/dev/null | tail -1)" \
             "$(grep -oE '\[ckda_branch\][^|]*' "$f" 2>/dev/null | tail -1 | cut -c1-150)"
      fi
    done
    for a in add6 add3 add6b; do
      s=$D/${a}_src/ckpt
      [ -f "$s/latest_checkpointed_iteration.txt" ] && \
        echo "  surgery $a: done (iter $(cat "$s/latest_checkpointed_iteration.txt"))"
    done
  } >> "$LOG" 2>&1
  [ -n "$SAY" ] && tail -14 "$LOG"
  sleep "$INTERVAL"
done
