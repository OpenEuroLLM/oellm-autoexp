#!/bin/bash
# Rolling watchdog for the oellm_32b_dense campaign. Run ON jupiter.
#
# Polls the 1024-node speed test, the 1024-node 8k stability run, and the seven
# 16-node arms. Exits (so the caller is notified) on anything that needs a human
# decision; otherwise keeps quiet and periodically re-syncs wandb.
#
#   SPEED_JOB   1024-node speed test (50 iters)
#   STAB_JOB    1024-node stability run (8000 iters)
#
# Exit reasons are printed as "EVENT: ...".
set -u
B=/e/project1/e-sta-openeurollm/pre_production_training
SPEED_JOB=${SPEED_JOB:-}
STAB_JOB=${STAB_JOB:-}
POLL=${POLL:-300}
MAX_POLLS=${MAX_POLLS:-60}
SYNC_EVERY=${SYNC_EVERY:-6}     # every N polls
TARGET=${TARGET:-8000}
STALL_POLLS=${STALL_POLLS:-4}   # consecutive polls with no iteration progress -> alert

# Failure signatures that mean "already dead, stop waiting" (mirrors
# config/job/auto_cancel.yaml).
# Failure signatures. FT-AWARE: only faults that make progress impossible no
# matter how often ft_launcher retries. Rank-level fault strings
# (terminate called / ChildFailedError / RendezvousClosedError / NCCL / IMA)
# are DELIBERATELY ABSENT: under ft_launcher they appear and are then RECOVERED,
# so alerting on them stops the watchdog on a healthy run. Two 1024-node runs
# (1379130, 1380778) were cancelled that way, and this watchdog then tripped on
# 1380817 while it was advancing normally. Non-recoverable stalls are caught by
# the ITERATION-counter stall check below instead. Mirrors config/job/auto_cancel_ft.yaml.
ERR_RX='unrecognized arguments|Disk quota exceeded|Saved FP8 metadata does not match|mmap length is greater than file size|Invalid infrastructure rank|torch\.cuda\.OutOfMemoryError|CUDA out of memory\. Tried to allocate'

logfor() { ls -t "$B"/*"$1"*/logs/slurm-"$2".log 2>/dev/null | head -1; }
iters()  { [ -f "${1:-}" ] && grep -ao 'iteration *[0-9]*/' "$1" 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0; }
state()  { sacct -j "$1" --format=State -n -X 2>/dev/null | head -1 | tr -d ' '; }

for i in $(seq 1 "$MAX_POLLS"); do
  echo "--- poll $i  $(date '+%F %T') ---"

  for spec in "SPEED:$SPEED_JOB:fp8sp1024-gradfp32:50" "STAB:$STAB_JOB:sc-fp8-zloss1e-4-nobsramp:$TARGET"; do
    tag=${spec%%:*}; rest=${spec#*:}; job=${rest%%:*}; rest=${rest#*:}
    dir=${rest%%:*}; goal=${rest##*:}
    [ -z "$job" ] && continue
    st=$(state "$job"); L=$(logfor "$dir" "$job"); it=$(iters "$L")
    tf=$(grep -a "TFLOP/s/GPU" "$L" 2>/dev/null | grep -oE 'TFLOP/s/GPU\): [0-9.]+' | grep -oE '[0-9.]+' \
         | tail -100 | sort -n | awk '{a[NR]=$1} END {if(NR>0) printf "%.0f(%.0f-%.0f)", a[int(NR/2)], a[1], a[NR]}')
    echo "  $tag $job state=$st iters=${it:-0}/$goal tflops=${tf:-n/a}"

    err=$(grep -aoE "$ERR_RX" "$L" 2>/dev/null | head -1)
    [ -n "$err" ] && { echo "EVENT: $tag ($job) error signature: $err"; exit 0; }

    # STALL DETECTION. Added after job 1375720 sat at iteration 100 for 4 h
    # (~4000 node-hours) while its log kept GROWING with restart banners — so
    # neither an error pattern nor an inactivity-on-log check could catch it.
    # Progress is measured by the ITERATION COUNTER, not by log activity.
    prev_var="prev_it_$tag"; stuck_var="stuck_$tag"
    prev=$(eval "echo \${$prev_var:-}"); stuck=$(eval "echo \${$stuck_var:-0}")
    if [ "$st" = "RUNNING" ] && [ -n "$prev" ] && [ "${it:-0}" = "$prev" ]; then
      stuck=$((stuck + 1))
      if [ "$stuck" -ge "$STALL_POLLS" ]; then
        echo "EVENT: $tag ($job) STALLED at iteration ${it:-0} for $stuck polls (~$((stuck*POLL/60)) min) — state still RUNNING"; exit 0
      fi
    else
      stuck=0
    fi
    eval "$prev_var=\${it:-0}"; eval "$stuck_var=$stuck"
    case "$st" in
      FAILED|TIMEOUT|NODE_FAIL|CANCELLED*) echo "EVENT: $tag ($job) terminal state=$st iters=${it:-0}"; exit 0;;
      COMPLETED) echo "EVENT: $tag ($job) COMPLETED iters=${it:-0}"; exit 0;;
    esac
  done

  # 16-node arms: report progress, and flag any that reached the target (its
  # queued afterany continuation would otherwise resume and run to 2x TARGET).
  for d in "$B"/oellm_32b_dense_sc-*_gbs128_lr2e-4; do
    [ -d "$d" ] || continue
    nm=$(basename "$d" | sed 's/oellm_32b_dense_//; s/_gbs128_lr2e-4//')
    L=$(ls -t "$d"/logs/slurm-*.log 2>/dev/null | head -1); [ -z "$L" ] && continue
    j=$(basename "$L" | tr -dc '0-9'); it=$(iters "$L")
    printf "  n16 %-22s %s it=%s\n" "$nm" "$j" "${it:-0}"
    # Only alert if the arm reached TARGET *and* a continuation is still queued.
    # Once that job is cancelled the arm is genuinely finished, so re-alerting
    # would make the watchdog exit on every poll and stop watching everything else.
    if [ "${it:-0}" -ge "$TARGET" ] 2>/dev/null; then
      # Count PENDING *and* RUNNING continuations. Checking only PENDING misses
      # the case where the continuation already launched and is overshooting —
      # which is exactly what happened twice: job 1370565 (caught after 2 min)
      # and job 1379770 (ran 8001->10885 unnoticed, ~24 node-hours).
      cont=$(squeue -u "$USER" -h -o "%i %T %j" 2>/dev/null \
             | awk -v n="$nm" -v cur="$j" '$3 ~ n"_gbs128" && $1 != cur && ($2=="PENDING" || $2=="RUNNING") {c++} END {print c+0}')
      if [ "${cont:-0}" -gt 0 ]; then
        echo "EVENT: n16 arm $nm ($j) reached $TARGET with $cont continuation(s) pending/RUNNING — cancel to avoid running to $((TARGET*2))"; exit 0
      fi
    fi
  done

  if [ $((i % SYNC_EVERY)) -eq 0 ]; then
    ( cd ~/work/Projects/oellm-autoexp && source ~/work/venv/bin/activate 2>/dev/null
      export WANDB_SILENT=true
      for dd in "$B"/oellm_32b_dense_sc-*_gbs128_lr2e-4 "$B"/oellm_32b_dense_sc-fp8-zloss1e-4-nobsramp_gbs4096_lr2e-4; do
        [ -d "$dd" ] && python3 scripts/sync_runs.py --folder "$(basename "$dd")" --results-dir "$B" --continue-on-error >/dev/null 2>&1
      done ) && echo "  [wandb re-synced]"
  fi
  sleep "$POLL"
done
echo "EVENT: watchdog reached MAX_POLLS without a terminal event"
