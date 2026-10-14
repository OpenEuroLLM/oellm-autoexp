#!/bin/bash
# Status snapshot for the oellm_32b_dense campaign: 1024-node speed test,
# 1024-node 8k stability run, and the seven 16-node stability arms.
# Run ON jupiter (needs squeue/sacct + the production_training tree).
BASE=/e/project1/e-sta-openeurollm/pre_production_training
STEPS_TARGET=${STEPS_TARGET:-8000}

iters_of() {   # $1 = log path -> "iter/target"
  [ -f "$1" ] || { echo "-"; return; }
  local it
  it=$(grep -ao 'iteration *[0-9]*/' "$1" 2>/dev/null | tail -1 | tr -dc '0-9')
  echo "${it:-0}"
}
tflops_of() {  # $1 = log -> "median(min-max) n"
  [ -f "$1" ] || { echo "-"; return; }
  grep -a "TFLOP/s/GPU" "$1" 2>/dev/null \
    | grep -oE 'TFLOP/s/GPU\): [0-9.]+' | grep -oE '[0-9.]+' | tail -200 | sort -n \
    | awk '{a[NR]=$1} END {if(NR>0) printf "%.0f (%.0f-%.0f) n=%d", a[int(NR/2)], a[1], a[NR], NR; else printf "-"}'
}

echo "############ $(date '+%F %T') ############"
echo
echo "=== QUEUE (>=512 nodes) ==="
squeue -u "$USER" -o "%.10i %.10T %.6D %.42j %.12L %R" -h 2>/dev/null | awk '$3>=512' | cut -c1-150
echo
echo "=== 16-NODE STABILITY ARMS (target ${STEPS_TARGET} steps) ==="
printf "  %-26s %-9s %-8s %-8s %s\n" ARM JOB STATE ITERS "TFLOP/s med(min-max)"
for d in "$BASE"/oellm_32b_dense_sc-*_gbs128_lr2e-4*/; do
  [ -d "$d" ] || continue
  name=$(basename "$d" | sed 's/oellm_32b_dense_//; s/_gbs128_lr2e-4.*//')
  log=$(ls -t "$d"logs/slurm-*.log 2>/dev/null | head -1)
  [ -z "$log" ] && continue
  jid=$(basename "$log" | tr -dc '0-9')
  st=$(sacct -j "$jid" --format=State -n -X 2>/dev/null | head -1 | tr -d ' ')
  printf "  %-26s %-9s %-8s %-8s %s\n" "$name" "$jid" "${st:0:8}" "$(iters_of "$log")" "$(tflops_of "$log")"
done
echo
echo "=== 1024-NODE RUNS ==="
printf "  %-30s %-9s %-10s %-8s %s\n" RUN JOB STATE ITERS "TFLOP/s med(min-max)"
for d in "$BASE"/oellm_32b_dense_*1024*_gbs4096_lr2e-4/ "$BASE"/oellm_32b_dense_sc-fp8-zloss1e-4-nobsramp_gbs4096_lr2e-4/; do
  [ -d "$d" ] || continue
  name=$(basename "$d" | sed 's/oellm_32b_dense_//; s/_gbs4096_lr2e-4//')
  log=$(ls -t "$d"logs/slurm-*.log 2>/dev/null | head -1)
  [ -z "$log" ] && continue
  jid=$(basename "$log" | tr -dc '0-9')
  st=$(sacct -j "$jid" --format=State -n -X 2>/dev/null | head -1 | tr -d ' ')
  printf "  %-30s %-9s %-10s %-8s %s\n" "$name" "$jid" "${st:0:10}" "$(iters_of "$log")" "$(tflops_of "$log")"
done
echo
echo "=== WANDB ==="
echo -n "  mode: "; grep -rhoE 'WANDB_MODE: *[a-z]+' ~/work/Projects/oellm-autoexp/config/slurm/jupiter.yaml 2>/dev/null | head -1
find "$BASE" -maxdepth 3 -name "*.wandb" -newermt "-45 minutes" 2>/dev/null | wc -l | sed 's/^/  .wandb files touched in last 45min: /'
