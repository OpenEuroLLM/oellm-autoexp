#!/bin/bash
# Round-5 auto-start (docs/hybrid_moe_3b_campaign.md §12), detached on the MN5 login node.
# 1. wait for the shared-index smoke (h5_moe, job $CACHE_JOB) to reach step 40
# 2. smoke the not-yet-verified s1 arms and the s3 layout (two suites: no task-script race)
# 3. launch screen5 s1 / s2 / s3 with the arms whose smokes passed
set -u
R=/gpfs/projects/ehpc533/users/korbip/hybrid_v019
A=$R/autoexp
RUNS=/gpfs/scratch/ehpc533/users/korbip/hybrid_v019_runs/autoexp/campaign_3b
ST=$R/monitor_logs/round5_autostart.status
CACHE_JOB=$1
say() { echo "$(date '+%m-%d %H:%M') $*" >> $ST; }
passed() {  # suite arm -> 0 if some log of that arm reached step 40 without a traceback
  local s=$1 a=$2
  for f in $(grep -l -a " save \.* .*/$s/$a/ckpt" $RUNS/$s/$s/slurm-*.log 2>/dev/null); do
    grep -a -q "iteration  *40/" $f && ! grep -a -q "Traceback" $f && return 0
  done
  return 1
}
wait_jobs() { while squeue -u $USER -h -o %j | grep -q "$1"; do sleep 120; done; }
cd $A
say "start; waiting for the shared-index smoke (job $CACHE_JOB)"
while squeue -h -j $CACHE_JOB | grep -q .; do sleep 120; done
if ! passed screen5smoke_s1 h5_moe; then say "ABORT: the shared-index smoke (h5_moe) did not reach step 40"; exit 1; fi
say "shared index built ($(du -sh /gpfs/scratch/ehpc533/users/korbip/hybrid_v019_runs/data_cache_flag8k_r5 | cut -f1)); smoking the remaining s1 arms and s3"
NEW1="attn_moe,qwen35_moe,h5_moe_rr28_mix,h5_moe_lat2,h5_moe_rr28_mix_lat2"
scripts/korbi/mn5_autoexp_bg.sh r5smoke_s1b scripts/run_autoexp.py --yes --config-name experiments/korbi/campaign_3b/screen5smoke_s1_mn5 "aux.only=[$NEW1]" >/dev/null
scripts/korbi/mn5_autoexp_bg.sh r5smoke_s3 scripts/run_autoexp.py --yes --config-name experiments/korbi/campaign_3b/screen5smoke_s3_mn5 >/dev/null
sleep 300
wait_jobs screen5smoke
ALL1="attn_moe qwen35_moe h5_moe h5_moe_mhc4 h5_moe_mhc2 h5_moe_rr28 h5_moe_attnres h5_moe_g2 h5_moe_rr28_mix h5_moe_lat2 h5_moe_rr28_mix_lat2 h5_moe_seed2 h5_moe_rr28_seed2"
OK1=""; BAD1=""
for a in $ALL1; do if passed screen5smoke_s1 $a; then OK1="$OK1,$a"; else BAD1="$BAD1 $a"; fi; done
OK3=""; BAD3=""
for a in qwen35_moe h5_moe h5_moe_mhc4 h5_moe_rr28; do if passed screen5smoke_s3 $a; then OK3="$OK3,$a"; else BAD3="$BAD3 $a"; fi; done
OK1=${OK1#,}; OK3=${OK3#,}
OK2=$(echo $OK1 | tr ',' '\n' | grep -v seed2 | paste -sd,)
say "smoke: s1 passed [$OK1] failed [$BAD1]; s3 passed [$OK3] failed [$BAD3]"
[ -n "$OK1" ] && scripts/korbi/mn5_autoexp_bg.sh round5_s1 scripts/run_autoexp.py --yes --config-name experiments/korbi/campaign_3b/screen5_s1_mn5 "aux.only=[$OK1]" >/dev/null && say "launched round5 s1: [$OK1]"
[ -n "$OK2" ] && scripts/korbi/mn5_autoexp_bg.sh round5_s2 scripts/run_autoexp.py --yes --config-name experiments/korbi/campaign_3b/screen5_s2_mn5 "aux.only=[$OK2]" >/dev/null && say "launched round5 s2: [$OK2]"
[ -n "$OK3" ] && scripts/korbi/mn5_autoexp_bg.sh round5_s3 scripts/run_autoexp.py --yes --config-name experiments/korbi/campaign_3b/screen5_s3_mn5 "aux.only=[$OK3]" >/dev/null && say "launched round5 s3: [$OK3]"
sleep 120
say "queue: $(squeue -u $USER -h -o '%j' | grep -c screen5_) round-5 jobs"
say "done"
