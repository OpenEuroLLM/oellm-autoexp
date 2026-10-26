#!/bin/bash
# The MHA -> GQA switch of the ambitious run (user 2026-10-26: fresh start with GQA, no checkpoint
# conversion). The GQA run (suite main_gqa) is queued at 128 nodes twice: ehpc533
# (ambitious_gqa_128n_e533) and cns144 (ambitious_gqa_128n). When either runs:
#   1. the other GQA job is retired: its monitor (by PID), then the queued job;
#   2. the MHA run (suite main, ambitious_128n_e533) is stopped: its monitor (by PID -- the MHA and
#      GQA monitors share config names, a pkill -f pattern would hit both), then SIGUSR2 to its
#      training ranks (a final MHA checkpoint for reference), cancelled if not out in 30 min.
# The MHA cns144 job (ambitious_128n) stays queued: flip_32_to_128.sh's watcher records its start
# (account priority evidence) and cancels it. Run detached on the MN5 login node:
#   setsid nohup scripts/korbi/production/flip_mha_to_gqa.sh > $R/monitor_logs/flip_mha_to_gqa.log 2>&1 &
set -u
R=${R:-/gpfs/projects/ehpc533/users/korbip/hybrid_v019}
log() { echo "$(date '+%F %T') $*"; }
stop_monitor() {  # monitor name (monitor_logs/<name>.pid): kill only if it runs that config
    local name=$1 config=$2 pid
    pid=$(cat "$R/monitor_logs/$name.pid" 2>/dev/null) || { log "no pid file for $name"; return; }
    if ps -o args= -p "$pid" 2>/dev/null | grep -q -- "--config-name experiments/korbi/production/$config "; then
        kill "$pid" && log "stopped the $name monitor (pid $pid)"
    else
        log "the $name monitor (pid $pid) is not running $config: left alone"
    fi
}
GQA=ambitious_gqa_128n,ambitious_gqa_128n_e533
log "waiting for one of $GQA to run"
until squeue -u "$USER" -h -n "$GQA" -t R -o %i | grep -q .; do sleep 60; done
WINNER=$(squeue -u "$USER" -h -n "$GQA" -t R -o %j | head -1)
log "GQA job running: $WINNER $(squeue -u "$USER" -h -n "$WINNER" -t R -o '%i %N' | head -1)"
if [ "$WINNER" = ambitious_gqa_128n_e533 ]; then
    stop_monitor ambitious_gqa128 hybrid_ambitious_128_mn5
    scancel -u "$USER" -n ambitious_gqa_128n && log "cancelled the other GQA job (ambitious_gqa_128n)"
else
    stop_monitor ambitious_gqa128e533 hybrid_ambitious_128_ehpc533_mn5
    scancel -u "$USER" -n ambitious_gqa_128n_e533 && log "cancelled the other GQA job (ambitious_gqa_128n_e533)"
fi
stop_monitor ambitious128e533 hybrid_ambitious_128_ehpc533_mn5
for J in $(squeue -u "$USER" -h -n ambitious_128n_e533 -t R -o %i); do
    "$(dirname "$0")/signal_save_exit.sh" "$J" && log "sent SIGUSR2 to $J's ranks (final MHA save + exit)"
    for _ in $(seq 60); do
        squeue -h -j "$J" -o %i 2>/dev/null | grep -q . || break
        sleep 30
    done
    if squeue -h -j "$J" -o %i 2>/dev/null | grep -q .; then
        scancel "$J"; log "cancelled $J (no exit within 30 min)"
    else
        log "$J exited"
    fi
done
scancel -u "$USER" -n ambitious_128n_e533 -t PD && log "cancelled queued MHA restarts"
log "switch done"
