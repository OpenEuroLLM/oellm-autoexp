#!/bin/bash
# Fresh-start switch of the conservative run from run_2 (MHA by mistake) to run_3 (GQA; user
# 2026-10-26). Run detached on the MN5 login node that runs the monitors:
#   setsid nohup scripts/korbi/production/flip_q35_run2_to_run3.sh > $R/monitor_logs/flip_q35_run3.log 2>&1 &
# Waits until one run_3 128-node job runs (q35c_run_3_128e on ehpc533 or q35c_run_3_128n on
# cns144), cancels the other one and its monitor, stops run_2's monitor (by PID: run_2 and run_3
# monitors share the config name), then asks run_2's job to save a final MHA checkpoint and exit
# (SIGUSR2 to its ranks; reference for MHA vs GQA), cancelling it after 30 minutes otherwise.
set -u
cd "$(dirname "$0")/../../.."
R=${R:-/gpfs/projects/ehpc533/users/korbip/hybrid_v019}
# run_3 candidates: job name:monitor pid file
N3S=${N3S:-"q35c_run_3_128e:$R/monitor_logs/q35c_run_3_128e.pid q35c_run_3_128n:$R/monitor_logs/q35c_run_3_128n.pid"}
RUN2_JOB_NAME=${RUN2_JOB_NAME:-q35c_128e}
RUN2_MONITOR_PID_FILE=${RUN2_MONITOR_PID_FILE:-$R/monitor_logs/q35c_128e.pid}
log() { echo "$(date '+%F %T') $*"; }
stop_pid() { [ -f "$1" ] && kill "$(cat "$1")" 2>/dev/null && log "stopped monitor pid $(cat "$1") ($1)"; }
log "waiting for one of: $N3S"
RUNNING=""
until [ -n "$RUNNING" ]; do
    for c in $N3S; do
        if squeue -u "$USER" -h -n "${c%%:*}" -t R -o %i | grep -q .; then RUNNING=$c; break; fi
    done
    [ -n "$RUNNING" ] || sleep 60
done
log "run_3 job running: ${RUNNING%%:*} $(squeue -u "$USER" -h -n "${RUNNING%%:*}" -t R -o '%i %N' | head -1)"
for c in $N3S; do
    [ "$c" = "$RUNNING" ] && continue
    stop_pid "${c#*:}"
    scancel -u "$USER" -n "${c%%:*}" && log "cancelled ${c%%:*}"
done
stop_pid "$RUN2_MONITOR_PID_FILE"
for J in $(squeue -u "$USER" -h -n "$RUN2_JOB_NAME" -t R -o %i); do
    scripts/korbi/production/signal_save_exit.sh "$J" && log "sent SIGUSR2 to the ranks of run_2 job $J (final MHA save + exit)"
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
scancel -u "$USER" -n "$RUN2_JOB_NAME" -t PD && log "cancelled queued run_2 jobs"
log "switch done"
