#!/bin/bash
# The 32 -> 128-node flip of the ambitious run (user 2026-10-24). Run detached on the MN5 login node
# that runs the monitors:
#   setsid nohup scripts/korbi/production/flip_32_to_128.sh > $R/monitor_logs/flip_32_to_128.log 2>&1 &
# Waits until the 128-node job (ambitious_128n) runs, then stops the 32-node run: first its monitor
# (no restart), then the job itself with SIGUSR2 -- Megatron's exit signal handler
# (exit_signal_handler, exit_signal 12) saves a checkpoint and exits; the 128-node job is still in
# its startup (> 10 min) and loads the newest complete checkpoint (the tracker is written last).
# A job that has not exited after 30 minutes is cancelled; queued 32-node restarts are cancelled.
set -u
log() { echo "$(date '+%F %T') $*"; }
log "waiting for ambitious_128n to run"
until squeue -u "$USER" -h -n ambitious_128n -t R -o %i | grep -q .; do sleep 60; done
log "128-node job running: $(squeue -u "$USER" -h -n ambitious_128n -t R -o '%i %N' | head -1)"
if pkill -f "hybrid_ambitious_32_mn5"; then log "stopped the 32-node monitor"; fi
for J in $(squeue -u "$USER" -h -n ambitious_32n -t R -o %i); do
    scancel --signal=USR2 --full "$J" && log "sent SIGUSR2 to $J (save + exit)"
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
scancel -u "$USER" -n ambitious_32n -t PD && log "cancelled queued 32-node jobs"
log "flip done"
