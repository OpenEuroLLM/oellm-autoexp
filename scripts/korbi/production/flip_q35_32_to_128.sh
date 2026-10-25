#!/bin/bash
# The 32 -> 128-node flip of the conservative (q35) run, as flip_32_to_128.sh does for the
# ambitious run. Run detached on the MN5 login node that runs the monitors:
#   setsid nohup scripts/korbi/production/flip_q35_32_to_128.sh > $R/monitor_logs/flip_q35.log 2>&1 &
# Waits until the 128-node job (q35c_128n) runs, then stops the 32-node run: its monitor first (no
# restart), then the job's worker ranks with SIGUSR2 -- Megatron's exit signal handler (exit_signal 12) saves a
# checkpoint and exits; the 128-node job is still starting up and loads the newest complete
# checkpoint (the tracker is written last). Not exited after 30 minutes: cancelled. Queued 32-node
# restarts are cancelled.
set -u
cd "$(dirname "$0")/../../.."
N128=${N128:-q35c_128n}
N32=${N32:-q35c_32n}
MONITOR_PATTERN=${MONITOR_PATTERN:-q35_conservative_32_mn5}
log() { echo "$(date '+%F %T') $*"; }
log "waiting for $N128 to run"
until squeue -u "$USER" -h -n "$N128" -t R -o %i | grep -q .; do sleep 60; done
log "128-node job running: $(squeue -u "$USER" -h -n "$N128" -t R -o '%i %N' | head -1)"
if pkill -f "$MONITOR_PATTERN"; then log "stopped the 32-node monitor"; fi
for J in $(squeue -u "$USER" -h -n "$N32" -t R -o %i); do
    # SIGUSR2 to the training ranks only (scancel --signal=USR2 --full killed torchrun, no save)
    scripts/korbi/production/signal_save_exit.sh "$J" && log "sent SIGUSR2 to the ranks of $J (save + exit)"
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
scancel -u "$USER" -n "$N32" -t PD && log "cancelled queued 32-node jobs"
log "flip done"
