#!/bin/bash
# The 32 -> 128-node flip of the ambitious run (user 2026-10-24). Run detached on the MN5 login node
# that runs the monitors:
#   setsid nohup scripts/korbi/production/flip_32_to_128.sh > $R/monitor_logs/flip_32_to_128.log 2>&1 &
# Waits until a 128-node job (ambitious_128n or ambitious_128n_e533) runs, retires the other one,
# then stops the 32-node run: first its monitor
# (no restart), then the job's training ranks with SIGUSR2 -- Megatron's exit signal handler
# (exit_signal_handler, exit_signal SIGUSR2) saves a checkpoint and exits; the 128-node job waits
# for that (WAIT_FOR_JOB_NAME=ambitious_32n in its sbatch) before it loads the newest checkpoint.
# A job that has not exited after 30 minutes is cancelled; queued 32-node restarts are cancelled.
set -u
log() { echo "$(date '+%F %T') $*"; }
# the 128-node job is queued twice (user 2026-10-25): cns144 (ambitious_128n) and ehpc533
# (ambitious_128n_e533); whichever runs first takes over, the other is retired: its monitor first
# (so it is not resubmitted), then the queued job. job name -> its monitor's config name
declare -A MONITOR=([ambitious_128n]=hybrid_ambitious_128_mn5
                    [ambitious_128n_e533]=hybrid_ambitious_128_ehpc533_mn5)
NAMES=ambitious_128n,ambitious_128n_e533
log "waiting for one of $NAMES to run"
until squeue -u "$USER" -h -n "$NAMES" -t R -o %i | grep -q .; do sleep 60; done
WINNER=$(squeue -u "$USER" -h -n "$NAMES" -t R -o %j | head -1)
log "128-node job running: $WINNER $(squeue -u "$USER" -h -n "$WINNER" -t R -o '%i %N' | head -1)"
# The cns144 job stays queued when it loses (user 2026-10-25: its actual start time is the argument
# for that account's priority): its monitor stops (no resubmits); a watcher records when it gets its
# nodes and cancels it within a minute -- still in its startup, long before a first save (it would
# otherwise train next to the winner in the same run directory). Start time -> $R/monitor_logs/
# cns144_start.log.
watch_cns144() {
    until squeue -u "$USER" -h -n ambitious_128n -t R -o %i | grep -q .; do
        squeue -u "$USER" -h -n ambitious_128n -o %i | grep -q . || { log "cns144 job left the queue unstarted"; return; }
        sleep 30
    done
    local info; info=$(squeue -u "$USER" -h -n ambitious_128n -t R -o '%i submitted=%V started=%S')
    scancel -u "$USER" -n ambitious_128n
    log "cns144 128-node job STARTED: $info -- cancelled (not needed)"
    echo "$(date '+%F %T') $info" >> "${R:-/gpfs/projects/ehpc533/users/korbip/hybrid_v019}/monitor_logs/cns144_start.log"
}
for N in ambitious_128n ambitious_128n_e533; do
    [ "$N" = "$WINNER" ] && continue
    if pkill -f "${MONITOR[$N]} "; then log "stopped the $N monitor"; fi
    if [ "$N" = ambitious_128n ]; then
        watch_cns144 &
        log "cns144 job kept queued; its start is watched (pid $!)"
    else
        scancel -u "$USER" -n "$N" && log "cancelled the other 128-node job ($N)"
    fi
done
if pkill -f "hybrid_ambitious_32_mn5 "; then log "stopped the 32-node monitor"; fi
for J in $(squeue -u "$USER" -h -n ambitious_32n -t R -o %i); do
    # SIGUSR2 to the training ranks only: scancel --signal=USR2 --full also hits torchrun, whose
    # default action (terminate) killed a q35 job without a save (signal_save_exit.sh)
    "$(dirname "$0")/signal_save_exit.sh" "$J" && log "sent SIGUSR2 to $J's ranks (save + exit)"
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
wait  # the cns144 watcher, if any
