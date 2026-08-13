#!/usr/bin/env bash
# =============================================================================
# Permanent-node FAULT INJECTOR for the resilience harness.
# =============================================================================
# Runs OUTSIDE the container, in the srun task shell (apptainer shares the host
# PID namespace, so it can see + signal the in-container training processes).
# Called from the ft_launcher slurm config's launcher_cmd as a backgrounded
# one-shot, BEFORE `apptainer exec ft_launcher ...`.
#
# WHAT IT KILLS AND WHY (learned the hard way, 2026-07-09):
#   The reliable way to exercise recovery is to kill the TRAINING WORKERS and
#   LEAVE THE ft_launcher AGENT ALIVE, so the agent observes its workers exit,
#   reports WorkerState.FAILED, and restarts the worker group (which reloads the
#   latest checkpoint at fixed world size). This is the path that recovered
#   cleanly in job 49007050.
#
#   The OLD `pkill -9 -f "python.*pretrain_gpt"` was flaky (~1/3 landed): it also
#   matched / killed the AGENT. When the agent died first, the workers ORPHANED
#   and kept training (all ranks still answered collectives), the surviving
#   agent saw everyone HEALTHY, no failure was ever detected, and training just
#   ran to the wall clock. That produced a FALSE "recovery stall".
#
# SELECTION: kill every process whose /proc cmdline contains "pretrain_gpt" but
#   NOT "ft_launcher" (the agent) and NOT "saboteur" (this script / the launcher
#   shell, whose argv embeds the whole launcher_cmd incl. pretrain_gpt.py). A
#   short BURST of passes defeats a match/visibility race at fault time, but is
#   over in ~1s so it does NOT touch the workers the agent respawns during the
#   restart (~seconds later) -> recovery is left a clean field.
#
# Env: SABOTEUR_NODE (SLURM_NODEID to hit; -1 = inert), SABOTEUR_DELAY (seconds
#      before the kill), SABOTEUR_PASSES (burst length, default 3),
#      SABOTEUR_PASS_GAP (seconds between passes, default 0.4).
# =============================================================================
node="${SABOTEUR_NODE:--1}"
delay="${SABOTEUR_DELAY:-999999}"
passes="${SABOTEUR_PASSES:-3}"
gap="${SABOTEUR_PASS_GAP:-0.4}"

# Only the target node fires; everyone else is a no-op.
[ "x${SLURM_NODEID:-x}" = "x${node}" ] || exit 0
[ "${node}" = "-1" ] && exit 0

sleep "${delay}"
# Emit the intervention wall-clock (epoch seconds) so the harness can measure the
# fault->recovery overhead. fire_epoch= is the authoritative t0 (runs on the
# target node the instant before the kill).
echo "[saboteur] node=${node} SLURM_NODEID=${SLURM_NODEID:-?} firing after ${delay}s (workers only, sparing agent) fire_epoch=$(date +%s) at $(date '+%Y-%m-%d %H:%M:%S')" >&2

total=0
for pass in $(seq 1 "${passes}"); do
  hit=0
  for p in $(pgrep -f pretrain_gpt 2>/dev/null); do
    c="$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null)" || continue
    case "$c" in
      *ft_launcher*) continue ;;   # spare the elastic AGENT (must survive to report FAILED)
      *saboteur*)    continue ;;   # spare this script / the launcher shell
      *pretrain_gpt*) : ;;         # a training worker -> kill
      *) continue ;;
    esac
    kill -9 "$p" 2>/dev/null && { hit=$((hit+1)); total=$((total+1)); }
  done
  echo "[saboteur] node=${node} pass=${pass}/${passes} killed_workers=${hit}" >&2
  sleep "${gap}"
done
echo "[saboteur] node=${node} DONE total_killed_workers=${total} done_epoch=$(date +%s)" >&2
