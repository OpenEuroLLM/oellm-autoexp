#!/usr/bin/env bash
# =============================================================================
# PERMANENT NODE-LOSS injector for hot-spare (standby promotion) validation.
# =============================================================================
# Sibling of scripts/saboteur.sh. The difference is deliberate:
#
#   saboteur.sh          kills the WORKERS, SPARES the ft_launcher agent
#                        -> the agent reports FAILED and restarts the worker
#                           group on the SAME nodes. Tests restart, not promotion.
#   saboteur_nodeloss.sh kills the workers AND the agent
#                        -> the node leaves the rendezvous entirely, so a STANDBY
#                           node has to take its group rank. Tests promotion.
#
# WHY IT SELF-SELECTS INSTEAD OF TAKING A NODE ID
# -----------------------------------------------
# Which node parks as standby is decided by the barrier rendezvous, not by SLURM,
# and it is NOT predictable from SLURM_NODEID. Observed on JUPITER job 1354846
# with 3 nodes: infra ranks 0 and 2 went ACTIVE, infra rank 1 became the standby.
# A run targeting "node 1" therefore killed nothing (total_killed_workers=0) and
# the test silently passed without ever injecting a fault.
#
# So each node decides locally: "do I actually have training workers?" Only an
# ACTIVE node does. Among those, exactly one must fire — otherwise we lose more
# nodes than we have spares and the job legitimately dies. Two guards:
#
#   1. SABOTEUR_SKIP_NODE0=1 (default) leaves SLURM_NODEID 0 alone. It is
#      normally the c10d store host, and killing it tests a different, harsher
#      thing than spare promotion.
#   2. An atomic mkdir claim on shared scratch. mkdir either creates the
#      directory or fails; there is no race. The first eligible node wins and
#      every other node exits. This is what makes "exactly one" true even when
#      several nodes are eligible.
#
# Runs OUTSIDE the container, in the srun task shell — apptainer shares the host
# PID namespace, so it can see and signal the in-container processes.
#
# Env:
#   SABOTEUR_DELAY      seconds before firing. Huge (default) = inert.
#                       Measured from the srun task shell, so it must cover
#                       container start + NCCL init + enough training iterations
#                       to have written a rolling checkpoint.
#   SABOTEUR_CLAIM_DIR  shared-FS directory for the claim (required to fire).
#   SABOTEUR_SKIP_NODE0 1 (default) = never fire on SLURM_NODEID 0.
# =============================================================================
delay="${SABOTEUR_DELAY:-999999}"
claim_dir="${SABOTEUR_CLAIM_DIR:-}"
skip0="${SABOTEUR_SKIP_NODE0:-1}"
me="${SLURM_NODEID:-?}"

log() { echo "[saboteur-nodeloss] node=${me} $*" >&2; }

# Inert unless both a sane delay and a claim directory are given.
case "${delay}" in ''|999999|-1) exit 0 ;; esac
[ -z "${claim_dir}" ] && { log "no SABOTEUR_CLAIM_DIR -> inert"; exit 0; }

sleep "${delay}"

# --- am I an ACTIVE node? (a standby node runs no training workers) ----------
# Same selection as saboteur.sh: cmdline contains pretrain_gpt, but exclude the
# ft_launcher AGENT and this script / the launcher shell (whose argv embeds the
# whole launcher_cmd, including "pretrain_gpt.py" and "saboteur").
workers=""
for p in $(pgrep -f pretrain_gpt 2>/dev/null); do
  c="$(cat "/proc/$p/cmdline" 2>/dev/null | tr '\0' ' ')" || continue
  [ -z "$c" ] && continue   # pid vanished between pgrep and read
  case "$c" in
    *ft_launcher*) continue ;;
    *saboteur*)    continue ;;
    *pretrain_gpt*) workers="${workers} $p" ;;
  esac
done

if [ -z "${workers}" ]; then
  log "no training workers here -> I am standby (or not up yet); standing down"
  exit 0
fi

if [ "${skip0}" = "1" ] && [ "${me}" = "0" ]; then
  log "active, but SLURM_NODEID 0 is skipped (rendezvous store host); standing down"
  exit 0
fi

# --- exactly one node may fire ----------------------------------------------
mkdir -p "${claim_dir}" 2>/dev/null
claim="${claim_dir}/nodeloss_claim_${SLURM_JOB_ID:-nojob}"
if ! mkdir "${claim}" 2>/dev/null; then
  log "another active node already claimed the fault; standing down"
  exit 0
fi

log "CLAIMED. firing after ${delay}s: killing workers AND agent (permanent node loss) fire_epoch=$(date +%s) at $(date '+%Y-%m-%d %H:%M:%S')"

# --- kill the workers, then the agent ----------------------------------------
# Workers first so the agent cannot respawn them, then the agent so the node
# actually leaves the rendezvous. A short burst defeats a match/visibility race.
total=0
for pass in 1 2 3; do
  for p in $(pgrep -f pretrain_gpt 2>/dev/null); do
    c="$(cat "/proc/$p/cmdline" 2>/dev/null | tr '\0' ' ')" || continue
  [ -z "$c" ] && continue   # pid vanished between pgrep and read
    case "$c" in
      *saboteur*)     continue ;;                 # never self-kill
      *ft_launcher*)  : ;;                        # the AGENT -> kill it too
      *pretrain_gpt*) : ;;                        # a worker  -> kill
      *)              continue ;;
    esac
    kill -9 "$p" 2>/dev/null && total=$((total+1))
  done
  sleep 0.4
done

log "DONE total_killed=${total} done_epoch=$(date +%s)"
