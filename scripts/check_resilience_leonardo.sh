#!/usr/bin/env bash
# =============================================================================
# Leonardo FIXED-WORLD HOT-SPARE resilience check (end-to-end).
# =============================================================================
# Submits the fixed-world hot-spare config, injects a PERMANENT node failure via
# the saboteur, monitors the SLURM log, and VERIFIES:
#   1. RESERVE MODE  - world initializes at N active (min-healthy), a spare held
#   2. LOCAL SAVES   - the frequent /dev/shm local checkpoint is being written
#   3. FAULT FIRES   - the saboteur permanently kills an active node
#   4. SPARE PROMOTED- the world re-forms at N (fixed size), no "Invalid
#                      infrastructure rank", not shrunk, not force-terminated
#   5. RELOAD        - it resumes from a checkpoint (NOT "start from random")
#   6. WEIGHT CORRECTNESS - loss continuity: the loss right after the reload is
#                      ~ the loss the run had at the reloaded iteration. A
#                      mismatched-shard / wrong-weights reload shows up as a loss
#                      SPIKE (toward ln(vocab)) and FAILS this check. This is the
#                      one that catches a "resumed, but from the wrong weights".
#
# Run ON Leonardo (this script does NOT ssh anywhere itself):
#   ssh leonardo "cd ~/work/Projects/oellm-autoexp && bash scripts/check_resilience_leonardo.sh"
#
# Env knobs (all overridable):
#   CONFIG            experiment config (default: experiments/korbi/megatron_moe_resilient_leonardo)
#   ACCOUNT           SLURM account   (default: euhpc_d29_026; login env's OELLM_prod2026 is invalid)
#   QOS               SLURM qos       (default: boost_qos_dbg; falls back to normal)
#   NVRX_SIDECAR      patched NVRx dir (default: .../nvrx05_patched, use_infra_group_rank=False)
#   ACTIVE_WORLD      expected GPU world size of the ACTIVE set (default: 8 = 2 nodes x 4)
#   SABOTEUR_NODE     active SLURM_NODEID to kill (default: 1)
#   SABOTEUR_DELAY    seconds before the kill    (default: 150)
#   CLEAR_GLOBAL      1 -> clear the global ckpt dir so a fallback = obvious cold-start (default: 1)
#   LOSS_SPIKE_FACTOR post-reload loss must be < factor * pre-fault loss at reload iter (default: 3.0)
#   MIN_POST_ITERS    post-reload iters to collect before cancelling (default: 5)
#   POLL_SECS/MAX_POLLS  monitor cadence (default: 20 / 90)
#   MAX_SUBMITS       resubmit attempts on transient sbatch failure (default: 3)
#   OUTPUT_BASE / GLOBAL_CKPT_DIR   paths on /leonardo_scratch
#   REPEAT / --repeat N  run the whole check N times, tally PASS/FAIL, print the
#                        recovery success RATE (each run streams its own output,
#                        incl. the STALL DIAGNOSIS on any hang). Default 1.
set -uo pipefail

# --- repeat mode wrapper (must run before anything else) --------------------
REPEAT="${REPEAT:-1}"
while [ $# -gt 0 ]; do
  case "$1" in
    --repeat)   REPEAT="${2:-1}"; shift 2 ;;
    --repeat=*) REPEAT="${1#*=}"; shift ;;
    *)          shift ;;
  esac
done
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"   # absolute, for a clean re-exec
if [ "${REPEAT}" -gt 1 ] 2>/dev/null; then
  echo "[leo-resil] ###### REPEAT MODE: ${REPEAT} runs (measuring recovery success rate) ######"
  npass=0; nfail=0; ninc=0; results=()
  for run in $(seq 1 "${REPEAT}"); do
    echo "[leo-resil]"
    echo "[leo-resil] ########################## RUN ${run}/${REPEAT} ##########################"
    REPEAT=1 bash "${SELF}"   # single pass; inherits any exported env knobs, streams its own output
    rc=$?
    case "${rc}" in
      0) npass=$((npass+1)); results+=("run ${run}: PASS") ;;
      2) ninc=$((ninc+1));   results+=("run ${run}: INCONCLUSIVE") ;;
      *) nfail=$((nfail+1)); results+=("run ${run}: FAIL (see its STALL DIAGNOSIS above)") ;;
    esac
  done
  echo "[leo-resil]"
  echo "[leo-resil] ################## REPEAT SUMMARY (${REPEAT} runs) ##################"
  for r in "${results[@]}"; do echo "[leo-resil]   ${r}"; done
  echo "[leo-resil] PASS=${npass}  FAIL=${nfail}  INCONCLUSIVE=${ninc}  =>  recovery success rate = ${npass}/${REPEAT}"
  exit 0
fi

CONFIG="${CONFIG:-experiments/korbi/megatron_moe_resilient_leonardo}"
# The saboteur lives in a slurm config; override the group so the fault fires
# regardless of the experiment's default slurm (empty -> use the config's own).
SLURM_CFG="${SLURM_CFG:-leonardo_ftlauncher_saboteur}"
ACCOUNT="${ACCOUNT:-euhpc_d29_026}"
QOS="${QOS:-boost_qos_dbg}"
NVRX_SIDECAR="${NVRX_SIDECAR:-/leonardo_scratch/large/userexternal/kpoeppel/nvrx05_patched}"
ACTIVE_WORLD="${ACTIVE_WORLD:-8}"
SABOTEUR_NODE="${SABOTEUR_NODE:-1}"
SABOTEUR_DELAY="${SABOTEUR_DELAY:-150}"
# Hard SLURM time cap so a HUNG recovery is killed in minutes, not left idling to
# the default limit (a single hang otherwise wastes ~nodes*time GPU-hr).
TIME_LIMIT="${TIME_LIMIT:-00:08:00}"
# NCCL_DEBUG so a re-init failure prints WARNings into the log; the stall
# diagnosis below uses them to tell a rendezvous stall from an NCCL re-init fail.
# WARN is enough (INFO is huge); override to INFO for a deep dive.
NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
CLEAR_GLOBAL="${CLEAR_GLOBAL:-1}"
LOSS_SPIKE_FACTOR="${LOSS_SPIKE_FACTOR:-3.0}"
MIN_POST_ITERS="${MIN_POST_ITERS:-5}"
POLL_SECS="${POLL_SECS:-20}"
MAX_POLLS="${MAX_POLLS:-90}"
MAX_SUBMITS="${MAX_SUBMITS:-3}"
OUTPUT_BASE="${OUTPUT_BASE:-/leonardo_scratch/large/userexternal/kpoeppel/output}"
GLOBAL_CKPT_DIR="${GLOBAL_CKPT_DIR:-/leonardo_scratch/large/userexternal/kpoeppel/output/resilient_demo/ckpt}"

cd "$(dirname "$0")/.." || { echo "cannot cd to repo root"; exit 2; }
[ -d ~/work/venv ] && source ~/work/venv/bin/activate 2>/dev/null

BASE_NAME="$(basename "$CONFIG")"
LOG_DIR="${OUTPUT_BASE}/${BASE_NAME}"
NODE_ROUND=$(( ACTIVE_WORLD / 4 ))   # node-level rendezvous size (4 GPUs/node on booster)
# The ft_launcher AGENT's failure-detection signature -- the authoritative "a
# real fault landed and recovery began" marker. We anchor everything on this
# instead of srun's "task N: Killed" (which is absent now that the saboteur
# spares the agent, and which fired even when workers survived).
DETECT_RE="state\.name='FAILED'|restarting worker group|worker_state_cnt=Counter\("

say() { echo "[leo-resil] $*"; }

# --- submit once; echo the job id ------------------------------------------
submit_once() {
  # ++ = force add-or-override (works whether or not the slurm config already
  # defines these). NB: the saboteur only actually fires if the config's slurm
  # carries the gated saboteur launcher_cmd (leonardo_ftlauncher*); if it uses a
  # plain slurm config, check #3 (fault fired) will FAIL and say so.
  local overrides=(
    "++slurm.env.SABOTEUR_NODE=\"${SABOTEUR_NODE}\""
    "++slurm.env.SABOTEUR_DELAY=\"${SABOTEUR_DELAY}\""
  )
  [ -n "${SLURM_CFG}" ] && overrides=("slurm=${SLURM_CFG}" "${overrides[@]}")
  overrides+=("++slurm.sbatch.time=\"${TIME_LIMIT}\"")   # cap wasted time on a hang
  overrides+=("++backend.env.NCCL_DEBUG=\"${NCCL_DEBUG}\"")   # so a re-init fail prints WARNings
  SLURM_ACCOUNT="${ACCOUNT}" SLURM_QOS="${QOS}" NVRX_SIDECAR="${NVRX_SIDECAR}" PYTHONPATH=. \
    python scripts/run_autoexp.py --config-name "${CONFIG}" --submit-and-exit "${overrides[@]}" 2>&1 \
    | grep -oE "Submitted job \([0-9]+\)" | grep -oE "[0-9]+" | head -1
}

log_path_for() { echo "${LOG_DIR}/slurm-$1.log"; }

# --- numeric helpers (bash has no floats -> use awk) -----------------------
# lm loss logged at a given iteration, restricted to lines BEFORE $3 (0 = whole file)
loss_at_iter() {  # $1=iter  $2=log  $3=before_line
  awk -v it="$1" -v bl="${3:-0}" '
    (bl==0 || NR<bl) && $0 ~ ("iteration +" it "/") {
      if (match($0, /lm loss: [0-9.eE+-]+/)) v = substr($0, RSTART+9, RLENGTH-9)
    } END { if (v!="") print v }' "$2"
}
first_loss_after_line() {  # $1=line  $2=log
  awk -v k="$1" 'NR>k && match($0, /lm loss: [0-9.eE+-]+/) { print substr($0, RSTART+9, RLENGTH-9); exit }' "$2"
}

# Classify WHY a recovery didn't complete: rendezvous stall (before re-init) vs
# NCCL re-init failure (after re-rendezvous). Needs NCCL_DEBUG>=WARN in the job.
diagnose_stall() {  # $1=log  $2=kill_line
  local L="$1" kl="${2:-0}"
  echo
  say "----------------- STALL DIAGNOSIS (recovery did not complete) -----------------"
  # after the fault: did a NEW rendezvous round form? did Megatron re-init? NCCL errors?
  local re_round re_init rdzv nccl
  re_round="$(awk -v k="$kl" 'NR>k' "$L" 2>/dev/null | grep -acE "joined round [1-9]")"
  re_init="$( awk -v k="$kl" 'NR>k' "$L" 2>/dev/null | grep -acE "using world size")"
  rdzv="$(awk -v k="$kl" 'NR>k' "$L" 2>/dev/null | grep -aoE "joined round [0-9]+ .*world of size [0-9]+|failure_detected|nodes_waiting=[0-9]+|will restart worker group|Detected cluster changes|RendezvousClosed[A-Za-z]*|RendezvousTimeout[A-Za-z]*|Assigned active rank [0-9]+|Invalid infrastructure rank" | sort | uniq -c | tail -12)"
  nccl="$(awk -v k="$kl" 'NR>k' "$L" 2>/dev/null | grep -aiE "ncclInternalError|Cuda failure|NCCL.*abort|Internal check failed|unhandled cuda error|Connect res|invalid usage|Watchdog|collective.*timeout" | grep -aviE "health_check" | tail -8)"
  say "post-fault rendezvous activity:";  echo "${rdzv:-  (none)}"
  say "post-fault NCCL errors:";          echo "${nccl:-  (none)}"
  say "post-fault: re-rendezvous rounds=${re_round:-0}  megatron re-inits=${re_init:-0}"
  if [ "${re_round:-0}" -eq 0 ]; then
    say "VERDICT: RENDEZVOUS-STALL - no re-rendezvous round formed after the fault"
    say "         (stuck detecting / waiting for the dead node). Lever = FT rendezvous"
    say "         timeouts / promotion config on 0.5.0-patched, NOT NCCL/0.6.0."
  elif [ "${re_init:-0}" -eq 0 ]; then
    say "VERDICT: RENDEZVOUS->INIT-STALL - a round re-formed but Megatron never re-init'd."
  elif [ -n "${nccl}" ]; then
    say "VERDICT: NCCL-REINIT-FAILURE - re-rendezvous + re-init happened but NCCL errored on"
    say "         comm re-init (the 2.28.9 churn). Lever = NCCL version / --max-restarts retries,"
    say "         NOT NVRx 0.6.0 (which doesn't change NCCL)."
  else
    say "VERDICT: UNKNOWN - re-init attempted, no clear NCCL error; inspect ${L}"
  fi
}

# =============================================================================
PASS=1
for attempt in $(seq 1 "${MAX_SUBMITS}"); do
  say "=== submit ${attempt}/${MAX_SUBMITS}: config=${CONFIG} account=${ACCOUNT} qos=${QOS} ==="
  say "sidecar=${NVRX_SIDECAR}  expect world=${ACTIVE_WORLD} (nodes=${NODE_ROUND})  saboteur: node ${SABOTEUR_NODE} @ ${SABOTEUR_DELAY}s"
  [ "${CLEAR_GLOBAL}" = "1" ] && { rm -rf "${GLOBAL_CKPT_DIR}" 2>/dev/null; say "cleared global ckpt dir (fallback => cold-start)"; }
  # cancel stale runs of this config
  for j in $(squeue -u "${USER}" -h -o "%i %j" 2>/dev/null | grep -F "${BASE_NAME}" | awk '{print $1}'); do scancel "$j" 2>/dev/null; done

  JOBID="$(submit_once)"
  if [ -z "${JOBID}" ]; then say "submit failed (slurmctld/account?) - retrying"; sleep 20; continue; fi
  LOG="$(log_path_for "${JOBID}")"
  say "job ${JOBID}  log ${LOG}"

  # --- monitor until we have enough post-reload data, or the job ends -------
  killed=""; reloaded_line=""
  for i in $(seq 1 "${MAX_POLLS}"); do
    state="$(squeue -j "${JOBID}" -h -o %T 2>/dev/null | tr -d '[:space:]')"
    [ -f "${LOG}" ] || { sleep "${POLL_SECS}"; continue; }
    # once the FT AGENT DETECTED the worker failure (the real fault anchor -- NOT
    # the srun "task Killed" string, which no longer fires now that we spare the
    # agent), the reload happened, and we have >= MIN_POST_ITERS post-reload
    # iterations, cancel to bound GPU usage and analyze.
    if grep -qaE "${DETECT_RE}" "${LOG}" 2>/dev/null; then
      rlline="$(grep -anE "successfully loaded checkpoint" "${LOG}" 2>/dev/null | tail -1 | cut -d: -f1)"
      if [ -n "${rlline}" ]; then
        npost="$(awk -v k="${rlline}" 'NR>k && /iteration +[0-9]+\// {c++} END{print c+0}' "${LOG}")"
        if [ "${npost:-0}" -ge "${MIN_POST_ITERS}" ]; then
          say "recovery observed (+${npost} post-reload iters) - cancelling to analyze"
          scancel "${JOBID}" 2>/dev/null; break
        fi
      fi
    fi
    [ -z "${state}" ] && { say "[poll $i] job ${JOBID} left the queue"; break; }
    sleep "${POLL_SECS}"
  done

  # retry only on a pre-training infra flake (never reached training)
  if [ ! -f "${LOG}" ] || ! grep -qaE "using world size" "${LOG}" 2>/dev/null; then
    say "job never reached training (infra flake?) - retrying"; continue
  fi
  # retry a MISSED FAULT: the injector fired but the FT layer never saw a worker
  # failure -> the kill raced / didn't land. Resubmit rather than score a false
  # negative (this is exactly the artifact the old harness mistook for a stall).
  if ! grep -qaE "${DETECT_RE}" "${LOG}" 2>/dev/null \
     && grep -qaE "\[saboteur\].*firing|task [0-9]+: Killed" "${LOG}" 2>/dev/null \
     && [ "${attempt}" -lt "${MAX_SUBMITS}" ]; then
    say "saboteur fired but no worker-failure detected (fault missed) - resubmitting"; continue
  fi
  break
done

# =============================================================================
# ANALYSIS + PASS/FAIL checklist
# =============================================================================
[ -f "${LOG}" ] || { say "FAIL: no log at ${LOG}"; exit 1; }
# FL = the line where the FT agent DETECTED the worker failure (the fault anchor).
# SAB = did the injector actually fire? (used to distinguish "fault missed" from
# "saboteur never ran"). All post-fault analysis keys off FL, not srun's message.
FL="$(grep -anE "${DETECT_RE}" "${LOG}" 2>/dev/null | head -1 | cut -d: -f1)"
SAB="$(grep -acE "\[saboteur\].*firing|task [0-9]+: Killed" "${LOG}" 2>/dev/null)"
KL="${FL}"   # downstream `NR>k` post-fault scans anchor on the detection line
RLLINE="$(awk -v k="${KL:-0}" 'NR>k && /successfully loaded checkpoint/ {print NR; exit}' "${LOG}")"

# 1. reserve mode: first world-init == ACTIVE_WORLD (not larger = elastic)
first_world="$(grep -aoE "using world size: [0-9]+" "${LOG}" 2>/dev/null | head -1 | grep -oE "[0-9]+")"
[ "${first_world:-0}" = "${ACTIVE_WORLD}" ] && r1="PASS" || r1="FAIL(world=${first_world:-none})"

# 2. local checkpoint saves happened
nsaves="$(grep -acE "saved local checkpoint from iteration" "${LOG}" 2>/dev/null)"
[ "${nsaves:-0}" -ge 1 ] && r2="PASS(${nsaves})" || r2="FAIL(0 local saves)"

# 3. fault fired AND was detected by the FT agent (a real, recoverable fault --
#    not just an srun-visible kill that left the workers alive).
if [ -n "${FL}" ]; then r3="PASS"
elif [ "${SAB:-0}" -ge 1 ]; then r3="FAIL(saboteur fired but no worker-failure detected -> fault did not land)"
else r3="FAIL(saboteur never fired)"; fi

# Checks 4-6 only make sense once a real fault was detected; otherwise mark N/A so
# a missed-fault run reports honestly instead of scanning the initial (cold) start.
if [ -z "${FL}" ]; then
  r4="N/A (no fault)"; r5="N/A (no fault)"; r6="N/A (no fault)"
else
  # 4. spare promoted: after the fault a world re-forms at ACTIVE_WORLD, and NO
  #    "Invalid infrastructure rank".
  post_world8="$(awk -v k="${KL}" 'NR>k && /using world size: '"${ACTIVE_WORLD}"',/' "${LOG}" | wc -l)"
  badrank="$(grep -acE "Invalid infrastructure rank" "${LOG}" 2>/dev/null)"
  if [ "${post_world8:-0}" -ge 1 ] && [ "${badrank:-0}" -eq 0 ]; then r4="PASS"
  elif [ "${badrank:-0}" -ge 1 ]; then r4="FAIL(Invalid infrastructure rank)"
  else r4="FAIL(no world=${ACTIVE_WORLD} re-init after fault)"; fi

  # 5. reload, not cold-start (after the fault)
  cold="$(awk -v k="${KL}" 'NR>k && /will not load any checkpoints|start from random/' "${LOG}" | wc -l)"
  if [ -n "${RLLINE}" ] && [ "${cold:-0}" -eq 0 ]; then r5="PASS"
  elif [ "${cold:-0}" -ge 1 ]; then r5="FAIL(cold-start after fault)"
  else r5="FAIL(no reload after fault)"; fi

  # 6. WEIGHT CORRECTNESS: loss continuity across the reload
  reload_iter="$(awk -v k="${KL}" 'NR>k && match($0,/at iteration [0-9]+/){print substr($0,RSTART+13,RLENGTH-13); exit}' "${LOG}")"
  loss_pre="$(loss_at_iter "${reload_iter:-x}" "${LOG}" "${KL}")"       # loss at reload-iter, BEFORE the fault
  # fallback: if that exact iteration wasn't a log-interval multiple, use the LAST
  # loss logged before the fault (still a valid continuity baseline). Avoids a
  # spurious INCONCLUSIVE on a run that actually recovered fine.
  [ -z "${loss_pre}" ] && loss_pre="$(awk -v k="${KL}" 'NR<k && match($0,/lm loss: [0-9.eE+-]+/){v=substr($0,RSTART+9,RLENGTH-9)} END{if(v!="")print v}' "${LOG}")"
  loss_post="$(first_loss_after_line "${RLLINE:-999999999}" "${LOG}")"  # first loss after the reload
  r6="$(awk -v pre="${loss_pre}" -v post="${loss_post}" -v f="${LOSS_SPIKE_FACTOR}" 'BEGIN{
    if (pre=="" || post=="") { print "UNKNOWN(pre=" pre " post=" post ")"; exit }
    if (post+0 <= pre*f + 0.05) print "PASS(pre=" pre " post=" post ")";
    else print "FAIL(LOSS SPIKE pre=" pre " -> post=" post " => wrong weights?)";
  }')"
fi

echo
say "================= RESILIENCE CHECK: ${BASE_NAME} (job ${JOBID:-?}) ================="
say "1. reserve mode (world=${ACTIVE_WORLD}) ....... ${r1}"
say "2. local ckpt saves ........................... ${r2}"
say "3. fault fired (saboteur) ..................... ${r3}"
say "4. spare promoted (world stays ${ACTIVE_WORLD}) ${r4}"
say "5. reload (not cold-start) .................... ${r5}"
say "6. WEIGHT CORRECTNESS (loss continuity) ....... ${r6}"

# if the fault fired but recovery didn't complete, say WHY (rendezvous vs NCCL)
if [ -n "${KL}" ]; then case "${r4}${r5}" in *FAIL*) diagnose_stall "${LOG}" "${KL}" ;; esac; fi

case "${r1}${r2}${r3}${r4}${r5}${r6}" in
  *FAIL*) say "RESULT: FAIL"; exit 1 ;;
  *UNKNOWN*) say "RESULT: INCONCLUSIVE (loss check had no data - inspect ${LOG})"; exit 2 ;;
  *) say "RESULT: PASS - recovered at fixed world=${ACTIVE_WORLD} from the correct weights (loss continuous)"; exit 0 ;;
esac
