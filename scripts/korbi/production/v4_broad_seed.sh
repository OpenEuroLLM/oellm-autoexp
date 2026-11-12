#!/bin/bash
# Seed the broad-session arms' own checkpoint directories (docs/recall_surgery_ambitious.md §10) with HARD LINKS to
# their start, so each arm runs with load = save = <root>/<arm>/ckpt: a pool restart resumes from the arm's own latest
# save instead of re-running from the start (pool restarts re-run from `load`).
#
#   v4_broad_seed.sh <IT> [arm ...]           arms default: the four of hybrid_v4_broad23k_mn5; start = iteration IT
#   v4_broad_seed.sh --dec <IT> [arm ...]     decay copies dec_<arm>, from <root>/<arm>/ckpt/iter_<IT + DEC_FROM>
#                                             (DEC_FROM default 1500 = hybrid_v4_broad23k_mn5's steps)
#                                             (else <root>/<arm>/ckpt_rolling/iter_<IT + DEC_FROM>)
#   DRY=1 ...                                 print only
#
# Defaults = the 23300 session (hybrid_v4_broad23k_mn5; user 2026-11-12):
#   v4_broad_seed.sh 23300                    ctrl cgqa_lr4 kv8_lr4 kv8_post4_lr4 into production/v4_broad23k
#   v4_broad_seed.sh --dec 23300              their decay copies from +1500
# The superseded phase-2 plan (hybrid_v4_broad_mn5, 7 arms, +800):
#   ROOT=/gpfs/scratch/ehpc533/hybrid_nucleus/production/v4_broad DEC_FROM=800 ARMS_ALL="ctrl kv8_lr4 post4_lr4 \
#       kv8_post4_lr4 conv_lr4 kv16_lr8 k6_L49" v4_broad_seed.sh <IT>
#
# Sources: ctrl / k6_L49 -> the main run's frozen copy ckpt_eval/iter_<IT> (hard-linked there first, runbook §10.3);
# the surgery arms -> <root>/src/<surgery>/ckpt/iter_<IT> (v4_broad_surgery.sh; only when its tracker = IT, i.e. the
# surgery verified). Login node is fine: hard links only, a few seconds. Never writes into a source; refuses a target
# that already has a tracker (an arm's progress is never clobbered).
set -euo pipefail

ROOT=${ROOT:-/gpfs/scratch/ehpc533/hybrid_nucleus/production/v4_broad23k}
MAIN_EVAL=${MAIN_EVAL:-/gpfs/scratch/ehpc533/hybrid_nucleus/production/main_mhc4/ambitious_e360_eng80_v4_2ph_hd256/ckpt_eval}
DEC_FROM=${DEC_FROM:-1500}
ALL_ARMS=${ARMS_ALL:-"ctrl cgqa_lr4 kv8_lr4 kv8_post4_lr4"}

DEC=0
if [ "${1:-}" = "--dec" ]; then DEC=1; shift; fi
IT=${1:?usage: v4_broad_seed.sh [--dec] <IT> [arm ...]}
shift
case "$IT" in ''|*[!0-9]*) echo "IT must be an integer: $IT" >&2; exit 2;; esac
ARMS=${*:-$ALL_ARMS}

run() { if [ "${DRY:-0}" = 1 ]; then echo "+ $*"; else "$@"; fi; }

surgery_of() {
  case "$1" in
    ctrl|k6_L49) echo "" ;;
    kv8_lr4) echo kv8 ;;
    kv16_lr8) echo kv16 ;;
    post4_lr4) echo post4 ;;
    kv8_post4_lr4) echo kv8post4 ;;
    conv_lr4) echo conv ;;
    cgqa_lr4) echo cgqa ;;
    *) echo "unknown arm $1" >&2; return 1 ;;
  esac
}

rc=0
for arm in $ARMS; do
  if [ $DEC = 1 ]; then
    step=$((IT + DEC_FROM))
    src_ckpt=$ROOT/$arm/ckpt
    dst=$ROOT/dec_$arm/ckpt
  else
    step=$IT
    s=$(surgery_of "$arm")
    if [ -z "$s" ]; then
      src_ckpt=$MAIN_EVAL
    else
      src_ckpt=$ROOT/src/$s/ckpt
      tr=$src_ckpt/latest_checkpointed_iteration.txt
      if [ ! -f "$tr" ] || [ "$(tr -d '[:space:]' < "$tr")" != "$IT" ]; then
        echo "[seed] $arm: surgery $s not done for $IT ($tr) -- skipped (seed again once it is)"; rc=1; continue
      fi
    fi
    dst=$ROOT/$arm/ckpt
  fi
  name=$(printf 'iter_%07d' "$step")
  src=$src_ckpt/$name
  # relaunched branch arms (§10.6) save their exit step as a rolling save: <arm>/ckpt_rolling/iter_<step>
  if [ $DEC = 1 ] && [ ! -f "$src/.metadata" ] && [ -f "$ROOT/$arm/ckpt_rolling/$name/.metadata" ]; then
    src=$ROOT/$arm/ckpt_rolling/$name
  fi
  if [ ! -f "$src/.metadata" ]; then
    echo "[seed] $arm: no checkpoint $src/.metadata -- skipped"; rc=1; continue
  fi
  if [ -f "$dst/latest_checkpointed_iteration.txt" ]; then
    echo "[seed] $arm: $dst already seeded (tracker $(cat "$dst/latest_checkpointed_iteration.txt")) -- left alone"; continue
  fi
  if [ -e "$dst/$name" ]; then
    echo "[seed] $arm: $dst/$name exists without a tracker (interrupted seed?) -- remove it by hand after checking"; rc=1; continue
  fi
  run mkdir -p "$dst"
  # hard links: no data copied; the source tree is only read (a later save of the arm writes new iter_* dirs)
  run cp -al "$src" "$dst/$name"
  if [ "${DRY:-0}" = 1 ]; then
    echo "+ echo $step > $dst/latest_checkpointed_iteration.txt"
  else
    n_src=$(find "$src" -type f | wc -l); n_dst=$(find "$dst/$name" -type f | wc -l)
    [ "$n_src" = "$n_dst" ] || { echo "[seed] $arm: file count $n_dst != $n_src" >&2; rc=1; continue; }
    echo "$step" > "$dst/latest_checkpointed_iteration.txt"
    printf '{"arm": "%s", "start": %s, "source": "%s", "files": %s, "seeded": "%s"}\n' \
      "$arm" "$step" "$src" "$n_src" "$(date -Is)" > "$dst/seeded_from.json"
  fi
  echo "[seed] $arm: $dst/$name <- $src"
done
exit $rc
