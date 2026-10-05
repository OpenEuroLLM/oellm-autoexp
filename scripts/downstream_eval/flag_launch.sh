#!/bin/bash
# Launch the FLAG suite (vLLM + lighteval halves) for ONE converted export and record the run.
#
#   flag_launch.sh NAME
#
# Called by `eval_checkpoints.sh flag` on the LOGIN node: it submits the two halves as job arrays
# (also from an oellm-autoexp `job.local` stage, see chain_flag_130M_jupiter.yaml).
# Writes $FLAG_STATE_DIR/NAME.env with the two run directories and job ids; `flag-status` and
# `flag-rerun` read that file instead of guessing which timestamped directory belongs to the run.
#
# Environment (set by eval_checkpoints.sh / sites/*.env): EXPORT_ROOT EVAL_REPO EXPECTED_EVAL_REV
#   FLAG_ACCOUNT FLAG_WORK FLAG_CONCURRENCY FLAG_TIME FLAG_STATE_DIR
#   FLAG_ARRAY_VLLM / FLAG_ARRAY_LIGHTEVAL   optional subset of array indices, e.g. "1,2,30" (a test);
#                                            recorded, so flag-status ignores the indices not submitted
#   FLAG_SUBSET=1                            (oellm-autoexp) only some evals are run; recorded for flag-collect
#   DRY_RUN=1                                render, but print the sbatch commands instead
#   PREPARE_ONLY=1                           views + render, no submission (LAUNCHER=autoexp in
#                                            the state): oellm-autoexp array stages run the rows
#                                            (jupiter_flag_evals.sh run-one); idempotent
set -euo pipefail
NAME="${1:?usage: flag_launch.sh NAME}"
source "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/flag_lib.sh"
EXPORT="$EXPORT_ROOT/$NAME"
STATE="$FLAG_STATE_DIR/$NAME.env"
FLAG_SCRIPT="$EVAL_REPO/scripts/jupiter_flag_evals.sh"

[ -f "$EXPORT/config.json" ] || { echo "error: no export $EXPORT" >&2; exit 1; }
[ -x "$FLAG_SCRIPT" ] || { echo "error: $FLAG_SCRIPT missing (git submodule update --init submodules/oellm-eval)" >&2; exit 1; }
if [ "${PREPARE_ONLY:-0}" = 1 ] && [ -f "$STATE" ] && grep -q '^LAUNCHER=autoexp$' "$STATE" \
        && [ "${FORCE:-0}" != 1 ]; then
    # A second prepare would render NEW run directories, orphaning the rows already done.
    echo "$NAME already prepared ($STATE)"; exit 0
fi
if [ -f "$STATE" ] && [ "${FORCE:-0}" != 1 ]; then
    echo "error: $NAME already launched ($STATE); use flag-status / flag-rerun, or FORCE=1 for a fresh run" >&2
    exit 1
fi

# The suite definition (task groups, n_shot) is read from the INSTALLED oellm-eval, so it must be
# this submodule at the pinned revision -- otherwise results silently come from another suite.
# Checked on the login node; eval_checkpoints.sh runs it once and passes FLAG_PREFLIGHT_OK=1.
rev="${FLAG_EVAL_REV:-}"
if [ "${FLAG_PREFLIGHT_OK:-0}" != 1 ]; then
    rev=$(flag_preflight)
fi

export ACCOUNT="$FLAG_ACCOUNT" FLAG_WORK CONCURRENCY="$FLAG_CONCURRENCY" TIME="$FLAG_TIME"
"$FLAG_SCRIPT" views "$EXPORT" > /dev/null
rendered=$("$FLAG_SCRIPT" render "$NAME")
echo "$rendered"
vllm_sb=$(awk '$1 == "sbatch" && $2 ~ /\/vllm\// {print $2}' <<< "$rendered")
light_sb=$(awk '$1 == "sbatch" && $2 ~ /\/lighteval\// {print $2}' <<< "$rendered")
[ -f "$vllm_sb" ] && [ -f "$light_sb" ] || { echo "error: render did not produce both launchers" >&2; exit 1; }

submit_half() {  # <sbatch file> <array override> -> job id
    local args=(--parsable --account="$FLAG_ACCOUNT")
    [ -n "$2" ] && args+=(--array="$2%${FLAG_CONCURRENCY:-40}")
    if [ "${DRY_RUN:-0}" = 1 ]; then
        printf 'sbatch' >&2; printf ' %q' "${args[@]}" "$1" >&2; printf '\n' >&2; echo DRYRUN
    else
        sbatch "${args[@]}" "$1"
    fi
}
launcher=flag_launch
if [ "${PREPARE_ONLY:-0}" = 1 ]; then
    launcher=autoexp vllm_job="" light_job=""
else
    vllm_job=$(submit_half "$vllm_sb" "${FLAG_ARRAY_VLLM:-}")
    light_job=$(submit_half "$light_sb" "${FLAG_ARRAY_LIGHTEVAL:-}")
fi

[ "${DRY_RUN:-0}" = 1 ] && { echo "dry run: no state written"; exit 0; }
mkdir -p "$FLAG_STATE_DIR"
cat > "$STATE" <<STATE_EOF
# written by flag_launch.sh $(date -Is)
NAME=$NAME
LAUNCHER=$launcher
EXPORT=$EXPORT
EVAL_REV=$rev
VLLM_RUN=$(dirname "$vllm_sb")
LIGHTEVAL_RUN=$(dirname "$light_sb")
VLLM_JOB=$vllm_job
LIGHTEVAL_JOB=$light_job
VLLM_INDICES=${FLAG_ARRAY_VLLM:-}
LIGHTEVAL_INDICES=${FLAG_ARRAY_LIGHTEVAL:-}
SUBSET=${FLAG_SUBSET:-}
STATE_EOF
if [ "$launcher" = autoexp ]; then
    echo "prepared $NAME: $(dirname "$vllm_sb") $(dirname "$light_sb")  (state: $STATE)"
else
    echo "launched $NAME: vllm=$vllm_job lighteval=$light_job  (state: $STATE)"
fi
