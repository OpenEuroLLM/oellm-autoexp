#!/bin/bash
# Launch the FLAG suite (vLLM + lighteval halves) for ONE converted export and record the run.
#
#   flag_launch.sh NAME
#
# Called by `eval_checkpoints.sh flag` directly for exports that already exist, and at the end of
# convert.sbatch (FLAG_AFTER_CONVERT=1) for exports it just wrote, so no node is spent on waiting.
# Writes $FLAG_STATE_DIR/NAME.env with the two run directories and job ids; `flag-status` and
# `flag-rerun` read that file instead of guessing which timestamped directory belongs to the run.
#
# Environment (set by eval_checkpoints.sh / sites/*.env): EXPORT_ROOT EVAL_REPO EXPECTED_EVAL_REV
#   FLAG_ACCOUNT FLAG_WORK FLAG_CONCURRENCY FLAG_TIME FLAG_STATE_DIR
#   FLAG_ARRAY_VLLM / FLAG_ARRAY_LIGHTEVAL   optional --array override (e.g. a subset for a test)
#   DRY_RUN=1                                render, but print the sbatch commands instead
set -euo pipefail
NAME="${1:?usage: flag_launch.sh NAME}"
source "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/flag_lib.sh"
EXPORT="$EXPORT_ROOT/$NAME"
STATE="$FLAG_STATE_DIR/$NAME.env"
FLAG_SCRIPT="$EVAL_REPO/scripts/jupiter_flag_evals.sh"

[ -f "$EXPORT/config.json" ] || { echo "error: no export $EXPORT" >&2; exit 1; }
[ -x "$FLAG_SCRIPT" ] || { echo "error: $FLAG_SCRIPT missing (git submodule update --init submodules/oellm-eval)" >&2; exit 1; }
if [ -f "$STATE" ] && [ "${FORCE:-0}" != 1 ]; then
    echo "error: $NAME already launched ($STATE); use flag-status / flag-rerun, or FORCE=1 for a fresh run" >&2
    exit 1
fi

# The suite definition (task groups, n_shot) is read from the INSTALLED oellm-eval, so it must be
# this submodule at the pinned revision -- otherwise results silently come from another suite.
# Checked on the login node (compute nodes have no git); convert.sbatch passes FLAG_PREFLIGHT_OK=1.
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
    [ -n "$2" ] && args+=(--array="$2")
    if [ "${DRY_RUN:-0}" = 1 ]; then
        printf 'sbatch' >&2; printf ' %q' "${args[@]}" "$1" >&2; printf '\n' >&2; echo DRYRUN
    else
        sbatch "${args[@]}" "$1"
    fi
}
vllm_job=$(submit_half "$vllm_sb" "${FLAG_ARRAY_VLLM:-}")
light_job=$(submit_half "$light_sb" "${FLAG_ARRAY_LIGHTEVAL:-}")

[ "${DRY_RUN:-0}" = 1 ] && { echo "dry run: no state written"; exit 0; }
mkdir -p "$FLAG_STATE_DIR"
cat > "$STATE" <<STATE_EOF
# written by flag_launch.sh $(date -Is)
NAME=$NAME
EXPORT=$EXPORT
EVAL_REV=$rev
VLLM_RUN=$(dirname "$vllm_sb")
LIGHTEVAL_RUN=$(dirname "$light_sb")
VLLM_JOB=$vllm_job
LIGHTEVAL_JOB=$light_job
STATE_EOF
echo "launched $NAME: vllm=$vllm_job lighteval=$light_job  (state: $STATE)"
