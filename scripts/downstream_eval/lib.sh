# Shared setup for the downstream-eval scripts: pick a site profile and a model profile,
# then expose WORK_ROOT and the per-step paths. Everything can be overridden in the environment.
set -euo pipefail

HERE=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
PROFILE_DIR="$HERE/models"

case "$(hostname -s)" in jp*) SITE="${SITE:-jupiter}" ;; esac
SITE="${SITE:-}"
[ -n "$SITE" ] || { echo "error: set SITE=<name>; no profile matches $(hostname -s)" >&2; exit 2; }
[ -f "$HERE/sites/$SITE.env" ] || { echo "error: no site profile $HERE/sites/$SITE.env" >&2; exit 2; }
source "$HERE/sites/$SITE.env"

MODEL="${MODEL:-32b_dense}"
[ -f "$PROFILE_DIR/$MODEL.env" ] || { echo "error: no model profile $PROFILE_DIR/$MODEL.env" >&2; exit 2; }
source "$PROFILE_DIR/$MODEL.env"

# Layout under WORK_ROOT (data only; the scripts live in the repo).
# EXPORT_ROOT may point at a shared export tree; prompt views must stay on the same filesystem
# as the exports, because they hard-link the weights.
: "${EXPORT_ROOT:=$WORK_ROOT/hf}"               # converted HF checkpoints, one dir per name
: "${PROMPT_VIEW_ROOT:=$(dirname "$EXPORT_ROOT")/prompt_views/identity}"
: "${RUN_ROOT:=$WORK_ROOT/reasoning_vllm}"      # vLLM/Evalchemy runs
: "${HF_CACHE:=$WORK_ROOT/hf_home}"             # HF backend (12 lm-eval tasks)
: "${REASONING_CACHE:=$WORK_ROOT/hf_home_reasoning}"
: "${LOG_DIR:=$WORK_ROOT/logs}"

declare -A RUN_PATH ALIAS_SPEC
for _entry in "${RUNS[@]:-}"; do [ -n "$_entry" ] && RUN_PATH[${_entry%%:*}]=${_entry#*:}; done
for _entry in "${ALIASES[@]:-}"; do [ -n "$_entry" ] && ALIAS_SPEC[${_entry%%:*}]=${_entry#*:}; done

# An export is named <run>_<iter/1000>k, or <run>_<iter> when the iteration is not a whole thousand.
checkpoint_name() {
    if [ $(($2 % 1000)) -eq 0 ]; then echo "$1_$(($2 / 1000))k"; else echo "$1_$2"; fi
}

# "run:iter", "run:latest", an alias name, or the name of an existing export -> "name:checkpoint path"
resolve_spec() {
    local spec="$1" run iter dir name
    [ -n "${ALIAS_SPEC[$spec]:-}" ] && { name="$spec"; spec="${ALIAS_SPEC[$spec]}"; }
    if [[ "$spec" == *:* ]]; then
        run=${spec%%:*}; iter=${spec#*:}
        [ -n "${RUN_PATH[$run]:-}" ] || { echo "error: unknown run '$run'" >&2; return 1; }
        if [ "$iter" = latest ]; then
            dir=$(ls -d "${RUN_PATH[$run]}"/iter_* 2>/dev/null | tail -1)
        else
            dir=$(printf '%s/iter_%07d' "${RUN_PATH[$run]}" "$iter")
        fi
        [ -d "$dir" ] || { echo "error: no checkpoint $dir" >&2; return 1; }
        iter=$((10#$(basename "$dir" | sed 's/iter_//')))
        printf '%s:%s\n' "${name:-$(checkpoint_name "$run" "$iter")}" "$dir"
        return
    fi
    [ -d "$EXPORT_ROOT/$spec" ] || { echo "error: '$spec' is not an alias, a run:iter, or an export" >&2; return 1; }
    printf '%s:\n' "$spec"   # already converted; no checkpoint path needed
}

select_checkpoints() {
    local spec
    for spec in "$@"; do resolve_spec "$spec" || return 1; done
}

require_exports() {  # every named export must exist before evaluating
    local name
    for name in "$@"; do
        [ -d "$EXPORT_ROOT/$name" ] || { echo "error: no export $EXPORT_ROOT/$name (run 'convert' first)" >&2; exit 1; }
    done
}
