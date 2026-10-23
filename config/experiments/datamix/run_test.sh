#!/usr/bin/env bash
# Test of the whole DataMix chain on dev-g (chain-datamix-test.yaml): training, HF
# conversion and lm-eval of every scale, 100 iterations each, at most 2 jobs at a time.
# Run dirs output/test/datamix/<scale> in the repo. The monitor must keep running until
# the last eval is submitted, so outside tmux the script starts itself in the tmux
# session datamix-test (detach: Ctrl-b d, reattach: tmux attach -t datamix-test).
# Extra arguments go to run_autoexp.py:
#
#   bash config/experiments/datamix/run_test.sh                       # all scales
#   bash config/experiments/datamix/run_test.sh 'aux.scales=[0.1B]'   # one scale only
#   bash config/experiments/datamix/run_test.sh --dry-run             # render, don't submit (no tmux)
#
# Jobs that already finished are skipped; delete output/test/datamix to start over.
set -euo pipefail

SESSION=datamix-test
script=$(realpath "$0")
cd "$(dirname "$script")/../../.."   # repo root
export HF_HOME=${HF_HOME:-/scratch/project_465002530/cache}

# prerequisites, see chain_guide.md
fail() { echo "error: $*" >&2; exit 1; }
[ -d "$HF_HOME/hub/models--openeurollm--tokenizer-256k" ] \
    || fail "tokenizer not found in HF_HOME=$HF_HOME"
[ -f artifacts/MegatronTrainingLumi_bridge_x86_64.sif ] \
    || fail "convert container artifacts/MegatronTrainingLumi_bridge_x86_64.sif not found"
[ -e "$HOME/eval_local" ] \
    || fail "eval env ~/eval_local not found (scripts/install_eval_env.sh)"

# outside tmux: rerun this script in a new tmux session (a dry run needs no monitor)
if [ -z "${TMUX:-}" ] && [[ " $* " != *" --dry-run "* ]]; then
    ! tmux has-session -t "=$SESSION" 2>/dev/null \
        || fail "tmux session $SESSION exists already: tmux attach -t $SESSION"
    # the shell stays open after the monitor exits, to keep its output readable
    tmux new-session -d -s "$SESSION" -c "$PWD" \
        "HF_HOME=$(printf '%q' "$HF_HOME") bash $(printf '%q ' "$script" "$@"); exec bash"
    echo "started in tmux session $SESSION (detach: Ctrl-b d, reattach: tmux attach -t $SESSION)"
    if [ -t 1 ]; then exec tmux attach -t "=$SESSION"; fi
    exit 0
fi

# the test needs both dev-g job slots; a job submitted while they are taken is dropped
jobs=$(squeue -h -u "$USER" -p dev-g | wc -l)
[ "$jobs" -eq 0 ] || echo "warning: $jobs of your jobs are on dev-g; the test shares its 2-job limit with them" >&2

exec uv run python scripts/run_autoexp.py --config-name experiments/datamix/chain-datamix-test "$@"
