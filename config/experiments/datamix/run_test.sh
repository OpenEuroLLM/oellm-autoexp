#!/usr/bin/env bash
# Test of the whole DataMix chain on dev-g (chain-datamix-test.yaml): training, HF
# conversion and lm-eval of every scale, 100 iterations each, one job at a time.
# Run dirs output/test/datamix/<scale> in the repo. The monitor must keep running until
# the last eval is submitted, so outside tmux the script starts itself in the tmux
# session datamix-test (detach: Ctrl-b d, reattach: tmux attach -t datamix-test).
# Extra arguments go to run_autoexp.py; the training data is required:
#
#   D='aux.data_path=[/path/to/bin_00_text_document]'
#   bash config/experiments/datamix/run_test.sh "$D"                       # all scales
#   bash config/experiments/datamix/run_test.sh "$D" 'aux.scales=[0.1B]'   # one scale only
#   bash config/experiments/datamix/run_test.sh "$D" --dry-run             # render, don't submit (no tmux)
#
# Before the run it downloads the eval datasets (prefetch_eval.sh). Jobs that already
# finished are skipped; delete output/test/datamix to start over.
set -euo pipefail

SESSION=datamix-test
script=$(realpath "$0")
cd "$(dirname "$script")/../../.."   # repo root
export HF_HOME=${HF_HOME:-/scratch/project_465002530/cache}
# project and containers, as in datamix.yaml (aux.account, aux.containers_dir); set by
# setup_lumi.sh in ~/datamix_env.sh
ACCOUNT=${DATAMIX_PROJECT:-project_465002530}
CONTAINERS=${DATAMIX_CONTAINERS:-/scratch/project_465002530/containers}

# prerequisites, see chain_guide.md
fail() { echo "error: $*" >&2; exit 1; }
[ -d "$HF_HOME/hub/models--openeurollm--tokenizer-256k" ] \
    || fail "tokenizer not found in HF_HOME=$HF_HOME"
for sif in AutoExp_2026-07-02 MegatronTrainingLumi_bridge_x86_64 laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0; do
    [ -r "$CONTAINERS/$sif.sif" ] || fail "container $CONTAINERS/$sif.sif not found (setup_lumi.sh)"
done
[ -e "$HOME/eval_local" ] \
    || fail "eval env ~/eval_local not found (scripts/install_eval_env.sh)"

# outside tmux: rerun this script in a new tmux session (a dry run needs no monitor)
if [ -z "${TMUX:-}" ] && [[ " $* " != *" --dry-run "* ]]; then
    ! tmux has-session -t "=$SESSION" 2>/dev/null \
        || fail "tmux session $SESSION exists already: tmux attach -t $SESSION"
    # A new tmux session gets the environment of the tmux server, not of this shell, so
    # pass on what the run needs. The shell stays open after the monitor exits, to keep
    # its output readable.
    envs=()
    for v in PATH HF_HOME UV_CACHE_DIR UV_PYTHON_INSTALL_DIR DATAMIX_PROJECT DATAMIX_CONTAINERS OUTPUT_ROOT; do
        [ -z "${!v:-}" ] || envs+=(-e "$v=${!v}")
    done
    tmux new-session -d -s "$SESSION" -c "$PWD" "${envs[@]}" \
        "bash $(printf '%q ' "$script" "$@"); exec bash"
    echo "started in tmux session $SESSION (detach: Ctrl-b d, reattach: tmux attach -t $SESSION)"
    if [ -t 1 ]; then exec tmux attach -t "=$SESSION"; fi
    exit 0
fi

# dev-g allows 1 job per project; the test waits while anyone in the project has one there
jobs=$(squeue -h -A "$ACCOUNT" -p dev-g -o '%u %i %j' || true)
[ -z "$jobs" ] || echo "note: dev-g is in use in your project, the test waits for it:"$'\n'"$jobs" >&2

# the eval jobs run offline: download the datasets of the test's task groups first
if [[ " $* " != *" --dry-run "* ]]; then
    bash config/experiments/datamix/prefetch_eval.sh --config-name experiments/datamix/chain-datamix-test "$@"
fi

exec uv run python scripts/run_autoexp.py --config-name experiments/datamix/chain-datamix-test "$@"
