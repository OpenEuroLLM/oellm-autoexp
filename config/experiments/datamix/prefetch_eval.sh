#!/usr/bin/env bash
# Downloads the eval datasets of a DataMix config into $HF_HOME, on the login node.
# Run it with the same config name and overrides as run_autoexp.py, before the first run
# and whenever aux.eval_task_groups changes:
#
#   bash config/experiments/datamix/prefetch_eval.sh --config-name experiments/datamix/chain-datamix
#   bash config/experiments/datamix/prefetch_eval.sh --config-name experiments/datamix/chain-datamix \
#       'aux.eval_task_groups=[generic-multilingual,xnli-eu]'
#
# Why this script exists:
# The eval jobs run offline (HF_HUB_OFFLINE=1, config/container/lumi_eval.yaml; LUMI
# compute nodes have no internet), so every dataset of aux.eval_task_groups must already be
# in $HF_HOME. When oellm-eval is used on its own, `oellm-eval schedule` runs on the login
# node and downloads the datasets itself before it submits the eval jobs. autoexp instead
# runs `oellm-eval schedule --local true` inside its own SLURM eval job, i.e. on an offline
# compute node, so the chain skips that download (backend.skip_checks in _chain-datamix.yaml)
# and nothing in autoexp downloads the datasets. The documented workaround,
# `scripts/install_eval_env.sh --prefetch`, takes the task groups from
# $OELLM_EVAL_TASK_GROUPS (default open-sci-0.01), not from the config, and reinstalls the
# eval env. This script does only oellm-eval's own download step (`--download_only true`),
# for the task groups and in the eval container of the given config.
#
# Needs: $HF_HOME (the same as for the runs), ~/eval_local (scripts/install_eval_env.sh).
# Datasets already in $HF_HOME are reused, so running it again is quick.
set -euo pipefail

cd "$(dirname "$(realpath "$0")")/../../.."   # repo root
fail() { echo "error: $*" >&2; exit 1; }
[ -n "${HF_HOME:-}" ] || fail "HF_HOME is not set (use the same as for the runs)"
[ -x "$HOME/eval_local/bin/oellm-eval" ] || fail "eval env ~/eval_local not found (scripts/install_eval_env.sh)"

# same arguments as run_autoexp.py: --config-name NAME and key=value overrides
config=experiments/datamix/chain-datamix
render_args=()
while [ $# -gt 0 ]; do
    case "$1" in
        --config-name) config=$2; shift 2 ;;
        --config-name=*) config=${1#*=}; shift ;;
        *=*) render_args+=(-o "$1"); shift ;;
        *) shift ;;   # run_autoexp.py flags such as --dry-run
    esac
done

# aux.eval_task_groups and aux.eval_container as the runs resolve them
read -r groups container < <(
    uv run python scripts/render_config.py --config-name "$config" "${render_args[@]}" 2>/dev/null |
    uv run python -c '
import json, sys
aux = json.load(sys.stdin)["aux"]
groups = aux.get("eval_task_groups")
if not isinstance(groups, list) or not groups:
    sys.exit("aux.eval_task_groups is not set")
print(",".join(groups), aux["eval_container"])
') || fail "could not read aux.eval_task_groups from $config (is it set?)"

echo "prefetching task groups $groups into HF_HOME=$HF_HOME"
# --models: an existing dir without model files, so only the datasets are downloaded
singularity exec --bind "$HOME" --bind /pfs --bind /scratch \
    --env HF_HUB_OFFLINE=0 --env HF_DATASETS_OFFLINE=0 --env HF_HOME="$HF_HOME" \
    --env PYTHONPATH="$HOME/eval_local/lib/python3.12/site-packages" \
    --env PATH="$HOME/eval_local/bin:/usr/bin:/bin" \
    "$container" \
    oellm-eval schedule --models "$PWD/oellm_autoexp" --task_groups "$groups" --download_only true
echo "done: datasets of $groups are in $HF_HOME"
