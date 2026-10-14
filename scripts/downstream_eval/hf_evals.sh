#!/bin/bash
# open-sci-0.01 (12 lm-eval tasks) on converted exports, through oellm-eval's HF backend.
# Called by eval_checkpoints.sh; the exports to evaluate are the arguments.
HERE=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
source "$HERE/lib.sh"
[ $# -gt 0 ] || { echo "usage: hf_evals.sh NAME [NAME...]" >&2; exit 2; }
require_exports "$@"

# uv cache and tool dir per user, so several people can use the same checkout.
C="${UV_ROOT:-$HOME/.cache/oellm-eval-tools}"
export UV_CACHE_DIR=$C/uv-cache UV_TOOL_DIR=$C/uv-tools UV_PYTHON_INSTALL_DIR=$C/uv-python
command -v oellm-eval >/dev/null || uv tool install -p 3.12 "$EVAL_REPO"

# Our own HF cache: the shared one is not writable for everyone (dataset lock files).
export HF_HOME="$HF_CACHE"
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE HF_HUB_ENABLE_HF_TRANSFER  # no hf_transfer in the container

TASKS='copa social_iqa openbookqa lambada_openai winogrande mmlu hellaswag arc_easy arc_challenge commonsense_qa piqa boolq'
# Download inside the eval container: oellm-eval's host-side download uses datasets 4.x,
# which rejects script datasets (social_i_qa).
apptainer exec --bind "$CONTAINER_BIND" \
    --env HF_HOME=$HF_HOME,HF_DATASETS_CACHE=$HF_HOME/datasets,HF_DATASETS_TRUST_REMOTE_CODE=1,HF_HUB_ENABLE_HF_TRANSFER=0 \
    "$HF_EVAL_SIF" python -c \
    "from lm_eval.tasks import TaskManager, get_task_dict; get_task_dict('$TASKS'.split(), TaskManager())"

MODELS=$(for n in "$@"; do printf '%s,' "$EXPORT_ROOT/$n"; done | sed 's/,$//')
echo "models: $MODELS"
oellm-eval schedule --models "$MODELS" --task_groups open-sci-0.01 --skip_checks true --dry_run true \
    --slurm_template_var "{\"ACCOUNT\":\"$ACCOUNT\",\"SLURM_MEM\":\"200G\"}"

# lm-eval 0.4.10 passes `dtype=`, which transformers 4.53 ignores -> fp32 load and OOM.
S=$(ls -td "$EVAL_SHARED_RUNS/$USER"/*/ | head -1)submit_evals.sbatch
sed -i 's/trust_remote_code=True \\$/trust_remote_code=True,torch_dtype=bfloat16 \\/' "$S"
grep -q "torch_dtype=bfloat16" "$S" || { echo "error: could not patch $S" >&2; exit 1; }
echo
echo "Rendered: $S"
echo "Submit with: sbatch ${ARRAY:+--array=$ARRAY} $S"
[ "${SUBMIT:-0}" = 1 ] && sbatch ${ARRAY:+--array=$ARRAY} "$S"
