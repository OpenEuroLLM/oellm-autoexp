#!/bin/bash
# DCLM-core-22 on converted exports, rendered the way jitsev1 ran it. NEVER SUBMITS.
#
# WHY THIS EXISTS (note to ourselves, 2026-09-18)
# We have never run dclm-core-22 ourselves; our tables are open-sci-0.01 (12 tasks, HF backend)
# plus the vLLM reasoning set. jitsev1 ran dclm-core-22 twice, and the two campaigns differ in a
# way that decides whether his numbers are comparable to ours:
#
#   /e/fscratch/e-sta-openeurollm/jj1/eval/oellm_base            2026-09-05..07  --model hf, gpu:1
#       results/<ckpt>/open-sci-0.01        the 12 tasks dclm-core shares with open-sci  (58 ckpts)
#       results_mathcode/<ckpt>/dclm-core-rest   the other 12 evals                      (57 ckpts)
#       -> same harness, backend and GPU count as our table: directly comparable.
#   /e/fscratch/e-sta-openeurollm/jj1/eval/production_harness_alignment_20260915   vLLM DP4, gpu:4
#       77 tasks x 17 v1 checkpoints (production_4000..60000), dclm-core-22 among them.
#       -> base-profile.json sets parallel_configs [{dp:4,tp:1}], which oellm-eval only accepts
#          with model_backend=vllm, so the whole campaign is vLLM.
#
# This script follows the second (vLLM) protocol, because 8 of the 22 evals are generative
# (5 bigbench *_generate_until, jeopardy exact_match, coqa and squadv2 f1) and those are the ones
# the HF backend is slow at. Set MODEL_BACKEND=hf to reproduce the first protocol instead.
#
#   ./dclm_evals.sh b1_174k control_64k        # renders one launcher, prints the sbatch command
#
HERE=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
source "$HERE/lib.sh"
[ $# -gt 0 ] || { echo "usage: dclm_evals.sh NAME [NAME...]   (exports under $EXPORT_ROOT)" >&2; exit 2; }
require_exports "$@"

: "${MODEL_BACKEND:=vllm}"                  # hf reproduces the oellm_base protocol
: "${DCLM_CACHE:=$WORK_ROOT/hf_home_dclm}"  # own cache: these datasets are not in hf_home
: "${DCLM_SIF:=$REASONING_SIF}"             # the complete image; the HF-only image has no vLLM
: "${QUEUE_LIMIT:=22}"                      # one array task per eval; each gets its own metrics cache
: "${DCLM_TIME:=02:00:00}"                  # coqa/squadv2/bigbench generate_until are the slow ones

# 21 tasks / 22 evals (hellaswag is run at 0 and 10 shots). jeopardy and the bigbench
# *_generate_until tasks are custom YAMLs; oellm-eval passes its bundled custom_lm_eval_tasks
# as --include_path automatically, so the download below must use the same include path.
TASKS='agieval_lsat_ar arc_easy arc_challenge boolq commonsense_qa copa hellaswag openbookqa piqa
       bigbench_language_identification_multiple_choice winogrande wsc273 lambada_openai
       bigbench_qa_wikidata_generate_until bigbench_dyck_languages_generate_until
       bigbench_operators_generate_until bigbench_repeat_copy_logic_generate_until
       bigbench_cs_algorithms_generate_until coqa squadv2 jeopardy'

# uv cache and tool dir per user, so several people can use the same checkout.
C="${UV_ROOT:-$HOME/.cache/oellm-eval-tools}"
export UV_CACHE_DIR=$C/uv-cache UV_TOOL_DIR=$C/uv-tools UV_PYTHON_INSTALL_DIR=$C/uv-python
command -v oellm-eval >/dev/null || uv tool install -p 3.12 "$EVAL_REPO"
INCLUDE_PATH=$("$EVAL_REPO/.venv/bin/python" -c \
    'from importlib.resources import files; print(files("oellm.resources") / "custom_lm_eval_tasks")' \
    2>/dev/null || echo "$EVAL_REPO/oellm/resources/custom_lm_eval_tasks")

mkdir -p "$DCLM_CACHE"
export HF_HOME="$DCLM_CACHE"
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE HF_HUB_ENABLE_HF_TRANSFER

# Download inside the container: oellm-eval's host-side download uses datasets 4.x, which rejects
# script datasets. Same pitfall as hf_evals.sh, and dclm-core adds more of them (agieval, bigbench).
apptainer exec --bind "$CONTAINER_BIND" \
    --env HF_HOME=$HF_HOME,HF_DATASETS_CACHE=$HF_HOME/datasets,HF_DATASETS_TRUST_REMOTE_CODE=1,HF_HUB_ENABLE_HF_TRANSFER=0 \
    "$DCLM_SIF" python -c \
    "from lm_eval.tasks import TaskManager, get_task_dict; get_task_dict('$TASKS'.split(), TaskManager(include_path='$INCLUDE_PATH'))"

MODELS=$(for n in "$@"; do printf '%s,' "$EXPORT_ROOT/$n"; done | sed 's/,$//')
echo "models: $MODELS"
echo "backend: $MODEL_BACKEND  image: $DCLM_SIF"

parallel_opts=()
[ "$MODEL_BACKEND" = vllm ] && parallel_opts=(
    --data_parallel_backend mp --data_parallel_size "$EVAL_DP" --tensor_parallel_size "$EVAL_TP"
    --model_args dtype=bfloat16,gpu_memory_utilization=0.9,max_num_seqs=32)

oellm-eval schedule --models "$MODELS" --task_groups dclm-core-22 \
    --model_backend "$MODEL_BACKEND" "${parallel_opts[@]}" \
    --queue_limit "$QUEUE_LIMIT" --log_samples true --skip_checks true --dry_run true \
    --slurm_template_var "{\"ACCOUNT\":\"$ACCOUNT\",\"PARTITION\":\"$PARTITION\",\"NODES\":1,
        \"CPUS_PER_TASK\":$CPUS_PER_TASK,\"THREADS_PER_CORE\":1,\"SLURM_MEM\":\"$SLURM_MEM\",
        \"TIME\":\"$DCLM_TIME\"}"

S=$(ls -td "$EVAL_SHARED_RUNS/$USER"/*/ | head -1)submit_evals.sbatch
if [ "$MODEL_BACKEND" = hf ]; then
    # lm-eval 0.4.10 passes `dtype=`, which transformers 4.53 ignores -> fp32 load and OOM.
    sed -i 's/trust_remote_code=True \\$/trust_remote_code=True,torch_dtype=bfloat16 \\/' "$S"
    grep -q "torch_dtype=bfloat16" "$S" || { echo "error: could not patch $S" >&2; exit 1; }
fi
bash -n "$S"
echo
echo "Rendered: $S"
echo "Check first:  grep -E '^#SBATCH|MODEL_BACKEND=' $S"
echo "Submit with:  sbatch $S     # ${#} model(s) x 22 evals"
echo
echo "Before trusting the numbers:"
echo "  * EVAL_REPO is $EVAL_REPO — use the checkout with commit 60c630c if you want a failed"
echo "    eval to leave the rest of its array task running (tag v0.01 predates that fix)."
echo "  * exports carry add_bos_token=true, as jitsev1's do; keep it for comparability."
echo "  * compare against jj1/eval/oellm_base only — the Sep-15 campaign is a different backend."
