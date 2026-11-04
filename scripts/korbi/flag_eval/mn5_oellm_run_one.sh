#!/bin/bash
# One FLAG row of the oellm-eval suite on MareNostrum 5, inside the MN5 image.
# The task names, n_shot values and suites come from config/eval_tasks/flag_evals.yaml.
# The lm-eval half uses the proven MN5 flag_lmeval.py runner; evalchemy and lighteval
# use the same harness commands as oellm-eval's JUPITER launcher.
#
#   mn5_oellm_run_one.sh <export_dir> <out_root> <task> <n_shot> <suite> <limit>
#
# <task> is the oellm-eval task path (e.g. arc_challenge, AMC23, xcopa:tr), <suite> is
# lm-eval-harness | evalchemy | lighteval. <limit> 0 means the full dataset.
set -euo pipefail
export_dir=$1 out_root=$2 task=$3 n_shot=$4 suite=$5 limit=$6
R=${R:-/gpfs/projects/ehpc533/users/korbip/hybrid_v019}
cd "$(dirname "$0")/../../.."

# vLLM runtime and the MN5 eval environment (sets INFER_PY, DEST, PYTHONPATH, HF_HOME).
source scripts/korbi/vllm/setup_vllm_uv.sh
source scripts/korbi/vllm/uva_tables_cleanup.sh
# The vLLM wheel pins an older typer; lighteval's CLI metadata needs the image's 0.27.
rm -rf "$DEST/typer" "$DEST"/typer-*.dist-info "$DEST"/typer_slim-*.dist-info
export PYLIBS_OELLM=${PYLIBS_OELLM:-$R/pylibs_oellm}
export OELLM_STAGE=${OELLM_STAGE:-$R/oellm_mn5/oellm_mn5_stage}
export PYTHONPATH="$PYLIBS_OELLM:$OELLM_STAGE:$PYTHONPATH"
# Stanza tokenizers are used by a few multilingual rows (e.g. flores200 Maltese);
# the model files were downloaded on JUPITER and copied here (no internet on MN5).
export STANZA_RESOURCES_DIR=${STANZA_RESOURCES_DIR:-$R/stanza_resources}
export OELLM_INCLUDE="$PWD/submodules/oellm-eval/oellm/resources/custom_lm_eval_tasks"
export OELLM_MHC_IMPL=${OELLM_MHC_IMPL:-triton}
export OELLM_FUSION_IMPL=${OELLM_FUSION_IMPL:-triton}
export OELLM_DECODE_GRAPHS=${OELLM_DECODE_GRAPHS:-1}
export OELLM_GRAPH_MAX_ROWS=${OELLM_GRAPH_MAX_ROWS:-256}
export OELLM_ENGRAM_DECODE_IMPL=${OELLM_ENGRAM_DECODE_IMPL:-batched}
export OELLM_ENGRAM_TABLES=${OELLM_ENGRAM_TABLES:-ram}
export OELLM_ENGRAM_LOOKUP=${OELLM_ENGRAM_LOOKUP:-leader}
export OELLM_ENGRAM_MADVISE=${OELLM_ENGRAM_MADVISE:-normal}
export OELLM_LOCAL_EXPERT_LOAD=${OELLM_LOCAL_EXPERT_LOAD:-1}
export OELLM_MAX_NUM_SEQS=${OELLM_MAX_NUM_SEQS:-256}
export OELLM_TOKEN_BUDGET=${OELLM_TOKEN_BUDGET:-32768}
export OELLM_REQUEST_BATCH_SIZE=${OELLM_REQUEST_BATCH_SIZE:-64}
case "${OELLM_VLLM_ENFORCE_EAGER:-1}" in
    0|false|False|no|off) VLLM_EAGER=False ;;
    *) VLLM_EAGER=True ;;
esac

export_name=$(basename "$export_dir")
# The oellm-eval views add the identity chat template that Evalchemy/lighteval need.
VIEWS_ROOT=${VIEWS_ROOT:-/gpfs/scratch/ehpc533/hybrid_nucleus/views}
view_dir="$VIEWS_ROOT/identity/$export_name"
if [ -d "$view_dir" ]; then
    export_dir="$view_dir"
fi
task_key=$(printf '%s' "$task" | tr '/:| ' '____')
out="$out_root/$export_name/$suite"
[ "$limit" = 0 ] || out="${out}_limit${limit}"
mkdir -p "$out/done"
done_marker="$out/done/${task_key}_${n_shot}.ok"
if [ -f "$done_marker" ]; then
    echo "[mn5_oellm] already done: $done_marker"
    exit 0
fi

limit_args=()
[ "$limit" = 0 ] || limit_args=(--limit "$limit")

echo "[mn5_oellm] export=$export_name task=$task n_shot=$n_shot suite=$suite limit=$limit"
echo "[mn5_oellm] PYTHONPATH=$PYTHONPATH"

case "$suite" in
    lm-eval-harness)
        # One row; the proven MN5 runner loads one engine and writes summary_shard0.tsv.
        # The suite's stable name is vllm/<task>/<n_shot>.
        lm_name="vllm/$task/$n_shot"
        "$INFER_PY" scripts/korbi/flag_eval/flag_lmeval.py \
            --export "$export_dir" --out "$out_root" \
            --names "$lm_name" \
            --shard 0 --num-shards 1 \
            --tp "${OELLM_TP:-4}" --dp "${OELLM_DP:-1}" --ep "${OELLM_EP:-1}" \
            --max-num-seqs "$OELLM_MAX_NUM_SEQS" \
            --max-num-batched-tokens "$OELLM_TOKEN_BUDGET" \
            --request-batch-size "$OELLM_REQUEST_BATCH_SIZE" \
            "${limit_args[@]}"
        ;;
    evalchemy)
        # Same command shape as oellm-eval's rendered launcher.
        model_args="pretrained=$export_dir,trust_remote_code=True,dtype=bfloat16,gpu_memory_utilization=0.9,max_num_seqs=$OELLM_MAX_NUM_SEQS,tensor_parallel_size=${OELLM_TP:-4},enforce_eager=$VLLM_EAGER"
        [ "${OELLM_DP:-1}" -gt 1 ] && model_args="$model_args,data_parallel_size=${OELLM_DP:-1},data_parallel_backend=mp"
        [ "${OELLM_EP:-1}" = 1 ] && model_args="$model_args,enable_expert_parallel=True"
        ( cd "$OELLM_STAGE/evalchemy" && \
          "$INFER_PY" -m eval.eval --model vllm --tasks "$task" --model_args "$model_args" \
              --batch_size auto --output_path "$out/evalchemy_${task_key}_${n_shot}" \
              "${limit_args[@]}" )
        ;;
    lighteval)
        light_task="$task|$n_shot"
        # lighteval pins its own math/metric deps (latex2sympy2_extended==1.0.6, nltk,
        # sacrebleu, ...). Keep them in a lighteval-only prefix so the lm-eval harness
        # keeps using the pylibs_flag versions.
        PYLIBS_LIGHTEVAL=${PYLIBS_LIGHTEVAL:-$R/oellm_mn5/pylibs_lighteval}
        if [ -d "$PYLIBS_LIGHTEVAL" ]; then
            export PYTHONPATH="$PYLIBS_LIGHTEVAL:$PYTHONPATH"
        fi
        echo "[mn5_oellm] lighteval libs=$PYLIBS_LIGHTEVAL PYTHONPATH_head=${PYTHONPATH%%:*}"
        # The 248 GB export does not fit lighteval's HF `accelerate` path on 4 GPUs
        # (CPU offload then CUDA-OOMs), so vLLM is the default backend.
        if [ "${OELLM_LIGHTEVAL_BACKEND:-vllm}" = vllm ]; then
            "$INFER_PY" "$PWD/scripts/korbi/flag_eval/lighteval_vllm_cli.py" vllm \
                "model_name=$export_dir,trust_remote_code=True,dtype=bfloat16,gpu_memory_utilization=0.9,max_num_seqs=$OELLM_MAX_NUM_SEQS,max_num_batched_tokens=$OELLM_TOKEN_BUDGET,max_model_length=$OELLM_TOKEN_BUDGET,tensor_parallel_size=${OELLM_TP:-4}" \
                "$light_task" \
                --load-tasks-multilingual --output-dir "$out/lighteval_${task_key}_${n_shot}" \
                ${limit:+--max-samples "$limit"}
        else
            "$INFER_PY" "$PWD/scripts/korbi/flag_eval/lighteval_accelerate_cli.py" accelerate \
                "model_name=$export_dir,trust_remote_code=True,batch_size=32" "$light_task" \
                --load-tasks-multilingual --output-dir "$out/lighteval_${task_key}_${n_shot}" \
                ${limit:+--max-samples "$limit"}
        fi
        ;;
    *)
        echo "[mn5_oellm] unknown suite: $suite" >&2
        exit 2
        ;;
esac

printf '{"task": "%s", "n_shot": %s, "suite": "%s", "limit": %s, "time": "%s"}\n' \
    "$task" "$n_shot" "$suite" "$limit" "$(date -Is)" > "$done_marker"
echo "[mn5_oellm] done: $done_marker"
