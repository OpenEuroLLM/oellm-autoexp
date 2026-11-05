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
#
# A GROUP of rows of one suite: <task> and <n_shot> comma-separated (one n_shot for all, or
# one per task). They run on ONE model load (an 84B export takes minutes to load; the rows
# often less): lm-eval in one flag_lmeval.py engine, lighteval / evalchemy as one harness call
# with the task list (output <suite>_group_<hash>, which the aggregator reads like a row's).
# Rows already done are skipped; each row gets its done marker once its result is there, so
# a rerun of a partly failed group runs only the rest.
set -euo pipefail
export_dir=$1 out_root=$2 tasks_arg=$3 shots_arg=$4 suite=$5 limit=$6
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
# Keep the lighteval vLLM model class (which reads the env var) in sync with VLLM_EAGER.
export OELLM_VLLM_ENFORCE_EAGER=$([ "$VLLM_EAGER" = False ] && echo 0 || echo 1)

export_name=$(basename "$export_dir")
# The oellm-eval views add the identity chat template that Evalchemy/lighteval need.
VIEWS_ROOT=${VIEWS_ROOT:-/gpfs/scratch/ehpc533/hybrid_nucleus/views}
view_dir="$VIEWS_ROOT/identity/$export_name"
if [ -d "$view_dir" ]; then
    export_dir="$view_dir"
fi
key_of() { printf '%s' "$1" | tr '/:| ' '____'; }
IFS=',' read -r -a all_tasks <<< "$tasks_arg"
IFS=',' read -r -a all_shots <<< "$shots_arg"
if [ "${#all_shots[@]}" -eq 1 ]; then
    for i in "${!all_tasks[@]}"; do all_shots[$i]=${all_shots[0]}; done
fi
if [ "${#all_shots[@]}" -ne "${#all_tasks[@]}" ]; then
    echo "[mn5_oellm] ${#all_tasks[@]} task(s) but ${#all_shots[@]} n_shot value(s)" >&2
    exit 2
fi
out="$out_root/$export_name/$suite"
[ "$limit" = 0 ] || out="${out}_limit${limit}"
mkdir -p "$out/done"
tasks=() shots=()
for i in "${!all_tasks[@]}"; do
    marker="$out/done/$(key_of "${all_tasks[$i]}")_${all_shots[$i]}.ok"
    if [ -f "$marker" ]; then
        echo "[mn5_oellm] already done: $marker"
    else
        tasks+=("${all_tasks[$i]}") shots+=("${all_shots[$i]}")
    fi
done
[ "${#tasks[@]}" -gt 0 ] || { echo "[mn5_oellm] all ${#all_tasks[@]} row(s) done"; exit 0; }
task=$(IFS=','; printf '%s' "${tasks[*]}") n_shot=$(IFS=','; printf '%s' "${shots[*]}")
# the output directory of this call: the row's own name, or one per group (by its rows)
if [ "${#tasks[@]}" -eq 1 ]; then
    task_key="$(key_of "${tasks[0]}")" n_shot=${shots[0]}
    run_key="${task_key}_${n_shot}"
else
    task_key="group_$(printf '%s\n' "$task" "$n_shot" | sha1sum | cut -c1-10)"
    run_key=$task_key
fi

limit_args=()
[ "$limit" = 0 ] || limit_args=(--limit "$limit")

echo "[mn5_oellm] export=$export_name task=$task n_shot=$n_shot suite=$suite limit=$limit"
echo "[mn5_oellm] PYTHONPATH=$PYTHONPATH"

rc=0
case "$suite" in
    lm-eval-harness)
        # The proven MN5 runner: one engine for all the rows, summary_shard0.tsv, and its
        # own done marker per row. The suite's stable name is vllm/<task>/<n_shot>.
        lm_names=()
        for i in "${!tasks[@]}"; do lm_names+=("vllm/${tasks[$i]}/${shots[$i]}"); done
        "$INFER_PY" scripts/korbi/flag_eval/flag_lmeval.py \
            --export "$export_dir" --out "$out_root" \
            --names "${lm_names[@]}" \
            --shard 0 --num-shards 1 \
            --tp "${OELLM_TP:-4}" --dp "${OELLM_DP:-1}" --ep "${OELLM_EP:-1}" \
            --max-num-seqs "$OELLM_MAX_NUM_SEQS" \
            --max-num-batched-tokens "$OELLM_TOKEN_BUDGET" \
            --request-batch-size "$OELLM_REQUEST_BATCH_SIZE" \
            "${limit_args[@]}" || rc=$?
        ;;
    evalchemy)
        # Same command shape as oellm-eval's rendered launcher.
        # Large evalchemy rows (GPQADiamond, JEEBench) OOMed at 256 seqs: ~2.7 GiB
        # sat in CUDA-graph private pools. Use a smaller batch/graph ceiling by default.
        eval_seqs=${OELLM_EVALCHEMY_MAX_NUM_SEQS:-64}
        eval_graph_rows=${OELLM_EVALCHEMY_GRAPH_MAX_ROWS:-64}
        model_args="pretrained=$export_dir,trust_remote_code=True,dtype=bfloat16,gpu_memory_utilization=0.9,max_num_seqs=$eval_seqs,tensor_parallel_size=${OELLM_TP:-4},enforce_eager=$VLLM_EAGER"
        [ "${OELLM_DP:-1}" -gt 1 ] && model_args="$model_args,data_parallel_size=${OELLM_DP:-1},data_parallel_backend=mp"
        [ "${OELLM_EP:-1}" = 1 ] && model_args="$model_args,enable_expert_parallel=True"
        ( cd "$OELLM_STAGE/evalchemy" && \
          OELLM_GRAPH_MAX_ROWS=$eval_graph_rows "$INFER_PY" -m eval.eval --model vllm --tasks "$task" --model_args "$model_args" \
              --batch_size auto --output_path "$out/evalchemy_${run_key}" \
              "${limit_args[@]}" ) || rc=$?
        ;;
    lighteval)
        light_task=$(for i in "${!tasks[@]}"; do printf '%s|%s\n' "${tasks[$i]}" "${shots[$i]}"; done | paste -sd, -)
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
        # Full 1012-sample rows OOMed at 256 seqs (~2.5 GiB in CUDA-graph pools),
        # so keep a smaller batch/graph ceiling for lighteval.
        lv_seqs=${OELLM_LIGHTEVAL_MAX_NUM_SEQS:-64}
        lv_graph_rows=${OELLM_LIGHTEVAL_GRAPH_MAX_ROWS:-64}
        if [ "${OELLM_LIGHTEVAL_BACKEND:-vllm}" = vllm ]; then
            OELLM_GRAPH_MAX_ROWS=$lv_graph_rows \
            "$INFER_PY" "$PWD/scripts/korbi/flag_eval/lighteval_vllm_cli.py" vllm \
                "model_name=$export_dir,trust_remote_code=True,dtype=bfloat16,gpu_memory_utilization=0.9,max_num_seqs=$lv_seqs,max_num_batched_tokens=$OELLM_TOKEN_BUDGET,max_model_length=$OELLM_TOKEN_BUDGET,tensor_parallel_size=${OELLM_TP:-4}" \
                "$light_task" \
                --load-tasks-multilingual --output-dir "$out/lighteval_${run_key}" \
                ${limit:+--max-samples "$limit"} || rc=$?
        else
            # the HF path loads the export's own model code: this checkout's (hf_fast_prefill_env)
            [ "${OELLM_HF_REFRESH_CODE:-0}" = 1 ] && bash scripts/korbi/hf_export/refresh_code.sh "$export_dir"
            "$INFER_PY" "$PWD/scripts/korbi/flag_eval/lighteval_accelerate_cli.py" accelerate \
                "model_name=$export_dir,trust_remote_code=True,batch_size=32" "$light_task" \
                --load-tasks-multilingual --output-dir "$out/lighteval_${run_key}" \
                ${limit:+--max-samples "$limit"} || rc=$?
        fi
        ;;
    *)
        echo "[mn5_oellm] unknown suite: $suite" >&2
        exit 2
        ;;
esac

# a done marker per row whose result is there (a failed harness call may still have written
# some of them: a rerun of the group runs only the rest)
lm_done="$out_root/$export_name"
[ "$limit" = 0 ] || lm_done="${lm_done}_limit${limit}"
missing=$(python3 scripts/korbi/flag_eval/oellm_rows_done.py --suite "$suite" \
    --results "$out/${suite}_${run_key}" --lm-done "$lm_done/done" --out-done "$out/done" \
    --limit "$limit" --rows "$(for i in "${!tasks[@]}"; do printf '%s|%s\n' "${tasks[$i]}" "${shots[$i]}"; done)")
if [ -n "$missing" ]; then
    echo "[mn5_oellm] rows without a result (rc $rc): $missing" >&2
    exit 1
fi
[ "$rc" = 0 ] || echo "[mn5_oellm] harness exit $rc, but every row has its result"
echo "[mn5_oellm] done: ${#tasks[@]} row(s) of $suite"
