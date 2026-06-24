#!/bin/bash
#SBATCH --job-name=lmeval-preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
## removed: gpus-per-node (using --gres=gpu:1 instead)
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --signal=B:TERM@90
#SBATCH --open-mode=append
#SBATCH --output=/shared_silo/scratch/%u/logs/lmeval-%j.out
#SBATCH --error=/shared_silo/scratch/%u/logs/lmeval-%j.err

# Generic preemptible lm-eval-harness job (single node, vLLM or hf-auto backend).
# All task/model/eval knobs are set by the caller via environment variables.
# On restart, --use_cache replays completed samples from SQLite.
#
# Required:
#   MODEL   — path to HuggingFace model directory
#   TASKS   — comma-separated lm-eval task names
#
# Common optional:
#   MODEL_BACKEND          vllm (default) | hf | hf-auto
#   NUM_FEWSHOT            0 (default)
#   BATCH_SIZE             auto (default)
#   DTYPE                  auto (default)            [hf-auto]
#   TRUST_REMOTE_CODE      set non-empty to enable   [hf-auto]
#   ATTN_IMPLEMENTATION    e.g. sdpa, flash_attention_2  [hf-auto]
#   PARALLELIZE            set non-empty to enable   [hf-auto]
#   DEVICE_MAP             auto (default)            [hf-auto]
#   TENSOR_PARALLEL_SIZE   (default unset)           [vllm]
#   DATA_PARALLEL_SIZE     (default unset)           [vllm]
#   GPU_MEM_UTIL           0.9 (default)             [vllm]
#   MAX_MODEL_LEN          (default unset)           [vllm]
#   MAX_GEN_TOKS           (default unset)
#   APPLY_CHAT_TEMPLATE    set non-empty to enable
#   LIMIT                  (default unset)
#   INCLUDE_PATH           dir with custom task YAMLs
#   EXTRA_ARGS             extra flags for lm_eval

# set -euo pipefail
# Always log why we exited — critical for diagnosing silent OOM kills
_on_exit() {
    local code=$?
    local sig=""
    [ $code -gt 128 ] && sig=" (signal $((code - 128)))"
    if [ $code -ne 0 ]; then
        echo "FATAL: job exited with code ${code}${sig} at $(date)" >&2
        echo "FATAL: job exited with code ${code}${sig} at $(date)"
        # Signal 9 = SIGKILL (OOM killer or admin cancel)
        [ $((code - 128)) -eq 9 ] && echo "  -> Likely cause: OOM kill. Try increasing --mem." >&2
    fi
    echo "Exit: code=${code} at $(date)"
}
trap _on_exit EXIT

SCRATCH="${SCRATCH:-/shared_silo/scratch/${USER}}"
IMG="${IMG:-/shared_silo/scratch/containers/vllm-dev_preview_releases_v0.20.0_20260422.sif}"
MODEL="${MODEL:?MODEL is required}"
TASKS="${TASKS:?TASKS is required}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
MODEL_BACKEND="${MODEL_BACKEND:-vllm}"

# hf / hf-auto knobs
DTYPE="${DTYPE:-auto}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-}"
PARALLELIZE="${PARALLELIZE:-}"
DEVICE_MAP="${DEVICE_MAP:-auto}"

# vllm knobs
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-}"

# misc
MAX_GEN_TOKS="${MAX_GEN_TOKS:-}"
APPLY_CHAT_TEMPLATE="${APPLY_CHAT_TEMPLATE:-}"
LIMIT="${LIMIT:-}"
INCLUDE_PATH="${INCLUDE_PATH:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

EVALS_DIR="${EVALS_DIR:-/shared_silo/scratch/shared/tw-dashboard/evals}"

# Local lm-eval checkout takes precedence; shared venv provides Python + other deps
_EVAL_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
LOCAL_LMEVAL="${LOCAL_LMEVAL:-$(cd "${_EVAL_SCRIPT_DIR}/../.." && pwd)/submodules/lm-evaluation-harness}"
LMEVAL_ENV="${LMEVAL_ENV:-/shared_silo/scratch/shared/tw-dashboard/lm-eval-env-v0.4.11}"
LMEVAL_SITE="${LMEVAL_ENV}/lib/python3.12/site-packages"

DASHBOARD_URL="${DASHBOARD_URL:-http://tus1-vm-amd-k8s-001:8080}"
DASHBOARD_API_KEY="${DASHBOARD_API_KEY:-${TW_API_KEY:-}}"
WANDB_SHIM="${WANDB_SHIM:-}"
FASTTRACKML_URL="${FASTTRACKML_URL:-http://tus1-vm-amd-k8s-001:5000}"

MODEL_SHORT="${MODEL_NAME:-$(basename "$MODEL")}"
TASK_SLUG=$(echo "$TASKS" | tr ',:|/ ' '_')

# Stable output dir — no $SLURM_JOBID so requeued jobs find previous results
OUTDIR="${SCRATCH}/eval_results/lmeval_${TASK_SLUG}_${MODEL_SHORT}"
CACHE_DIR="${SCRATCH}/lm_eval_cache/${MODEL_SHORT}"

mkdir -p "${SCRATCH}/eval_results" "$OUTDIR" "$CACHE_DIR"

# Remove any stale SQLite cache from a previous OOM-killed run — corrupt caches
# cause lm_eval to crash silently before printing anything.
CACHE_FILE="${CACHE_DIR}/${TASK_SLUG}"
if [ -f "${CACHE_FILE}" ] && ! python3 -c "import sqlite3; sqlite3.connect('${CACHE_FILE}').execute('PRAGMA integrity_check')" 2>/dev/null | grep -q "^ok$"; then
    echo "WARNING: corrupt/stale cache detected at ${CACHE_FILE} — removing"
    rm -f "${CACHE_FILE}"
fi

echo "=== lm-eval (preemptible): ${TASKS} ==="
echo "Model:      $MODEL"
echo "Backend:    $MODEL_BACKEND"
echo "Tasks:      $TASKS"
echo "Fewshot:    $NUM_FEWSHOT"
echo "Batch:      $BATCH_SIZE"
echo "Output:     $OUTDIR"
echo "Cache:      $CACHE_DIR"
echo "Job ID:     $SLURM_JOBID"
echo "Restart:    ${SLURM_RESTART_COUNT:-0}"
echo "Start:      $(date)"

# Task-level skip: if results JSON already exists, this run is done
if find "$OUTDIR" -maxdepth 2 -name 'results_*.json' -print -quit 2>/dev/null | grep -q .; then
    echo "Results already exist in $OUTDIR — skipping (already completed)"
    find "$OUTDIR" -maxdepth 2 -name 'results_*.json' -exec ls -lh {} +
    exit 0
fi
echo "No existing results JSON found in $OUTDIR — starting evaluation"

# ---------------------------------------------------------------------------
# Build model_args from backend + knobs
# ---------------------------------------------------------------------------
if [ "$MODEL_BACKEND" = "vllm" ]; then
    MODEL_ARGS="pretrained=${MODEL},gpu_memory_utilization=${GPU_MEM_UTIL}"
    [ -n "$DTYPE" ]                && MODEL_ARGS="${MODEL_ARGS},dtype=${DTYPE}"
    [ -n "$MAX_MODEL_LEN" ]        && MODEL_ARGS="${MODEL_ARGS},max_model_len=${MAX_MODEL_LEN}"
    [ -n "$TENSOR_PARALLEL_SIZE" ] && MODEL_ARGS="${MODEL_ARGS},tensor_parallel_size=${TENSOR_PARALLEL_SIZE}"
    [ -n "$DATA_PARALLEL_SIZE" ]   && MODEL_ARGS="${MODEL_ARGS},data_parallel_size=${DATA_PARALLEL_SIZE}"
else
    MODEL_ARGS="pretrained=${MODEL},device_map=${DEVICE_MAP},dtype=${DTYPE}"
    [ -n "$TRUST_REMOTE_CODE" ]    && MODEL_ARGS="${MODEL_ARGS},trust_remote_code=True"
    [ -n "$ATTN_IMPLEMENTATION" ]  && MODEL_ARGS="${MODEL_ARGS},attn_implementation=${ATTN_IMPLEMENTATION}"
    [ -n "$PARALLELIZE" ]          && MODEL_ARGS="${MODEL_ARGS},parallelize=True"
fi

# ---------------------------------------------------------------------------
# Build optional lm_eval flags
# ---------------------------------------------------------------------------
OPT_FLAGS=""
[ -n "$MAX_GEN_TOKS" ]        && OPT_FLAGS="$OPT_FLAGS --gen_kwargs max_gen_toks=$MAX_GEN_TOKS"
[ -n "$APPLY_CHAT_TEMPLATE" ] && OPT_FLAGS="$OPT_FLAGS --apply_chat_template"
[ -n "$LIMIT" ]               && OPT_FLAGS="$OPT_FLAGS --limit $LIMIT"
[ -n "$INCLUDE_PATH" ]        && OPT_FLAGS="$OPT_FLAGS --include_path ${INCLUDE_PATH}"

# Wandb shim bind mount (optional)
WANDB_BIND=""
WANDB_ARGS=""
if [ -n "$WANDB_SHIM" ] && [ -d "${WANDB_SHIM}/wandb" ]; then
    WANDB_BIND="-B ${WANDB_SHIM}/wandb:/usr/local/lib/python3.12/dist-packages/wandb"
    WANDB_ARGS="--wandb_args project=tw-evals,name=${MODEL_SHORT}_${TASK_SLUG}"
fi

echo "Launching singularity at $(date)"
echo "Image: ${IMG}"
echo "SLURM: job=${SLURM_JOBID:-unknown} node=${SLURMD_NODENAME:-unknown} gpus=${SLURM_GPUS_ON_NODE:-unknown} mem=${SLURM_MEM_PER_NODE:-unknown}"

singularity exec --rocm \
    -B /shared_silo/scratch:/shared_silo/scratch:rw \
    -B /usr/share/libdrm:/usr/share/libdrm:ro \
    $WANDB_BIND \
    "$IMG" bash <<INNEREOF &
# set -euo pipefail

echo "Inside singularity at $(date)"

# --- GPU visibility ---
# Pin to the ABSOLUTE physical GPU index SLURM assigned (SLURM_JOB_GPUS).
# ROCR_VISIBLE_DEVICES is cgroup-relative (always 0) and the --rocm container
# sees all physical GPUs, so co-located single-GPU jobs would all collide on
# physical GPU 0. SLURM_JOB_GPUS holds the distinct absolute index per job and
# is expanded on the host here (unquoted heredoc).
export HIP_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
unset ROCR_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES

# Clean stale pip packages that could shadow pre-installed ones
rm -rf ~/.local/lib/python3.12/site-packages 2>/dev/null || true

# Local lm-eval checkout first, shared venv second (provides Python + deps)
export PYTHONPATH="${LOCAL_LMEVAL}:${LMEVAL_SITE}:\${PYTHONPATH:-}"
export PATH="${LMEVAL_ENV}/bin:\$PATH"

# --- Per-job unique port to avoid TCP store collisions when multiple jobs share a node ---
# Derives a port in 20000-59999 from SLURM_JOBID so concurrent jobs never clash.
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$(( 20000 + (${SLURM_JOBID} % 40000) ))}"

# --- HuggingFace settings ---
export HF_HUB_DISABLE_XET=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_TELEMETRY=1
export TORCH_EXTENSIONS_DIR=/dev/shm/torch_ext

# --- vLLM / ROCm settings ---
export VLLM_USE_V1=1
export VLLM_TARGET_DEVICE=rocm
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# --- Dashboard tracking ---
export WANDB_API_KEY=local
export FASTTRACKML_URL=${FASTTRACKML_URL}
export DASHBOARD_URL=${DASHBOARD_URL}
export DASHBOARD_API_KEY=${DASHBOARD_API_KEY}
export EVALS_DIR=${EVALS_DIR}
export MODEL=${MODEL}
export MODEL_SHORT=${MODEL_SHORT}
export TASKS=${TASKS}
export TASK_SLUG=${TASK_SLUG}
export NUM_FEWSHOT=${NUM_FEWSHOT}

export LM_HARNESS_CACHE_PATH=${SCRATCH}/lm_eval_request_cache

python -c "import lm_eval, importlib.metadata as m; print(f'lm-eval: {m.version(\"lm_eval\")} from {lm_eval.__file__}')" 2>/dev/null || true

echo "Backend: ${MODEL_BACKEND}  args: ${MODEL_ARGS}"
echo "Running: tasks=${TASKS} fewshot=${NUM_FEWSHOT} batch=${BATCH_SIZE}"

python -m lm_eval \
    --model "${MODEL_BACKEND}" \
    --model_args "${MODEL_ARGS}" \
    --tasks "${TASKS}" \
    --num_fewshot "${NUM_FEWSHOT}" \
    --batch_size "${BATCH_SIZE}" \
    --output_path "${OUTDIR}" \
    --log_samples \
    --use_cache "${CACHE_DIR}/${TASK_SLUG}" \
    ${WANDB_ARGS} \
    ${OPT_FLAGS} \
    ${EXTRA_ARGS}

echo "lm-eval complete!"

# --- Post results to dashboard ---
RESULTS_JSON=\$(find "${OUTDIR}" -maxdepth 2 -name 'results_*.json' -print -quit 2>/dev/null)
if [ -n "\$RESULTS_JSON" ]; then
    echo "Posting results to dashboard: \$RESULTS_JSON"
    RESULTS_DIR=\$(dirname "\$RESULTS_JSON")
    python3 - "\$RESULTS_JSON" "\$RESULTS_DIR" <<'REPORTER'
import json, sys, os, glob

sys.path.insert(0, os.environ.get("EVALS_DIR", "."))
from dashboard_reporter import DashboardReporter

results_path = sys.argv[1]
results_dir = sys.argv[2]

with open(results_path) as f:
    results = json.load(f)

try:
    r = DashboardReporter()
    eid = r.create_run(
        model_name=os.environ.get("MODEL_SHORT", "unknown"),
        eval_framework="lm-eval",
        job_id=os.environ.get("SLURM_JOB_ID", ""),
        run_name=f'{os.environ.get("MODEL_SHORT","")}__{os.environ.get("TASK_SLUG","")}',
        config_json=json.dumps({
            "tasks": os.environ.get("TASKS", ""),
            "num_fewshot": os.environ.get("NUM_FEWSHOT", "0"),
            "model": os.environ.get("MODEL", ""),
        }),
    )
    r.report_lm_eval_results(eid, results)

    sample_files = sorted(glob.glob(os.path.join(results_dir, "samples_*.jsonl")))
    all_samples = []
    for sf in sample_files:
        task_name = os.path.basename(sf).split("samples_", 1)[1].rsplit("_2", 1)[0]
        with open(sf) as fh:
            for line in fh:
                s = json.loads(line)
                doc = s.get("doc", {})
                question = doc.get("question", "")
                if not question:
                    args = s.get("arguments", {})
                    if isinstance(args, dict):
                        first_arg = next(iter(args.values()), {})
                        question = first_arg.get("arg_0", "")[:500] if isinstance(first_arg, dict) else ""
                    elif isinstance(args, (list, tuple)) and args:
                        question = str(args[0][0])[:500] if isinstance(args[0], (list, tuple)) else str(args[0])[:500]

                target = s.get("target", "")
                choices = doc.get("choices", {})
                if isinstance(target, int) and isinstance(choices, dict):
                    labels = choices.get("label", choices.get("text", []))
                    target = labels[target] if target < len(labels) else str(target)
                elif isinstance(target, int) and isinstance(choices, list):
                    target = choices[target] if target < len(choices) else str(target)
                target = str(target)

                completion = ""
                resps = s.get("filtered_resps", s.get("resps", []))
                if resps and isinstance(resps[0], (list, tuple)):
                    vals = [r[0] if isinstance(r, (list, tuple)) and r else r for r in resps]
                    if all(isinstance(v, str) and v.replace("-", "").replace(".", "").isdigit() for v in vals if v):
                        scores = [float(v) for v in vals if v]
                        if scores:
                            chosen = scores.index(max(scores))
                            if isinstance(choices, dict):
                                labels = choices.get("label", [])
                                texts = choices.get("text", [])
                                if chosen < len(labels):
                                    completion = labels[chosen]
                                    if chosen < len(texts):
                                        completion = f"{labels[chosen]}: {texts[chosen][:200]}"
                            elif isinstance(choices, list) and chosen < len(choices):
                                completion = str(choices[chosen])[:200]
                    elif isinstance(vals[0], str):
                        completion = vals[0][:2000]

                metric_names = s.get("metrics", [])
                primary_score = None
                for m in metric_names:
                    val = s.get(m)
                    if isinstance(val, (int, float)) and primary_score is None:
                        primary_score = float(val)

                all_samples.append({
                    "task": task_name,
                    "sample_idx": s.get("doc_id", 0),
                    "input_text": question[:500],
                    "input_full": question if len(question) > 500 else None,
                    "completion": completion,
                    "target": target[:1000],
                    "score": primary_score,
                    "is_correct": primary_score == 1.0 if primary_score is not None else None,
                    "metrics": json.dumps({m: s.get(m) for m in metric_names if isinstance(s.get(m), (int, float))}) or None,
                })

    if all_samples:
        r.add_samples(eid, all_samples)
        print(f"[DASHBOARD] Posted eval {eid}: {len(all_samples)} samples across {len(sample_files)} tasks")
    else:
        print(f"[DASHBOARD] Posted eval {eid}: scores only (no sample JSONL files found)")

except Exception as e:
    import traceback
    print(f"[DASHBOARD] WARNING: Failed to post results: {e}", file=sys.stderr)
    traceback.print_exc()
REPORTER
else
    echo "[DASHBOARD] No results JSON found in ${OUTDIR} — skipping dashboard report"
fi
INNEREOF
SING_PID=$!

trap 'echo "SIGTERM received ($(date)), shutting down..."; kill -TERM $SING_PID 2>/dev/null; wait $SING_PID 2>/dev/null' TERM
wait $SING_PID
SING_EXIT=$?

if [ $SING_EXIT -ne 0 ]; then
    echo "ERROR: singularity exited with code ${SING_EXIT} at $(date)" >&2
    exit $SING_EXIT
fi

echo "End: $(date)"
