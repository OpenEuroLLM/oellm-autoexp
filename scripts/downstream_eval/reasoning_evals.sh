#!/usr/bin/env bash
# Render (but never submit) a JUPITER vLLM DP4 comparison on three reasoning tasks.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/downstream_eval/reasoning_evals.sh [MODEL|all]

Prepare identity prompt views and render one Slurm launcher for the full
GSM8K 4-shot, MATH500 0-shot and MBPP 3-shot evaluations. The script does not
submit the launcher.

MODEL defaults to all, meaning every export under EXPORT_ROOT.

Useful overrides:
  TASKS="..."             Tasks from the pinned `reasoning` group (default:
                          "gsm8k MATH500 mbpp"; e.g. "HumanEval GPQADiamond").
                          Run mbpp with QUEUE_LIMIT=1: its code_eval metric
                          shares one temp file across concurrent tasks.
  RUN_ID=...              Stable suffix for the run directory.
  OELLM_ACCOUNT=...       Slurm account (default: e-ext-2025e02-108).
  QUEUE_LIMIT=...         Array workers (default: 6 for all, 1 for one model).
  MAX_ARRAY_LEN=...       Concurrent workers (default: QUEUE_LIMIT).
  CHECK_ONLY=1            Validate inputs without creating prompt views/runs.
  VERIFY_IMAGE_SHA256=1   Hash the 8.1 GiB SIF before rendering.
  EVAL_REPO=...           Pinned oellm-eval checkout.
  EVAL_IMAGE=...          Complete Evalchemy/vLLM SIF.
  HF_HOME=...             Offline dataset cache.
  EXPORT_ROOT=...         Root containing the six HF exports.
  PROMPT_VIEW_ROOT=...    Same-filesystem destination for identity views.
  RUN_ROOT=...            Root for rendered launchers and results.
  EVAL_PYTHON=...         Python from an environment with oellm-eval deps.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi
if [[ $# -gt 1 ]]; then
    usage >&2
    exit 2
fi

MODEL="${1:-all}"

EXPECTED_EVAL_REV="${EXPECTED_EVAL_REV:-70fa2a1f7288cf70b8d29f9bd08fb07ed3bcd013}"
EXPECTED_IMAGE_SHA256="${EXPECTED_IMAGE_SHA256:-5ae0222748346026ae311845664fec45adfea19a715f2b60eb661082973d06c2}"
EVAL_REPO="${EVAL_REPO:-$(dirname "$(realpath "$0")")/../../submodules/oellm-eval}"
EVAL_IMAGE="${EVAL_IMAGE:-/e/fscratch/e-sta-openeurollm/jj1/eval_integrated_install_20260911/images/oellm-eval-complete.sif}"
HF_HOME="${HF_HOME:-/e/fscratch/e-sta-openeurollm/jj1/eval/hf_home}"
EXPORT_ROOT="${EXPORT_ROOT:-/e/scratch/e-sta-openeurollm/production_training/downstream_eval_32b/hf}"
PROMPT_VIEW_ROOT="${PROMPT_VIEW_ROOT:-/e/scratch/e-sta-openeurollm/production_training/downstream_eval_32b/prompt_views/identity}"
RUN_ROOT="${RUN_ROOT:-/e/scratch/e-sta-openeurollm/production_training/downstream_eval_32b/reasoning_vllm}"
OELLM_ACCOUNT="${OELLM_ACCOUNT:-e-ext-2025e02-108}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
VERIFY_IMAGE_SHA256="${VERIFY_IMAGE_SHA256:-0}"
CHECK_ONLY="${CHECK_ONLY:-0}"
# Tasks from the pinned `reasoning` group, space-separated.
TASKS="${TASKS:-gsm8k MATH500 mbpp}"
read -ra TASK_LIST <<< "$TASKS"
[[ ${#TASK_LIST[@]} -gt 0 ]] || { echo "error: TASKS is empty" >&2; exit 1; }
has_task() { [[ " $TASKS " == *" $1 "* ]]; }

if [[ "$MODEL" == all ]]; then
    mapfile -t MODELS < <(cd "$EXPORT_ROOT" && ls -d */ 2>/dev/null | sed 's#/$##')
    [[ ${#MODELS[@]} -gt 0 ]] || fail "no exports found in $EXPORT_ROOT"
else
    MODELS=("$MODEL")
fi
DEFAULT_QUEUE_LIMIT=${#MODELS[@]}

RUN_DIR="$RUN_ROOT/${MODEL}_${RUN_ID}"
HF_HUB_CACHE="$HF_HOME/hub"
HF_DATASETS_CACHE="$HF_HOME/datasets"
QUEUE_LIMIT="${QUEUE_LIMIT:-$DEFAULT_QUEUE_LIMIT}"
MAX_ARRAY_LEN="${MAX_ARRAY_LEN:-$QUEUE_LIMIT}"
EVAL_DP="${EVAL_DP:-4}"
EVAL_TP="${EVAL_TP:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-288}"
SLURM_MEM="${SLURM_MEM:-400G}"
EVAL_TIME="${EVAL_TIME:-01:00:00}"

fail() {
    echo "error: $*" >&2
    exit 1
}

nearest_existing_ancestor() {
    local candidate="$1"
    local parent
    while [[ ! -e "$candidate" ]]; do
        parent=$(dirname "$candidate")
        [[ "$parent" != "$candidate" ]] || fail "no existing ancestor for: $1"
        candidate="$parent"
    done
    printf '%s\n' "$candidate"
}

for command_name in dirname git find grep sha256sum tee wc; do
    command -v "$command_name" >/dev/null || fail "required command not found: $command_name"
done

[[ -e "$EVAL_REPO/.git" ]] || fail "evaluation checkout not found: $EVAL_REPO"  # a submodule's .git is a file
actual_eval_rev=$(git -C "$EVAL_REPO" rev-parse HEAD)
[[ "$actual_eval_rev" == "$EXPECTED_EVAL_REV" ]] || \
    fail "oellm-eval is at $actual_eval_rev; expected $EXPECTED_EVAL_REV"
[[ -z "$(git -C "$EVAL_REPO" status --porcelain)" ]] || \
    fail "oellm-eval checkout has local changes: $EVAL_REPO"

[[ -r "$EVAL_IMAGE" ]] || fail "evaluation image is not readable: $EVAL_IMAGE"
[[ -d "$HF_HOME" ]] || fail "offline HF cache not found: $HF_HOME"
[[ -d "$HF_HUB_CACHE" ]] || fail "offline hub cache not found: $HF_HUB_CACHE"
[[ -d "$HF_DATASETS_CACHE" ]] || fail "offline datasets cache not found: $HF_DATASETS_CACHE"
require_cached() {  # task, dataset dir under the datasets cache
    has_task "$1" || return 0
    find "$HF_DATASETS_CACHE/$2" -type f -name '*.arrow' -print -quit 2>/dev/null | \
        grep -q . || fail "$1 data is absent from the offline cache: $HF_DATASETS_CACHE/$2"
}
require_cached gsm8k openai___gsm8k
require_cached mbpp google-research-datasets___mbpp
require_cached MATH500 HuggingFaceH4___math-500
require_cached HumanEval openai___openai_humaneval
require_cached GPQADiamond Idavidrein___gpqa

[[ "$QUEUE_LIMIT" =~ ^[1-9][0-9]*$ ]] || fail "QUEUE_LIMIT must be a positive integer"
[[ "$MAX_ARRAY_LEN" =~ ^[1-9][0-9]*$ ]] || fail "MAX_ARRAY_LEN must be a positive integer"
[[ "$CHECK_ONLY" == 0 || "$CHECK_ONLY" == 1 ]] || fail "CHECK_ONLY must be 0 or 1"

prompt_view_parent=$(nearest_existing_ancestor "$PROMPT_VIEW_ROOT")
run_parent=$(nearest_existing_ancestor "$RUN_ROOT")
[[ -d "$prompt_view_parent" && -w "$prompt_view_parent" ]] || \
    fail "prompt-view parent is not writable: $prompt_view_parent"
[[ -d "$run_parent" && -w "$run_parent" ]] || \
    fail "run parent is not writable: $run_parent"

for model_name in "${MODELS[@]}"; do
    base_export="$EXPORT_ROOT/$model_name"
    [[ -d "$base_export" ]] || fail "HF export not found: $base_export"
    [[ -r "$base_export/config.json" ]] || fail "model config not found: $base_export/config.json"
    [[ -r "$base_export/tokenizer_config.json" ]] || fail "tokenizer config not found: $base_export/tokenizer_config.json"
    find "$base_export" -maxdepth 1 -type f \( -name '*.safetensors' -o -name '*.bin' \) -print -quit | \
        grep -q . || fail "model weights not found: $base_export"
done

if [[ -z "${EVAL_PYTHON:-}" ]]; then
    scheduler_bin=$(command -v oellm-eval || true)
    [[ -n "$scheduler_bin" ]] || fail "oellm-eval is not installed; set EVAL_PYTHON to its environment's Python"
    IFS= read -r scheduler_shebang < "$scheduler_bin"
    EVAL_PYTHON="${scheduler_shebang#\#!}"
fi
[[ -x "$EVAL_PYTHON" ]] || fail "evaluation Python is not executable: $EVAL_PYTHON"

run_oellm_eval() {
    PYTHONPATH="$EVAL_REPO${PYTHONPATH:+:$PYTHONPATH}" \
        "$EVAL_PYTHON" -c 'from oellm.main import main; main()' "$@"
}

schedule_help=$(run_oellm_eval schedule --help 2>&1)
[[ "$schedule_help" == *"--data_parallel_size"* ]] || \
    fail "the selected Python cannot load the pinned vLLM scheduler"
[[ "$schedule_help" == *"--confirm_run_unsafe_code"* ]] || \
    fail "the selected Python cannot load the pinned MBPP safety option"

if [[ "$VERIFY_IMAGE_SHA256" == 1 ]]; then
    actual_image_sha256=$(sha256sum "$EVAL_IMAGE")
    actual_image_sha256=${actual_image_sha256%% *}
    [[ "$actual_image_sha256" == "$EXPECTED_IMAGE_SHA256" ]] || \
        fail "image SHA-256 is $actual_image_sha256; expected $EXPECTED_IMAGE_SHA256"
elif [[ "$VERIFY_IMAGE_SHA256" != 0 ]]; then
    fail "VERIFY_IMAGE_SHA256 must be 0 or 1"
fi

PROMPT_VIEWS=()
for model_name in "${MODELS[@]}"; do
    base_export="$EXPORT_ROOT/$model_name"
    prompt_view="$PROMPT_VIEW_ROOT/$model_name"
    if [[ -e "$prompt_view" ]]; then
        PYTHONPATH= "$EVAL_PYTHON" - "$base_export" "$prompt_view" <<'PY'
import json
import sys
from pathlib import Path

source = Path(sys.argv[1]).resolve()
view = Path(sys.argv[2]).resolve()
manifest = json.loads((view / "prompt-view-manifest.json").read_text())
if Path(manifest["source"]).resolve() != source or manifest.get("template") != "identity":
    raise SystemExit(f"identity view manifest does not match {source}: {view}")
if any(path.is_symlink() for path in view.iterdir()):
    raise SystemExit(f"identity view contains symlinks: {view}")
PY
    elif [[ "$CHECK_ONLY" == 1 ]]; then
        echo "Would prepare hard-linked identity prompt view: $prompt_view"
    else
        echo "Preparing hard-linked identity prompt view: $prompt_view"
        "$EVAL_PYTHON" "$EVAL_REPO/containers/prepare_base_model_view.py" \
            --source "$base_export" --out "$prompt_view" --template identity
    fi
    PROMPT_VIEWS+=("$prompt_view")
done

if [[ "$CHECK_ONLY" == 1 ]]; then
    echo
    echo "Read-only preflight passed for ${#MODELS[@]} model(s): ${MODELS[*]}"
    echo "Evaluation revision: $actual_eval_rev"
    echo "Image: $EVAL_IMAGE"
    echo "Run destination: $RUN_DIR"
    exit 0
fi

[[ ! -e "$RUN_DIR" ]] || fail "run directory already exists: $RUN_DIR"
mkdir -p "$RUN_DIR"

export EVAL_BASE_DIR="$RUN_ROOT"
export EVAL_OUTPUT_DIR="$RUN_DIR"
export EVAL_CONTAINER_IMAGE="$EVAL_IMAGE"
export EVALCHEMY_DIR=/opt/evalchemy
export EVAL_WORKDIR=/opt/evalchemy
export NLTK_DATA=/opt/nltk_data
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export QUEUE_LIMIT
export HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE OELLM_ACCOUNT
export OELLM_CPUS_PER_TASK="$CPUS_PER_TASK" OELLM_SLURM_MEM="$SLURM_MEM" OELLM_EVAL_TIME="$EVAL_TIME"
printf -v OELLM_MODELS '%s\n' "${PROMPT_VIEWS[@]}"
export OELLM_MODELS
export OELLM_REFERENCE_TASKS="$TASKS"

PYTHONPATH="$EVAL_REPO${PYTHONPATH:+:$PYTHONPATH}" "$EVAL_PYTHON" - <<'PY'
import csv
import os
from pathlib import Path
from oellm.task_groups import _expand_task_groups

selected = set(os.environ["OELLM_REFERENCE_TASKS"].split())
tasks = [task for task in _expand_task_groups(["reasoning"]) if task.task in selected]
if {task.task for task in tasks} != selected or len(tasks) != len(selected):
    raise SystemExit(f"pinned registry did not resolve exactly {sorted(selected)}")
models = os.environ["OELLM_MODELS"].splitlines()
if not models:
    raise SystemExit("no model prompt views selected")
with (Path(os.environ["EVAL_OUTPUT_DIR"]) / "input.csv").open("w", newline="") as handle:
    writer = csv.writer(handle, lineterminator="\n")
    writer.writerow(["model_path", "task_path", "n_shot", "eval_suite"])
    for model in models:
        for task in tasks:
            writer.writerow([model, task.task, task.n_shot, task.suite])
PY

expected_evals=$((${#TASK_LIST[@]} * ${#MODELS[@]}))
actual_csv_lines=$(wc -l < "$RUN_DIR/input.csv")
[[ "$actual_csv_lines" -eq $((expected_evals + 1)) ]] || \
    fail "input manifest has $actual_csv_lines lines; expected $((expected_evals + 1))"

slurm_options=$("$EVAL_PYTHON" - <<'PY'
import json
import os

print(json.dumps({
    "ACCOUNT": os.environ["OELLM_ACCOUNT"],
    "PARTITION": "booster",
    "NODES": 1,
    "CPUS_PER_TASK": int(os.environ["OELLM_CPUS_PER_TASK"]),
    "THREADS_PER_CORE": 1,
    "SLURM_MEM": os.environ["OELLM_SLURM_MEM"],
    "TIME": os.environ["OELLM_EVAL_TIME"],
    "SINGULARITY_ARGS": (
        "--nv --cleanenv --containall --no-mount bind-paths,hostfs,cwd,home "
        "--env HF_HUB_CACHE=" + os.environ["HF_HUB_CACHE"] + " "
        "--env HF_DATASETS_OFFLINE=1 --env HF_ALLOW_CODE_EVAL=1 "
        "--env HF_EVALUATE_OFFLINE=1 --env RAY_USAGE_STATS_ENABLED=0 "
        "--env VLLM_USE_V2_MODEL_RUNNER=1 --env OPENBLAS_NUM_THREADS=1 "
        "--env OMP_NUM_THREADS=4"
    ),
}))
PY
)

run_oellm_eval schedule \
    --eval_csv_path "$RUN_DIR/input.csv" \
    --model_backend vllm \
    --data_parallel_backend mp \
    --data_parallel_size "$EVAL_DP" \
    --tensor_parallel_size "$EVAL_TP" \
    --model_args dtype=bfloat16,gpu_memory_utilization=0.9,max_num_seqs=32 \
    --log_samples true \
    --confirm_run_unsafe_code true \
    --max_array_len "$MAX_ARRAY_LEN" \
    --slurm_template_var "$slurm_options" \
    --skip_checks true \
    --dry_run true 2>&1 | tee "$RUN_DIR/render.log"

mapfile -t batch_scripts < <(find "$RUN_DIR" -type f -name submit_evals.sbatch)
[[ ${#batch_scripts[@]} -eq 1 ]] || fail "expected one generated launcher, found ${#batch_scripts[@]}"
BATCH_SCRIPT="${batch_scripts[0]}"
bash -n "$BATCH_SCRIPT"

grep -Fq "#SBATCH --gres=gpu:$((EVAL_DP * EVAL_TP))" "$BATCH_SCRIPT" || \
    fail "launcher does not request $((EVAL_DP * EVAL_TP)) GPU(s)"
grep -Fq '#SBATCH --nodes=1' "$BATCH_SCRIPT" || fail "launcher does not request one node"
grep -Fq "#SBATCH --cpus-per-task=$CPUS_PER_TASK" "$BATCH_SCRIPT" || \
    fail "launcher does not request $CPUS_PER_TASK CPUs"
grep -Fq '#SBATCH --threads-per-core=1' "$BATCH_SCRIPT" || fail "launcher does not request one thread per core"
grep -Fq 'MODEL_BACKEND="vllm"' "$BATCH_SCRIPT" || fail "launcher is not using the vLLM backend"
grep -Fq "data_parallel_size=$EVAL_DP" "$BATCH_SCRIPT" || fail "launcher is not using DP$EVAL_DP"
grep -Fq "tensor_parallel_size=$EVAL_TP" "$BATCH_SCRIPT" || fail "launcher is not using TP$EVAL_TP"
grep -Fq "$EVAL_IMAGE" "$BATCH_SCRIPT" || fail "launcher does not use the selected complete image"
grep -Fq 'LIMIT=""' "$BATCH_SCRIPT" || fail "launcher contains an example limit"
expected_workers=$QUEUE_LIMIT
if (( expected_workers > expected_evals )); then
    expected_workers=$expected_evals
fi
grep -Fq "#SBATCH --array=0-$((expected_workers - 1))%$MAX_ARRAY_LEN" "$BATCH_SCRIPT" || \
    fail "launcher array does not match $expected_workers workers with concurrency $MAX_ARRAY_LEN"

sha256sum "$BATCH_SCRIPT" "$RUN_DIR/input.csv" > "$RUN_DIR/launch.sha256"
{
    echo "oellm_eval_revision=$actual_eval_rev"
    echo "image=$EVAL_IMAGE"
    echo "expected_image_sha256=$EXPECTED_IMAGE_SHA256"
    echo "models=${MODELS[*]}"
    echo "prompt_views=${PROMPT_VIEWS[*]}"
    echo "tasks=$TASKS (n-shot and suite per task in input.csv)"
    echo "backend=vllm,dp=4,tp=1,data_parallel_backend=mp"
    echo "array_workers=$QUEUE_LIMIT,array_concurrency=$MAX_ARRAY_LEN"
} > "$RUN_DIR/provenance.txt"

echo
echo "Prepared without submitting:"
echo "  run directory: $RUN_DIR"
echo "  launcher:      $BATCH_SCRIPT"
echo
echo "Inspect, test, and submit explicitly:"
printf '  sha256sum --check %q\n' "$RUN_DIR/launch.sha256"
printf '  sed -n '\''1,80p'\'' %q\n' "$BATCH_SCRIPT"
printf '  sbatch --test-only %q\n' "$BATCH_SCRIPT"
printf '  sbatch --parsable %q\n' "$BATCH_SCRIPT"
