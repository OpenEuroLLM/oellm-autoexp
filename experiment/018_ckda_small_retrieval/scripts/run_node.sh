#!/usr/bin/env bash
set -euo pipefail
export CKDA_WORK=${CKDA_WORK:-/gpfs/projects/ehpc533/users/korbip/ckda_retrieval_018}
R=/gpfs/projects/ehpc533/users/korbip/hybrid_v019
export PYTHONPATH="$CKDA_WORK/source:$CKDA_WORK/source/lm_scaling:$CKDA_WORK/scripts:$R/pylibs_flag:$R/pylibs_hfx:$R/pylibs_ep"
export SKDA_PYLIBS="$R/pylibs_flag"
export HF_HOME=/gpfs/projects/ehpc533/users/korbip/data/flag_evals/hf_home
export HF_HUB_CACHE="$HF_HOME/hub" HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_METRICS_CACHE="$CKDA_WORK/cache/metrics"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_EVALUATE_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export NLTK_DATA="$R/nltk_data"
export TRITON_CACHE_DIR="/tmp/ckda018_triton_${SLURM_JOB_ID:-local}"
mkdir -p "$CKDA_WORK/results" "$HF_METRICS_CACHE"
case "${1:-probe}" in
  probe) python "$CKDA_WORK/scripts/probe.py" ;;
  baseline)
    pids=()
    for index in 0 1 2 3; do
      case "$index" in
        0) arm=ckda; model=skda-shipped-hybrid-lowrank; suite=focus ;;
        1) arm=kda; model=kda-sig-hybrid-lowrank; suite=focus ;;
        2) arm=ckda; model=skda-shipped-hybrid-lowrank; suite=recall ;;
        3) arm=kda; model=kda-sig-hybrid-lowrank; suite=recall ;;
      esac
      CUDA_VISIBLE_DEVICES=$index python "$CKDA_WORK/scripts/evaluate.py" \
        --source "$CKDA_WORK/source" \
        --model "$CKDA_WORK/assets/exports/${model}_983M_gbs256_decay80B" \
        --suite "$suite" --limit 256 --batch-size 4 \
        --out "$CKDA_WORK/results/baseline/$arm/$suite" \
        > "$CKDA_WORK/results/baseline_${arm}_${suite}.log" 2>&1 &
      pids+=("$!")
    done
    failed=0
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    [[ $failed == 0 ]]
    for suite in focus recall; do
      python "$CKDA_WORK/scripts/compare.py" \
        "$CKDA_WORK/results/baseline/ckda/$suite" \
        "$CKDA_WORK/results/baseline/kda/$suite" \
        --out "$CKDA_WORK/results/comparison_$suite.json"
    done
    ;;
  *) echo "unknown mode: $1" >&2; exit 2 ;;
esac
