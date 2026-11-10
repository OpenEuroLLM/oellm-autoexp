#!/usr/bin/env bash
set -euo pipefail
export CKDA_WORK=${CKDA_WORK:-/gpfs/projects/ehpc533/users/korbip/ckda_retrieval_018}
R=/gpfs/projects/ehpc533/users/korbip/hybrid_v019
export PYTHONPATH="$CKDA_WORK/source:$CKDA_WORK/source/lm_scaling:$CKDA_WORK/scripts:$R/pylibs_flag:$R/pylibs_hfx:$R/pylibs_ep"
export SKDA_PYLIBS="$R/pylibs_flag"
export HF_HOME=/gpfs/projects/ehpc533/users/korbip/data/flag_evals/hf_home
export HF_HUB_CACHE="$HF_HOME/hub" HF_DATASETS_CACHE="$CKDA_WORK/cache/datasets"
export HF_METRICS_CACHE="$CKDA_WORK/cache/metrics"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_EVALUATE_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export NLTK_DATA="$R/nltk_data"
export TRITON_CACHE_DIR="/tmp/ckda018_triton_${SLURM_JOB_ID:-local}"
mkdir -p "$CKDA_WORK/results" "$HF_METRICS_CACHE"
mkdir -p "$HF_DATASETS_CACHE"
for dataset in "$CKDA_WORK/cache/recall_datasets"/*; do
  [[ -d "$dataset" ]] || continue
  link="$HF_DATASETS_CACHE/$(basename "$dataset")"
  if [[ ! -e "$link" && ! -L "$link" ]]; then ln -s "$dataset" "$link"; fi
done
case "${1:-probe}" in
  probe) python "$CKDA_WORK/scripts/probe.py" ;;
  baseline)
    # Export only from native Megatron weights; no HF-to-Megatron conversion.
    for arm in ckda kda; do
      if [[ $arm == ckda ]]; then model=skda-shipped-hybrid-lowrank; else model=kda-sig-hybrid-lowrank; fi
      export_dir="$CKDA_WORK/assets/exports/${model}_983M_gbs256_decay80B"
      if [[ ! -f "$export_dir/.native_export_verified" ]]; then
        python "$CKDA_WORK/source/lm_scaling/megatron_ext/export_megatron.py" \
          --run "$CKDA_WORK/assets/native/$arm/iter_0076294/mp_rank_00/model_optim_rng.pt" \
          --arm "$model" --geometry 983M --out "$export_dir" \
          --tokenizer EleutherAI/gpt-neox-20b
        touch "$export_dir/.native_export_verified"
      fi
    done
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
        --tokenizer "$CKDA_WORK/assets/tokenizer" \
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
  diagnose)
    pids=()
    for index in 0 1; do
      if [[ $index == 0 ]]; then arm=ckda; model=skda-shipped-hybrid-lowrank; else arm=kda; model=kda-sig-hybrid-lowrank; fi
      CUDA_VISIBLE_DEVICES=$index python "$CKDA_WORK/source/lm_scaling/recall_diagnose.py" \
        --model "$CKDA_WORK/assets/exports/${model}_983M_gbs256_decay80B" \
        --tokenizer "$CKDA_WORK/assets/tokenizer" --arm "$arm" --docs 64 \
        --max-length 4096 --context-prefix none \
        --out "$CKDA_WORK/results/diagnosis_$arm.json" \
        > "$CKDA_WORK/results/diagnosis_$arm.log" 2>&1 &
      pids+=("$!")
    done
    failed=0
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    [[ $failed == 0 ]]
    ;;
  screen)
    # Four independent workers; each processes a fixed subset of the 16 cells.
    screen_worker() {
      local gpu=$1 index=0
      for arm in ckda kda; do
        if [[ $arm == ckda ]]; then model=skda-shipped-hybrid-lowrank; else model=kda-sig-hybrid-lowrank; fi
        for condition in baseline no_neg_alpha no_reflection both alpha_gain_0p5 alpha_gain_2p0 window_512 window_1024; do
          if [[ $((index % 4)) == "$gpu" ]]; then
            CUDA_VISIBLE_DEVICES=$gpu python "$CKDA_WORK/scripts/evaluate.py" \
              --source "$CKDA_WORK/source" \
              --model "$CKDA_WORK/assets/exports/${model}_983M_gbs256_decay80B" \
              --tokenizer "$CKDA_WORK/assets/tokenizer" --suite recall --limit 64 \
              --batch-size 4 --condition "$condition" \
              --out "$CKDA_WORK/results/screen/$arm/$condition" \
              > "$CKDA_WORK/results/screen_${arm}_${condition}.log" 2>&1 || return
          fi
          index=$((index + 1))
        done
      done
    }
    pids=()
    for gpu in 0 1 2 3; do screen_worker "$gpu" & pids+=("$!"); done
    failed=0
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    [[ $failed == 0 ]]
    ;;
  *) echo "unknown mode: $1" >&2; exit 2 ;;
esac
