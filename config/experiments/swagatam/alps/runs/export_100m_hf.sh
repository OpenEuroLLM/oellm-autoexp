#!/bin/bash
# Export every complete 100M checkpoint (two-bank and consolidated, every rho) and verify it.
# Invoked inside the training container by export_100m_hf.sbatch.
set -u

ROOT=/capstor/store/cscs/swissai/a0248/users/swagatam/oellm-autoexp
OUT=$ROOT/output
RESULTS=$ROOT/oellm_autoexp/hf_export/engram_scaling_clariden/results
cd "$ROOT"
export PATH=/opt/venv/bin:$PATH
export PYTHONPATH=.:submodules/Megatron-LM
export HF_HOME=${HF_HOME:-/iopsstor/scratch/cscs/$USER/hf_home}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
mkdir -p "$RESULTS"

parity_ok() {
  python - "$1" << 'PY'
import json, sys
p = sys.argv[1]
try:
    checks = json.load(open(p))["checks"]
except Exception:
    sys.exit(1)
sys.exit(0 if checks and all(c.get("ok") for c in checks) else 1)
PY
}

fail=0
n=0
for run in $(ls -d "$OUT"/engram_100m_1b_E*_clariden "$OUT"/engram_100m_1b_cons_E*_clariden | sort); do
  name=$(basename "$run")
  for ckpt in $(ls -d "$run"/ckpt/iter_* 2>/dev/null | sort); do
    [ -f "$ckpt/.metadata" ] || { echo "SKIP incomplete $ckpt"; continue; }
    iter=$((10#$(basename "$ckpt" | sed 's/iter_//')))
    export_dir=$run/hf/iter_$(printf '%07d' "$iter")
    tag=${name}_iter${iter}
    check_log=$RESULTS/check_${tag}.txt
    parity_json=$RESULTS/parity_${tag}.json
    n=$((n + 1))
    echo "======== $tag ($n) ========"
    if [ -f "$parity_json" ] && parity_ok "$parity_json" && [ -f "$check_log" ] && grep -q 'checks passed' "$check_log"; then
      echo "already verified: $tag"
      continue
    fi
    if ! python -m oellm_autoexp.hf_export.engram_scaling_clariden.export_hf \
        --run-dir "$run" --iter "$iter" --out-dir "$export_dir" --overwrite \
        > "$RESULTS/export_${tag}.log" 2>&1; then
      echo "EXPORT FAILED $tag"; tail -n 30 "$RESULTS/export_${tag}.log"; fail=1; continue
    fi
    if ! python -m oellm_autoexp.hf_export.engram_scaling_clariden.check_hf_export \
        "$export_dir" > "$check_log" 2>&1; then
      echo "CHECK FAILED $tag"; tail -n 20 "$check_log"; fail=1; continue
    fi
    if ! torchrun --standalone --nproc-per-node 1 \
        -m oellm_autoexp.hf_export.engram_scaling_clariden.megatron_parity \
        --export-dir "$export_dir" --out "$parity_json" \
        > "$RESULTS/parity_${tag}.log" 2>&1; then
      echo "PARITY FAILED $tag"; tail -n 30 "$RESULTS/parity_${tag}.log"; fail=1; continue
    fi
    echo "OK $tag"
  done
done
echo "done: $n checkpoints, fail=$fail"
exit "$fail"
