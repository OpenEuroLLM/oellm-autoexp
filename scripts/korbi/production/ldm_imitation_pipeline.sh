#!/bin/bash
# mHC-4 -> LDM imitation of one HF export on one node, every stage skipped when its output exists:
#   1. mhc_coeff_dump.py       per-token coefficients / streams (sampled tokens) -> <out>/layer_*.npz
#   2. mhc_imitation_fit.py    candidates vs mHC's layer inputs, normal equations, per-token arrays
#   3. ldm_imitation_init.py   fixed-address LDM fits (K:<lambda_min>:<key noise> runs) -> <out>/ldm_init.json
#   4. ldm_imitation_eval.py   the step-0 loss of mHC, collapse and the imitated variants -> <out>/eval.json
#   bash ldm_imitation_pipeline.sh <export dir> <out dir> <runs> <variants>
set -euo pipefail
cd "$(dirname "$0")/../../.."
EXPORT=$1
OUT=$2
RUNS=${3:-K8:0:0,K8:0.9:0,K8:0.9:0.02}
VARIANTS=${4:-mhc,collapse}
P=scripts/korbi/production
[ -f "$OUT/stats.json" ] || python3 $P/mhc_coeff_dump.py --model-dir "$EXPORT" --out "$OUT" --docs 16 --seq-len 4096 --samples 32
[ -f "$OUT/tokens.npz" ] || python3 $P/mhc_imitation_fit.py --dump "$OUT" --out "$OUT/fit.json" --device cuda \
    --gram "$OUT/gram.npz" --tokens "$OUT/tokens.npz"
[ -f "$OUT/ldm_init.json" ] || python3 $P/ldm_imitation_init.py --gram "$OUT/gram.npz" --tokens "$OUT/tokens.npz" \
    --runs "$RUNS" --out "$OUT/ldm_init.json"
python3 $P/ldm_imitation_eval.py --model-dir "$EXPORT" --init "$OUT/ldm_init.json" --out "$OUT/eval.json" \
    --variants "$VARIANTS"
