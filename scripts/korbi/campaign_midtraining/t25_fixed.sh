#!/bin/bash
# t25c: the constant-size per-token layout (TOKEN_FIXED_PAD 8192) vs the bucketed one (0) over the real cross-document
# layouts, one process each, smooth and scattered inputs, plus dense (token_layer_seq_bench.py, 40 layouts, 2 passes).
# Usage (from submodules/Megatron-LM): bash ../../scripts/korbi/campaign_midtraining/t25_fixed.sh <spans.json> <out dir>
spans=$1 out=$2
mkdir -p "$out"
B=../../scripts/korbi/campaign_midtraining/token_layer_seq_bench.py
for pad in 8192 0; do
    for rough in "" "--rough"; do
        python "$B" --spans "$spans" --variant tok --fixed-pad $pad $rough --out "$out/pad${pad}${rough:+_rough}.json" \
            || echo "pad $pad $rough failed"
    done
done
python "$B" --spans "$spans" --variant dense --out "$out/dense.json" || echo "dense failed"
