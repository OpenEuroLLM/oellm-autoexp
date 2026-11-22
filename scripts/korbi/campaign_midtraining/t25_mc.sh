#!/bin/bash
# t25b: the dk/dv kernel's row-chunk size (MC; chunks without a listed query skipped) on H100 at v4b shapes, over the
# real cross-document layouts, smooth and scattered inputs (token_layer_seq_bench.py, 20 layouts, one pass each).
# Usage (from submodules/Megatron-LM): bash ../../scripts/korbi/campaign_midtraining/t25_mc.sh <spans.json> <out dir>
spans=$1 out=$2
mkdir -p "$out"
B=../../scripts/korbi/campaign_midtraining/token_layer_seq_bench.py
for mc in 128 64 32 16; do
    for rough in "" "--rough"; do
        ROWMASK_HP_CONFIG="dkv=8,1,$mc" python "$B" --spans "$spans" --variant tok --n 10 --long 10 --passes 1 $rough \
            --out "$out/mc${mc}${rough:+_rough}.json" || echo "mc $mc $rough failed"
    done
done
