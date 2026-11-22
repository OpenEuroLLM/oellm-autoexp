#!/bin/bash
# t25: token_layer_seq_bench.py for per-token, per 16 queries, dense and per-token on scattered inputs (one process each).
# Usage (from submodules/Megatron-LM): bash ../../scripts/korbi/campaign_midtraining/t25_run.sh <spans.json> <out dir> [tag]
spans=$1 out=$2 tag=${3:-}
mkdir -p "$out"
B=../../scripts/korbi/campaign_midtraining/token_layer_seq_bench.py
for v in tok sb16 dense; do
    python "$B" --spans "$spans" --variant "$v" --out "$out/${tag}$v.json" || echo "variant $v failed"
done
python "$B" --spans "$spans" --variant tok --rough --out "$out/${tag}tok_rough.json" || echo "tok rough failed"
