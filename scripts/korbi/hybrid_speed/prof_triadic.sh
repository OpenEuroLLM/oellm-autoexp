#!/bin/bash
# Stack-attributed profiles of the triadic mixers (6 heads of 128, E = 4) at the speed-suite
# geometry:  bash scripts/korbi/hybrid_speed/prof_triadic.sh
set -u
for base in gdn kda; do
    python scripts/korbi/hybrid_speed/bench_block.py --pattern L-L-L-*- --name "tri_${base}_prof" \
        --config "{\"experimental_attention_variant\": \"triadic\", \"linear_key_head_dim\": 128, \"linear_value_head_dim\": 128, \"linear_num_key_heads\": 6, \"linear_num_value_heads\": 6, \"linear_triadic_base\": \"$base\", \"linear_triadic_e\": 4}" \
        --profile --stack 5
done
