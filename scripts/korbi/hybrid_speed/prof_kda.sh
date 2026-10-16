#!/bin/bash
# Profiled KDA / Complex KDA / GDN stacks (L-L-L-*-) with source locations, to attribute
# the glue around the chunk kernels:  bash scripts/korbi/hybrid_speed/prof_kda.sh
set -u
for v in kda complex_kda gated_delta_net; do
    python scripts/korbi/hybrid_speed/bench_block.py --pattern L-L-L-*- \
        --config "{\"experimental_attention_variant\": \"$v\"}" --name "${v}_prof" --profile --stack 8
done
