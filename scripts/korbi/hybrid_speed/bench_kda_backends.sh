#!/bin/bash
# FLA's KDA backends (FLA_KDA_BACKEND: auto picks tilelang when installed) on the KDA /
# Complex KDA stacks:  bash scripts/korbi/hybrid_speed/bench_kda_backends.sh
set -u
export FLA_DISABLE_BACKEND_DISPATCH=0
for v in kda complex_kda; do
    for be in auto triton tilelang triton_optimized hybrid_optimized; do
        FLA_KDA_BACKEND=$be python scripts/korbi/hybrid_speed/bench_block.py --pattern L-L-L-*- \
            --config "{\"experimental_attention_variant\": \"$v\"}" --name "${v}_$be" \
            || echo "FAIL ${v}_$be"
    done
done
