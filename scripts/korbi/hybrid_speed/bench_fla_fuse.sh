#!/bin/bash
# FLA layers with and without fused projections (FLA_FUSE_PROJECTIONS), one stack each,
# then a profiled complex_kda stack:  bash scripts/korbi/hybrid_speed/bench_fla_fuse.sh
set -u
for v in kda complex_kda gated_delta_net; do
    for f in 0 1; do
        FLA_FUSE_PROJECTIONS=$f python scripts/korbi/hybrid_speed/bench_block.py --pattern L-L-L-*- \
            --config "{\"experimental_attention_variant\": \"$v\"}" --name "${v}_fuse$f"
    done
done
python scripts/korbi/hybrid_speed/bench_block.py --pattern L-L-L-*- \
    --config '{"experimental_attention_variant": "complex_kda"}' --name complex_kda_prof --profile --stack 6
