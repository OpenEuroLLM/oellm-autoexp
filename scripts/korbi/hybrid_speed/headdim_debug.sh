#!/bin/bash
# Full tracebacks of the head-dim matrix failures (see headdim_matrix.sh).
set -u
H=1536
run() {  # name head_dim variant [extra json]
    local name=$1 d=$2 v=$3 extra=${4:-}
    local n=$((H / d))
    echo "################ $name"
    python scripts/korbi/hybrid_speed/bench_block.py --pattern 'L-*-' --hidden $H --heads $n --ffn 4096 \
        --mbs 1 --iters 2 --warmup 1 --name "$name" \
        --config "{\"experimental_attention_variant\": \"$v\", \"linear_key_head_dim\": $d, \"linear_value_head_dim\": $d, \"linear_num_key_heads\": $n, \"linear_num_value_heads\": $n $extra}" 2>&1 \
        | grep -v "^\[rank[1-9]" | grep -aE "Error|error|^\s+File|at [0-9]+:[0-9]+|^RESULT|^\s{4}\S" | grep -v Warning | tail -30
}
run tri_kda_hd128 128 triadic ', "linear_triadic_base": "kda", "linear_triadic_e": 4'
run complex_kda_hd256 256 complex_kda
run kda_hd512 512 kda
run power_hd256 256 power
