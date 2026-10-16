#!/bin/bash
# One 3:1 block per mixer with each FLA kernel backend (env-selected), and the GDN fused
# output norm on / off: fwd+bwd ms of the stack (RESULT lines), on one GPU.
cd "$(dirname "$0")/../../.."
B="python scripts/korbi/hybrid_speed/bench_block.py --seq 4096 --mbs 8 --iters 10 --warmup 3"
run() { echo "## $*"; "$@" 2>&1 | grep RESULT; }
for v in kda complex_kda; do
  for be in default triton tilelang triton_optimized hybrid_optimized; do
    if [ $be = default ]; then e=(); else e=(FLA_KDA_BACKEND=$be); fi
    run env "${e[@]}" $B --name "${v}_$be" --pattern L-L-L-*- --config "{\"experimental_attention_variant\":\"$v\"}"
  done
done
run $B --name gdn_native --pattern G-G-G-*- --config '{}'
run $B --name gdn_native_fusednorm --pattern G-G-G-*- --config '{"gdn_fused_out_norm":true}'
run env FLA_TILELANG=0 $B --name gdn_native_notilelang --pattern G-G-G-*- --config '{}'
