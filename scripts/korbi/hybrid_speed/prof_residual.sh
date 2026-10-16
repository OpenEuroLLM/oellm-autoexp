#!/bin/bash
# Residual-stream schemes on the 130M dense geometry (12 blocks): fwd+bwd time, peak memory
# and the activations each saves for backward (per module), then a kernel profile each.
#   bash scripts/korbi/hybrid_speed/prof_residual.sh
set -u
P='*-*-*-*-*-*-*-*-*-*-*-*-'
declare -A CFG=(
  [residual]='{}'
  [mhc4]='{"mhc": true, "mhc_expansion_rate": 4}'
  [routed_k2]='{"mhc": true, "mhc_expansion_rate": 4, "mhc_routed_topk": 2}'
  [routed_k2_mix]='{"mhc": true, "mhc_expansion_rate": 4, "mhc_routed_topk": 2, "mhc_routed_mix": true, "mhc_routed_balance_rate": 0.001}'
  [routed_k2_rev]='{"mhc": true, "mhc_expansion_rate": 4, "mhc_routed_topk": 2, "mhc_routed_reversible": true}'
  [attn_res4]='{"attn_res": true, "attn_res_num_blocks": 4}'
  [attn_res8]='{"attn_res": true, "attn_res_num_blocks": 8}'
)
for name in residual mhc4 routed_k2 routed_k2_mix routed_k2_rev attn_res4 attn_res8; do
    python scripts/korbi/hybrid_speed/bench_block.py --pattern "$P" --name "$name" \
        --config "${CFG[$name]}" --saved --iters 10
done
for name in mhc4 routed_k2 routed_k2_mix attn_res4; do
    python scripts/korbi/hybrid_speed/bench_block.py --pattern "$P" --name "${name}_prof" \
        --config "${CFG[$name]}" --profile --iters 5 2>&1 | grep -v "^\[" | head -40
done
