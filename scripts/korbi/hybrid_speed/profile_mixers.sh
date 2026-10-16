#!/bin/bash
# Stack-attributed profiles of one 3:1 block of each slow mixer (run on a GPU node from the
# repo root, e.g. via kernel_tests_mn5.yaml with aux.bench): which Python line launches the
# time the speed suite's kernel summaries only show as "elementwise".
#   scripts/korbi/hybrid_speed/profile_mixers.sh [case ...]
cd "$(dirname "$0")/../../.."
B="python scripts/korbi/hybrid_speed/bench_block.py --seq 4096 --mbs 8 --iters 5 --warmup 3 --profile --stack 4"
declare -A CASE=(
  [gate]='--pattern *-*- --config {"attention_output_gate":true}'
  [shortconv]='--pattern C*-C*- --config {}'
  [mamba2]='--pattern M-M-M-*- --config {"mamba_num_heads":24,"mamba_head_dim":64,"mamba_num_groups":8,"mamba_state_dim":128}'
  [gdn]='--pattern G-G-G-*- --config {}'
  [kda]='--pattern L-L-L-*- --config {"experimental_attention_variant":"kda"}'
  [ckda]='--pattern L-L-L-*- --config {"experimental_attention_variant":"complex_kda"}'
  [tri_gdn]='--pattern L-L-L-*- --config {"experimental_attention_variant":"triadic","linear_key_head_dim":128,"linear_value_head_dim":128,"linear_num_key_heads":6,"linear_num_value_heads":6,"linear_triadic_base":"gdn","linear_triadic_e":4,"linear_triadic_kernel":"cute"}'
  [tri_kda]='--pattern L-L-L-*- --config {"experimental_attention_variant":"triadic","linear_key_head_dim":128,"linear_value_head_dim":128,"linear_num_key_heads":6,"linear_num_value_heads":6,"linear_triadic_base":"kda","linear_triadic_e":4}'
  [power]='--pattern L-L-L-*- --mbs 4 --config {"experimental_attention_variant":"power","linear_power_chunk_size":256}'
  [mlstm]='--pattern L-L-L-*- --config {"experimental_attention_variant":"mlstm"}'
)
cases=("$@"); [ ${#cases[@]} -eq 0 ] && cases=(gate shortconv mamba2 gdn kda ckda tri_gdn tri_kda power mlstm)
for c in "${cases[@]}"; do
  echo "################ $c"
  read -r -a args <<< "${CASE[$c]}"
  $B --name "$c" "${args[@]}" 2>&1 | grep -v "Warning\|warn("
done
