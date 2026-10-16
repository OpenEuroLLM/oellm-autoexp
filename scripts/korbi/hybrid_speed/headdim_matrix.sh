#!/bin/bash
# Head-dim capability / speed matrix of the token mixers on a 1536-wide model (H100):
# head dims 64 / 128 / 256 / 512 (24 / 12 / 6 / 3 heads), three mixer layers + one
# attention layer, seq 4096, mbs 2. Prints one RESULT line per (mixer, head dim), or the
# first error line when the kernels do not support it.
#   bash scripts/korbi/hybrid_speed/headdim_matrix.sh
set -u
H=1536
for d in 64 128 256 512; do
  n=$((H / d))
  common="\"linear_key_head_dim\": $d, \"linear_value_head_dim\": $d, \"linear_num_key_heads\": $n, \"linear_num_value_heads\": $n"
  for v in gated_delta_net kda complex_kda mlstm power; do
    cfg="{\"experimental_attention_variant\": \"$v\", $common}"
    out=$(python scripts/korbi/hybrid_speed/bench_block.py --pattern 'L-L-L-*-' --hidden $H \
          --heads $n --ffn 4096 --mbs 2 --iters 8 --name "${v}_hd$d" --config "$cfg" 2>&1)
    echo "$out" | grep -a "^RESULT" || echo "FAIL ${v}_hd$d: $(echo "$out" | grep -aE "Error|error:|assert" | grep -v Warning | tail -1 | cut -c1-200)"
  done
  for b in gdn kda; do
    cfg="{\"experimental_attention_variant\": \"triadic\", \"linear_triadic_base\": \"$b\", \"linear_triadic_e\": 4, $common}"
    out=$(python scripts/korbi/hybrid_speed/bench_block.py --pattern 'L-L-L-*-' --hidden $H \
          --heads $n --ffn 4096 --mbs 2 --iters 8 --name "tri_${b}_hd$d" --config "$cfg" 2>&1)
    echo "$out" | grep -a "^RESULT" || echo "FAIL tri_${b}_hd$d: $(echo "$out" | grep -aE "Error|error:|assert" | grep -v Warning | tail -1 | cut -c1-200)"
  done
  # softmax attention with this head dim (cuDNN / flash)
  out=$(python scripts/korbi/hybrid_speed/bench_block.py --pattern '*-*-*-*-' --hidden $H \
        --heads $n --ffn 4096 --mbs 2 --iters 8 --name "attn_hd$d" 2>&1)
  echo "$out" | grep -a "^RESULT" || echo "FAIL attn_hd$d: $(echo "$out" | grep -aE "Error|error:|assert" | grep -v Warning | tail -1 | cut -c1-200)"
done
