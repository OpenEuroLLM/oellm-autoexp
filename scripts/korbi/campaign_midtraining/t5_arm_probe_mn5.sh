#!/bin/bash
# Retrieval / NLL of a trained step-5 arm (t5_s3_sparse_branch_mn5) with its own attention: HF export of the arm's
# step-1500 checkpoint, its trained sparse-attention parameters (extract_variant_params.py: NSA / CSA2), then
# long_attention_probe.py at 64k with the arm's policy (block-wise selection as trained; per-token selection
# in decode and on the prompt's last 128 tokens, as inference would). One MN5 node:
#
#   t5_arm_probe_mn5.sh <arm> "<probe policy>" [probe args...]
#   e.g. t5_arm_probe_mn5.sh nsa "nsa:topk=32,select_window=512,block=32,stride=16,select_block=64,decode=token"
set -euo pipefail
arm=$1; policy=$2; shift 2
R=${HYBRID_OUTPUT_ROOT:-/gpfs/scratch/ehpc533/hybrid_nucleus}
D=/gpfs/projects/ehpc533/users/korbip/data
run=$R/production/t5_s3_sparse/$arm
ckpt=$run/ckpt/iter_0001500
[ -d "$ckpt" ] || { echo "no step-1500 checkpoint under $run" >&2; exit 1; }
out=$R/hf_exports/t5_s3_${arm}_1500
probe_out=$R/campaign_midtraining/t5_probe/$arm
config=$(readlink -f "$(ls -t "$run"/current_*.yaml | head -1)")
tok=$(python - "$config" <<'PY'
import sys
from oellm_autoexp.hf_export.export_q35 import megatron_args
print(megatron_args(sys.argv[1])["tokenizer_model"])
PY
)
case $(basename "$tok") in
  tokenizer-256k-agent) valid=$D/v8ml-valid ;;
  tokenizer-256k)       valid=$D/v8ml-valid-basetok ;;
  *) echo "no validation set in the tokenization of $tok" >&2; exit 2 ;;
esac
echo "[t5_arm_probe] $arm: $ckpt ($config, $tok) -> $out; policy $policy"
if [ ! -f "$out/.export_done" ]; then
  rm -rf "$out"
  python -m oellm_autoexp.hf_export.export_ambitious --ckpt-dir "$ckpt" --config "$config" --out-dir "$out" \
    --tokenizer "$tok"
  touch "$out/.export_done"
fi
python -m oellm_autoexp.hf_export.refresh_export "$out"
[ -f "$out/.check_done" ] || { python -m oellm_autoexp.hf_export.check_export "$out" && touch "$out/.check_done"; }
variant_args=()
if python scripts/korbi/campaign_midtraining/extract_variant_params.py "$ckpt" "$out/variants.pt"; then
  variant_args=(--variant-params "$out/variants.pt")
fi
bash scripts/korbi/campaign_midtraining/run_hf_probe_mn5.sh long_attention_probe.py "$out" "$probe_out" \
  --docs "$valid/long_64k_text_document" --raw /gpfs/projects/ehpc533/users/korbip/hybrid_v019/data/downstream/raw.jsonl \
  --policies "$policy" --lengths 65536 --tasks nll,retrieval --n 16 --nll-docs 4 --chunk 16384 --parity 0 \
  "${variant_args[@]}" "$@"
