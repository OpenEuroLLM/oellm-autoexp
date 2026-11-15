#!/bin/bash
# Check of the additive kvshare export (block type "ckda_kvshare", §14.8 arm K) on one node, inside the training
# image: (1) export K@38146 (gamma 0) -> its v8ml-valid loss must equal b7@38146's (exports/small31_b7_38146: the
# same model plus a zero-weight branch); (2) with the branch's o_proj scaled up (synthetic gamma), the prefill and the
# cached step-by-step decode of a short sequence must agree, and the branch must change the logits.
#   ARM=K STEP=38146 k_export_check.sh   (a 1-node pool job: small31_readout_jupiter with backend.command overridden)
set -uo pipefail
: "${ARM:=K}" "${STEP:=38146}"
ROOT=${ROOT:-/e/scratch/e-sta-openeurollm/poeppel1/recall_surgery/small31}
EXPORT=$ROOT/exports/small31_${ARM}_$STEP
OUT=$ROOT/evals_check
mkdir -p "$OUT"
cd "$(dirname "$0")/../../.."
if [ ! -f "$EXPORT/.export_done" ]; then
  EXPERT_DTYPE=bf16 HF_EXPORTER=sequential bash scripts/korbi/hf_export/export_run.sh "$ROOT/$ARM" "$STEP" "$EXPORT" \
    || { echo "[k_check] export failed"; exit 1; }
fi
grep -o '"block_types": \[[^]]*\]' "$EXPORT/config.json" | head -1
grep -o '"kv_share_sources": {[^}]*}' "$EXPORT/config.json" | head -1
E=scripts/korbi/recall_surgery/small31_eval.py
CUDA_VISIBLE_DEVICES=0 python "$E" --export "$EXPORT" --out "$OUT" --valid-loss --valid-docs 16 &
CUDA_VISIBLE_DEVICES=1 python "$E" --export "$ROOT/exports/small31_b7_38146" --out "$OUT" --valid-loss --valid-docs 16 &
wait
for d in "small31_${ARM}_$STEP" small31_b7_38146; do echo "[k_check] $d $(cat "$OUT/$d/valid_loss.json")"; done
CUDA_VISIBLE_DEVICES=2 python - "$EXPORT" <<'PY'
import sys, torch
from transformers import AutoModelForCausalLM
path = sys.argv[1]
model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
layers = [i for i, t in enumerate(model.config.block_types) if t == "ckda_kvshare"]
print("[k_check] ckda_kvshare layers", layers)
ids = torch.randint(10, 50000, (1, 48), generator=torch.Generator().manual_seed(0)).cuda()
with torch.no_grad():
    base = model(ids).logits.float()
    for i in layers:  # a synthetic gamma: the branch at a sizable weight
        w = model.model.layers[i].mixer.attn.o_proj.weight
        w.copy_(torch.randn_like(w) * 0.02)
    full = model(ids).logits.float()
    out = model(ids[:, :40], use_cache=True)
    past, steps = out.past_key_values, [out.logits[:, -1].float()]
    for t in range(40, 48):
        out = model(ids[:, t:t + 1], past_key_values=past, use_cache=True)
        past = out.past_key_values
        steps.append(out.logits[:, -1].float())
    dec = torch.stack(steps[1:], 1)
    print(f"[k_check] branch changes logits: max |d| {(full - base).abs().max().item():.3e}")
    print(f"[k_check] prefill vs decode (last 8 tokens): max |d| {(full[:, 40:48] - dec).abs().max().item():.3e} "
          f"(logit scale {full.abs().max().item():.1f})")
PY
