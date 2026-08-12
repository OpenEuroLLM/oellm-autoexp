"""Sanity-check a converted HF checkpoint before it is fed to oellm-eval.

Three checks, cheapest first:

1. **Strict load** — every tensor in ``model.safetensors`` is consumed by the
   module tree and every parameter is filled. Catches renamed/missing/leftover
   keys and every shape-level layout mistake.
2. **Forward** — mean NLL on a short multilingual sample. A correctly converted
   base model of this size lands around 2–4 nats/token; a layout error (swapped
   gate/up, transposed QKV, wrong fused-section order) shows up as ~12 nats,
   i.e. ``log(vocab_size)``.
3. **Reference comparison** (optional) — max abs logit difference against a
   ``.pt`` dump of Megatron logits for the same token ids. This is the only
   check that proves numerical equivalence; produce the dump on the Megatron
   side for the same ``input_ids``.

Usage::

    python -m oellm_autoexp.hf_export.validate_hf_export --model-dir <hf_dir>
    python -m oellm_autoexp.hf_export.validate_hf_export --model-dir <hf_dir> \\
        --reference megatron_logits.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# Short multilingual probe (EU languages present in the training blend).
PROBE_TEXT = (
    "The European Union is a political and economic union of 27 member states. "
    "Die Europäische Union ist ein Staatenverbund von 27 Mitgliedstaaten. "
    "L'Union européenne est une union politique et économique de 27 États membres. "
    "La Unión Europea es una unión política y económica de 27 Estados miembros. "
    "Unia Europejska jest związkiem politycznym i gospodarczym 27 państw członkowskich."
)

RANDOM_BASELINE_MARGIN = 1.0  # nats below log(V) that still counts as broken


def check_strict_load(model_dir: Path, device: str, dtype: torch.dtype):
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, trust_remote_code=True, dtype=dtype, device_map=None
    )
    model.to(device).eval()

    from safetensors import safe_open

    with safe_open(str(model_dir / "model.safetensors"), framework="pt") as f:
        file_keys = set(f.keys())
    param_keys = {k for k, _ in model.named_parameters()}
    if cfg.tie_word_embeddings:
        param_keys.discard("lm_head.weight")

    unexpected = sorted(file_keys - param_keys)
    missing = sorted(param_keys - file_keys)
    return model, cfg, unexpected, missing


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--reference", type=Path, default=None)
    p.add_argument("--device", default="cuda")
    # bf16 forwards never agree bit-for-bit, so gate on distribution-level
    # agreement rather than raw logit deltas.
    # Calibrated across all five 0.1B/20BT variants on 1024 real tokens each.
    # Every variant sits at the same bf16 noise floor — mean KL 4.9-5.2e-4,
    # top-1 agreement 98.4-99.5%, |dNLL| 0.0-0.0027 — which is what says the
    # port is faithful: the floor is identical whether the layer is attention,
    # GDN, mLSTM or Mamba2. |dNLL| is the noisiest of the three at this sample
    # size, so it gets the loosest gate; even so, 0.01 is 4.5x below the
    # 0.045-nat spread between the architectures being compared, and a real
    # conversion bug misses by orders of magnitude (the RoPE-as-identity bug put
    # NLL at 5.8 vs 3.0).
    p.add_argument("--max-kl", type=float, default=2e-3,
                   help="max mean KL(megatron || hf) over all positions")
    p.add_argument("--min-agree", type=float, default=0.98,
                   help="min fraction of positions whose argmax token matches")
    p.add_argument("--max-nll-delta", type=float, default=1e-2,
                   help="max |NLL_megatron - NLL_hf| on the reference tokens")
    args = p.parse_args()

    from transformers import AutoTokenizer

    dtype = torch.bfloat16
    model, cfg, unexpected, missing = check_strict_load(args.model_dir, args.device, dtype)

    ok = True
    print(f"model_dir            : {args.model_dir}")
    print(f"mixer_types          : {json.dumps(cfg.mixer_types)}")
    print(f"params               : {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    if unexpected or missing:
        ok = False
        print(f"FAIL unexpected keys : {unexpected[:8]}")
        print(f"FAIL missing keys    : {missing[:8]}")
    else:
        print("OK   strict load     : all tensors matched")

    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    ids = tok(PROBE_TEXT, return_tensors="pt").input_ids.to(args.device)
    with torch.no_grad():
        out = model(input_ids=ids)
    logits = out.logits.float()
    nll = torch.nn.functional.cross_entropy(
        logits[0, :-1], ids[0, 1:], reduction="mean"
    ).item()
    random_nll = float(torch.log(torch.tensor(float(cfg.vocab_size))))
    print(f"tokens               : {ids.shape[1]}")
    print(f"mean NLL             : {nll:.4f}  (random baseline {random_nll:.4f})")
    if not (nll < random_nll - RANDOM_BASELINE_MARGIN):
        ok = False
        print("FAIL forward         : NLL is at the random baseline -> conversion is wrong")
    else:
        print("OK   forward         : NLL well below random")

    if args.reference is not None:
        ref = torch.load(args.reference, map_location="cpu")
        ref_ids = ref["input_ids"].to(args.device)
        ref_logits = ref["logits"].float()
        with torch.no_grad():
            mine = model(input_ids=ref_ids).logits.float().cpu()
        if mine.shape != ref_logits.shape:
            ok = False
            print(f"FAIL reference       : shape {tuple(mine.shape)} vs "
                  f"{tuple(ref_logits.shape)}")
        else:
            delta = (mine - ref_logits).abs()
            # Raw logit deltas are dominated by the bf16 forward, so the
            # decision is made on the *distributions* and on next-token
            # agreement — those are what any eval actually consumes.
            p_ref = torch.log_softmax(ref_logits, dim=-1)
            p_mine = torch.log_softmax(mine, dim=-1)
            kl = torch.nn.functional.kl_div(
                p_mine, p_ref, log_target=True, reduction="none"
            ).sum(-1)
            agree = (mine.argmax(-1) == ref_logits.argmax(-1)).float().mean().item()
            nll_ref = torch.nn.functional.cross_entropy(
                ref_logits[:, :-1].reshape(-1, ref_logits.size(-1)),
                ref_ids.cpu()[:, 1:].reshape(-1),
            ).item()
            nll_mine = torch.nn.functional.cross_entropy(
                mine[:, :-1].reshape(-1, mine.size(-1)),
                ref_ids.cpu()[:, 1:].reshape(-1),
            ).item()
            print(f"tokens compared      : {ref_logits.shape[0] * ref_logits.shape[1]}")
            print(f"max |Δlogit|         : {delta.max().item():.5f}")
            print(f"mean |Δlogit|        : {delta.mean().item():.6f}")
            print(f"mean KL(mega‖hf)     : {kl.mean().item():.3e}   max {kl.max().item():.3e}")
            print(f"top-1 agreement      : {agree * 100:.3f}%")
            print(f"NLL megatron / hf    : {nll_ref:.5f} / {nll_mine:.5f} "
                  f"(Δ {abs(nll_ref - nll_mine):.5f})")
            passed = (
                kl.mean().item() < args.max_kl
                and agree >= args.min_agree
                and abs(nll_ref - nll_mine) < args.max_nll_delta
            )
            if not passed:
                ok = False
                print("FAIL reference       : HF forward diverges from Megatron")
            else:
                print("OK   reference       : HF forward matches Megatron")
    else:
        print("SKIP reference       : no --reference dump given")

    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
