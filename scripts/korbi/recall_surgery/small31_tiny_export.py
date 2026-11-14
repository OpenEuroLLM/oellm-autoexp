#!/usr/bin/env python3
"""A tiny random oellm_ambitious HF export with the small31 block layout (attention / dense MLP / CKDA / Engram-free,
production tokenizer) to test the Phase-1 readout pipeline (small31_eval.py, small31_readout.sh) before a real
checkpoint exists (docs/recall_surgery_ambitious.md §14). The model code is the exporter's (oellm_ambitious +
oellm_hybrid files copied next to the weights, as export_ambitious does).

    small31_tiny_export.py --out <dir> --tokenizer <tokenizer dir>   (GPU: CKDA runs the FLA kernels)
"""

import argparse
import glob
import json
import os
import shutil
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "../../.."))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", required=True)
    a = ap.parse_args()
    pkg = os.path.join(a.out, "_pkg", "oellm_ambitious")
    os.makedirs(pkg, exist_ok=True)
    src = [*glob.glob(f"{REPO}/oellm_autoexp/hf_export/oellm_ambitious/*.py"),
           *glob.glob(f"{REPO}/oellm_autoexp/hf_export/oellm_hybrid/*.py")]
    for f in src:
        shutil.copy(f, pkg)
    sys.path.insert(0, os.path.dirname(pkg))
    from oellm_ambitious.configuration_oellm_ambitious import OellmAmbitiousConfig
    from oellm_ambitious.modeling_oellm_ambitious import OellmAmbitiousForCausalLM

    fla = {"class": "fla.layers.complex_kda_layer.ComplexKimiDeltaAttention",
           "kwargs": {"hidden_size": 256, "num_heads": 4, "head_dim": 64, "allow_neg_eigval": True,
                      "gate": "signed_sigmoid2", "gate_init_style": "shipped", "lower_bound": -5.0,
                      "output_gate": "lowrank", "drop_silu": True, "conv_size": 4, "norm_eps": 1e-05}}
    cfg = OellmAmbitiousConfig(
        vocab_size=262144, hidden_size=256, rms_norm_eps=1e-5, residual="mhc", mhc_expansion_rate=4,
        block_types=["attention", "mlp", "ckda", "mlp", "ckda", "mlp", "attention", "mlp"],
        num_attention_heads=4, num_key_value_heads=2, head_dim=64, qk_layernorm=True, attention_output_gate=True,
        exclusive_self_attention=True, fla_layer=fla, num_experts=0, num_experts_per_tok=0, moe_intermediate_size=0,
        shared_expert_intermediate_size=0, mlp_intermediate_size=512, activation_clamp_value=10.0,
        mlp_post_act_norm=True, post_act_norm_floor=1.0, engram_table_sizes=None, tie_word_embeddings=False,
        max_position_embeddings=100000000, bos_token_id=1, eos_token_id=2, pad_token_id=2,
        auto_map={"AutoConfig": "configuration_oellm_ambitious.OellmAmbitiousConfig",
                  "AutoModelForCausalLM": "modeling_oellm_ambitious.OellmAmbitiousForCausalLM"})
    torch.manual_seed(0)
    model = OellmAmbitiousForCausalLM(cfg).to(torch.bfloat16)
    for _, p in model.named_parameters():
        if p.dim() >= 2:
            torch.nn.init.normal_(p, std=0.02)
    model.save_pretrained(a.out)
    for f in src:
        shutil.copy(f, a.out)
    for f in os.listdir(a.tokenizer):
        if os.path.isfile(os.path.join(a.tokenizer, f)):
            shutil.copy(os.path.join(a.tokenizer, f), a.out)
    shutil.rmtree(os.path.join(a.out, "_pkg"))
    with open(os.path.join(a.out, ".export_done"), "w") as f:
        f.write("tiny test export\n")
    print(json.dumps({"out": a.out, "params": sum(p.numel() for p in model.parameters())}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
