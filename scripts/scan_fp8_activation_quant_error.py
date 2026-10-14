#!/usr/bin/env python3
"""Measure saved-scale FP8 activation quantization error on a fixed batch.

The model runs a read-only BF16 forward so hooks can capture the inputs
to the four GEMMs that production executes in FP8.  Each input is then
quantized and dequantized with that layer's saved Transformer Engine
delayed forward scale. This isolates representation/scale quality
without updating parameters.
"""

import argparse
import csv
import io
import math
import os
import re
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp

from scan_attention_entropy import build_config, fixed_batch, load_weights, verify_load
from scan_fp8_amax import decode


FP8_MAX = 448.0
MODULES = (
    "self_attention.linear_qkv",
    "self_attention.linear_proj",
    "mlp.linear_fc1",
    "mlp.linear_fc2",
)


def saved_activation_scales(ckpt):
    reader = dcp.FileSystemReader(str(ckpt))
    meta = reader.read_metadata().state_dict_metadata
    pat = re.compile(
        r"decoder\.layers\.(?P<module>self_attention\.linear_(?:qkv|proj)|"
        r"mlp\.linear_fc[12])\._extra_state/shard_(?P<layer>\d+)_\d+$"
    )
    pairs = [(key, pat.match(key)) for key in meta]
    pairs = [(key, match) for key, match in pairs if match]
    state = {key: io.BytesIO() for key, _ in pairs}
    dcp.load(state, storage_reader=reader)
    out = {}
    for key, match in pairs:
        payload = decode(state[key])
        scale = payload.get("scale_fwd") if isinstance(payload, dict) else None
        if scale is not None and scale.numel():
            out[(int(match["layer"]), match["module"])] = float(scale.float()[0])
    return out


def first_tensor(obj):
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, (tuple, list)):
        for value in obj:
            found = first_tensor(value)
            if found is not None:
                return found
    return None


def quant_stats(value, scale):
    x = value.detach().float()
    scaled = x * scale
    # Match Transformer Engine's saturating cast rather than PyTorch's plain
    # conversion, which may produce NaN above the E4M3 finite maximum.
    q = scaled.clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    dq = q.float() / scale
    err = dq - x
    signal = x.pow(2).sum()
    noise = err.pow(2).sum()
    return {
        "numel": x.numel(),
        "scale_act": scale,
        "amax": x.abs().max().item(),
        "amax_scaled": scaled.abs().max().item(),
        "rel_l2": (noise / signal.clamp_min(1e-30)).sqrt().item(),
        "sqnr_db": 10.0 * math.log10(max(signal.item(), 1e-30) / max(noise.item(), 1e-30)),
        "cosine": torch.nn.functional.cosine_similarity(x.flatten(), dq.flatten(), dim=0).item(),
        "zero_frac": (q.float() == 0).float().mean().item(),
        "clip_frac": (scaled.abs() > FP8_MAX).float().mean().item(),
    }


def attach(model, scales):
    rows, handles = [], []
    for layer_idx, layer in enumerate(model.decoder.layers):
        for module_name in MODULES:
            module = layer
            for part in module_name.split("."):
                module = getattr(module, part)
            scale = scales.get((layer_idx, module_name))
            if scale is None:
                continue

            def hook(_module, args, _name=module_name, _layer=layer_idx, _scale=scale):
                value = first_tensor(args)
                if value is not None:
                    # qkv and fc1 are TE LayerNormLinear modules: their Python
                    # input is the residual stream, while the activation that
                    # is quantized is the fused RMSNorm output.
                    if _name in ("self_attention.linear_qkv", "mlp.linear_fc1"):
                        x = value.float()
                        eps = _module.config.layernorm_epsilon
                        value = (
                            x
                            * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
                            * _module.layer_norm_weight.float()
                        ).to(value.dtype)
                    rows.append({"layer": _layer, "tensor": _name, **quant_stats(value, _scale)})

            handles.append(module.register_forward_pre_hook(hook))
    return rows, handles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--doc-offset", type=int, default=0)
    args = ap.parse_args()

    for key, value in (
        ("MASTER_ADDR", "127.0.0.1"),
        ("MASTER_PORT", "29520"),
        ("RANK", "0"),
        ("WORLD_SIZE", "1"),
        ("LOCAL_RANK", "0"),
    ):
        os.environ.setdefault(key, value)
    torch.distributed.init_process_group("nccl", rank=0, world_size=1)

    from megatron.core import parallel_state as ps
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_spec,
    )

    ps.initialize_model_parallel(1, 1)
    torch.manual_seed(0)
    cfg, leaf = build_config(args.config, args.seq_len)
    spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=None, moe_grouped_gemm=False, qk_layernorm=cfg.qk_layernorm
    )
    model = (
        GPTModel(
            config=cfg,
            transformer_layer_spec=spec,
            vocab_size=leaf["padded_vocab_size"],
            max_sequence_length=args.seq_len,
            pre_process=True,
            post_process=True,
            share_embeddings_and_output_weights=not leaf["untie_embeddings_and_output_weights"],
            position_embedding_type=leaf["position_embedding_type"],
            rotary_base=leaf["rotary_base"],
        )
        .cuda()
        .eval()
    )
    load_weights(model, args.ckpt)
    iteration = int(re.search(r"\d+", args.ckpt.name).group())
    verify_load(model, "docs/64k-debug/data/norm_gains.csv", iteration)
    scales = saved_activation_scales(args.ckpt)
    rows, handles = attach(model, scales)

    tokens, _ = fixed_batch(args.data, args.seq_len, args.batch, args.doc_offset)
    positions = torch.arange(args.seq_len, device=tokens.device).unsqueeze(0)
    mask = torch.triu(
        torch.ones(args.seq_len, args.seq_len, device=tokens.device, dtype=torch.bool), 1
    ).view(1, 1, args.seq_len, args.seq_len)
    with torch.no_grad():
        model(tokens, positions, mask)
    for handle in handles:
        handle.remove()

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv, "w", newline="") as f:
        fields = ["iter", "tensor", "layer"] + list(rows[0])[2:]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {"iter": iteration}
                | {k: (f"{v:.8g}" if isinstance(v, float) else v) for k, v in row.items()}
            )

    for name in MODULES:
        group = [row for row in rows if row["tensor"] == name]
        sqnr = sorted(row["sqnr_db"] for row in group)
        zeros = sorted(row["zero_frac"] for row in group)
        clips = sorted(row["clip_frac"] for row in group)
        print(
            f"{iteration} {name}: median SQNR={sqnr[len(sqnr) // 2]:.2f} dB, "
            f"median zeros={zeros[len(zeros) // 2]:.5f}, "
            f"max clipping={max(clips):.3e}"
        )
    print(f"{len(rows)} activation tensors -> {args.csv}")


if __name__ == "__main__":
    main()
