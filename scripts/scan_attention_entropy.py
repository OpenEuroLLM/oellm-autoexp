#!/usr/bin/env python3
"""Attention entropy and logit scale per layer, from a checkpoint, on one GPU.

Tests the QK-norm hypothesis: the q/k norm gains grow all run (rms(gamma_q) *
rms(gamma_k) is up 1.6x by 75k), attention logits scale with their product, and
a large enough logit scale collapses the attention softmax onto one key. Unlike
every other drift we measured, this one has a built-in nonlinearity -- softmax
saturation -- so a SMOOTH driver can still produce a SUDDEN turn. That is why
the entropy itself has to be measured rather than inferred from the gains.

One checkpoint per invocation. ~64 GB of weights land on the GPU, so this needs
a whole GH200; it fits on a login node.

  APPTAINERENV_LD_LIBRARY_PATH=<stub>:/usr/local/lib \\
  apptainer exec --nv <sif> python3 scripts/scan_attention_entropy.py \\
      <iter_dir> --config <frozen config-*.yaml> --data <shard prefix> \\
      --csv docs/64k-debug/data/attn_entropy.csv

Every checkpoint must see the SAME tokens or the comparison is meaningless, so
the batch is taken from a fixed shard at a fixed offset.
"""

import argparse
import csv
import dataclasses
import math
import os
import re
from pathlib import Path

import torch
import yaml


def flat(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from flat(v, f"{p}.{k}" if p else k)
    else:
        yield p, o


def build_config(config_yaml, seq_len):
    """TransformerConfig from the run's own frozen config, filtered to valid
    fields."""
    from megatron.core.transformer.transformer_config import TransformerConfig

    raw = dict(flat(yaml.safe_load(open(config_yaml))))
    # The frozen config is deeply nested; key on the leaf name.
    leaf = {}
    for k, v in raw.items():
        leaf.setdefault(k.split(".")[-1], v)

    valid = {f.name for f in dataclasses.fields(TransformerConfig)}
    kw = {k: v for k, v in leaf.items() if k in valid}

    # Megatron's CLI names are not all TransformerConfig field names. Filtering
    # by field name alone silently drops these, and the model then builds with
    # the wrong shape -- `swiglu` in particular halves linear_fc1's width, which
    # only shows up as a size mismatch once the weights are already loading.
    if leaf.get("swiglu"):
        import torch.nn.functional as F

        kw["gated_linear_unit"] = True
        kw["activation_func"] = F.silu
    if leaf.get("apply_layernorm_1p"):
        kw["layernorm_zero_centered_gamma"] = True
    if leaf.get("squared_relu"):
        kw["activation_func"] = lambda x: torch.pow(F.relu(x), 2)

    # Single-GPU, inference-only, and never FP8: we are measuring the attention
    # distribution of the stored weights, not reproducing training numerics.
    # Filtered against the dataclass, so a field this Megatron does not have is
    # skipped rather than raising.
    override = dict(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        pipeline_model_parallel_layout=None,
        account_for_embedding_in_pipeline_split=False,
        account_for_loss_in_pipeline_split=False,
        microbatch_group_size_per_vp_stage=None,
        num_layers_per_virtual_pipeline_stage=None,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        sequence_parallel=False,
        bf16=True,
        fp16=False,
        fp8=None,
        fp8_param=False,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        perform_initialization=False,
        use_cpu_initialization=False,
        gradient_accumulation_fusion=False,
        cuda_graph_impl="none",  # the sentinel is the string, not None
        enable_cuda_graph=False,
        recompute_granularity=None,
        recompute_method=None,
        recompute_num_layers=None,
        moe_token_dispatcher_type=None,
        # Every one of these is a training/pipelining optimisation that either
        # asserts or is meaningless at PP=1, TP=1, no-grad.
        defer_embedding_wgrad_compute=False,
        delay_wgrad_compute=False,
        overlap_p2p_comm=False,
        overlap_p2p_comm_warmup_flush=False,
        batch_p2p_comm=False,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
        align_param_gather=False,
        tp_comm_overlap=False,
        cpu_offloading=False,
        cpu_offloading_num_layers=0,
        finalize_model_grads_func=None,
        grad_scale_func=None,
        no_sync_func=None,
        grad_sync_func=None,
        param_sync_func=None,
        use_te_rng_tracker=False,
    )
    kw.update({k: v for k, v in override.items() if k in valid})
    kw.pop("num_layers_in_first_pipeline_stage", None)
    kw.pop("num_layers_in_last_pipeline_stage", None)
    return TransformerConfig(**kw), leaf


ENTROPY_ROWS = []


def patch_attention(head_stride):
    """Record entropy from the q/k that core attention actually receives.

    Those are post-QK-norm and post-RoPE, which is exactly the tensor
    pair whose dot product becomes the softmax logits -- so we do not
    have to reproduce either transform, and it works whatever attention
    backend is fused underneath.
    """
    from megatron.core.extensions.transformer_engine import TEDotProductAttention

    original = TEDotProductAttention.forward
    counter = {"layer": 0}

    def forward(self, query, key, value, attention_mask, *a, **kw):
        with torch.no_grad():
            # sbhd -> b h s d
            q = query.permute(1, 2, 0, 3).float()
            k = key.permute(1, 2, 0, 3).float()
            if k.shape[1] != q.shape[1]:  # GQA: expand kv groups
                k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
            b, h, s, d = q.shape
            causal = torch.triu(torch.ones(s, s, device=q.device, dtype=torch.bool), 1)

            ents, maxls = [], []
            for hi in range(0, h, head_stride):  # subsample heads: 4096^2 each
                logits = torch.matmul(q[:, hi], k[:, hi].transpose(-1, -2)) / math.sqrt(d)
                logits = logits.masked_fill(causal, float("-inf"))
                maxls.append(logits[:, 1:, :].max().item())
                p = torch.softmax(logits, dim=-1)
                ent = -(p * torch.log(p.clamp_min(1e-12))).sum(-1)
                # Row 0 attends to one key by construction; it is not informative.
                ents.append(ent[:, 1:].mean().item())
                del logits, p, ent

            pos = torch.arange(1, s, device=q.device, dtype=torch.float)
            ENTROPY_ROWS.append(
                {
                    "layer": counter["layer"],
                    "entropy_nats": sum(ents) / len(ents),
                    # Uniform attention over the causal window, as a yardstick.
                    "uniform_nats": torch.log(pos + 1).mean().item(),
                    "max_logit": max(maxls),
                    "heads_sampled": len(ents),
                }
            )
            counter["layer"] += 1
        return original(self, query, key, value, attention_mask, *a, **kw)

    TEDotProductAttention.forward = forward
    return counter


def load_weights(model, ckpt):
    """Load a torch_dist checkpoint that has no `common.pt`.

    These were saved as raw torch_dist shards, so megatron's
    dist_checkpointing.load() refuses them -- it wants a common.pt that is not
    there. Read them with plain DCP instead. Megatron stacks the per-layer
    tensors into one [num_layers, ...] tensor, so each stacked tensor is read
    once and scattered across that layer's parameters; reading per-parameter
    would re-read a 33 GB tensor 64 times.
    """
    import torch.distributed.checkpoint as dcp

    reader = dcp.FileSystemReader(str(ckpt))
    meta = reader.read_metadata().state_dict_metadata
    sd = model.state_dict()

    plan = {}
    for mk in sd:
        if "_extra_state" in mk:
            continue
        m = re.match(r"(decoder\.layers)\.(\d+)\.(.*)", mk)
        ck, idx = (f"{m.group(1)}.{m.group(3)}", int(m.group(2))) if m else (mk, None)
        if ck in meta:
            plan.setdefault(ck, []).append((idx, mk))

    missing = [
        k
        for k in sd
        if "_extra_state" not in k and not any(k in [t[1] for t in v] for v in plan.values())
    ]
    if missing:
        print(f"  warning: {len(missing)} params not in checkpoint, e.g. {missing[:3]}")

    for ck, targets in plan.items():
        m = meta[ck]
        buf = {ck: torch.empty(m.size, dtype=m.properties.dtype)}
        dcp.load(buf, storage_reader=reader)
        t = buf[ck]
        for idx, mk in targets:
            src = t if idx is None else t[idx]
            sd[mk].copy_(src.to(device=sd[mk].device, dtype=sd[mk].dtype))
        del buf, t
    print(
        f"  loaded {sum(len(v) for v in plan.values())} params from {len(plan)} checkpoint tensors"
    )


def verify_load(model, gains_csv, it):
    """Cross-check against scan_norm_gains.py, which read the same checkpoint
    independently.

    A silent layer-index mix-up would otherwise look fine.
    """
    want = {}
    for r in csv.DictReader(open(gains_csv)):
        if int(r["iter"]) == it and r["tensor"] == "q_layernorm":
            want[int(r["layer"])] = float(r["mean"])
    if not want:
        print("  (no gains row for this iter; load unverified)")
        return
    sd = model.state_dict()
    bad = 0
    for layer, expect in sorted(want.items())[:8]:
        got = sd[f"decoder.layers.{layer}.self_attention.q_layernorm.weight"].float().mean().item()
        if abs(got - expect) > 2e-3:
            bad += 1
            print(f"  MISMATCH layer {layer}: loaded {got:.6f} vs csv {expect:.6f}")
    print(
        "  load verified against norm_gains.csv"
        if not bad
        else f"  *** {bad} mismatches -- the load is wrong ***"
    )


def fixed_batch(prefix, seq_len, batch, offset):
    """The same tokens for every checkpoint, or the comparison means
    nothing."""
    from megatron.core.datasets.indexed_dataset import IndexedDataset

    ds = IndexedDataset(str(prefix))
    toks = []
    i = offset
    while len(toks) < seq_len * batch + 1 and i < len(ds):
        toks.extend(ds[i].tolist())
        i += 1
    if len(toks) < seq_len * batch + 1:
        raise SystemExit(f"{prefix}: not enough tokens from offset {offset}")
    t = torch.tensor(toks[: seq_len * batch + 1], dtype=torch.long)
    return t[:-1].view(batch, seq_len).cuda(), t[1:].view(batch, seq_len).cuda()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--config", required=True, help="the run's frozen config-*.yaml")
    ap.add_argument("--data", required=True, help="megatron shard prefix (no .bin)")
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--doc-offset", type=int, default=0)
    ap.add_argument("--head-stride", type=int, default=8, help="sample every Nth head")
    args = ap.parse_args()

    for k, v in [
        ("MASTER_ADDR", "127.0.0.1"),
        ("MASTER_PORT", "29518"),
        ("RANK", "0"),
        ("WORLD_SIZE", "1"),
        ("LOCAL_RANK", "0"),
    ]:
        os.environ.setdefault(k, v)
    torch.distributed.init_process_group("nccl", rank=0, world_size=1)

    from megatron.core import parallel_state as ps
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_spec,
    )

    ps.initialize_model_parallel(1, 1)
    torch.manual_seed(0)

    cfg, leaf = build_config(args.config, args.seq_len)
    counter = patch_attention(args.head_stride)

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

    verify_load(
        model,
        "docs/64k-debug/data/norm_gains.csv",
        int(re.search(r"\d+", args.ckpt.name).group()),
    )

    tokens, _ = fixed_batch(args.data, args.seq_len, args.batch, args.doc_offset)
    pos = torch.arange(args.seq_len, device=tokens.device).unsqueeze(0)
    mask = torch.triu(
        torch.ones(args.seq_len, args.seq_len, device=tokens.device, dtype=torch.bool), 1
    ).view(1, 1, args.seq_len, args.seq_len)

    ENTROPY_ROWS.clear()
    counter["layer"] = 0
    with torch.no_grad():
        model(tokens, pos, mask)

    it = int(re.search(r"\d+", args.ckpt.name).group())
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    new = not args.csv.exists()
    with open(args.csv, "a", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "iter",
                "layer",
                "entropy_nats",
                "uniform_nats",
                "max_logit",
                "heads_sampled",
            ],
        )
        if new:
            w.writeheader()
        for r in ENTROPY_ROWS:
            w.writerow(
                {
                    "iter": it,
                    **{k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()},
                }
            )
    ent = [r["entropy_nats"] for r in ENTROPY_ROWS]
    print(
        f"{it}: {len(ENTROPY_ROWS)} layers, mean entropy "
        f"{sum(ent) / len(ent):.3f} nats, min {min(ent):.3f} -> {args.csv}"
    )


if __name__ == "__main__":
    main()
