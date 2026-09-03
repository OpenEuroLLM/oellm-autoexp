#!/usr/bin/env python3
"""Sharpness (lambda_max) of one decoder layer, from a checkpoint, on one GPU.

Tests the edge-of-stability hypothesis. A constant learning rate is stable only
while lr < 2 / lambda_max, and lambda_max typically GROWS as a model trains. So
a rate that was safe early can cross the boundary later with nothing changing in
the config -- which is what this run looks like: no event at 66,000, yet the
loss turns there in every fork regardless of seed, numerics or hardware.

Read-off: the TREND in lambda_max across checkpoints.

⚠️ Do NOT compare lr * lambda_max against 2. That is the SGD stability
criterion. This run uses Adam, whose step is lr * m / (sqrt(v) + eps) -- the
update is preconditioned, so the relevant curvature is that of P^-1 H with
P = diag(sqrt(v) + eps), not of H. With median sqrt(v) ~ 6e-8 (see item 19) the
preconditioner is enormous, so the raw and preconditioned quantities differ by
orders of magnitude. The absolute number here is only meaningful RELATIVE to
other checkpoints measured the same way.

  apptainer exec --nv <sif> python3 scripts/scan_sharpness.py <iter_dir> \\
      --config <frozen config-*.yaml> --data <shard prefix> \\
      --csv docs/64k-debug/data/sharpness.csv

Sequence length defaults to 512, not 4096: activation memory for the backward
pass is what binds here, and the question is how curvature MOVES across
checkpoints, which a consistent shorter context answers just as well.
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


def block_params(model, layer):
    """Curvature of one decoder layer's weights.

    The full Hessian is out of reach: the model alone is 64 GB of a 96 GB card,
    so a global gradient does not fit. Restricting to one layer's weight
    matrices keeps the gradient at ~2 GB and still answers the question we have,
    which is whether curvature GROWS across checkpoints -- lambda_max of a
    principal submatrix is a lower bound on the global lambda_max, so a block
    crossing the stability boundary is already meaningful.
    """
    ps = []
    for n, p in model.named_parameters():
        if f"layers.{layer}." in n and n.endswith(".weight") and "layernorm" not in n:
            p.requires_grad_(True)
            ps.append(p)
        else:
            p.requires_grad_(False)
    return ps


def grad_at(model, ps, tokens, pos, mask, labels, offset=None, eps=0.0):
    """Gradient of the loss wrt `ps`, optionally at theta + eps*offset."""
    if offset is not None and eps:
        with torch.no_grad():
            for p, o in zip(ps, offset):
                p.add_(o, alpha=eps)
    for p in ps:
        p.grad = None
    loss = model(tokens, pos, mask, labels=labels).mean()
    g = torch.autograd.grad(loss, ps)
    if offset is not None and eps:
        with torch.no_grad():
            for p, o in zip(ps, offset):
                p.sub_(o, alpha=eps)
    return [x.detach().float() for x in g], loss.item()


def power_iteration(model, ps, batch, iters, rel_eps):
    """lambda_max by finite-difference Hessian-vector products.

    Hv = (g(theta + e v) - g(theta - e v)) / 2e. Finite differences rather than
    double-backprop on purpose: create_graph would retain the whole forward
    graph and roughly double activation memory, which this card cannot spare.
    """
    tokens, pos, mask, labels = batch
    gen = torch.Generator(device="cpu").manual_seed(0)  # same v0 every checkpoint
    v = [torch.randn(p.shape, generator=gen).to(p.device, torch.float32) for p in ps]
    nrm = math.sqrt(sum(x.pow(2).sum().item() for x in v))
    v = [x / nrm for x in v]

    theta_nrm = math.sqrt(sum(p.detach().float().pow(2).sum().item() for p in ps))
    e = rel_eps * theta_nrm
    lam = float("nan")
    prev = None
    for k in range(iters):
        vb = [x.to(p.dtype) for x, p in zip(v, ps)]
        gp, _ = grad_at(model, ps, tokens, pos, mask, labels, vb, +e)
        gm, _ = grad_at(model, ps, tokens, pos, mask, labels, vb, -e)
        hv = [(a - b) / (2 * e) for a, b in zip(gp, gm)]
        lam = sum((a * b).sum().item() for a, b in zip(v, hv))
        nrm = math.sqrt(sum(x.pow(2).sum().item() for x in hv))
        if nrm == 0:
            break
        v = [x / nrm for x in hv]
        print(f"    power iter {k + 1}: lambda = {lam:.6f}")
        if k > 2 and prev and abs(lam - prev) < 1e-3 * abs(lam):
            print(f"    converged after {k + 1} iterations")
            break
        prev = lam
    return lam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--layer", type=int, default=32, help="which decoder layer")
    ap.add_argument("--iters", type=int, default=25, help="power iterations")
    ap.add_argument("--rel-eps", type=float, default=1e-3)
    ap.add_argument("--lr", type=float, default=3e-4, help="recorded for reference only")
    args = ap.parse_args()

    for k, v in [
        ("MASTER_ADDR", "127.0.0.1"),
        ("MASTER_PORT", "29519"),
        ("RANK", "0"),
        ("WORLD_SIZE", "1"),
        ("LOCAL_RANK", "0"),
    ]:
        os.environ.setdefault(k, v)
    torch.distributed.init_process_group("nccl", rank=0, world_size=1)

    from megatron.core import parallel_state as ps_
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_spec,
    )

    ps_.initialize_model_parallel(1, 1)
    torch.manual_seed(0)

    cfg, leaf = build_config(args.config, args.seq_len)
    spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=None, moe_grouped_gemm=False, qk_layernorm=cfg.qk_layernorm
    )
    model = GPTModel(
        config=cfg,
        transformer_layer_spec=spec,
        vocab_size=leaf["padded_vocab_size"],
        max_sequence_length=args.seq_len,
        pre_process=True,
        post_process=True,
        share_embeddings_and_output_weights=not leaf["untie_embeddings_and_output_weights"],
        position_embedding_type=leaf["position_embedding_type"],
        rotary_base=leaf["rotary_base"],
    ).cuda()
    model.train()

    load_weights(model, args.ckpt)
    it = int(re.search(r"\d+", args.ckpt.name).group())
    verify_load(model, "docs/64k-debug/data/norm_gains.csv", it)

    tokens, labels = fixed_batch(args.data, args.seq_len, 1, 0)
    pos = torch.arange(args.seq_len, device=tokens.device).unsqueeze(0)
    mask = torch.triu(
        torch.ones(args.seq_len, args.seq_len, device=tokens.device, dtype=torch.bool), 1
    ).view(1, 1, args.seq_len, args.seq_len)

    params = block_params(model, args.layer)
    n = sum(p.numel() for p in params)
    print(f"  layer {args.layer}: {len(params)} weight tensors, {n / 1e6:.0f}M params")
    lam = power_iteration(model, params, (tokens, pos, mask, labels), args.iters, args.rel_eps)

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    new = not args.csv.exists()
    with open(args.csv, "a", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["iter", "layer", "seq_len", "lambda_max", "lr", "lr_times_lambda"]
        )
        if new:
            w.writeheader()
        w.writerow(
            {
                "iter": it,
                "layer": args.layer,
                "seq_len": args.seq_len,
                "lambda_max": f"{lam:.6g}",
                "lr": args.lr,
                "lr_times_lambda": f"{args.lr * lam:.6g}",
            }
        )
    print(
        f"{it}: lambda_max ~ {lam:.5f}  (compare across checkpoints only; "
        f"the SGD lr*lambda<2 rule does NOT apply to Adam) -> {args.csv}"
    )


if __name__ == "__main__":
    main()
