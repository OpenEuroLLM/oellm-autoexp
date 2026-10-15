#!/usr/bin/env python3
"""MuP coordinate check for Megatron HybridModels (any layer pattern / mixer).

Builds the same model at several widths, trains a few steps with Megatron's own muP
setup (``use_mup`` init / forward multipliers + ``get_mup_config_overrides`` param-group
LRs) and records the RMS of every layer's output and of the logits. Under a correct
parametrization these stay O(1) in width at every step: the slope of log(RMS) against
log(width) is ~0. A layer whose slope drifts away from 0 over the steps is mis-scaled
(its init, its LR class, or a forward multiplier). Single GPU::

    PYTHONPATH=<autoexp>/submodules/Megatron-LM[:<ComplexKDA fla>] \\
    python scripts/korbi/mup_coord_check.py --pattern "L-*-C-N-" \\
        --variant gated_delta_net --widths 256 512 1024 2048 --base-width 256 \\
        --out coord.json [--extra '{"engram_table_size": 20000}'] [--no-mup]

``--backend fake`` runs on a box where NCCL cannot initialise (world size 1 only).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict


def _init_dist(backend: str) -> None:
    import torch.distributed as dist

    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29533")
    if backend == "fake":
        from torch.testing._internal.distributed.fake_pg import FakeStore

        dist.init_process_group(backend="fake", store=FakeStore(), world_size=1, rank=0)
        new_group = dist.new_group

        def _new_group(*a, **kw):  # gloo subgroups do not exist on the fake backend
            kw.pop("backend", None)
            return new_group(*a, **kw)

        dist.new_group = _new_group
    else:
        dist.init_process_group(backend=backend, world_size=1, rank=0)


def build(args, width: int):
    import torch
    from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
    from megatron.core.models.hybrid.hybrid_model import HybridModel
    from megatron.core.transformer import TransformerConfig
    from megatron.core.transformer.enums import AttnBackend

    heads = width // args.head_dim
    extra = json.loads(args.extra) if args.extra else {}
    variant = {} if args.variant is None else {"experimental_attention_variant": args.variant}
    config = TransformerConfig(
        num_layers=len(args.pattern),
        hidden_size=width,
        num_attention_heads=heads,
        kv_channels=args.head_dim,
        ffn_hidden_size=int(round(width * args.ffn_ratio / 64)) * 64,
        normalization="RMSNorm",
        add_bias_linear=False,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        attention_backend=AttnBackend.unfused,
        is_hybrid_model=True,
        linear_key_head_dim=args.head_dim,
        linear_value_head_dim=args.head_dim,
        linear_num_key_heads=heads,
        linear_num_value_heads=heads,
        use_mup=not args.no_mup,
        mup_base_hidden_size=args.base_width,
        mup_base_head_dim=args.head_dim,
        init_method_std=args.init_std,
        **variant,
        **extra,
    )
    model = HybridModel(
        config=config,
        hybrid_stack_spec=hybrid_stack_spec,
        vocab_size=args.vocab,
        max_sequence_length=args.seq,
        hybrid_layer_pattern=args.pattern,
        position_embedding_type="none",
        share_embeddings_and_output_weights=False,
    ).cuda()
    return model, config


def optimizer_for(args, model, config):
    import torch
    from megatron.core.optimizer import _get_param_groups, get_mup_config_overrides
    from megatron.core.optimizer.optimizer_config import OptimizerConfig

    opt_config = OptimizerConfig(optimizer="adam", lr=args.lr, min_lr=args.lr, weight_decay=0.0)
    overrides = (
        {} if args.no_mup else get_mup_config_overrides(opt_config, config.mup_width_mult, "adam")
    )
    groups = _get_param_groups([model], opt_config, overrides)
    torch_groups = [
        {
            "params": g["params"],
            "lr": g.get("max_lr", args.lr),
            "eps": g.get("eps", opt_config.adam_eps),
        }
        for g in groups
    ]
    return torch.optim.Adam(torch_groups, betas=(0.9, 0.95))


def run_width(args, width: int) -> dict:
    import torch
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    records: dict = defaultdict(lambda: defaultdict(list))  # name -> step -> [rms per seed]
    for seed in range(args.seeds):
        parallel_state.destroy_model_parallel()
        parallel_state.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(1234 + seed)
        torch.manual_seed(1234 + seed)
        model, config = build(args, width)
        opt = optimizer_for(args, model, config)
        step = {"i": 0}

        def hook(name):
            def fn(_module, _inp, out):
                t = out[0] if isinstance(out, tuple) else out
                if torch.is_tensor(t) and t.is_floating_point():
                    records[name][step["i"]].append(t.detach().float().pow(2).mean().sqrt().item())

            return fn

        model.embedding.register_forward_hook(hook("embedding"))
        for i, layer in enumerate(model.decoder.layers):
            layer.register_forward_hook(hook(f"layer{i:02d}_{args.pattern[i]}"))
        model.output_layer.register_forward_hook(hook("logits"))

        gen = torch.Generator(device="cuda").manual_seed(99 + seed)
        ids = torch.randint(0, args.vocab, (args.batch, args.seq), device="cuda", generator=gen)
        labels = torch.randint(0, args.vocab, (args.batch, args.seq), device="cuda", generator=gen)
        for i in range(args.steps):
            step["i"] = i
            loss = model(ids, position_ids=None, attention_mask=None, labels=labels).float().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            records["loss"][i].append(loss.item())
        del model, opt
        torch.cuda.empty_cache()
    return {name: {s: sum(v) / len(v) for s, v in steps.items()} for name, steps in records.items()}


def slope(xs, ys) -> float:
    lx = [math.log(x) for x in xs]
    ly = [math.log(max(y, 1e-30)) for y in ys]
    mx, my = sum(lx) / len(lx), sum(ly) / len(ly)
    var = sum((a - mx) ** 2 for a in lx)
    return sum((a - mx) * (b - my) for a, b in zip(lx, ly)) / var if var else 0.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--pattern", default="L-*-L-*-")
    p.add_argument("--variant", default=None, help="experimental_attention_variant for `L`")
    p.add_argument("--widths", type=int, nargs="+", default=[256, 512, 1024, 2048])
    p.add_argument("--base-width", type=int, default=256)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--ffn-ratio", type=float, default=2.75)
    p.add_argument("--vocab", type=int, default=4096)
    p.add_argument("--seq", type=int, default=256)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-2, help="large on purpose: exposes blow-ups")
    p.add_argument("--init-std", type=float, default=0.02)
    p.add_argument("--no-mup", action="store_true", help="standard parametrization, to compare")
    p.add_argument("--extra", default=None, help="JSON dict of extra TransformerConfig fields")
    p.add_argument("--backend", default="nccl")
    p.add_argument("--out", default=None)
    p.add_argument("--tolerance", type=float, default=0.25, help="|slope| flagged above this")
    args = p.parse_args()

    _init_dist(args.backend)
    results = {w: run_width(args, w) for w in args.widths}
    names = [n for n in results[args.widths[0]] if n != "loss"]
    last = args.steps - 1
    print(f"log-log slope of activation RMS vs width ({'standard' if args.no_mup else 'muP'})")
    print(f"{'module':24s} {'step0':>8s} {'step' + str(last):>8s}")
    flagged = []
    summary = {}
    for name in names:
        s0 = slope(args.widths, [results[w][name][0] for w in args.widths])
        sl = slope(args.widths, [results[w][name][last] for w in args.widths])
        summary[name] = {"slope_step0": s0, "slope_last": sl}
        mark = "  <-- " if max(abs(s0), abs(sl)) > args.tolerance else ""
        if mark:
            flagged.append(name)
        print(f"{name:24s} {s0:8.3f} {sl:8.3f}{mark}")
    losses = {w: results[w]["loss"][last] for w in args.widths}
    print("final loss by width:", {w: round(v, 4) for w, v in losses.items()})
    print("flagged:", flagged or "none")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(
                {"args": vars(args), "summary": summary, "losses": losses, "raw": results},
                f,
                indent=1,
            )


if __name__ == "__main__":
    main()
