"""Forward+backward time of a HybridModel decoder stack, outside the training loop.

    python scripts/korbi/hybrid_speed/bench_block.py --pattern L-L-L-*- \
        --config '{"experimental_attention_variant": "kda"}' [--profile] [--compile]

One GPU, one process (TP = PP = 1). Builds the 130M geometry of the speed suite (override
with --hidden / --heads / --ffn / --config), runs the decoder on random bf16 hidden states
[seq, mbs, hidden] (no embedding, no loss), and reports the median fwd+bwd time per stack
and per mixer layer, the peak memory and -- with --profile -- the top GPU kernels. For
iterating on one feature's kernels without a full training job. Prints one RESULT line
(JSON) for scripts.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics

import torch


def build(args):
    from megatron.core import parallel_state
    from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
    from megatron.core.models.hybrid.hybrid_model import HybridModel
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer import TransformerConfig

    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(29500 + os.getpid() % 1000))
        torch.distributed.init_process_group("nccl", rank=0, world_size=1)
        parallel_state.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(0)
    kwargs = dict(
        num_layers=1, hidden_size=args.hidden, num_attention_heads=args.heads,
        num_query_groups=args.heads, kv_channels=args.hidden // args.heads,
        ffn_hidden_size=args.ffn, normalization="RMSNorm", add_bias_linear=False,
        gated_linear_unit=True, activation_func=torch.nn.functional.silu,
        hidden_dropout=0.0, attention_dropout=0.0, is_hybrid_model=True, bf16=True,
        params_dtype=torch.bfloat16, qk_layernorm=True,
        linear_key_head_dim=64, linear_value_head_dim=64,
        linear_num_key_heads=args.heads, linear_num_value_heads=args.heads,
    )
    kwargs.update(json.loads(args.config))
    config = TransformerConfig(**kwargs)
    model = HybridModel(
        config=config, hybrid_stack_spec=hybrid_stack_spec, vocab_size=1024,
        max_sequence_length=args.seq, hybrid_layer_pattern=args.pattern,
        position_embedding_type="none", share_embeddings_and_output_weights=False,
    ).cuda().to(torch.bfloat16)
    return model.decoder


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pattern", required=True)
    p.add_argument("--config", default="{}", help="TransformerConfig overrides (JSON)")
    p.add_argument("--name", default=None)
    p.add_argument("--hidden", type=int, default=768)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--ffn", type=int, default=2048)
    p.add_argument("--seq", type=int, default=4096)
    p.add_argument("--mbs", type=int, default=8)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--stack", type=int, default=0,
                   help="with --profile: group GPU time by this many Python stack frames "
                        "(attributes elementwise kernels to the code that launched them)")
    p.add_argument("--compile", action="store_true", help="torch.compile the stack")
    args = p.parse_args()

    stack = build(args)
    run = torch.compile(stack) if args.compile else stack
    x = torch.randn(args.seq, args.mbs, args.hidden, device="cuda", dtype=torch.bfloat16,
                    requires_grad=True)
    dy = torch.randn_like(x)

    def step():
        out = run(hidden_states=x, attention_mask=None)
        out = out[0] if isinstance(out, tuple) else out
        out.backward(dy)

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(args.iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        step()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    ms = statistics.median(times)
    mixers = sum(c not in "-|/ " for c in args.pattern)
    result = {"name": args.name or args.pattern, "pattern": args.pattern, "config": json.loads(args.config),
              "ms": ms, "ms_per_layer": ms / max(mixers, 1), "spread": (max(times) - min(times)) / ms,
              "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
              "tokens_per_s": args.seq * args.mbs / (ms / 1000)}
    print("RESULT " + json.dumps(result), flush=True)

    if args.profile:
        from torch.profiler import ProfilerActivity, profile

        acts = [ProfilerActivity.CUDA] + ([ProfilerActivity.CPU] if args.stack else [])
        with profile(activities=acts, with_stack=bool(args.stack)) as prof:
            for _ in range(3):
                step()
            torch.cuda.synchronize()
        if args.stack:
            avg = prof.key_averages(group_by_stack_n=args.stack)
            print(avg.table(sort_by="self_cuda_time_total", row_limit=40, max_name_column_width=60,
                            max_src_column_width=110))
        else:
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25,
                                            max_name_column_width=80))


if __name__ == "__main__":
    main()
