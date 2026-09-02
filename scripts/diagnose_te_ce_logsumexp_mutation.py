#!/usr/bin/env python3
"""Check whether TE parallel cross-entropy corrupts a precomputed TP logsumexp graph.

Launch with torchrun. Each process owns one vocabulary shard; the reference concatenates all
shards so it remains valid when tensor parallel world size is greater than one.
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist


class _VocabParallelLogsumexp(torch.autograd.Function):
    """Self-contained copy of the proposed differentiable TP logsumexp helper."""

    @staticmethod
    def forward(ctx, logits: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        logits_fp32 = logits.float()
        logits_max = logits_fp32.amax(dim=-1)
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=group)

        exp_logits = torch.exp(logits_fp32 - logits_max.unsqueeze(-1))
        sum_exp_logits = exp_logits.sum(dim=-1)
        dist.all_reduce(sum_exp_logits, op=dist.ReduceOp.SUM, group=group)

        softmax = exp_logits / sum_exp_logits.unsqueeze(-1)
        ctx.save_for_backward(softmax)
        return torch.log(sum_exp_logits) + logits_max

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (softmax,) = ctx.saved_tensors
        return softmax * grad_output.unsqueeze(-1), None


def vocab_parallel_logsumexp(logits: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    return _VocabParallelLogsumexp.apply(logits, group)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--local-vocab-size", type=int, default=8192)
    parser.add_argument("--logit-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--failure-threshold",
        type=float,
        default=0.5,
        help="Exit nonzero when the post-TE relative gradient error reaches this value.",
    )
    return parser.parse_args()


def _global_reference(base: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    """Return the BF16-rounded reference gradient for this rank's vocabulary shard."""
    shards = [torch.empty_like(base) for _ in range(world_size)]
    dist.all_gather(shards, base)
    full_logits = torch.cat(shards, dim=-1).float().requires_grad_()
    (torch.logsumexp(full_logits, dim=-1) ** 2).sum().backward()

    width = base.size(-1)
    local_grad = full_logits.grad[..., rank * width : (rank + 1) * width]
    # A BF16 input leaf accumulates a BF16 gradient. Match that rounding before comparison.
    return local_grad.to(base.dtype).float()


def _relative_error(actual: torch.Tensor, expected: torch.Tensor, group) -> float:
    parts = torch.stack(
        [
            (actual.float() - expected.float()).square().sum(),
            expected.float().square().sum(),
        ]
    )
    dist.all_reduce(parts, op=dist.ReduceOp.SUM, group=group)
    return (parts[0].sqrt() / parts[1].sqrt()).item()


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        group = dist.group.WORLD
        device = torch.device("cuda", local_rank)

        # Different rank seeds model distinct contiguous vocabulary shards.
        torch.manual_seed(args.seed + rank)
        shape = (args.sequence_length, args.batch_size, args.local_vocab_size)
        base = torch.randn(shape, device=device, dtype=torch.bfloat16) * args.logit_scale

        # Every TP rank must receive the same global token labels.
        label_generator = torch.Generator(device=device).manual_seed(args.seed + 10_000)
        labels = torch.randint(
            0,
            args.local_vocab_size * world_size,
            (args.sequence_length, args.batch_size),
            generator=label_generator,
            device=device,
        )

        expected_grad = _global_reference(base, rank, world_size)

        # Control: the standalone helper without a subsequent TE call.
        control = base.clone().requires_grad_()
        control_lse = vocab_parallel_logsumexp(control, group)
        (control_lse**2).sum().backward()
        control_error = _relative_error(control.grad, expected_grad, group)

        # Suspect path: build the LSE graph, then let TE run before LSE backward.
        suspect = base.clone().requires_grad_()
        suspect_lse = vocab_parallel_logsumexp(suspect, group)
        before_te = suspect.detach().clone()

        from megatron.core.extensions.transformer_engine import te_parallel_cross_entropy

        if te_parallel_cross_entropy is None:
            raise RuntimeError("Transformer Engine parallel cross-entropy is unavailable")
        ce_loss = te_parallel_cross_entropy(suspect, labels, group)
        torch.cuda.synchronize(device)

        mutation = (suspect.detach().float() - before_te.float()).abs()
        mutation_stats = torch.stack(
            [mutation.max(), mutation.ne(0).sum().to(torch.float32)]
        )
        dist.all_reduce(mutation_stats[:1], op=dist.ReduceOp.MAX, group=group)
        dist.all_reduce(mutation_stats[1:], op=dist.ReduceOp.SUM, group=group)

        (suspect_lse**2).sum().backward()
        suspect_error = _relative_error(suspect.grad, expected_grad, group)

        # Keep TE's result alive through the tested backward and make synchronization explicit.
        del ce_loss
        torch.cuda.synchronize(device)

        if rank == 0:
            if control_error >= args.failure_threshold:
                verdict = "INVALID: standalone helper already disagrees with the reference"
            elif suspect_error >= args.failure_threshold:
                verdict = "BUG REPRODUCED: TE forward corrupts the earlier LSE backward"
            else:
                verdict = "PASS: no material corruption detected"

            print(f"TP world size:             {world_size}")
            print(f"Global vocabulary size:    {args.local_vocab_size * world_size}")
            print(f"Standalone relative error: {control_error:.6e}")
            print(f"Post-TE relative error:    {suspect_error:.6e}")
            print(f"TE max input mutation:     {mutation_stats[0].item():.6e}")
            print(f"TE changed input elements: {int(mutation_stats[1].item())}")
            print(f"Verdict: {verdict}")

        if control_error >= args.failure_threshold or suspect_error >= args.failure_threshold:
            raise SystemExit(1)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
