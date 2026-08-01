#!/usr/bin/env python3
"""Numerical parity check for the TP=1 Liger LM-head adapter.

Run inside the JUPITER training container after installing liger-kernel:
  python experiment/003_multilingual_throughput_tuning/scripts/test_liger_fused_linear_ce.py
"""

import torch
import torch.nn.functional as F

from megatron.core.fusions.liger_fused_linear_cross_entropy import (
    liger_fused_linear_cross_entropy,
)


def main() -> None:
    torch.manual_seed(20260731)
    device = "cuda"
    dtype = torch.bfloat16
    sequence, batch, hidden, vocab = 17, 3, 128, 4096

    states = torch.randn(sequence, batch, hidden, device=device, dtype=dtype)
    weight = torch.randn(vocab, hidden, device=device, dtype=dtype)
    labels = torch.randint(vocab, (batch, sequence), device=device)
    # Match the binary loss masks supplied by Megatron's pretraining datasets.
    # The adapter passes zero-mask labels to Liger as ignore_index, avoiding a
    # released Liger 0.8.0 issue with arbitrary reduction='none' upstream
    # gradients.
    loss_mask = torch.randint(0, 2, (batch, sequence), device=device, dtype=torch.int32).float()

    reference_states = states.detach().clone().requires_grad_(True)
    reference_weight = weight.detach().clone().requires_grad_(True)
    reference_logits = F.linear(reference_states.reshape(-1, hidden), reference_weight)
    reference_loss = F.cross_entropy(
        reference_logits.float(), labels.transpose(0, 1).reshape(-1), reduction="none"
    ).view(sequence, batch).transpose(0, 1)
    (reference_loss * loss_mask).sum().backward()

    for chunk_size in (None, 16):
        liger_states = states.detach().clone().requires_grad_(True)
        liger_weight = weight.detach().clone().requires_grad_(True)
        liger_loss = liger_fused_linear_cross_entropy(
            liger_states, liger_weight, labels, loss_mask, chunk_size
        )
        (liger_loss * loss_mask).sum().backward()

        torch.testing.assert_close(
            (liger_loss * loss_mask).float(), reference_loss * loss_mask, rtol=3e-3, atol=3e-3
        )
        torch.testing.assert_close(
            liger_states.grad.float(), reference_states.grad.float(), rtol=3e-2, atol=3e-2
        )
        torch.testing.assert_close(
            liger_weight.grad.float(), reference_weight.grad.float(), rtol=3e-2, atol=3e-2
        )
    print("PASS: default and explicit-chunk Liger loss/gradients match PyTorch FP32 CE for TP=1")


if __name__ == "__main__":
    main()
