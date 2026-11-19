"""Greedy generation for very long prompts with the oellm_ambitious HF model: the prompt is prefilled in chunks
through the model's cache (a one-shot 256k prefill needs more than a 64 GB GPU: 2026-11-19 multi-document QA at 256k
went out of memory, the chunked NLL did not), then the answer is decoded token by token from the cache. Block-wise
sparse selection is exact under any chunking (sparse_attention.py); with OELLM_SPARSE_SELECT=token the last
OELLM_SPARSE_TAIL queries of each chunk select per token, the prompt's end included."""

from __future__ import annotations

import torch


@torch.no_grad()
def prefill(model, ids: list[int], cache=None, chunk: int = 32768):
    """Feed ``ids`` in chunks through the model's cache (a new one, or ``cache`` continued in place): (cache, the last
    position's logits [1, vocab])."""
    dev = next(model.parameters()).device
    x = torch.tensor([ids], device=dev)
    logits = None
    for a in range(0, x.shape[1], chunk):
        out = model(input_ids=x[:, a:a + chunk], past_key_values=cache, use_cache=True, logits_to_keep=1)
        cache, logits = out.past_key_values, out.logits[:, -1]
    return cache, logits


@torch.no_grad()
def greedy(model, ids: list[int], max_new_tokens: int, eos: int | None, chunk: int = 32768,
           stop=None) -> list[int]:
    """The greedy continuation of ``ids`` (up to max_new_tokens, stopping at eos, or once ``stop(new tokens)`` is
    true, e.g. at the end of the first line of the answer)."""
    dev = next(model.parameters()).device
    cache, logits = prefill(model, ids, chunk=chunk)
    new = []
    for _ in range(max_new_tokens):
        t = int(logits.argmax(-1))
        new.append(t)
        if (eos is not None and t == eos) or (stop is not None and stop(new)):
            break
        out = model(input_ids=torch.tensor([[t]], device=dev), past_key_values=cache, use_cache=True,
                    logits_to_keep=1)
        cache, logits = out.past_key_values, out.logits[:, -1]
    return new
