"""Tests for the KV-cache block-diffusion sampler (spec section 14) and
invariant I8.

The model is a tiny randomly initialized Qwen3 causal LM, built from a config and never
downloaded. I8: with block size 1 and token shift 1, static decoding reproduces greedy
autoregressive decoding. The other tests cover the pass counts of the static and dynamic modes,
the cache against a full forward pass without a cache, a full-recompute oracle of the whole
algorithm, the cuts at EOS and at max_new_tokens, and the argument checks. The tests with
``clean_causal`` check the token-causal clean rule of objective nld (spec section 15) in the masks,
in the logits, and in the whole decoding.

Run with:
    .claude/dllm/state/pytest2.sh tests/unit/dllm/test_sampler.py -v
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

import torch
import transformers
from torch.utils.hooks import RemovableHandle

from oellm_autoexp.dllm.eval.sampler import BlockDiffusionSampler, SampleResult
from oellm_autoexp.dllm.masks import dense_bd3_reference, make_ar_mask_mod, materialize_mask
from oellm_autoexp.dllm.targets import prediction_sources

_REAL_VOCAB = 47  # len(tokenizer) of the tiny model
_MASK_ID = 47  # M = real_vocab, as in spec section 12
_NO_EOS = -1  # never emitted, so the EOS stop is off
_BLOCK = 4
_PROMPT = [3, 9, 17, 5]  # length 4, so the first block is block 1 when B = 4


def _tiny_model() -> transformers.Qwen3ForCausalLM:
    """Return the tiny Qwen3 causal LM (vocabulary 48) in eval mode, after
    torch.manual_seed(0)."""
    config = transformers.Qwen3Config(
        vocab_size=48,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=512,
        tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    return transformers.Qwen3ForCausalLM(config).eval()


def _sampler(
    model: transformers.Qwen3ForCausalLM,
    block_size: int = _BLOCK,
    token_shift: int = 0,
    eos_id: int = _NO_EOS,
    clean_causal: bool = False,
) -> BlockDiffusionSampler:
    """Return a sampler for the tiny model with the mask and real vocabulary of
    these tests."""
    return BlockDiffusionSampler(
        model, block_size, _MASK_ID, token_shift, _REAL_VOCAB, eos_id, clean_causal=clean_causal
    )


def _ar_greedy(model: transformers.Qwen3ForCausalLM, prompt: list[int], count: int) -> list[int]:
    """Decode greedily without a cache, recomputing the whole sequence at every
    step."""
    seq = list(prompt)
    with torch.no_grad():
        for _ in range(count):
            logits = model(input_ids=torch.tensor([seq])).logits[0, -1, :_REAL_VOCAB]
            seq.append(int(torch.argmax(logits)))
    return seq[len(prompt) :]


def _reference_logits(
    model: transformers.Qwen3ForCausalLM,
    clean: list[int],
    block: list[int],
    clean_causal: bool = False,
) -> torch.Tensor:
    """Return the logits of ``clean + block`` from one full forward pass
    without a cache.

    The clean rows follow the clean rule of the sampler: block-causal, or token-causal when
    ``clean_causal`` is set. The block rows see every key, which is the denoise rule of spec
    section 14. The clean rows come from the dense reference of invariant I1 (and I10).

    Args:
        model: The tiny model in eval mode.
        clean: Clean ids before the block. Their count is a multiple of the block size.
        block: Ids of the block, with mask ids at the masked positions.
        clean_causal: True for the token-causal clean rule of spec section 15.

    Returns:
        Logits of shape ``[len(clean) + len(block), V]``.
    """
    total = len(clean) + len(block)
    mask = dense_bd3_reference(total, len(block), None, clean_causal=clean_causal)
    allowed = mask[total:, total:]
    allowed[len(clean) :] = True  # the block rows see every key of the sequence
    with torch.no_grad():
        out = model(
            input_ids=torch.tensor([clean + block]),
            position_ids=torch.arange(total).unsqueeze(0),
            attention_mask=allowed.reshape(1, 1, total, total),
            use_cache=False,
        )
    return out.logits[0]


def _oracle_decode(
    model: transformers.Qwen3ForCausalLM,
    prompt: list[int],
    max_new: int,
    block_size: int,
    token_shift: int,
    mode: str,
    tau: float,
    steps: int | None,
    clean_causal: bool = False,
) -> tuple[list[int], int]:
    """Decode by full forward passes without a cache, following spec section 14
    step by step.

    Each denoise pass recomputes the whole sequence. The unmask rule is written out again here, so
    the test does not reuse the sampler's helpers. The clean rows of each pass follow
    ``clean_causal``.

    Returns:
        A tuple ``(tokens, denoise_passes)``.
    """
    prompt_len = len(prompt)
    per_pass = -(-block_size // (block_size if steps is None else steps))
    history = list(prompt)  # decided ids of positions 0 .. start - 1
    generated: list[int] = []
    denoise_passes = 0
    block_index = prompt_len // block_size
    while True:
        start = block_index * block_size
        block = [
            prompt[i] if i < prompt_len else _MASK_ID for i in range(start, start + block_size)
        ]
        masked = [j for j in range(block_size) if start + j >= prompt_len]
        while masked:
            full = _reference_logits(model, history[:start], block, clean_causal)
            denoise_passes += 1
            # Spec section 6 gives src[i] as the stream index that predicts position i. With one
            # sequence of length total, a clean-stream source total + i - 1 is the position i - 1.
            total = start + block_size
            src, _ = prediction_sources(total, block_size, token_shift)
            pred = full[src[start:] % total]
            restricted = pred[:, :_REAL_VOCAB]
            ids = restricted.argmax(dim=-1).tolist()
            conf = torch.softmax(restricted.float(), dim=-1).max(dim=-1).values.tolist()
            by_confidence = sorted(masked, key=lambda j: (-conf[j], j))
            if mode == "static":
                chosen = by_confidence[:per_pass]
            else:
                chosen = [j for j in masked if conf[j] >= tau] or by_confidence[:1]
            for j in chosen:
                block[j] = ids[j]
            masked = [j for j in masked if j not in chosen]
        generated.extend(block[max(0, prompt_len - start) :])
        history = history[:start] + block
        if len(generated) >= max_new:
            return generated[:max_new], denoise_passes
        block_index += 1


def _record_calls(
    model: transformers.Qwen3ForCausalLM,
) -> tuple[list[dict], RemovableHandle]:
    """Record the input ids, the attention mask, and the logits of every model
    call.

    Returns:
        A tuple ``(records, handle)``. Call ``handle.remove()`` when done.
    """
    records: list[dict] = []

    def hook(module: torch.nn.Module, args: tuple, kwargs: dict, output: object) -> None:
        records.append(
            {
                "input_ids": kwargs["input_ids"].clone(),
                "mask": kwargs["attention_mask"].clone(),
                "logits": output.logits.detach().clone(),
            }
        )

    return records, model.register_forward_hook(hook, with_kwargs=True)


def _first_denoise_call(records: list[dict], block_size: int, cache_len: int) -> dict:
    """Return the first denoise call, the one over a cache of ``cache_len``
    tokens.

    A denoise call returns the logits of every block position. A cache-
    fill or prefill call returns the last position only, so it is not
    selected.
    """
    for record in records:
        logits_positions = record["logits"].shape[1]
        if logits_positions == block_size and record["mask"].shape[-1] == cache_len + block_size:
            return record
    raise AssertionError(f"no denoise call over a cache of {cache_len} tokens")


@pytest.mark.parametrize("clean_causal", [False, True])
@pytest.mark.parametrize("prompt", [[3], [3, 9, 17, 5, 22], [3, 9, 17, 5, 22, 30, 11]])
def test_i8_block_one_shift_one_static_matches_ar_greedy(
    prompt: list[int], clean_causal: bool
) -> None:
    model = _tiny_model()
    expected = _ar_greedy(model, prompt, 12)
    sampler = _sampler(model, block_size=1, token_shift=1, clean_causal=clean_causal)
    result = sampler.generate(prompt, 12, mode="static")
    assert result.tokens == expected
    assert len(result.tokens) == 12
    assert max(result.tokens) < _REAL_VOCAB  # the mask id and the padding rows are never emitted


@pytest.mark.parametrize("prompt_len", [4, 8])
@pytest.mark.parametrize("token_shift", [0, 1])
def test_static_block_four_unmasks_one_token_per_denoise_pass(
    prompt_len: int, token_shift: int
) -> None:
    model = _tiny_model()
    prompt = [3, 9, 17, 5, 22, 30, 11, 2][:prompt_len]
    result = _sampler(model, token_shift=token_shift).generate(prompt, 12, mode="static")
    assert len(result.tokens) == 12
    assert result.denoise_passes == 3 * _BLOCK  # B denoise passes in each of the 3 blocks
    assert result.cache_fill_passes == 3
    assert result.tpf_decode == 1.0
    assert max(result.tokens) < _REAL_VOCAB


@pytest.mark.parametrize(("steps", "passes_per_block"), [(1, 1), (2, 2), (3, 2), (4, 4)])
def test_static_steps_unmask_ceil_of_block_over_steps_per_pass(
    steps: int, passes_per_block: int
) -> None:
    result = _sampler(_tiny_model()).generate(_PROMPT, 8, mode="static", steps=steps)
    assert len(result.tokens) == 8
    assert result.denoise_passes == 2 * passes_per_block
    assert result.tpf_decode == _BLOCK / passes_per_block


def test_dynamic_tau_zero_unmasks_each_block_in_one_pass() -> None:
    result = _sampler(_tiny_model()).generate(_PROMPT, 8, mode="dynamic", tau=0.0)
    assert len(result.tokens) == 8
    assert result.denoise_passes == 2
    assert result.tpf_decode == float(_BLOCK)


def test_dynamic_tau_above_one_unmasks_exactly_one_position_per_pass() -> None:
    result = _sampler(_tiny_model()).generate(_PROMPT, 8, mode="dynamic", tau=1.01)
    assert len(result.tokens) == 8
    assert result.denoise_passes == 8
    assert result.tpf_decode == 1.0


@pytest.mark.parametrize("clean_causal", [False, True])
@pytest.mark.parametrize("prompt", [_PROMPT, [3, 9, 17, 5, 22, 30]])
def test_first_denoise_logits_match_full_forward_without_cache(
    prompt: list[int], clean_causal: bool
) -> None:
    model = _tiny_model()
    records, handle = _record_calls(model)
    try:
        _sampler(model, token_shift=0, clean_causal=clean_causal).generate(prompt, 4)
    finally:
        handle.remove()
    start = (len(prompt) // _BLOCK) * _BLOCK
    block = [prompt[i] if i < len(prompt) else _MASK_ID for i in range(start, start + _BLOCK)]
    call = _first_denoise_call(records, _BLOCK, cache_len=start)
    expected = _reference_logits(model, prompt[:start], block, clean_causal)[start:]
    torch.testing.assert_close(call["logits"][0], expected, atol=1e-5, rtol=0.0)


@pytest.mark.parametrize("clean_causal", [False, True])
def test_first_denoise_logits_of_second_block_use_the_filled_cache(clean_causal: bool) -> None:
    model = _tiny_model()
    records, handle = _record_calls(model)
    try:
        result = _sampler(model, token_shift=0, clean_causal=clean_causal).generate(_PROMPT, 8)
    finally:
        handle.remove()
    clean = _PROMPT + result.tokens[:_BLOCK]
    call = _first_denoise_call(records, _BLOCK, cache_len=len(clean))
    expected = _reference_logits(model, clean, [_MASK_ID] * _BLOCK, clean_causal)[len(clean) :]
    torch.testing.assert_close(call["logits"][0], expected, atol=1e-5, rtol=0.0)


@pytest.mark.parametrize("clean_causal", [False, True])
@pytest.mark.parametrize("token_shift", [0, 1])
@pytest.mark.parametrize(
    ("mode", "tau", "steps"),
    [
        ("static", 0.9, None),
        ("static", 0.9, 2),
        ("dynamic", 0.0, None),
        ("dynamic", 1.01, None),
    ],
)
def test_generate_matches_full_recompute_oracle(
    token_shift: int, mode: str, tau: float, steps: int | None, clean_causal: bool
) -> None:
    model = _tiny_model()
    prompt = [3, 9, 17, 5, 22, 30]  # the first block starts with two prompt tokens
    expected_tokens, expected_passes = _oracle_decode(
        model, prompt, 10, _BLOCK, token_shift, mode, tau, steps, clean_causal
    )
    sampler = _sampler(model, token_shift=token_shift, clean_causal=clean_causal)
    result = sampler.generate(prompt, 10, mode=mode, tau=tau, steps=steps)
    assert result.tokens == expected_tokens
    assert result.denoise_passes == expected_passes
    assert result.cache_fill_passes == 3


_MASK_PROMPT = [3, 9, 17, 5, 22, 30, 11, 2, 7, 13]  # 10 tokens: the prefill covers blocks 0 and 1


def _lower_triangular(size: int) -> torch.Tensor:
    """Return the boolean lower-triangular mask of ``size`` positions."""
    return torch.tril(torch.ones(size, size)).bool()


def _run_recorded(
    model: transformers.Qwen3ForCausalLM, clean_causal: bool, prompt: list[int], count: int
) -> list[dict]:
    """Decode statically and return the record of every model call."""
    records, handle = _record_calls(model)
    try:
        _sampler(model, clean_causal=clean_causal).generate(prompt, count)
    finally:
        handle.remove()
    return records


def _split_calls(records: list[dict], block_size: int) -> tuple[list[dict], list[dict]]:
    """Split the recorded calls into the last-position calls and the denoise
    calls.

    The prefill and the cache-fill passes return the logits of the last position only, so their
    logits have one position. A denoise pass returns the logits of the whole block.

    Returns:
        A tuple ``(last_only, denoise)``, each in call order.
    """
    last_only = [record for record in records if record["logits"].shape[1] == 1]
    denoise = [record for record in records if record["logits"].shape[1] == block_size]
    return last_only, denoise


@pytest.mark.parametrize("clean_causal", [False, True])
def test_prefill_and_cache_fill_masks_follow_the_clean_rule(clean_causal: bool) -> None:
    records = _run_recorded(_tiny_model(), clean_causal, _MASK_PROMPT, 6)
    last_only, _ = _split_calls(records, _BLOCK)
    # The prefill covers blocks 0 and 1 (8 tokens). Blocks 2 and 3 are generated, and each of them
    # gets a cache-fill pass that starts at position 8 and at position 12.
    prefill, fills = last_only[0], last_only[1:]
    assert prefill["input_ids"].shape[1] == 8
    assert len(fills) == 2
    if clean_causal:
        expected_prefill = _lower_triangular(8)
    else:
        expected_prefill = dense_bd3_reference(8, _BLOCK, None)[8:, 8:]
    assert torch.equal(prefill["mask"][0, 0], expected_prefill)
    for fill, start in zip(fills, (8, 12)):
        inside = _lower_triangular(_BLOCK) if clean_causal else torch.ones(_BLOCK, _BLOCK).bool()
        cached = torch.ones(_BLOCK, start, dtype=torch.bool)
        assert fill["mask"].shape == (1, 1, _BLOCK, start + _BLOCK)
        assert torch.equal(fill["mask"][0, 0], torch.cat([cached, inside], dim=1))


def test_denoise_masks_are_full_and_do_not_depend_on_clean_causal() -> None:
    model = _tiny_model()
    masks: dict[bool, list[torch.Tensor]] = {}
    for clean_causal in (False, True):
        _, denoise = _split_calls(_run_recorded(model, clean_causal, _MASK_PROMPT, 6), _BLOCK)
        masks[clean_causal] = [record["mask"] for record in denoise]
    # Block 2 has two masked positions and block 3 has four, so there are six denoise passes.
    assert [mask.shape[-1] for mask in masks[False]] == [12, 12, 16, 16, 16, 16]
    assert len(masks[True]) == len(masks[False])
    for mask_block, mask_causal in zip(masks[False], masks[True]):
        assert bool(mask_block.all())
        assert torch.equal(mask_block, mask_causal)


def test_rejects_clean_causal_that_is_not_a_bool() -> None:
    with pytest.raises(TypeError, match="clean_causal"):
        BlockDiffusionSampler(
            _tiny_model(), _BLOCK, _MASK_ID, 0, _REAL_VOCAB, _NO_EOS, clean_causal=1
        )


def test_eos_cuts_after_first_eos_and_stops_after_its_block() -> None:
    model = _tiny_model()
    full = _sampler(model).generate(_PROMPT, 12).tokens
    eos = full[5]
    first = full.index(eos)
    result = _sampler(model, eos_id=eos).generate(_PROMPT, 12)
    assert result.tokens == full[: first + 1]
    assert result.tokens[-1] == eos
    blocks = first // _BLOCK + 1
    assert result.denoise_passes == _BLOCK * blocks
    assert result.cache_fill_passes == blocks


def test_max_new_tokens_cuts_the_text_and_keeps_the_passes_of_its_block() -> None:
    model = _tiny_model()
    full = _sampler(model).generate(_PROMPT, 8).tokens
    result = _sampler(model).generate(_PROMPT, 6)
    assert result.tokens == full[:6]
    assert result.denoise_passes == 2 * _BLOCK
    assert result.cache_fill_passes == 2


def test_zero_max_new_tokens_returns_no_tokens_and_no_passes() -> None:
    result = _sampler(_tiny_model()).generate(_PROMPT, 0)
    assert result == SampleResult(tokens=[], denoise_passes=0, cache_fill_passes=0)
    assert result.tpf_decode == 0.0
    assert result.tpf_all == 0.0


def test_tpf_properties_divide_tokens_by_the_matching_passes() -> None:
    result = SampleResult(tokens=[1, 2, 3, 4], denoise_passes=2, cache_fill_passes=2)
    assert result.tpf_decode == 2.0
    assert result.tpf_all == 1.0


def test_bool_4d_causal_mask_matches_plain_causal_forward() -> None:
    model = _tiny_model()
    ids = torch.tensor([[3, 9, 17, 5, 22, 30, 11]])
    length = ids.shape[1]
    causal = materialize_mask(make_ar_mask_mod(length, None), 1, length, length).unsqueeze(1)
    with torch.no_grad():
        plain = model(input_ids=ids, use_cache=False).logits
        explicit = model(input_ids=ids, attention_mask=causal, use_cache=False).logits
    torch.testing.assert_close(explicit, plain, atol=1e-5, rtol=0.0)


def test_rejects_empty_prompt() -> None:
    with pytest.raises(ValueError, match="empty"):
        _sampler(_tiny_model()).generate([], 4)


@pytest.mark.parametrize("mode", ["greedy", "Static", ""])
def test_rejects_mode_other_than_static_or_dynamic(mode: str) -> None:
    with pytest.raises(ValueError, match="mode"):
        _sampler(_tiny_model()).generate(_PROMPT, 4, mode=mode)


@pytest.mark.parametrize("token_shift", [2, -1])
def test_rejects_token_shift_other_than_zero_or_one(token_shift: int) -> None:
    with pytest.raises(ValueError, match="token_shift"):
        BlockDiffusionSampler(_tiny_model(), _BLOCK, _MASK_ID, token_shift, _REAL_VOCAB, _NO_EOS)


def test_rejects_mask_id_below_real_vocab() -> None:
    with pytest.raises(ValueError, match="mask_id"):
        BlockDiffusionSampler(_tiny_model(), _BLOCK, 40, 0, _REAL_VOCAB, _NO_EOS)


def test_rejects_block_size_below_one() -> None:
    with pytest.raises(ValueError, match="block_size"):
        BlockDiffusionSampler(_tiny_model(), 0, _MASK_ID, 0, _REAL_VOCAB, _NO_EOS)


def test_rejects_attn_implementation_other_than_sdpa() -> None:
    model = _tiny_model()
    model.config._attn_implementation = "eager"
    with pytest.raises(ValueError, match="sdpa"):
        BlockDiffusionSampler(model, _BLOCK, _MASK_ID, 0, _REAL_VOCAB, _NO_EOS)


def test_rejects_layer_types_other_than_full_attention() -> None:
    model = _tiny_model()
    model.config.layer_types = ["sliding_attention", "full_attention"]
    with pytest.raises(ValueError, match="full-attention"):
        BlockDiffusionSampler(model, _BLOCK, _MASK_ID, 0, _REAL_VOCAB, _NO_EOS)


def test_rejects_mask_id_at_the_model_vocabulary_size() -> None:
    with pytest.raises(ValueError, match="mask_id"):
        BlockDiffusionSampler(_tiny_model(), _BLOCK, 48, 0, _REAL_VOCAB, _NO_EOS)  # vocab size 48


def test_rejects_real_vocab_above_the_model_vocabulary_size() -> None:
    with pytest.raises(ValueError, match="real_vocab"):
        BlockDiffusionSampler(_tiny_model(), _BLOCK, _MASK_ID, 0, 49, _NO_EOS)  # vocab size is 48


def test_rejects_steps_in_dynamic_mode() -> None:
    with pytest.raises(ValueError, match="steps"):
        _sampler(_tiny_model()).generate(_PROMPT, 4, mode="dynamic", steps=2)


def test_rejects_steps_below_one() -> None:
    with pytest.raises(ValueError, match="steps"):
        _sampler(_tiny_model()).generate(_PROMPT, 4, mode="static", steps=0)


@pytest.mark.parametrize("tau", [float("nan"), float("inf"), -0.1])
def test_rejects_tau_that_is_not_finite_or_negative(tau: float) -> None:
    with pytest.raises(ValueError, match="tau"):
        _sampler(_tiny_model()).generate(_PROMPT, 4, mode="dynamic", tau=tau)


def test_rejects_negative_max_new_tokens() -> None:
    with pytest.raises(ValueError, match="max_new_tokens"):
        _sampler(_tiny_model()).generate(_PROMPT, -1)


def test_rejects_prompt_id_outside_the_real_vocabulary() -> None:
    with pytest.raises(ValueError, match="prompt ids"):
        _sampler(_tiny_model()).generate([3, _MASK_ID], 4)


_PADDED_REAL_VOCAB = 40  # M = 40 and ids 41..47 are padding rows, so the vocabulary has 48 ids


def _model_with_padding_on_top() -> transformers.Qwen3ForCausalLM:
    """Return the tiny model with a large logit bias on ids 40 and up.

    The bias makes the mask id and the padding rows win every argmax
    over the full vocabulary, so only the real-vocabulary restriction of
    the sampler keeps them out.
    """
    model = _tiny_model()
    bias = torch.zeros(model.config.vocab_size)
    bias[_PADDED_REAL_VOCAB:] = 1e3

    def add_bias(module: torch.nn.Module, args: tuple, output: torch.Tensor) -> torch.Tensor:
        return output + bias.to(output)

    model.lm_head.register_forward_hook(add_bias)
    return model


def test_padding_rows_are_never_emitted_even_when_they_win_the_argmax() -> None:
    model = _model_with_padding_on_top()
    with torch.no_grad():
        full = model(input_ids=torch.tensor([_PROMPT])).logits[0, -1]
    assert int(full.argmax()) >= _PADDED_REAL_VOCAB  # over all ids, M or padding wins
    sampler = BlockDiffusionSampler(
        model, _BLOCK, _PADDED_REAL_VOCAB, 0, _PADDED_REAL_VOCAB, _NO_EOS
    )
    static = sampler.generate(_PROMPT, 12, mode="static")
    dynamic = sampler.generate(_PROMPT, 12, mode="dynamic", tau=0.0)
    for result in (static, dynamic):
        assert len(result.tokens) == 12
        assert max(result.tokens) < _PADDED_REAL_VOCAB
