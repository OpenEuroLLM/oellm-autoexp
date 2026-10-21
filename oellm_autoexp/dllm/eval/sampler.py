"""KV-cache block-diffusion sampler for Hugging Face Qwen3 models (spec section
14).

Decoding runs block by block over absolute positions, which are multiples of the block size, as in
training. A block starts as mask tokens, and the prompt tokens that fall inside it stay fixed. Each
denoise pass runs the block on top of the cache of the finished clean blocks, then unmasks some of
its positions. The cache is cropped back after every denoise pass. When no mask is left, a
cache-fill pass stores the finished block as clean tokens.

The model is called with explicit 4D boolean masks of shape ``[1, 1, q, kv]``, where True means
that the query may attend to the key, and with explicit ``position_ids``. The model must use
``attn_implementation="sdpa"``, which passes such masks through unchanged.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from transformers import DynamicCache

from oellm_autoexp.dllm.masks import make_bd3_mask_mod

_MODES: tuple[str, ...] = ("static", "dynamic")
_TOKEN_SHIFTS: tuple[int, ...] = (0, 1)


@dataclass
class SampleResult:
    """Generated tokens and pass counts of one
    :meth:`BlockDiffusionSampler.generate` call.

    Attributes:
        tokens: Generated ids after the prompt. They are cut to ``max_new_tokens`` and then after
            the first EOS, which is kept.
        denoise_passes: Number of denoise passes, that is model calls on a block.
        cache_fill_passes: Number of cache-fill passes, one per finished block.
    """

    tokens: list[int]
    denoise_passes: int
    cache_fill_passes: int

    @property
    def tpf_decode(self) -> float:
        """Generated tokens per denoise pass, or 0.0 when there was no denoise
        pass."""
        if self.denoise_passes == 0:
            return 0.0
        return len(self.tokens) / self.denoise_passes

    @property
    def tpf_all(self) -> float:
        """Generated tokens per model pass, counting denoise and cache-fill
        passes.

        Returns:
            The token count divided by the total pass count, or 0.0 when there was no pass.
        """
        passes = self.denoise_passes + self.cache_fill_passes
        if passes == 0:
            return 0.0
        return len(self.tokens) / passes


class BlockDiffusionSampler:
    """Decode from a block-diffusion Qwen3 model with a KV cache (spec section
    14).

    The sampler keeps no state between calls, so one instance can decode many prompts.

    Attributes:
        model: Hugging Face causal LM in eval mode, loaded with ``attn_implementation="sdpa"``.
        block_size: Diffusion block size B.
        mask_id: Mask token id M.
        token_shift: 0 or 1, as in spec section 6.
        real_vocab: Number of real token ids, ``len(tokenizer)``. Ids at or above it are never
            emitted.
        eos_id: End-of-sequence id that stops generation. Use a value that the model never emits,
            such as -1, to disable the stop.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        block_size: int,
        mask_id: int,
        token_shift: int,
        real_vocab: int,
        eos_id: int,
    ) -> None:
        """Check the decoding constants against the model and store them.

        Args:
            model: Hugging Face Qwen3 causal LM with full-attention layers, loaded with
                ``attn_implementation="sdpa"``. Call ``model.eval()`` before decoding.
            block_size: Diffusion block size B, at least 1.
            mask_id: Mask token id M. It must be at least ``real_vocab`` and below the model
                vocabulary size, so the argmax over the real ids never emits it.
            token_shift: 0 predicts each block position from its own logits. 1 predicts it from
                the logits of the position before it (spec section 6).
            real_vocab: Number of real token ids, ``len(tokenizer)``. It must be at least 1 and at
                most the model vocabulary size.
            eos_id: End-of-sequence id. Use a value that the model never emits, such as -1, to
                disable the stop.

        Raises:
            ValueError: If an argument is unsupported, or if the model does not use SDPA attention
                or has layers other than full attention.
        """
        config = model.config
        attn_implementation = getattr(config, "_attn_implementation", None)
        if attn_implementation != "sdpa":
            raise ValueError(
                "the sampler needs attn_implementation='sdpa' so that 4D boolean masks are used "
                f"as given, got {attn_implementation!r}"
            )
        layer_types = getattr(config, "layer_types", None) or []
        if any(layer_type != "full_attention" for layer_type in layer_types):
            raise ValueError("the sampler supports full-attention layers only")
        vocab_size = config.vocab_size
        self.model = model
        self.block_size = _require_int("block_size", block_size, minimum=1)
        self.real_vocab = _require_int("real_vocab", real_vocab, minimum=1)
        if self.real_vocab > vocab_size:
            raise ValueError(
                f"real_vocab must be <= the model vocabulary size {vocab_size}, got {real_vocab}"
            )
        self.mask_id = _require_int("mask_id", mask_id, minimum=self.real_vocab)
        if self.mask_id >= vocab_size:
            raise ValueError(
                f"mask_id must be < the model vocabulary size {vocab_size}, got {mask_id}"
            )
        self.token_shift = _require_int("token_shift", token_shift)
        if self.token_shift not in _TOKEN_SHIFTS:
            raise ValueError(f"token_shift must be 0 or 1, got {token_shift}")
        self.eos_id = _require_int("eos_id", eos_id)

    def generate(
        self,
        prompt_ids: Sequence[int],
        max_new_tokens: int,
        mode: str = "static",
        tau: float = 0.9,
        steps: int | None = None,
    ) -> SampleResult:
        """Generate up to ``max_new_tokens`` tokens after ``prompt_ids``.

        Blocks are decoded whole, so the pass counts include the tokens that are cut from the last
        block. Under ``mode="static"``, each denoise pass unmasks the ``ceil(B / steps)`` most
        confident masked positions. Under ``mode="dynamic"``, each pass unmasks every masked
        position whose confidence is at least ``tau``, and at least the single most confident one.
        The confidence of a position is the largest softmax probability over the real ids, and the
        prediction is the argmax over the same ids. Ties go to the lower position.

        Args:
            prompt_ids: Non-empty prompt token ids, each in ``[0, real_vocab)``.
            max_new_tokens: Maximum number of generated tokens, at least 0.
            mode: ``"static"`` or ``"dynamic"``.
            tau: Confidence threshold of ``"dynamic"``. It must be finite and at least 0. A value
                above 1 unmasks one position per pass.
            steps: Denoise passes per block of ``"static"``, at least 1. It defaults to the block
                size. It must be None for ``"dynamic"``.

        Returns:
            A :class:`SampleResult`. Its tokens are cut to ``max_new_tokens`` and then after the
            first EOS, which is kept.

        Raises:
            ValueError: If the prompt is empty or out of range, or if an argument is unsupported.
        """
        prompt = _check_prompt(prompt_ids, self.real_vocab)
        max_new_tokens = _require_int("max_new_tokens", max_new_tokens, minimum=0)
        if mode not in _MODES:
            raise ValueError(f"mode must be 'static' or 'dynamic', got {mode!r}")
        if not math.isfinite(tau) or tau < 0:
            raise ValueError(f"tau must be finite and >= 0, got {tau}")
        if mode == "dynamic" and steps is not None:
            raise ValueError("steps applies to mode 'static' only")
        unmask_per_pass = 0
        if mode == "static":
            steps_per_block = self.block_size if steps is None else steps
            steps_per_block = _require_int("steps", steps_per_block, minimum=1)
            # ceil(B / steps) positions per denoise pass (spec section 14, step 4)
            unmask_per_pass = (self.block_size + steps_per_block - 1) // steps_per_block
        if max_new_tokens == 0:
            return SampleResult(tokens=[], denoise_passes=0, cache_fill_passes=0)
        with torch.no_grad():
            return self._decode(prompt, max_new_tokens, mode, tau, unmask_per_pass)

    def _decode(
        self,
        prompt: list[int],
        max_new_tokens: int,
        mode: str,
        tau: float,
        unmask_per_pass: int,
    ) -> SampleResult:
        """Run the block loop of spec section 14 for a checked prompt.

        Args:
            prompt: Checked, non-empty prompt token ids.
            max_new_tokens: Positive maximum number of generated tokens.
            mode: ``"static"`` or ``"dynamic"``.
            tau: Confidence threshold of ``"dynamic"``.
            unmask_per_pass: Positions unmasked per denoise pass under ``"static"``.

        Returns:
            The :class:`SampleResult` of this call.
        """
        block_size = self.block_size
        prompt_len = len(prompt)
        first_block = prompt_len // block_size
        cache = DynamicCache(config=self.model.config)
        prev_last: Tensor | None = None
        if first_block > 0:
            prev_last = self._prefill(cache, prompt[: first_block * block_size])
        generated: list[int] = []
        denoise_passes = 0
        cache_fill_passes = 0
        block_index = first_block
        while True:
            start = block_index * block_size
            block = [self.mask_id] * block_size
            masked: list[int] = []
            for j in range(block_size):
                if start + j < prompt_len:
                    block[j] = prompt[start + j]
                else:
                    masked.append(j)
            while masked:
                logits = self._denoise_logits(cache, block, start)
                denoise_passes += 1
                source = _source_logits(logits, self.token_shift, prev_last)
                ids, conf = _greedy_and_confidence(source, self.real_vocab)
                chosen = _select_positions(conf.tolist(), masked, mode, tau, unmask_per_pass)
                ids_list = ids.tolist()
                for j in chosen:
                    block[j] = ids_list[j]
                chosen_set = set(chosen)
                masked = [j for j in masked if j not in chosen_set]
            prev_last = self._fill_block(cache, block, start)
            cache_fill_passes += 1
            fresh = block[max(0, prompt_len - start) :]
            generated.extend(fresh)
            if self.eos_id in fresh or len(generated) >= max_new_tokens:
                break
            block_index += 1
        tokens = generated[:max_new_tokens]
        if self.eos_id in tokens:
            tokens = tokens[: tokens.index(self.eos_id) + 1]
        return SampleResult(
            tokens=tokens,
            denoise_passes=denoise_passes,
            cache_fill_passes=cache_fill_passes,
        )

    def _device(self) -> torch.device:
        """Return the device of the model parameters."""
        return next(self.model.parameters()).device

    def _call(
        self,
        cache: DynamicCache,
        tokens: list[int],
        start: int,
        allowed: Tensor,
        last_only: bool,
    ) -> Tensor:
        """Run the model on ``tokens`` at absolute positions from ``start`` on.

        The call appends the keys of ``tokens`` to ``cache``.

        Args:
            cache: Cache that holds ``start`` tokens before the call.
            tokens: Input ids of the pass.
            start: Absolute position of ``tokens[0]``.
            allowed: Boolean mask of shape ``[1, 1, len(tokens), start + len(tokens)]``, where
                True means that the query may attend to the key.
            last_only: If True, compute the logits of the last position only.

        Returns:
            Logits with shape ``[len(tokens), V]``, or ``[1, V]`` when ``last_only`` is True.
        """
        device = self._device()
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
        position_ids = torch.arange(start, start + len(tokens), device=device).unsqueeze(0)
        extra: dict[str, int] = {"logits_to_keep": 1} if last_only else {}
        out = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=allowed,
            past_key_values=cache,
            use_cache=True,
            **extra,
        )
        return out.logits[0]

    def _prefill(self, cache: DynamicCache, prefix: list[int]) -> Tensor:
        """Fill the cache with the whole prompt blocks under the block-causal
        rule.

        Args:
            cache: Empty cache, filled with the keys of ``prefix``.
            prefix: Prompt ids, a multiple of the block size in length.

        Returns:
            Logits of the last prefix position with shape ``[V]``.
        """
        allowed = _prefix_mask(len(prefix), self.block_size, self._device())
        return self._call(cache, prefix, 0, allowed, last_only=True)[-1]

    def _denoise_logits(self, cache: DynamicCache, block: list[int], start: int) -> Tensor:
        """Run one denoise pass over the block and crop the cache back to
        ``start``.

        The block attends to every cached key and to every key of the block (spec section 14,
        step 3).

        Args:
            cache: Cache that holds the ``start`` tokens of the finished clean blocks.
            block: Current ids of the block. Masked positions hold ``mask_id``.
            start: Absolute position of the first block token, a multiple of the block size.

        Returns:
            Logits of the block with shape ``[B, V]``.
        """
        size = len(block)
        allowed = _full_mask(size, start + size, self._device())
        logits = self._call(cache, block, start, allowed, last_only=False)
        _crop_cache(cache, start)
        return logits

    def _fill_block(self, cache: DynamicCache, block: list[int], start: int) -> Tensor:
        """Store the finished block in the cache as clean tokens (spec section
        14, step 5).

        Args:
            cache: Cache that holds the ``start`` tokens before the block. The keys of the block
                are appended to it.
            block: Final ids of the block.
            start: Absolute position of the first block token.

        Returns:
            Logits of the last block position with shape ``[V]``.
        """
        size = len(block)
        allowed = _full_mask(size, start + size, self._device())
        return self._call(cache, block, start, allowed, last_only=True)[-1]


def _require_int(name: str, value: object, minimum: int | None = None) -> int:
    """Return ``value`` if it is an int, not a bool, that is at least
    ``minimum``.

    Args:
        name: Argument name, used in the error message.
        value: Value to check.
        minimum: Smallest allowed value, or None for no lower bound.

    Returns:
        The value unchanged.

    Raises:
        TypeError: If the value is not an int, or is a bool.
        ValueError: If the value is below ``minimum``.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {value!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _check_prompt(prompt_ids: Sequence[int], real_vocab: int) -> list[int]:
    """Return the prompt as a list of ints after checking it.

    Args:
        prompt_ids: Prompt token ids.
        real_vocab: Number of real token ids.

    Returns:
        The prompt as a list of Python ints.

    Raises:
        ValueError: If the prompt is empty or holds an id outside ``[0, real_vocab)``.
    """
    prompt = [operator.index(token) for token in prompt_ids]
    if not prompt:
        raise ValueError("prompt_ids must not be empty")
    for token in prompt:
        if not 0 <= token < real_vocab:
            raise ValueError(f"prompt ids must be in [0, {real_vocab}), got {token}")
    return prompt


def _prefix_mask(length: int, block_size: int, device: torch.device) -> Tensor:
    """Return the block-causal (c2c) mask of ``length`` clean positions.

    A query in block a may attend to a key in block b when b <= a (spec section 6, clean to clean).
    The rule is the clean half of ``make_bd3_mask_mod``, which is read at indices
    ``length..2*length-1`` of its input.

    Args:
        length: Number of clean positions, a multiple of ``block_size``.
        block_size: Diffusion block size B.
        device: Device of the returned mask.

    Returns:
        Boolean tensor of shape ``[1, 1, length, length]``. True means that the query may attend.
    """
    mask_mod = make_bd3_mask_mod(length, block_size, None)
    clean = torch.arange(length, device=device) + length
    batch = torch.zeros(1, dtype=torch.long, device=device)
    head = torch.zeros(1, dtype=torch.long, device=device)
    allowed = mask_mod(batch, head, clean.view(1, 1, length, 1), clean.view(1, 1, 1, length))
    return allowed.reshape(1, 1, length, length).contiguous()


def _full_mask(q_len: int, kv_len: int, device: torch.device) -> Tensor:
    """Return a mask in which every query may attend to every key.

    Args:
        q_len: Number of query positions.
        kv_len: Number of key positions.
        device: Device of the returned mask.

    Returns:
        Boolean tensor of shape ``[1, 1, q_len, kv_len]``, all True.
    """
    return torch.ones(1, 1, q_len, kv_len, dtype=torch.bool, device=device)


def _crop_cache(cache: DynamicCache, length: int) -> None:
    """Crop the cache back to ``length`` tokens.

    In transformers 5.x, ``DynamicCache.crop`` takes the number of tokens to remove as a
    non-positive integer, so the surplus is passed negated.

    Args:
        cache: Cache to crop.
        length: Number of tokens to keep.

    Raises:
        RuntimeError: If the cache holds fewer than ``length`` tokens.
    """
    surplus = cache.get_seq_length() - length
    if surplus < 0:
        raise RuntimeError(f"cache holds fewer than {length} tokens")
    if surplus > 0:
        cache.crop(-surplus)


def _greedy_and_confidence(logits: Tensor, real_vocab: int) -> tuple[Tensor, Tensor]:
    """Return the greedy id and its confidence over the real ids (spec section
    14, step 4).

    Args:
        logits: Tensor of shape ``[..., V]`` with ``V >= real_vocab``.
        real_vocab: Number of real ids. Only ids below it are considered.

    Returns:
        A tuple ``(ids, conf)``. ``ids`` is a LongTensor of shape ``[...]`` with the argmax over
        the real ids. ``conf`` is a float32 tensor of the same shape with the largest softmax
        probability over the real ids.

    Raises:
        ValueError: If the vocabulary is smaller than ``real_vocab``.
    """
    if logits.shape[-1] < real_vocab:
        raise ValueError(f"logits have {logits.shape[-1]} ids, fewer than real_vocab={real_vocab}")
    restricted = logits[..., :real_vocab].float()
    ids = restricted.argmax(dim=-1)
    conf = torch.softmax(restricted, dim=-1).amax(dim=-1)
    return ids, conf


def _source_logits(block_logits: Tensor, token_shift: int, prev_last: Tensor | None) -> Tensor:
    """Return the logits that predict each block position (spec section 6).

    Under token shift 0, position j is predicted by the logits of position j. Under token shift 1,
    it is predicted by the logits of position j - 1, and position 0 uses ``prev_last``, the logits
    of the position before the block. When ``prev_last`` is None, the block starts at position 0,
    which is a prompt token. That row is never read, because only masked positions are predicted.

    Args:
        block_logits: Logits of the block, shape ``[B, V]``.
        token_shift: 0 or 1.
        prev_last: Logits of the position before the block, shape ``[V]``, or None for block 0.

    Returns:
        Logits of shape ``[B, V]`` that predict the block positions.
    """
    if token_shift == 0:
        return block_logits
    head = block_logits[:1] if prev_last is None else prev_last.unsqueeze(0)
    return torch.cat([head, block_logits[:-1]], dim=0)


def _select_positions(
    conf: list[float],
    masked: list[int],
    mode: str,
    tau: float,
    unmask_per_pass: int,
) -> list[int]:
    """Choose the masked block positions to unmask in one denoise pass.

    Args:
        conf: Confidence of every block position, length B.
        masked: Masked block positions in increasing order. It must not be empty.
        mode: ``"static"`` takes the ``unmask_per_pass`` most confident positions. ``"dynamic"``
            takes every position with confidence at least ``tau``.
        tau: Confidence threshold of ``"dynamic"``.
        unmask_per_pass: Number of positions to unmask under ``"static"``, at least 1.

    Returns:
        The chosen positions. There is at least one. Ties go to the lower position.
    """
    by_confidence = sorted(masked, key=lambda j: (-conf[j], j))
    if mode == "static":
        return by_confidence[:unmask_per_pass]
    chosen = [j for j in masked if conf[j] >= tau]
    return chosen if chosen else by_confidence[:1]
