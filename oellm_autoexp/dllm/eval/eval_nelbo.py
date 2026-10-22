"""Offline metric M1 for block-diffusion and AR models (spec section 9, D-035).

M1 is the validation NELBO of a block-diffusion model, or the exact NLL of an AR model, in bits per
token over the scored positions of a fixed set of held-out sequences. The block-diffusion value uses
the deterministic estimator of spec section 9. For each block it draws ``l_k`` masked positions and
weights their cross entropy by ``B / l_k``. The model reads ``[xt | x0]`` under the dense mask of
spec section 5. For the nld objective (clean attention ``causal``, spec section 15) the clean
stream is token-causal.

The scored positions are the same for every model (D-035). Position ``i`` is scored when ``i >= 1``
and it is not a document start, that is ``doc_ids[i] == doc_ids[i - 1]``. Document ids are computed
even when the document mask is off, so the scored set depends only on the data.

The sequences are the first windows of a Megatron ``.bin/.idx`` dataset, see :func:`load_windows`.
Run the command in the GPU environment, for example::

    python -m oellm_autoexp.dllm.eval.eval_nelbo --model DIR --data-prefix PREFIX \\
        --objective bd3 --out m1.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    _REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from oellm_autoexp.dllm.data_prep import read_idx_header
from oellm_autoexp.dllm.masks import (
    CLEAN_ATTENTIONS,
    compute_doc_ids,
    make_ar_mask_mod,
    make_bd3_mask_mod,
    materialize_mask,
    resolve_clean_attention,
)
from oellm_autoexp.dllm.noise import sample_validation_noise, validation_nelbo
from oellm_autoexp.dllm.targets import ar_sources, prediction_sources

_OBJECTIVES: tuple[str, ...] = ("bd3", "ar")
_DTYPES: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}
_INTEGER_DTYPES: tuple[torch.dtype, ...] = (
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)
_LN2 = math.log(2.0)


def m1_scored_positions(x0: Tensor, eod_id: int) -> Tensor:
    """Return the M1 scored positions of every row (spec section 9, D-035).

    Position ``i`` is scored when ``i >= 1`` and ``doc_ids[i] == doc_ids[i - 1]``. Since
    ``doc_ids[i] - doc_ids[i - 1]`` is 1 exactly when ``x0[i - 1]`` is the EOD token, this excludes
    position 0 and every position that starts a document.

    Args:
        x0: Clean token ids with shape ``[b, L]``.
        eod_id: Token id of the end-of-document token.

    Returns:
        A BoolTensor ``[b, L]``. True marks a scored position.

    Raises:
        ValueError: If ``x0`` is not 2-D.
    """
    doc_ids = compute_doc_ids(x0, eod_id)
    scored = torch.zeros(x0.shape, dtype=torch.bool, device=x0.device)
    scored[:, 1:] = doc_ids[:, 1:] == doc_ids[:, :-1]
    return scored


def load_windows(prefix: str, num_sequences: int, seq_len: int) -> Tensor:
    """Read the first windows of a Megatron dataset as token ids.

    The windows are consecutive and do not overlap. They start at the first token of the ``.bin``
    file, and document boundaries are ignored. The element type comes from the ``.idx`` header.

    Args:
        prefix: Path prefix without the ``.bin`` or ``.idx`` suffix.
        num_sequences: Number of windows, at least 1.
        seq_len: Window length L, at least 1.

    Returns:
        A LongTensor ``[num_sequences, seq_len]``.

    Raises:
        ValueError: If a size is below 1, the element type is not supported, or the ``.bin`` file
            holds fewer than ``num_sequences * seq_len`` tokens.
    """
    if num_sequences < 1 or seq_len < 1:
        raise ValueError(f"num_sequences and seq_len must be >= 1, got {num_sequences}, {seq_len}")
    header = read_idx_header(Path(prefix + ".idx"))
    if header["dtype_size"] == 0:
        raise ValueError(f"{prefix}.idx has an unsupported element type {header['dtype']}")
    bin_path = prefix + ".bin"
    dtype = np.dtype(header["dtype"])
    available = os.path.getsize(bin_path) // dtype.itemsize
    needed = num_sequences * seq_len
    if available < needed:
        raise ValueError(
            f"{bin_path} holds {available} tokens, but {num_sequences} windows of {seq_len} "
            f"tokens need {needed}"
        )
    tokens = np.memmap(bin_path, dtype=dtype, mode="r", shape=(needed,))
    windows = np.array(tokens, dtype=np.int64).reshape(num_sequences, seq_len)
    return torch.from_numpy(windows)


def _from_flag_or_config(flag: int | None, config: Any, name: str) -> int | None:
    """Return the command-line value if it is given, else the config attribute,
    else None.

    Args:
        flag: Value of the command-line flag, or None when the flag is absent.
        config: Hugging Face model config.
        name: Attribute name in the config, such as ``dllm_block_size``.

    Returns:
        The resolved value, or None.
    """
    return flag if flag is not None else getattr(config, name, None)


def _resolve_clean_attention(flag: str | None, config: Any) -> str:
    """Return the clean attention of a bd3 model: the flag, else the config field, else block.

    A config without ``dllm_clean_attention`` is a block-causal model (spec section 15).

    Args:
        flag: Value of ``--clean-attention``, or None when the flag is absent.
        config: Hugging Face model config.

    Returns:
        ``"block"`` or ``"causal"``.

    Raises:
        ValueError: If the resolved value is not supported.
    """
    value = flag if flag is not None else getattr(config, "dllm_clean_attention", None)
    return resolve_clean_attention(value)


def _check_token_batch(x0: Tensor) -> None:
    """Raise ValueError unless ``x0`` is a 2-D integer tensor ``[b, L]``.

    Args:
        x0: Clean token ids.

    Raises:
        ValueError: If the shape or the dtype is unsupported.
    """
    if x0.dim() != 2:
        raise ValueError(f"x0 must have shape [b, L], got {tuple(x0.shape)}")
    if x0.dtype not in _INTEGER_DTYPES:
        raise ValueError(f"x0 must have an integer dtype, got {x0.dtype}")


def _token_cross_entropy(logits: Tensor, src: Tensor, x0: Tensor) -> Tensor:
    """Return the float32 cross entropy of ``x0`` against the logits at
    ``src``.

    Args:
        logits: Model logits with shape ``[b, S, V]``.
        src: LongTensor ``[L]`` of indices into dim 1 of ``logits``. Entry ``i`` selects the
            logits that predict ``x0[:, i]``.
        x0: Clean token ids with shape ``[b, L]``.

    Returns:
        A float32 tensor ``[b, L]``.
    """
    batch, length = x0.shape
    picked = logits.index_select(1, src).float()
    vocab = picked.shape[-1]
    ce = F.cross_entropy(
        picked.reshape(batch * length, vocab), x0.long().reshape(-1), reduction="none"
    )
    return ce.reshape(batch, length)


@torch.no_grad()
def bd3_nelbo_batch(
    model: nn.Module,
    x0: Tensor,
    block_size: int,
    token_shift: int,
    mask_id: int,
    eod_id: int,
    doc_mask: bool,
    generator: torch.Generator,
    clean_causal: bool = False,
) -> tuple[Tensor, Tensor]:
    """Compute the validation NELBO sum of one batch (spec section 9, D-035).

    The noise comes from :func:`sample_validation_noise`, which draws from ``generator``. The model
    reads ``cat([xt, x0])`` with positions ``0..L-1`` for both copies, under the dense mask of
    :func:`make_bd3_mask_mod`. The cross entropy of ``x0[i]`` is read from the logits at the input
    index given by :func:`prediction_sources`.

    Args:
        model: Hugging Face causal LM in eval mode, loaded with ``attn_implementation="sdpa"``.
        x0: Clean token ids with shape ``[b, L]``. ``L`` must be a multiple of ``block_size``.
        block_size: Diffusion block size B.
        token_shift: 0 or 1, as in spec section 6.
        mask_id: Mask token id M.
        eod_id: Token id of the end-of-document token.
        doc_mask: Whether the attention mask uses the document ids.
        generator: Generator on the device of ``x0``. Its state advances with the noise draws.
        clean_causal: True for the token-causal clean stream of the nld objective (spec section
            15). False for the block-causal clean stream of bd3 (spec section 5).

    Returns:
        A tuple ``(nelbo_sum, scored)`` of 0-dim float32 tensors. ``nelbo_sum`` is the sum of
        ``B / l_k * ce`` over the masked scored positions, and ``scored`` is the number of scored
        positions.

    Raises:
        ValueError: If an input is unsupported.
        TypeError: If ``clean_causal`` is not a bool.
    """
    _check_token_batch(x0)
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1:
        raise ValueError(f"block_size must be an integer >= 1, got {block_size!r}")
    if not isinstance(clean_causal, bool):
        raise TypeError(f"clean_causal must be a bool, got {clean_causal!r}")
    batch, length = x0.shape
    device = x0.device
    src, has_src = prediction_sources(length, block_size, token_shift, device=device)
    doc_ids = compute_doc_ids(x0, eod_id)
    attention_doc_ids = doc_ids if doc_mask else None
    xt, masked, block_weight = sample_validation_noise(x0, block_size, mask_id, generator)
    input_ids = torch.cat([xt, x0], dim=1)
    positions = torch.arange(length, device=device)
    position_ids = torch.cat([positions, positions]).unsqueeze(0).expand(batch, -1)
    attention_mask = materialize_mask(
        make_bd3_mask_mod(length, block_size, attention_doc_ids, clean_causal=clean_causal),
        batch,
        2 * length,
        2 * length,
        device,
    ).unsqueeze(1)
    logits = model(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        use_cache=False,
    ).logits
    ce = _token_cross_entropy(logits, src, x0)
    scored = m1_scored_positions(x0, eod_id)
    if token_shift == 1:
        scored = scored & has_src
    return validation_nelbo(ce, masked, block_weight, scored)


@torch.no_grad()
def ar_nll_batch(
    model: nn.Module,
    x0: Tensor,
    eod_id: int,
    doc_mask: bool,
) -> tuple[Tensor, Tensor]:
    """Compute the exact NLL sum of one batch of an AR model (spec section 9,
    D-035).

    The model reads ``x0`` with positions ``0..L-1`` under the causal mask of
    :func:`make_ar_mask_mod`. The output at index ``i - 1`` predicts ``x0[i]``. Position 0 has no
    prediction and contributes 0.

    Args:
        model: Hugging Face causal LM in eval mode, loaded with ``attn_implementation="sdpa"``.
        x0: Clean token ids with shape ``[b, L]``.
        eod_id: Token id of the end-of-document token.
        doc_mask: Whether the attention mask uses the document ids.

    Returns:
        A tuple ``(nll_sum, scored)`` of 0-dim float32 tensors. ``nll_sum`` is the sum of the
        next-token cross entropy over the scored positions, and ``scored`` is their number.

    Raises:
        ValueError: If an input is unsupported.
    """
    _check_token_batch(x0)
    batch, length = x0.shape
    device = x0.device
    src, has_src = ar_sources(length, device=device)
    doc_ids = compute_doc_ids(x0, eod_id)
    attention_doc_ids = doc_ids if doc_mask else None
    attention_mask = materialize_mask(
        make_ar_mask_mod(length, attention_doc_ids), batch, length, length, device
    ).unsqueeze(1)
    position_ids = torch.arange(length, device=device).unsqueeze(0).expand(batch, -1)
    logits = model(
        input_ids=x0,
        position_ids=position_ids,
        attention_mask=attention_mask,
        use_cache=False,
    ).logits
    zero = torch.zeros((), dtype=torch.float32, device=device)
    ce = torch.where(has_src, _token_cross_entropy(logits, src, x0), zero)
    scored = m1_scored_positions(x0, eod_id)
    nll_sum = torch.where(scored, ce, zero).sum().to(torch.float32)
    return nll_sum, scored.sum().to(torch.float32)


def evaluate(
    model: nn.Module,
    windows: Tensor,
    objective: str,
    block_size: int | None,
    token_shift: int | None,
    mask_id: int | None,
    eod_id: int,
    doc_mask: bool,
    seed: int,
    batch_size: int,
    device: str | torch.device,
    clean_attention: str = "block",
) -> dict[str, Any]:
    """Evaluate M1 over a set of windows and return the value with its
    settings.

    One generator seeded with ``seed`` supplies the noise of the whole run, and the batches are
    taken in order. For ``objective="ar"``, the arguments ``block_size``, ``token_shift``,
    ``mask_id``, and ``clean_attention`` are ignored, and the result reports ``block_size``,
    ``token_shift``, and ``clean_attention`` as None.

    Args:
        model: Hugging Face causal LM in eval mode, on ``device``.
        windows: Integer token ids with shape ``[N, L]``. They may be on any device.
        objective: ``"bd3"`` for the validation NELBO or ``"ar"`` for the exact NLL.
        block_size: Diffusion block size B (bd3 only).
        token_shift: 0 or 1 (bd3 only).
        mask_id: Mask token id M (bd3 only).
        eod_id: Token id of the end-of-document token.
        doc_mask: Whether the attention mask uses the document ids.
        seed: Seed of the noise generator.
        batch_size: Number of sequences per forward pass, at least 1.
        device: Device for the batches and for the generator.
        clean_attention: ``"block"`` (bd3, spec section 5) or ``"causal"`` (nld, spec section 15).
            The value must be one of these for both objectives, and bd3 uses it.

    Returns:
        A dict with the keys ``objective``, ``block_size``, ``token_shift``, ``clean_attention``,
        ``doc_mask``, ``num_sequences``, ``seq_len``, ``scored_positions``, ``nats_sum``,
        ``bits_per_token``, and ``seed``. ``bits_per_token`` equals
        ``nats_sum / (scored_positions * ln 2)``.

    Raises:
        ValueError: If an input is unsupported, if no position is scored, or if the sum is not
            finite.
    """
    if objective not in _OBJECTIVES:
        raise ValueError(f"objective must be one of {_OBJECTIVES}, got {objective!r}")
    if windows.dim() != 2 or windows.dtype not in _INTEGER_DTYPES:
        raise ValueError(
            f"windows must be an integer tensor [N, L], got {windows.dtype} "
            f"with shape {tuple(windows.shape)}"
        )
    num_sequences, seq_len = windows.shape
    if num_sequences < 1 or seq_len < 1:
        raise ValueError(f"windows must not be empty, got shape {tuple(windows.shape)}")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError(f"batch_size must be an integer >= 1, got {batch_size!r}")
    clean_attention = resolve_clean_attention(clean_attention)
    dev = torch.device(device)
    generator = torch.Generator(device=dev).manual_seed(seed)
    nats_sum = 0.0
    scored_positions = 0
    for start in range(0, num_sequences, batch_size):
        x0 = windows[start : start + batch_size].to(dev)
        if objective == "bd3":
            batch_nats, batch_scored = bd3_nelbo_batch(
                model,
                x0,
                block_size,
                token_shift,
                mask_id,
                eod_id,
                doc_mask,
                generator,
                clean_causal=clean_attention == "causal",
            )
        else:
            batch_nats, batch_scored = ar_nll_batch(model, x0, eod_id, doc_mask)
        nats_sum += float(batch_nats.item())
        scored_positions += int(batch_scored.item())
    if scored_positions == 0:
        raise ValueError(
            f"no position is scored in {num_sequences} windows of length {seq_len}; "
            "M1 needs a length of at least 2 and tokens that are not EOD"
        )
    if not math.isfinite(nats_sum):
        raise ValueError(f"the NLL sum is not finite: {nats_sum}")
    bits_per_token = nats_sum / (scored_positions * _LN2)
    is_bd3 = objective == "bd3"
    return {
        "objective": objective,
        "block_size": block_size if is_bd3 else None,
        "token_shift": token_shift if is_bd3 else None,
        "clean_attention": clean_attention if is_bd3 else None,
        "doc_mask": bool(doc_mask),
        "num_sequences": num_sequences,
        "seq_len": seq_len,
        "scored_positions": scored_positions,
        "nats_sum": nats_sum,
        "bits_per_token": bits_per_token,
        "seed": seed,
    }


def _default_device() -> str:
    """Return ``"cuda"`` when a CUDA device is visible, else ``"cpu"``."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser of :func:`main`.

    Returns:
        The configured ``argparse.ArgumentParser``.
    """
    parser = argparse.ArgumentParser(description="Offline M1: validation NELBO or exact NLL.")
    parser.add_argument("--model", required=True, help="model directory or Hugging Face id")
    parser.add_argument(
        "--data-prefix", required=True, help="Megatron .bin/.idx prefix of the held-out split"
    )
    parser.add_argument("--num-sequences", type=int, default=1000, help="number of windows")
    parser.add_argument("--seq-len", type=int, default=2048, help="window length L")
    parser.add_argument("--objective", required=True, choices=_OBJECTIVES, help="bd3 or ar")
    parser.add_argument(
        "--block-size", type=int, default=None, help="block size B (default dllm_block_size)"
    )
    parser.add_argument(
        "--token-shift",
        type=int,
        choices=(0, 1),
        default=None,
        help="token shift (default dllm_token_shift, else 0)",
    )
    parser.add_argument(
        "--mask-token-id",
        type=int,
        default=None,
        help="mask id M (default dllm_mask_token_id, else len(tokenizer))",
    )
    parser.add_argument(
        "--clean-attention",
        choices=CLEAN_ATTENTIONS,
        default=None,
        help="clean stream of bd3: block or causal (default dllm_clean_attention, else block)",
    )
    parser.add_argument(
        "--eod-id", type=int, default=None, help="EOD id (default tokenizer.eos_token_id)"
    )
    parser.add_argument(
        "--doc-mask", type=int, choices=(0, 1), default=1, help="document attention mask"
    )
    parser.add_argument("--seed", type=int, default=1234, help="seed of the noise generator")
    parser.add_argument("--batch-size", type=int, default=1, help="sequences per forward pass")
    parser.add_argument("--dtype", choices=tuple(_DTYPES), default="bfloat16", help="weights dtype")
    parser.add_argument(
        "--device",
        default=_default_device(),
        help="torch device (default: cuda if available, else cpu)",
    )
    parser.add_argument("--out", type=Path, required=True, help="write the JSON result here")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the offline M1 evaluation from the command line.

    The JSON result is written to ``--out`` and printed. It holds the model and the data prefix,
    the fields of :func:`evaluate`, and the resolved ``eod_id``, ``mask_token_id``, and ``dtype``.
    The block size, token shift, mask id, and clean attention default to the ``dllm_*`` fields of
    the model config. The clean attention falls back to ``block`` when the config has no
    ``dllm_clean_attention`` field. The mask id falls back to ``len(tokenizer)``, and the EOD id
    defaults to ``tokenizer.eos_token_id``. The tokenizer is loaded only when one of these is
    needed.

    Args:
        argv: Arguments without the program name, or None to read ``sys.argv``.

    Returns:
        Exit code 0.

    Raises:
        ValueError: If a setting is unsupported, or a value has no flag and no default, for example
            a bd3 run whose config has no ``dllm_block_size`` and no ``--block-size`` is given.
    """
    import transformers

    args = _build_parser().parse_args(argv)
    windows = load_windows(args.data_prefix, args.num_sequences, args.seq_len)
    config = transformers.AutoConfig.from_pretrained(args.model)
    block_size: int | None = None
    token_shift: int | None = None
    mask_id: int | None = None
    # The clean attention of the ar objective is not used, so ar takes the default here.
    clean_attention = "block"
    if args.objective == "bd3":
        block_size = _from_flag_or_config(args.block_size, config, "dllm_block_size")
        if block_size is None:
            raise ValueError("bd3 needs --block-size or dllm_block_size in config.json")
        token_shift = _from_flag_or_config(args.token_shift, config, "dllm_token_shift")
        token_shift = 0 if token_shift is None else token_shift
        mask_id = _from_flag_or_config(args.mask_token_id, config, "dllm_mask_token_id")
        clean_attention = _resolve_clean_attention(args.clean_attention, config)
    tokenizer = None
    if args.eod_id is None or (args.objective == "bd3" and mask_id is None):
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    eod_id = args.eod_id if args.eod_id is not None else tokenizer.eos_token_id
    if eod_id is None:
        raise ValueError("the tokenizer has no eos_token_id; pass --eod-id")
    if args.objective == "bd3" and mask_id is None:
        mask_id = len(tokenizer)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, dtype=_DTYPES[args.dtype], attn_implementation="sdpa"
    )
    model = model.to(args.device).eval()
    result = evaluate(
        model=model,
        windows=windows,
        objective=args.objective,
        block_size=block_size,
        token_shift=token_shift,
        mask_id=mask_id,
        eod_id=eod_id,
        doc_mask=bool(args.doc_mask),
        seed=args.seed,
        batch_size=args.batch_size,
        device=args.device,
        clean_attention=clean_attention,
    )
    payload = {
        "model": args.model,
        "data_prefix": args.data_prefix,
        **result,
        "eod_id": eod_id,
        "mask_token_id": mask_id,
        "dtype": args.dtype,
    }
    text = json.dumps(payload, indent=2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
