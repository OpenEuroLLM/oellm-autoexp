"""GSM8K 4-shot evaluation of AR greedy decoding and block-diffusion sampling
(metrics M2, M3).

Problems come from the local GSM8K ``main`` parquet files under ``$HF_HOME``. Each prompt is a
fixed few-shot prefix drawn from the training split, then ``Question: <q>`` and ``Answer:``.
The modes are:

- ``ar``: greedy decoding with a KV cache, one forward pass per generated token (TPF 1.0).
- ``static``: the block-diffusion sampler (spec section 14) with ``steps = block_size``.
- ``dynamic``: the block-diffusion sampler that unmasks positions with confidence >= ``tau``.

M2 is the accuracy in percent. M3 is the mean tokens per forward pass, with TPF_decode and TPF_all
as defined in spec section 14.

The block-diffusion modes read ``dllm_clean_attention`` from the model config. The value
``"causal"`` (objective nld, spec section 15) makes the sampler use token-causal attention among
the clean tokens. The value ``"block"`` (objective bd3, spec section 14) is also the value when the
config has no such field.

Run it in the GPU environment, for example:

    python oellm_autoexp/dllm/eval/eval_gsm8k.py --model DIR --mode static --out result.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    _REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

import torch
from torch import nn

from oellm_autoexp.dllm.masks import CLEAN_ATTENTIONS, resolve_clean_attention

_MODES = ("ar", "static", "dynamic")
_SPLITS = ("train", "test")
_DATASET_DIR = "datasets--openai--gsm8k"
_QUESTION_PREFIX = "Question:"
# An optional sign, digits with optional thousands commas, and an optional decimal part.
_NUMBER = r"-?\d+(?:,\d{3})*(?:\.\d+)?"
_NUMBER_RE = re.compile(_NUMBER)
_HASH_NUMBER_RE = re.compile(r"####\s*(" + _NUMBER + ")")
_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def _normalize_number(raw: str) -> str:
    """Remove the commas, a trailing period, and a ``.0`` suffix from a matched
    number.

    Args:
        raw: A number as matched in text, such as ``"1,234"`` or ``"3.0"``.

    Returns:
        The normalized number, such as ``"1234"`` or ``"3"``.
    """
    text = raw.replace(",", "").removesuffix(".")
    return text.removesuffix(".0")


def extract_answer(text: str) -> str | None:
    """Extract the predicted final number from generated text.

    The text is cut at the first line that starts with ``Question:``, where the model starts the
    next few-shot example. The answer is the number after the last ``####`` that is followed by a
    number. If there is none, the answer is the last number in the cut text.

    Args:
        text: Generated text.

    Returns:
        The normalized number as a string, or None if the cut text has no number.
    """
    kept: list[str] = []
    for line in text.split("\n"):
        if line.startswith(_QUESTION_PREFIX):
            break
        kept.append(line)
    cut = "\n".join(kept)
    hashed = _HASH_NUMBER_RE.findall(cut)
    if hashed:
        return _normalize_number(hashed[-1])
    numbers = _NUMBER_RE.findall(cut)
    if numbers:
        return _normalize_number(numbers[-1])
    return None


def gold_answer(answer: str) -> str:
    """Return the normalized gold number of a GSM8K answer.

    Args:
        answer: Full GSM8K answer text that ends with ``#### <number>``.

    Returns:
        The normalized number after the last ``####``.

    Raises:
        ValueError: If the answer has no ``#### <number>``.
    """
    hashed = _HASH_NUMBER_RE.findall(answer)
    if not hashed:
        raise ValueError(f"GSM8K answer has no '#### <number>' line: {answer!r}")
    return _normalize_number(hashed[-1])


def build_fewshot_prefix(train: Sequence[dict[str, str]], shots: int = 4, seed: int = 1234) -> str:
    """Build the few-shot prefix from a seeded draw of training items.

    Args:
        train: Training items with ``question`` and ``answer`` keys.
        shots: Number of examples, from 0 to ``len(train)``.
        seed: Seed of the ``random.Random`` instance that draws the examples.

    Returns:
        For each drawn item in draw order: ``Question: <q>``, a newline, ``Answer: <a>`` (the full
        answer, including its ``####`` line), and two newlines.

    Raises:
        ValueError: If ``shots`` is outside ``[0, len(train)]``.
    """
    if not 0 <= shots <= len(train):
        raise ValueError(f"shots must be in [0, {len(train)}], got {shots}")
    chosen = random.Random(seed).sample(train, shots)
    return "".join(f"Question: {item['question']}\nAnswer: {item['answer']}\n\n" for item in chosen)


def format_prompt(prefix: str, question: str) -> str:
    """Return the prompt of one problem.

    Args:
        prefix: Few-shot prefix from ``build_fewshot_prefix``.
        question: GSM8K question text.

    Returns:
        The prefix, then ``Question: `` and the question, a newline, and ``Answer:``.
    """
    return prefix + "Question: " + question + "\n" + "Answer:"


def load_gsm8k(split: str, hf_home: str | None = None) -> list[dict[str, str]]:
    """Load a GSM8K split (``main`` config) from the local Hugging Face cache.

    No network access is used. The parquet file is read with pyarrow.

    Args:
        split: ``"train"`` or ``"test"``.
        hf_home: Hugging Face home directory. Defaults to the ``HF_HOME`` environment variable.

    Returns:
        Items with ``question`` and ``answer`` keys, in file order.

    Raises:
        ValueError: If the split is unknown, or ``hf_home`` is None and ``HF_HOME`` is not set,
            or the cache holds several snapshots of the split.
        FileNotFoundError: If the cache has no snapshot of the split.
    """
    import pyarrow.parquet as pq

    if split not in _SPLITS:
        raise ValueError(f"split must be one of {_SPLITS}, got {split!r}")
    if hf_home is None:
        hf_home = os.environ.get("HF_HOME")
        if not hf_home:
            raise ValueError("hf_home is None and the HF_HOME environment variable is not set")
    pattern = f"hub/{_DATASET_DIR}/snapshots/*/main/{split}-00000-of-00001.parquet"
    matches = sorted(Path(hf_home).glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no GSM8K {split} parquet file matches {pattern} in {hf_home}")
    if len(matches) > 1:
        raise ValueError(f"several GSM8K {split} parquet files match: {[str(m) for m in matches]}")
    return pq.read_table(matches[0], columns=["question", "answer"]).to_pylist()


def generate_ar(
    model: nn.Module,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    real_vocab: int,
    eos_id: int,
) -> tuple[list[int], int]:
    """Decode greedily, one token at a time, with a KV cache (the AR baseline
    of M2 and M3).

    The prompt prefill gives the first token. Every later token costs one single-token forward
    pass. The pass count therefore equals the number of generated tokens, so AR has TPF 1.0.

    Args:
        model: Causal LM whose forward returns ``logits`` and ``past_key_values``.
        prompt_ids: Non-empty list of prompt token ids.
        max_new_tokens: Maximum number of generated tokens, at least 1.
        real_vocab: Only ids below this value can be generated.
        eos_id: Stop token, in ``[0, real_vocab)``. It is kept in the output.

    Returns:
        The generated token ids, which end with ``eos_id`` if it was produced, and the number of
        forward passes.

    Raises:
        ValueError: If the prompt is empty, ``max_new_tokens`` is below 1, ``eos_id`` is not in
            ``[0, real_vocab)``, or ``real_vocab`` exceeds the model's vocabulary.
    """
    if len(prompt_ids) == 0:
        raise ValueError("prompt_ids must not be empty")
    if max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")
    if not 0 <= eos_id < real_vocab:
        raise ValueError(f"eos_id must be in [0, real_vocab), got {eos_id}")
    device = next(model.parameters()).device
    generated: list[int] = []
    passes = 0
    past: Any = None
    step_ids = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    with torch.no_grad():
        while len(generated) < max_new_tokens:
            out = model(input_ids=step_ids, past_key_values=past, use_cache=True, logits_to_keep=1)
            passes += 1
            if real_vocab > out.logits.shape[-1]:
                raise ValueError(
                    f"real_vocab={real_vocab} exceeds the model vocabulary {out.logits.shape[-1]}"
                )
            past = out.past_key_values
            token = int(out.logits[0, -1, :real_vocab].argmax().item())
            generated.append(token)
            if token == eos_id:
                break
            step_ids = torch.tensor([[token]], dtype=torch.long, device=device)
    return generated, passes


def evaluate(
    model: nn.Module,
    tokenizer: Any,
    problems: Sequence[dict[str, str]],
    prefix: str,
    mode: str,
    block_size: int | None,
    token_shift: int | None,
    mask_id: int | None,
    max_new_tokens: int,
    real_vocab: int,
    eos_id: int,
    tau: float = 0.9,
    clean_attention: str = "block",
) -> dict[str, Any]:
    """Measure GSM8K accuracy (M2) and tokens per forward pass (M3) for one
    decoding mode.

    Args:
        model: Causal LM in the Hugging Face layout, or the converted block-diffusion model.
        tokenizer: Object with ``encode(text, add_special_tokens=...)`` and ``decode(ids, ...)``.
        problems: GSM8K items with ``question`` and ``answer`` keys, evaluated in order.
        prefix: Few-shot prefix from ``build_fewshot_prefix``.
        mode: ``"ar"``, ``"static"`` (``steps = block_size``), or ``"dynamic"`` (threshold tau).
        block_size: Diffusion block size B. Required for ``static`` and ``dynamic``.
        token_shift: Token shift, 0 or 1. Required for ``static`` and ``dynamic``.
        mask_id: Mask token id M. Required for ``static`` and ``dynamic``.
        max_new_tokens: Maximum number of generated tokens per problem, at least 1.
        real_vocab: Number of real token ids. Only ids below it are generated.
        eos_id: End-of-sequence id in ``[0, real_vocab)``. Generation stops at it.
        tau: Confidence threshold of ``dynamic`` mode, in [0, 1].
        clean_attention: Clean-token attention of the block-diffusion modes, ``"block"`` (spec
            section 14) or ``"causal"`` (objective nld, spec section 15). ``ar`` mode ignores it.

    Returns:
        A dict with ``accuracy`` (percent), ``n``, ``correct``, ``mode``, ``mean_tpf_decode``,
        ``mean_tpf_all`` (the means of the per-problem values), and ``records``. Each record has
        ``index``, ``prediction``, ``gold``, ``correct``, and ``completion`` (the decoded text up
        to the first EOS).

    Raises:
        ValueError: If the mode, a setting, or the problem list is not supported.
    """
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
    if max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")
    if not 0 <= eos_id < real_vocab:
        raise ValueError(f"eos_id must be in [0, real_vocab), got {eos_id}")
    if not problems:
        raise ValueError("problems must not be empty")
    if mode != "ar":
        if block_size is None or block_size < 1:
            raise ValueError(f"{mode} mode needs block_size >= 1, got {block_size}")
        if token_shift not in (0, 1):
            raise ValueError(f"{mode} mode needs token_shift in (0, 1), got {token_shift}")
        if mask_id is None:
            raise ValueError(f"{mode} mode needs mask_id")
    if mode == "dynamic" and not 0.0 <= tau <= 1.0:
        raise ValueError(f"tau must be in [0, 1], got {tau}")
    if clean_attention not in CLEAN_ATTENTIONS:
        raise ValueError(f"clean_attention must be 'block' or 'causal', got {clean_attention!r}")

    sampler: Any = None
    if mode != "ar":
        from oellm_autoexp.dllm.eval.sampler import BlockDiffusionSampler

        sampler = BlockDiffusionSampler(
            model=model,
            block_size=block_size,
            mask_id=mask_id,
            token_shift=token_shift,
            real_vocab=real_vocab,
            eos_id=eos_id,
            clean_causal=clean_attention == "causal",
        )

    records: list[dict[str, Any]] = []
    tpf_decode: list[float] = []
    tpf_all: list[float] = []
    with torch.no_grad():
        for index, problem in enumerate(problems):
            gold = gold_answer(problem["answer"])
            prompt = format_prompt(prefix, problem["question"])
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            if mode == "ar":
                generated, passes = generate_ar(
                    model, prompt_ids, max_new_tokens, real_vocab, eos_id
                )
                decode_tpf = len(generated) / passes
                all_tpf = decode_tpf
            else:
                if mode == "static":
                    result = sampler.generate(
                        prompt_ids, max_new_tokens, mode="static", steps=block_size
                    )
                else:
                    result = sampler.generate(prompt_ids, max_new_tokens, mode="dynamic", tau=tau)
                generated = [int(token) for token in result.tokens]
                decode_tpf = float(result.tpf_decode)
                all_tpf = float(result.tpf_all)
            cut = generated.index(eos_id) if eos_id in generated else len(generated)
            text = tokenizer.decode(generated[:cut], skip_special_tokens=True)
            prediction = extract_answer(text)
            correct = prediction is not None and prediction == gold
            records.append(
                {
                    "index": index,
                    "prediction": prediction,
                    "gold": gold,
                    "correct": correct,
                    "completion": text,
                }
            )
            tpf_decode.append(decode_tpf)
            tpf_all.append(all_tpf)

    n = len(records)
    n_correct = sum(1 for record in records if record["correct"])
    return {
        "accuracy": 100.0 * n_correct / n,
        "n": n,
        "correct": n_correct,
        "mode": mode,
        "mean_tpf_decode": sum(tpf_decode) / n,
        "mean_tpf_all": sum(tpf_all) / n,
        "records": records,
    }


def _resolve_int(flag: int | None, config: Any, attr: str) -> int:
    """Return the command-line value, or the config attribute when the flag is
    None.

    Args:
        flag: Value of the command-line flag, or None.
        config: Model config. Keys of config.json that the config class does not know are
            attributes.
        attr: Attribute name, such as ``dllm_block_size``.

    Returns:
        The integer value.

    Raises:
        ValueError: If neither the flag nor the config provides a value.
    """
    if flag is not None:
        return flag
    value = getattr(config, attr, None)
    if value is None:
        raise ValueError(f"{attr} is missing from config.json; pass the matching flag")
    return int(value)


def _resolve_clean_attention(config: Any) -> str:
    """Return the clean-token attention of a block-diffusion model from its
    config.

    Args:
        config: Model config. The field ``dllm_clean_attention`` is ``"causal"`` for objective nld
            (spec section 15) and ``"block"`` for objective bd3 (spec section 14). A config
            without the field, or with null, means ``"block"``.

    Returns:
        ``"block"`` or ``"causal"``.

    Raises:
        ValueError: If the field holds any other value.
    """
    return resolve_clean_attention(getattr(config, "dllm_clean_attention", None))


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The argument parser.
    """
    parser = argparse.ArgumentParser(description="GSM8K 4-shot evaluation (metrics M2 and M3).")
    parser.add_argument("--model", required=True, help="model directory or Hugging Face id")
    parser.add_argument("--mode", choices=_MODES, default="ar", help="decoding mode")
    parser.add_argument("--tau", type=float, default=0.9, help="confidence threshold (dynamic)")
    parser.add_argument("--n", type=int, default=250, help="number of leading test problems")
    parser.add_argument("--shots", type=int, default=4, help="number of few-shot examples")
    parser.add_argument("--seed", type=int, default=1234, help="seed of the example draw")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="generation budget")
    parser.add_argument(
        "--block-size", type=int, default=None, help="block size B (default: dllm_block_size)"
    )
    parser.add_argument(
        "--token-shift",
        type=int,
        choices=(0, 1),
        default=None,
        help="token shift (default: dllm_token_shift)",
    )
    parser.add_argument(
        "--mask-token-id", type=int, default=None, help="mask id M (default: dllm_mask_token_id)"
    )
    parser.add_argument("--device", default="cuda", help="torch device")
    parser.add_argument("--dtype", choices=tuple(_DTYPES), default="bfloat16", help="weights dtype")
    parser.add_argument("--out", default=None, help="write the JSON result, with records, here")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the GSM8K evaluation from the command line.

    The first ``--n`` problems of the test split are evaluated, in file order. The result is
    printed as one JSON line, without the records. With ``--out`` the full result is written too.

    Args:
        argv: Arguments without the program name, or None to read ``sys.argv``.

    Returns:
        Exit code 0.

    Raises:
        ValueError: If a setting is not supported, for example ``--n`` above the test size, or an
            unknown ``dllm_clean_attention`` in the config.
    """
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    args = _build_parser().parse_args(argv)
    test = load_gsm8k("test")
    if not 1 <= args.n <= len(test):
        raise ValueError(f"--n must be in [1, {len(test)}], got {args.n}")
    problems = test[: args.n]
    prefix = build_fewshot_prefix(load_gsm8k("train"), shots=args.shots, seed=args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=_DTYPES[args.dtype], attn_implementation="sdpa"
    )
    model = model.to(args.device).eval()
    real_vocab = len(tokenizer)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("the tokenizer has no eos_token_id")
    block_size: int | None = None
    token_shift: int | None = None
    mask_id: int | None = None
    clean_attention = "block"
    if args.mode != "ar":
        config = AutoConfig.from_pretrained(args.model)
        block_size = _resolve_int(args.block_size, config, "dllm_block_size")
        token_shift = _resolve_int(args.token_shift, config, "dllm_token_shift")
        mask_id = _resolve_int(args.mask_token_id, config, "dllm_mask_token_id")
        clean_attention = _resolve_clean_attention(config)

    result = evaluate(
        model,
        tokenizer,
        problems,
        prefix,
        args.mode,
        block_size,
        token_shift,
        mask_id,
        args.max_new_tokens,
        real_vocab,
        eos_id,
        tau=args.tau,
        clean_attention=clean_attention,
    )
    payload = {
        **result,
        "model": args.model,
        "shots": args.shots,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "tau": args.tau,
        "real_vocab": real_vocab,
        "eos_id": eos_id,
        "block_size": block_size,
        "token_shift": token_shift,
        "mask_token_id": mask_id,
        "clean_attention": clean_attention if args.mode != "ar" else None,
    }
    summary = {key: value for key, value in payload.items() if key != "records"}
    print(json.dumps(summary), flush=True)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
