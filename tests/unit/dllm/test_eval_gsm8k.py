"""Tests for oellm_autoexp/dllm/eval/eval_gsm8k.py (GSM8K 4-shot evaluation,
metrics M2 and M3)."""

from __future__ import annotations

import os
import random
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("torch")

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import transformers
from torch import nn

from oellm_autoexp.dllm.eval.eval_gsm8k import (
    build_fewshot_prefix,
    evaluate,
    extract_answer,
    format_prompt,
    generate_ar,
    gold_answer,
    load_gsm8k,
)

# Fixed character map of the tiny vocabulary: token id i is _CHARS[i]. The end-of-sequence id comes
# right after the characters, and the mask id is the first id outside the real vocabulary.
_CHARS = " \nabcdefghijklmnopqrstuvwxyz0123456789QA:#.?=+"
_CHAR_TO_ID = {char: index for index, char in enumerate(_CHARS)}
_EOS_ID = len(_CHARS)
_REAL_VOCAB = len(_CHARS) + 1
_MASK_ID = _REAL_VOCAB
_TINY_VOCAB = 48


class _CharTokenizer:
    """Character-level tokenizer with the fixed map above (no files, no
    network)."""

    eos_token_id = _EOS_ID

    def __len__(self) -> int:
        return _REAL_VOCAB

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [_CHAR_TO_ID[char] for char in text]

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        return "".join(_CHARS[token_id] for token_id in token_ids if token_id < len(_CHARS))


def _tiny_model() -> transformers.Qwen3ForCausalLM:
    """Return a seeded, randomly initialized tiny Qwen3 causal LM (no weights
    are downloaded)."""
    config = transformers.Qwen3Config(
        vocab_size=_TINY_VOCAB,
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


def _train_items() -> list[dict[str, str]]:
    """Return 10 short synthetic training items whose answers end with a
    ``####`` line."""
    return [
        {
            "question": f"tom has {i} apples and gets {i + 2} more. how many now?",
            "answer": f"{i} + {i + 2} = {2 * i + 2}.\n#### {2 * i + 2}",
        }
        for i in range(10)
    ]


def _problems(count: int) -> list[dict[str, str]]:
    """Return synthetic test problems; problem ``i`` has the gold answer ``2 *
    i + 1``."""
    return [
        {
            "question": f"tom has {i} pets and finds {i + 1} more. how many now?",
            "answer": f"{i} + {i + 1} = {2 * i + 1}.\n#### {2 * i + 1}",
        }
        for i in range(count)
    ]


def _greedy_reference(
    model: transformers.Qwen3ForCausalLM,
    prompt: Sequence[int],
    max_new_tokens: int,
    real_vocab: int,
    eos_id: int,
) -> list[int]:
    """Decode greedily without a cache: one full forward pass over the sequence per token."""
    ids = list(prompt)
    generated: list[int] = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids=torch.tensor([ids]), use_cache=False).logits
            token = int(logits[0, -1, :real_vocab].argmax().item())
            generated.append(token)
            if token == eos_id:
                break
            ids.append(token)
    return generated


def _evaluate(mode: str, **overrides: Any) -> dict[str, Any]:
    """Run ``evaluate`` on 3 synthetic problems with the tiny model and the
    character tokenizer."""
    settings: dict[str, Any] = {
        "model": _tiny_model(),
        "tokenizer": _CharTokenizer(),
        "problems": _problems(3),
        "prefix": build_fewshot_prefix(_train_items(), shots=4, seed=1234),
        "mode": mode,
        "block_size": None,
        "token_shift": None,
        "mask_id": None,
        "max_new_tokens": 8,
        "real_vocab": _REAL_VOCAB,
        "eos_id": _EOS_ID,
        "tau": 0.9,
    }
    settings.update(overrides)
    return evaluate(**settings)


def _check_result(result: dict[str, Any], mode: str, count: int) -> None:
    """Check the structure of an ``evaluate`` result."""
    assert result["mode"] == mode
    assert result["n"] == count
    assert 0 <= result["correct"] <= count
    assert 0.0 <= result["accuracy"] <= 100.0
    assert result["accuracy"] == pytest.approx(100.0 * result["correct"] / count)
    assert [record["index"] for record in result["records"]] == list(range(count))
    assert [record["gold"] for record in result["records"]] == [
        str(2 * i + 1) for i in range(count)
    ]
    for record in result["records"]:
        assert set(record) == {"index", "prediction", "gold", "correct", "completion"}
        assert record["correct"] == (record["prediction"] == record["gold"])


def test_char_tokenizer_round_trip_and_mask_id() -> None:
    tokenizer = _CharTokenizer()
    text = "tom has 3 apples? 3 + 4 = 7.\n#### 7"
    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert len(tokenizer) == _REAL_VOCAB
    # Spec section 12: M = len(tokenizer), and M must fit inside the padded model vocabulary.
    assert _MASK_ID == len(tokenizer) < _TINY_VOCAB


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The answer is 42.", "42"),
        ("#### 1,234", "1234"),
        ("x = 3.0\n#### 3.0", "3"),
        ("The total is 12.\nQuestion: how many apples are left after 99 are eaten?", "12"),
        ("Question: 7 apples?", None),
        ("no numbers in this text", None),
    ],
)
def test_extract_answer(text: str, expected: str | None) -> None:
    assert extract_answer(text) == expected


def test_extract_answer_prefers_the_last_hash_line_then_the_last_number() -> None:
    assert extract_answer("#### 4\nsome 9 apples\n#### 6") == "6"
    assert extract_answer("#### 5 then 9 apples") == "5"
    assert extract_answer("so it is 8 and\n####") == "8"


def test_gold_answer_reads_the_number_after_the_hash_line() -> None:
    assert gold_answer("2 + 3 = 5 and 4.0\n#### 1,234") == "1234"
    assert gold_answer("work\n#### 3.0") == "3"
    with pytest.raises(ValueError):
        gold_answer("the answer is 42")


def test_build_fewshot_prefix_is_deterministic_and_draws_four_distinct_items() -> None:
    train = _train_items()
    prefix = build_fewshot_prefix(train, shots=4, seed=1234)
    assert prefix == build_fewshot_prefix(train, shots=4, seed=1234)
    chosen = random.Random(1234).sample(train, 4)
    assert len({item["question"] for item in chosen}) == 4
    expected = "".join(
        f"Question: {item['question']}\nAnswer: {item['answer']}\n\n" for item in chosen
    )
    assert prefix == expected
    assert prefix.count("Question: ") == 4
    assert build_fewshot_prefix(train, shots=0) == ""
    with pytest.raises(ValueError):
        build_fewshot_prefix(train, shots=11)


def test_format_prompt_appends_the_question_and_the_answer_cue() -> None:
    assert format_prompt("Q0\n\n", "how many?") == "Q0\n\nQuestion: how many?\nAnswer:"


class _ScriptedModel(nn.Module):
    """Wrap a causal LM so that its greedy choices follow a script.

    The wrapper records the true logits of the last position for each
    forward call. It then replaces the logits with a row whose argmax is
    the scripted token, so decoding follows the script whatever the
    random weights predict. The recorded rows are unmodified model
    outputs.
    """

    def __init__(self, model: nn.Module, script: Sequence[int]) -> None:
        super().__init__()
        self.model = model
        self.script = list(script)
        self.recorded: list[torch.Tensor] = []

    def forward(self, **kwargs: Any) -> SimpleNamespace:
        out = self.model(**kwargs)
        self.recorded.append(out.logits[0, -1].detach().clone())
        forced = torch.full_like(out.logits, -1e9)
        forced[..., self.script[len(self.recorded) - 1]] = 0.0
        return SimpleNamespace(logits=forced, past_key_values=out.past_key_values)


def test_generate_ar_matches_cacheless_greedy_decoding() -> None:
    model = _tiny_model()
    prompt = [1, 5, 9, 20]
    tokens, passes = generate_ar(model, prompt, 6, _REAL_VOCAB, _EOS_ID)
    assert passes == len(tokens)
    assert 1 <= len(tokens) <= 6
    assert all(0 <= token < _REAL_VOCAB for token in tokens)
    assert tokens == _greedy_reference(model, prompt, 6, _REAL_VOCAB, _EOS_ID)
    # With a smaller real vocabulary, the unrestricted argmax (20 here) is not allowed any more.
    unrestricted = _greedy_reference(model, prompt, 6, _TINY_VOCAB, eos_id=-1)
    assert any(token >= 20 for token in unrestricted)
    small, _ = generate_ar(model, prompt, 6, 20, 19)
    assert all(token < 20 for token in small)
    assert small == _greedy_reference(model, prompt, 6, 20, 19)


def test_generate_ar_feeds_the_decoded_prefix_and_stops_at_eos() -> None:
    model = _tiny_model()
    prompt = [1, 5, 9, 20]
    script = [5, 7, _EOS_ID, 3, 3, 3]
    scripted = _ScriptedModel(model, script)
    tokens, passes = generate_ar(scripted, prompt, 6, _REAL_VOCAB, _EOS_ID)
    # The EOS is kept, and generation stops there, before the scripted tokens 3, 3, 3.
    assert tokens == [5, 7, _EOS_ID]
    assert passes == 3
    assert len(scripted.recorded) == 3
    # Each pass must see the prompt plus the tokens decoded so far. A cacheless pass over that
    # prefix gives the same last-position logits.
    prefix = list(prompt)
    with torch.no_grad():
        for step in range(3):
            full = model(input_ids=torch.tensor([prefix]), use_cache=False).logits[0, -1]
            assert torch.allclose(scripted.recorded[step], full, atol=1e-5, rtol=1e-4)
            prefix.append(script[step])


def test_generate_ar_rejects_unsupported_settings() -> None:
    model = _tiny_model()
    with pytest.raises(ValueError):
        generate_ar(model, [], 4, _REAL_VOCAB, _EOS_ID)
    with pytest.raises(ValueError):
        generate_ar(model, [1], 0, _REAL_VOCAB, _EOS_ID)
    with pytest.raises(ValueError):
        generate_ar(model, [1], 4, _REAL_VOCAB, _REAL_VOCAB)
    with pytest.raises(ValueError):
        generate_ar(model, [1], 4, _TINY_VOCAB + 1, 0)


def test_evaluate_ar_end_to_end_structure_and_unit_tpf() -> None:
    result = _evaluate("ar")
    _check_result(result, mode="ar", count=3)
    assert result["mean_tpf_decode"] == pytest.approx(1.0)
    assert result["mean_tpf_all"] == pytest.approx(1.0)


def test_evaluate_static_block_size_two_end_to_end_structure() -> None:
    result = _evaluate("static", block_size=2, token_shift=1, mask_id=_MASK_ID)
    _check_result(result, mode="static", count=3)
    assert 0.0 <= result["mean_tpf_all"] <= result["mean_tpf_decode"] + 1e-9


def test_evaluate_dynamic_block_size_two_end_to_end_structure() -> None:
    result = _evaluate("dynamic", block_size=2, token_shift=1, mask_id=_MASK_ID, tau=0.9)
    _check_result(result, mode="dynamic", count=3)
    assert 0.0 <= result["mean_tpf_all"] <= result["mean_tpf_decode"] + 1e-9


def test_evaluate_rejects_unsupported_settings() -> None:
    with pytest.raises(ValueError):
        _evaluate("beam")
    with pytest.raises(ValueError):
        _evaluate("ar", max_new_tokens=0)
    with pytest.raises(ValueError):
        _evaluate("ar", problems=[])
    with pytest.raises(ValueError):
        _evaluate("static", block_size=None, token_shift=1, mask_id=_MASK_ID)
    with pytest.raises(ValueError):
        _evaluate("dynamic", block_size=2, token_shift=1, mask_id=_MASK_ID, tau=1.5)


def test_load_gsm8k_reads_the_snapshot_under_hf_home(tmp_path: Path) -> None:
    main_dir = tmp_path / "hub" / "datasets--openai--gsm8k" / "snapshots" / "abc123" / "main"
    main_dir.mkdir(parents=True)
    rows = [
        {"question": "q one", "answer": "a one\n#### 1"},
        {"question": "q two", "answer": "a two\n#### 2"},
    ]
    pq.write_table(pa.Table.from_pylist(rows), main_dir / "test-00000-of-00001.parquet")
    assert load_gsm8k("test", hf_home=str(tmp_path)) == rows
    with pytest.raises(FileNotFoundError):
        load_gsm8k("train", hf_home=str(tmp_path))
    with pytest.raises(ValueError):
        load_gsm8k("validation", hf_home=str(tmp_path))


def test_load_gsm8k_needs_hf_home_when_no_directory_is_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_HOME", raising=False)
    with pytest.raises(ValueError):
        load_gsm8k("test")


def test_load_gsm8k_real_local_split_sizes() -> None:
    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        pytest.skip("HF_HOME is not set")
    for split in ("train", "test"):
        pattern = f"hub/datasets--openai--gsm8k/snapshots/*/main/{split}-00000-of-00001.parquet"
        if not list(Path(hf_home).glob(pattern)):
            pytest.skip(f"local GSM8K {split} parquet file is missing")
    assert len(load_gsm8k("train")) == 7473
    assert len(load_gsm8k("test")) == 1319
