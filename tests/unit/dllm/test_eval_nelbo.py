"""Tests for oellm_autoexp/dllm/eval/eval_nelbo.py (offline metric M1, spec
section 9, D-035).

The models are tiny randomly initialized Qwen3 causal LMs, built from a config and never
downloaded. Invariant I3 is checked on the Hugging Face path: with block size 1, token shift 1 and
every position masked, the bd3 NELBO sum equals the AR NLL sum, with and without the document
mask.

Run with:
    .claude/dllm/state/pytest2.sh tests/unit/dllm/test_eval_nelbo.py -v
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")

import numpy as np
import torch
import torch.nn.functional as F
import transformers

from oellm_autoexp.dllm.eval.eval_nelbo import (
    ar_nll_batch,
    bd3_nelbo_batch,
    evaluate,
    load_windows,
    m1_scored_positions,
    main,
)
from oellm_autoexp.dllm.masks import compute_doc_ids, dense_ar_reference, dense_bd3_reference
from oellm_autoexp.dllm.noise import sample_validation_noise

_VOCAB = 48
_EOD = 46  # end-of-document id used by the tests
_MASK = 47  # mask id M, the first id after the tiny real vocabulary
_RESULT_KEYS = (
    "model",
    "data_prefix",
    "objective",
    "block_size",
    "token_shift",
    "doc_mask",
    "num_sequences",
    "seq_len",
    "scored_positions",
    "nats_sum",
    "bits_per_token",
    "seed",
    "eod_id",
    "mask_token_id",
    "dtype",
)


def _tiny_model(dtype: torch.dtype = torch.float64) -> transformers.Qwen3ForCausalLM:
    """Return the tiny Qwen3 causal LM with sdpa attention, cast to ``dtype``,
    in eval mode.

    The weights come from ``torch.manual_seed(0)``, and no weights are downloaded.
    """
    config = transformers.Qwen3Config(
        vocab_size=_VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=512,
        tie_word_embeddings=True,
        attn_implementation="sdpa",
    )
    torch.manual_seed(0)
    model = transformers.Qwen3ForCausalLM(config)
    return model.to(dtype).eval()


def _windows(num: int, length: int, seed: int = 0) -> torch.Tensor:
    """Return random token windows of the tiny vocabulary with an EOD token
    every fifth position."""
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, _EOD, (num, length), generator=generator)
    tokens[:, ::5] = _EOD
    return tokens


def _write_indexed_dataset(prefix: Path, sequences: list[list[int]]) -> None:
    """Write a Megatron-style int32 ``<prefix>.bin`` and ``<prefix>.idx`` pair.

    The header follows ``read_idx_header``: the magic, version 1, dtype code 4 (int32), the
    sequence count, the document count (sequence count + 1), then the int32 lengths, the int64
    byte pointers, and the int64 document indices.
    """
    lengths = np.array([len(seq) for seq in sequences], dtype="<i4")
    pointers = np.zeros(len(sequences), dtype="<i8")
    pointers[1:] = np.cumsum(lengths[:-1].astype("<i8")) * 4
    doc_indices = np.arange(len(sequences) + 1, dtype="<i8")
    tokens = np.concatenate([np.asarray(seq, dtype="<i4") for seq in sequences])
    tokens.tofile(f"{prefix}.bin")
    with open(f"{prefix}.idx", "wb") as handle:
        handle.write(b"MMIDIDX\x00\x00")
        handle.write(struct.pack("<Q", 1))
        handle.write(struct.pack("<B", 4))
        handle.write(struct.pack("<Q", len(sequences)))
        handle.write(struct.pack("<Q", len(sequences) + 1))
        handle.write(lengths.tobytes())
        handle.write(pointers.tobytes())
        handle.write(doc_indices.tobytes())


def test_load_windows_reads_consecutive_windows_from_the_file_start(tmp_path: Path) -> None:
    prefix = tmp_path / "data"
    _write_indexed_dataset(prefix, [list(range(8)), list(range(8, 16)), list(range(16, 24))])
    windows = load_windows(str(prefix), 3, 5)
    assert windows.dtype == torch.long
    assert windows.shape == (3, 5)
    assert torch.equal(windows, torch.arange(15).reshape(3, 5))
    assert torch.equal(load_windows(str(prefix), 4, 6), torch.arange(24).reshape(4, 6))


def test_load_windows_rejects_short_files_and_empty_requests(tmp_path: Path) -> None:
    prefix = tmp_path / "data"
    _write_indexed_dataset(prefix, [list(range(8)), list(range(8, 16)), list(range(16, 24))])
    with pytest.raises(ValueError, match="need 30"):
        load_windows(str(prefix), 5, 6)
    with pytest.raises(ValueError, match="must be >= 1"):
        load_windows(str(prefix), 0, 6)


def test_m1_scored_positions_drops_position_zero_and_document_starts() -> None:
    x0 = torch.tensor([[5, 7, 5, 5, 7, 5]])
    scored = m1_scored_positions(x0, eod_id=7)
    assert scored.dtype == torch.bool
    assert scored.tolist() == [[False, True, False, True, True, False]]


@pytest.mark.parametrize("doc_mask", [False, True])
def test_i3_bd3_with_block_size_one_and_shift_one_equals_ar_on_the_hf_path(doc_mask: bool) -> None:
    model = _tiny_model(torch.float64)
    x0 = torch.tensor([[5, _EOD, 9, 5, 12, _EOD, 3, 8, 8, _EOD, 2, 5]])
    generator = torch.Generator().manual_seed(0)
    nelbo_sum, bd3_scored = bd3_nelbo_batch(
        model,
        x0,
        block_size=1,
        token_shift=1,
        mask_id=_MASK,
        eod_id=_EOD,
        doc_mask=doc_mask,
        generator=generator,
    )
    nll_sum, ar_scored = ar_nll_batch(model, x0, _EOD, doc_mask)
    # Positions 0, 2, 6, and 10 are excluded: position 0 and the three document starts.
    expected_scored = int(m1_scored_positions(x0, _EOD).sum())
    assert expected_scored == 8
    assert bd3_scored.item() == ar_scored.item() == expected_scored
    assert nll_sum.item() > 1.0
    assert abs(nelbo_sum.item() - nll_sum.item()) <= 1e-8


def _manual_bd3_nelbo(
    model: transformers.Qwen3ForCausalLM,
    x0: torch.Tensor,
    block_size: int,
    token_shift: int,
    doc_mask: bool,
    seed: int,
) -> tuple[float, int]:
    """Return the bd3 NELBO sum and scored count of a batch, written out one
    row at a time.

    The noise comes from ``sample_validation_noise`` with the same seed. Each row runs the model
    under the dense reference mask of invariant I1. The source of each position follows spec
    section 6 as written here, and the scored positions follow D-035. Document ids are counted
    directly from the EOD tokens.
    """
    length = x0.shape[1]
    generator = torch.Generator().manual_seed(seed)
    xt, masked, block_weight = sample_validation_noise(x0, block_size, _MASK, generator)
    positions = torch.tensor(list(range(length)) * 2)
    total, count = 0.0, 0
    for row in range(x0.shape[0]):
        docs, eods_before = [], 0
        for token in x0[row].tolist():
            docs.append(eods_before)
            eods_before += int(token == _EOD)
        allowed = dense_bd3_reference(length, block_size, docs if doc_mask else None)
        input_ids = torch.cat([xt[row], x0[row]])
        with torch.no_grad():
            logits = (
                model(
                    input_ids=input_ids[None],
                    position_ids=positions[None],
                    attention_mask=allowed[None, None],
                )
                .logits[0]
                .float()
            )
        for i in range(length):
            is_scored = i >= 1 and docs[i] == docs[i - 1]
            if token_shift == 0:
                src: int | None = i
            elif i == 0:
                src = None
            elif i % block_size != 0:
                src = i - 1
            else:
                src = length + i - 1
            if not is_scored or src is None:
                continue
            count += 1
            if masked[row, i]:
                ce = F.cross_entropy(logits[src][None], x0[row, i][None]).item()
                total += float(block_weight[row, i // block_size]) * ce
    return total, count


@pytest.mark.parametrize("token_shift", [0, 1])
@pytest.mark.parametrize("doc_mask", [False, True])
def test_bd3_nelbo_matches_a_manual_reference_with_block_size_four(
    token_shift: int, doc_mask: bool
) -> None:
    model = _tiny_model()
    x0 = _windows(2, 12, seed=4)
    seed = 11
    nelbo_sum, scored = bd3_nelbo_batch(
        model,
        x0,
        block_size=4,
        token_shift=token_shift,
        mask_id=_MASK,
        eod_id=_EOD,
        doc_mask=doc_mask,
        generator=torch.Generator().manual_seed(seed),
    )
    expected_sum, expected_scored = _manual_bd3_nelbo(model, x0, 4, token_shift, doc_mask, seed)
    assert expected_sum > 0
    assert scored.item() == expected_scored
    assert nelbo_sum.item() == pytest.approx(expected_sum, rel=1e-5)


@pytest.mark.parametrize("doc_mask", [False, True])
def test_ar_nll_matches_a_dense_reference_mask(doc_mask: bool) -> None:
    model = _tiny_model()
    x0 = _windows(2, 10, seed=1)
    rows = []
    for row in x0:
        doc_row = compute_doc_ids(row[None], _EOD)[0].tolist() if doc_mask else None
        rows.append(dense_ar_reference(10, doc_row))
    allowed = torch.stack(rows).unsqueeze(1)
    with torch.no_grad():
        logits = model(input_ids=x0, attention_mask=allowed).logits.float()
    ce = F.cross_entropy(
        logits[:, :-1].reshape(-1, _VOCAB), x0[:, 1:].reshape(-1), reduction="none"
    )
    scored = m1_scored_positions(x0, _EOD)[:, 1:].reshape(-1)
    nll_sum, count = ar_nll_batch(model, x0, _EOD, doc_mask)
    assert count.item() == int(scored.sum())
    assert nll_sum.item() == pytest.approx(float(ce[scored].sum()), rel=1e-5)


def test_evaluate_repeats_for_a_seed_and_changes_with_the_seed() -> None:
    model = _tiny_model()
    windows = _windows(4, 16)

    def run(seed: int) -> dict[str, Any]:
        return evaluate(
            model=model,
            windows=windows,
            objective="bd3",
            block_size=4,
            token_shift=0,
            mask_id=_MASK,
            eod_id=_EOD,
            doc_mask=True,
            seed=seed,
            batch_size=2,
            device="cpu",
        )

    first, second, other = run(7), run(7), run(8)
    assert first["nats_sum"] == second["nats_sum"]
    assert first["bits_per_token"] == second["bits_per_token"]
    assert first["bits_per_token"] != other["bits_per_token"]


@pytest.mark.parametrize("objective", ["bd3", "ar"])
def test_bits_per_token_is_nats_over_scored_positions_times_ln2(objective: str) -> None:
    model = _tiny_model()
    windows = _windows(3, 12, seed=2)
    result = evaluate(
        model=model,
        windows=windows,
        objective=objective,
        block_size=4,
        token_shift=1,
        mask_id=_MASK,
        eod_id=_EOD,
        doc_mask=False,
        seed=5,
        batch_size=2,
        device="cpu",
    )
    assert result["scored_positions"] == int(m1_scored_positions(windows, _EOD).sum())
    expected = result["nats_sum"] / (result["scored_positions"] * math.log(2))
    assert result["bits_per_token"] == pytest.approx(expected, rel=1e-12)
    assert result["bits_per_token"] > 0


def test_evaluate_draws_every_batch_from_one_generator_in_order() -> None:
    model = _tiny_model()
    windows = _windows(5, 8, seed=3)
    result = evaluate(
        model=model,
        windows=windows,
        objective="bd3",
        block_size=4,
        token_shift=0,
        mask_id=_MASK,
        eod_id=_EOD,
        doc_mask=True,
        seed=3,
        batch_size=2,
        device="cpu",
    )
    generator = torch.Generator().manual_seed(3)
    expected = 0.0
    for start in range(0, 5, 2):
        nelbo_sum, _ = bd3_nelbo_batch(
            model,
            windows[start : start + 2],
            block_size=4,
            token_shift=0,
            mask_id=_MASK,
            eod_id=_EOD,
            doc_mask=True,
            generator=generator,
        )
        expected += nelbo_sum.item()
    assert result["nats_sum"] == pytest.approx(expected, rel=1e-12)
    assert result["num_sequences"] == 5
    assert result["seq_len"] == 8


def test_evaluate_rejects_unsupported_settings() -> None:
    model = _tiny_model()
    windows = _windows(2, 8)

    def run(**overrides: Any) -> dict[str, Any]:
        settings: dict[str, Any] = {
            "model": model,
            "windows": windows,
            "objective": "bd3",
            "block_size": 4,
            "token_shift": 0,
            "mask_id": _MASK,
            "eod_id": _EOD,
            "doc_mask": True,
            "seed": 0,
            "batch_size": 1,
            "device": "cpu",
        }
        settings.update(overrides)
        return evaluate(**settings)

    with pytest.raises(ValueError, match="objective"):
        run(objective="mdm")
    with pytest.raises(ValueError, match="multiple of block_size"):
        run(block_size=3)
    with pytest.raises(ValueError, match="block_size must be an integer"):
        run(block_size=None)
    with pytest.raises(ValueError, match="token_shift must be 0 or 1"):
        run(token_shift=2)
    with pytest.raises(ValueError, match="batch_size"):
        run(batch_size=0)
    with pytest.raises(ValueError, match="no position is scored"):
        run(objective="ar", windows=_windows(2, 1))


def test_main_writes_the_json_with_all_keys(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model = _tiny_model(torch.float32)
    model.config.dllm_block_size = 4
    model.config.dllm_token_shift = 0
    model.config.dllm_mask_token_id = _MASK
    model_dir = tmp_path / "model"
    model.save_pretrained(model_dir)
    prefix = tmp_path / "data"
    _write_indexed_dataset(prefix, _windows(3, 16).tolist())
    out = tmp_path / "results" / "m1.json"

    code = main(
        [
            "--model",
            str(model_dir),
            "--data-prefix",
            str(prefix),
            "--objective",
            "bd3",
            "--eod-id",
            str(_EOD),
            "--dtype",
            "float32",
            "--num-sequences",
            "2",
            "--seq-len",
            "16",
            "--device",
            "cpu",
            "--out",
            str(out),
        ]
    )

    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert set(_RESULT_KEYS) <= set(payload)
    assert payload["model"] == str(model_dir)
    assert payload["data_prefix"] == str(prefix)
    assert payload["objective"] == "bd3"
    assert payload["block_size"] == 4
    assert payload["token_shift"] == 0
    assert payload["mask_token_id"] == _MASK
    assert payload["eod_id"] == _EOD
    assert payload["doc_mask"] is True
    assert payload["num_sequences"] == 2
    assert payload["seq_len"] == 16
    assert payload["seed"] == 1234
    assert payload["dtype"] == "float32"
    assert payload["scored_positions"] == int(m1_scored_positions(_windows(3, 16)[:2], _EOD).sum())
    assert payload["bits_per_token"] == pytest.approx(
        payload["nats_sum"] / (payload["scored_positions"] * math.log(2)), rel=1e-12
    )
    assert json.loads(capsys.readouterr().out) == payload


def test_main_ar_objective_reports_no_block_settings(tmp_path: Path) -> None:
    model = _tiny_model(torch.float32)
    model_dir = tmp_path / "model"
    model.save_pretrained(model_dir)
    prefix = tmp_path / "data"
    _write_indexed_dataset(prefix, _windows(2, 12).tolist())
    out = tmp_path / "ar.json"

    code = main(
        [
            "--model",
            str(model_dir),
            "--data-prefix",
            str(prefix),
            "--objective",
            "ar",
            "--eod-id",
            str(_EOD),
            "--dtype",
            "float32",
            "--num-sequences",
            "2",
            "--seq-len",
            "12",
            "--device",
            "cpu",
            "--out",
            str(out),
        ]
    )

    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["objective"] == "ar"
    assert payload["block_size"] is None
    assert payload["token_shift"] is None
    assert payload["mask_token_id"] is None
    assert payload["bits_per_token"] > 0


def test_main_bd3_without_a_block_size_raises(tmp_path: Path) -> None:
    model = _tiny_model(torch.float32)
    model_dir = tmp_path / "model"
    model.save_pretrained(model_dir)
    prefix = tmp_path / "data"
    _write_indexed_dataset(prefix, _windows(2, 12).tolist())
    out = tmp_path / "bd3.json"

    with pytest.raises(ValueError, match="block-size"):
        main(
            [
                "--model",
                str(model_dir),
                "--data-prefix",
                str(prefix),
                "--objective",
                "bd3",
                "--eod-id",
                str(_EOD),
                "--dtype",
                "float32",
                "--num-sequences",
                "2",
                "--seq-len",
                "12",
                "--device",
                "cpu",
                "--out",
                str(out),
            ]
        )
    assert not out.exists()
