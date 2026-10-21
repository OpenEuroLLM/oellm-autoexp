"""Tests for the pure helpers of oellm_autoexp/dllm/eval/parity_check.py (spec
13, I3)."""

from __future__ import annotations

import argparse

import pytest

pytest.importorskip("torch")

import torch

from oellm_autoexp.dllm.eval.parity_check import (
    KL_MAX,
    TOP1_MIN,
    add_parity_args,
    compare_logits,
    i3_compare,
    parity_verdict,
)


def test_compare_logits_identical_and_padding_rows_ignored() -> None:
    torch.manual_seed(0)
    ref = torch.randn(2, 5, 10)
    cand = ref.clone()
    cand[..., 8:] = 100.0  # padding rows beyond the real vocabulary must not matter
    stats = compare_logits(ref, cand, real_vocab=8)
    assert stats["top1"] == 1.0
    assert stats["kl_mean"] == pytest.approx(0.0, abs=1e-6)
    assert stats["positions"] == 10


def test_compare_logits_detects_a_difference() -> None:
    torch.manual_seed(0)
    ref = torch.randn(1, 4, 6)
    cand = ref + torch.randn(1, 4, 6)
    stats = compare_logits(ref, cand, real_vocab=6)
    assert stats["kl_mean"] > 0.0
    assert 0.0 <= stats["top1"] <= 1.0
    with pytest.raises(ValueError):
        compare_logits(ref, cand[..., :5], real_vocab=6)


def test_i3_compare_matches_shifted_ar_logits() -> None:
    torch.manual_seed(0)
    ar = torch.randn(2, 6, 7)
    bd3 = torch.zeros_like(ar)
    bd3[:, 1:] = ar[:, :-1]  # bd3 position i reads the clean output of position i - 1
    x0 = torch.randint(0, 7, (2, 6))
    stats = i3_compare(bd3, ar, x0)
    assert stats["mean_abs_diff"] == pytest.approx(0.0, abs=1e-6)
    assert stats["max_abs_diff"] == pytest.approx(0.0, abs=1e-6)


def test_parity_verdict_thresholds() -> None:
    good = {"top1": TOP1_MIN, "kl_mean": KL_MAX}
    assert parity_verdict(good, {"mean_abs_diff": 0.0}, 1e-2) == (True, True)
    assert parity_verdict({"top1": 0.98, "kl_mean": 0.0}, {"mean_abs_diff": 0.02}, 1e-2) == (
        False,
        False,
    )
    assert (
        parity_verdict({"top1": 1.0, "kl_mean": 2e-3}, {"mean_abs_diff": float("nan")}, 1.0)[1]
        is False
    )


def test_add_parity_args_defaults() -> None:
    args = add_parity_args(argparse.ArgumentParser()).parse_args(["--parity-hf", "x"])
    assert args.parity_hf == "x"
    assert args.parity_num_seqs == 4
    assert args.parity_seq_len == 512
