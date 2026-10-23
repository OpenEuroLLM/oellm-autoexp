"""Tests for the pure helpers of oellm_autoexp/dllm/eval/parity_check.py (spec
13, I3)."""

from __future__ import annotations

import argparse

import pytest

pytest.importorskip("torch")

import torch

from oellm_autoexp.dllm.eval.parity_check import (
    I3_MAX_TOL,
    I3_MEAN_TOL,
    KL_MAX,
    TOP1_MIN,
    add_parity_args,
    compare_logits,
    i3_compare,
    parity_dtype,
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
    exact = {"mean_abs_diff": 0.0, "max_abs_diff": 0.0}
    assert parity_verdict(good, exact, 1e-2, 1e-1) == (True, True)
    worse = {"mean_abs_diff": 0.02, "max_abs_diff": 0.02}
    assert parity_verdict({"top1": 0.98, "kl_mean": 0.0}, worse, 1e-2, 1e-1) == (False, False)
    nan = {"mean_abs_diff": float("nan"), "max_abs_diff": 0.0}
    assert parity_verdict({"top1": 1.0, "kl_mean": 2e-3}, nan, 1.0, 1.0)[1] is False


def test_i3_gate_rejects_a_localized_error_and_accepts_the_recorded_fp32_runs() -> None:
    """D-063: four positions off by 5 nats keep the mean below 1e-2 but fail the
    maximum; the worst recorded fp32 conversion (8B TP 2) passes."""
    torch.manual_seed(0)
    ar = torch.randn(4, 512, 16)
    x0 = torch.randint(0, 16, (4, 512))
    bd3 = torch.zeros_like(ar)
    bd3[:, 1:] = ar[:, :-1]
    target = x0[:, 1:]
    for row in range(4):
        # Moving the target logit by 5 changes its cross entropy by about 5 nats.
        bd3[row, 7, target[row, 6]] -= 5.0
    stats = i3_compare(bd3, ar, x0)
    assert stats["mean_abs_diff"] < 1e-2  # the old mean-only gate passed this
    good = {"top1": 1.0, "kl_mean": 0.0}
    assert parity_verdict(good, stats)[1] is False
    recorded_8b = {"mean_abs_diff": 4.29e-4, "max_abs_diff": 6.00e-3}
    assert parity_verdict(good, recorded_8b) == (True, True)
    assert I3_MEAN_TOL < 1e-2 and I3_MAX_TOL < 1.0


def test_add_parity_args_defaults() -> None:
    args = add_parity_args(argparse.ArgumentParser()).parse_args(["--parity-hf", "x"])
    assert args.parity_hf == "x"
    assert args.parity_num_seqs == 4
    assert args.parity_seq_len == 512
    assert args.parity_nld_block_size == 32  # the R11 block size; divides 512
    assert (args.parity_i3_tol, args.parity_i3_max_tol) == (I3_MEAN_TOL, I3_MAX_TOL)


def test_parity_dtype_follows_the_megatron_run() -> None:
    """The HF side runs in the Megatron dtype; without --bf16 or --fp16 both
    sides are fp32."""
    assert parity_dtype(bf16=True, fp16=False) is torch.bfloat16
    assert parity_dtype(bf16=False, fp16=True) is torch.float16
    assert parity_dtype(bf16=False, fp16=False) is torch.float32
