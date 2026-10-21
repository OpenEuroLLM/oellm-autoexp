"""Tests for the block-diffusion launcher flags, the chained MegaDLMs provider,
and validation.

No torch is needed. The provider test is skipped when the MegaDLMs
submodule is not checked out.
"""

from __future__ import annotations

import argparse
import types
from pathlib import Path

import pytest

from oellm_autoexp.dllm.args import add_bd_args, extra_args_provider, validate_bd_args

REPO_ROOT = Path(__file__).resolve().parents[3]
MEGADLMS_DIR = REPO_ROOT / "submodules" / "MegaDLMs"
DIFFLM_SOURCE = MEGADLMS_DIR / "custom_args" / "difflm.py"

_BD_FLAGS: tuple[str, ...] = (
    "--bd-objective",
    "--bd-block-size",
    "--bd-block-schedule",
    "--bd-token-shift",
    "--bd-complementary-mask",
    "--bd-loss-weighting",
    "--bd-t-min",
    "--bd-t-max",
    "--bd-doc-mask",
    "--bd-val-seed",
)


def _dest(flag: str) -> str:
    """Return the argparse destination name of a long option."""
    return flag.removeprefix("--").replace("-", "_")


def _bd_parser() -> argparse.ArgumentParser:
    """Return a fresh parser that holds only the block-diffusion flags."""
    return add_bd_args(argparse.ArgumentParser())


def _valid_args(**overrides: object) -> types.SimpleNamespace:
    """Return a namespace with a valid v1 configuration, with optional
    overrides."""
    values: dict[str, object] = {
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
        "sequence_parallel": False,
        "seq_length": 2049,
        "bd_block_size": 8,
        "bd_block_schedule": "",
        "bd_t_min": 0.0,
        "bd_t_max": 1.0,
        "bd_complementary_mask": 0,
        "bd_objective": "bd3",
        "bd_token_shift": 0,
        "mask_token": 151669,
        "padded_vocab_size": 151680,
        "reset_position_ids": False,
        "reset_attention_mask": False,
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


def test_add_bd_args_returns_the_same_parser() -> None:
    """add_bd_args extends the given parser and returns that same object."""
    parser = argparse.ArgumentParser()
    assert add_bd_args(parser) is parser


def test_exactly_the_ten_block_diffusion_flags_are_added() -> None:
    """add_bd_args adds the ten documented flags and no other destinations."""
    args = _bd_parser().parse_args([])
    assert set(vars(args)) == {_dest(flag) for flag in _BD_FLAGS}


def test_flags_are_listed_under_a_block_diffusion_group() -> None:
    """The help text shows a block diffusion group that contains every flag."""
    help_text = _bd_parser().format_help()
    assert "block diffusion:" in help_text
    for flag in _BD_FLAGS:
        assert flag in help_text


def test_defaults_of_all_ten_flags() -> None:
    """Every block-diffusion flag has its documented default when none is
    given."""
    args = _bd_parser().parse_args([])
    assert args.bd_objective == "bd3"
    assert args.bd_block_size == 8
    assert args.bd_block_schedule == ""
    assert args.bd_token_shift == 0
    assert args.bd_complementary_mask == 0
    assert args.bd_loss_weighting == "elbo"
    assert args.bd_t_min == 0.0
    assert args.bd_t_max == 1.0
    assert args.bd_doc_mask == 1
    assert args.bd_val_seed == 1234


def test_explicit_values_parse_for_all_ten_flags() -> None:
    """Passing all ten flags sets each attribute to the parsed value."""
    argv = [
        "--bd-objective",
        "ar",
        "--bd-block-size",
        "4",
        "--bd-block-schedule",
        "0:4,0.15:8",
        "--bd-token-shift",
        "1",
        "--bd-complementary-mask",
        "1",
        "--bd-loss-weighting",
        "uniform",
        "--bd-t-min",
        "0.1",
        "--bd-t-max",
        "0.9",
        "--bd-doc-mask",
        "0",
        "--bd-val-seed",
        "7",
    ]
    args = _bd_parser().parse_args(argv)
    assert args.bd_objective == "ar"
    assert args.bd_block_size == 4
    assert args.bd_block_schedule == "0:4,0.15:8"
    assert args.bd_token_shift == 1
    assert args.bd_complementary_mask == 1
    assert args.bd_loss_weighting == "uniform"
    assert args.bd_t_min == 0.1
    assert args.bd_t_max == 0.9
    assert args.bd_doc_mask == 0
    assert args.bd_val_seed == 7


@pytest.mark.parametrize(
    "argv",
    [
        ["--bd-token-shift", "2"],
        ["--bd-objective", "xyz"],
        ["--bd-loss-weighting", "foo"],
        ["--bd-complementary-mask", "2"],
        ["--bd-doc-mask", "2"],
    ],
)
def test_values_outside_the_choices_exit(argv: list[str]) -> None:
    """Argparse rejects a value outside the declared choices with
    SystemExit."""
    with pytest.raises(SystemExit):
        _bd_parser().parse_args(argv)


def test_valid_v1_configuration_passes() -> None:
    """The v1 default configuration passes validation."""
    validate_bd_args(_valid_args())


def test_valid_schedule_with_complementary_mask_passes() -> None:
    """A divisible schedule with bd3 complementary masking and block sizes >= 2
    passes."""
    validate_bd_args(
        _valid_args(bd_block_schedule="0:4,0.15:8", bd_block_size=2, bd_complementary_mask=1)
    )


def test_ar_objective_accepts_token_shift_one() -> None:
    """With the ar objective, token shift 1 is allowed and ignored."""
    validate_bd_args(_valid_args(bd_objective="ar", bd_token_shift=1))


def test_optional_attributes_may_be_absent() -> None:
    """Parallelism, reset, and padded vocabulary attributes default to off when
    missing."""
    args = _valid_args()
    for name in (
        "pipeline_model_parallel_size",
        "context_parallel_size",
        "sequence_parallel",
        "reset_position_ids",
        "reset_attention_mask",
        "padded_vocab_size",
    ):
        delattr(args, name)
    validate_bd_args(args)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"pipeline_model_parallel_size": 2}, "pipeline_model_parallel_size"),
        ({"context_parallel_size": 2}, "context_parallel_size"),
        ({"sequence_parallel": True}, "sequence parallelism"),
        ({"bd_block_size": 3}, "does not divide"),
        ({"bd_block_schedule": "0:3"}, "does not divide"),
        ({"bd_block_size": 0}, "block size must be at least 1"),
        ({"bd_t_min": 0.5, "bd_t_max": 0.4}, "bd_t_min"),
        ({"bd_t_max": 1.5}, "bd_t_max"),
        ({"bd_t_min": -0.1}, "bd_t_min"),
        ({"bd_complementary_mask": 1, "bd_block_size": 1}, "complementary"),
        ({"bd_complementary_mask": 1, "bd_objective": "ar"}, "complementary"),
        ({"mask_token": 151680}, "padded vocabulary"),
        ({"mask_token": -1}, "must be >= 0"),
        ({"reset_attention_mask": True}, "reset-attention-mask"),
        ({"reset_position_ids": True}, "reset-position-ids"),
    ],
)
def test_each_unsupported_setting_raises_on_its_own(
    overrides: dict[str, object], match: str | None
) -> None:
    """Each unsupported setting raises ValueError when it is the only
    violation."""
    with pytest.raises(ValueError, match=match):
        validate_bd_args(_valid_args(**overrides))


def test_extra_args_provider_chains_megadlms_and_bd_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The provider adds the MegaDLMs difflm flags and the block-diffusion
    flags."""
    if not DIFFLM_SOURCE.exists():
        pytest.skip("submodules/MegaDLMs is not checked out")
    monkeypatch.syspath_prepend(str(MEGADLMS_DIR))

    parser = extra_args_provider(argparse.ArgumentParser())
    args = parser.parse_args(["--mask-token", "5", "--bd-block-size", "4"])
    assert args.mask_token == 5
    assert args.bd_block_size == 4

    args = parser.parse_args(["--difflm-varilen-prob", "0"])
    assert args.difflm_varilen_prob == 0.0
