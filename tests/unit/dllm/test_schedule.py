"""Tests for the block-size schedule parser and lookup (spec section 8).

The schedule module has no dependency on the GPU stack, so these tests
need no skip guard and also run in the CI job without it.
"""

from __future__ import annotations

import re

import pytest

from oellm_autoexp.dllm.schedule import (
    block_size_at,
    parse_block_schedule,
    validate_block_sizes,
)


@pytest.mark.parametrize("spec", ["", "  ", "\t \n", "none", " None "])
def test_blank_or_none_spec_means_no_schedule(spec: str) -> None:
    assert parse_block_schedule(spec) == []


def test_parse_simple_schedule() -> None:
    assert parse_block_schedule("0:4,0.15:8") == [(0.0, 4), (0.15, 8)]


def test_parse_ignores_spaces_around_items_and_colons() -> None:
    assert parse_block_schedule(" 0 : 4 , 0.15 : 8 ") == [(0.0, 4), (0.15, 8)]


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ("0.1:4", "exactly 0"),
        ("0:4,0:8", "greater than the previous fraction"),
        ("0:8,0.5:4,0.3:2", "greater than the previous fraction"),
        ("0:4,1.0:8", "must be in [0, 1)"),
        ("0:0", "integer >= 1"),
        ("abc", "exactly one ':'"),
        ("0:4:5", "exactly one ':'"),
        ("0:x", "is not an integer"),
    ],
)
def test_parse_rejects_invalid_specs(spec: str, message: str) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        parse_block_schedule(spec)


def test_block_size_switches_at_fraction_boundary() -> None:
    schedule = parse_block_schedule("0:4,0.15:8")
    sizes = [block_size_at(step, 1000, schedule, 16) for step in (0, 149, 150, 999)]
    assert sizes == [4, 4, 8, 8]


def test_empty_schedule_returns_default() -> None:
    assert block_size_at(0, 1000, [], 16) == 16
    assert block_size_at(999, 1000, [], 16) == 16
    assert block_size_at(0, 0, [], 16) == 16


def test_train_iters_must_be_positive_when_schedule_is_set() -> None:
    with pytest.raises(ValueError, match="train_iters"):
        block_size_at(0, 0, [(0.0, 4)], 4)


def test_lookup_rejects_schedule_that_does_not_start_at_zero() -> None:
    with pytest.raises(ValueError, match="exactly 0"):
        block_size_at(0, 100, [(0.5, 8)], 4)


def test_lookup_rejects_negative_iteration() -> None:
    with pytest.raises(ValueError, match="covers iteration"):
        block_size_at(-1, 100, [(0.0, 4)], 4)


def test_validate_accepts_block_sizes_that_divide_length() -> None:
    validate_block_sizes(2048, 8, [(0.0, 4), (0.15, 8)])


def test_validate_rejects_scheduled_block_size_that_does_not_divide() -> None:
    with pytest.raises(ValueError, match="does not divide"):
        validate_block_sizes(2048, 8, [(0.0, 3)])


def test_validate_rejects_default_block_size_that_does_not_divide() -> None:
    with pytest.raises(ValueError, match="does not divide"):
        validate_block_sizes(2048, 3, [])


def test_validate_rejects_non_positive_default_block_size() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        validate_block_sizes(2048, 0, [])


def test_validate_rejects_non_positive_scheduled_block_size() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        validate_block_sizes(2048, 8, [(0.0, 0)])


def test_validate_rejects_non_positive_effective_length() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        validate_block_sizes(0, 1, [])
