"""Invariant tests for the block-diffusion noise samplers and the validation
estimator.

Covers invariant I4 (every block has a mask, and complementary copies are disjoint and cover each
block) and invariant I6 (the validation estimator is deterministic for identical inputs and a
seed). Spec sections 3, 9, and 11.

Run with:
    pytest tests/unit/dllm/test_noise.py -v
"""

from __future__ import annotations

import importlib
from collections.abc import Callable

import pytest

torch = pytest.importorskip("torch")
# Imported after the importorskip above, so a missing torch skips this module. importlib keeps the
# import out of the module-level import block, which avoids E402 without a noqa comment.
noise = importlib.import_module("oellm_autoexp.dllm.noise")
expand_block_values = noise.expand_block_values
sample_block_noise = noise.sample_block_noise
sample_complementary_noise = noise.sample_complementary_noise
sample_validation_noise = noise.sample_validation_noise
validation_nelbo = noise.validation_nelbo

MASK_ID = 1000
BATCH = 3
SEQ_LEN = 64
SEEDS = (0, 1, 2)
BLOCK_SIZES = (1, 2, 4, 8, 16)
COMPLEMENTARY_BLOCK_SIZES = (2, 4, 8, 16)
T_RANGES = ((0.0, 1.0), (0.0, 0.0), (1.0, 1.0), (0.2, 0.8))

Draw = Callable[[torch.Tensor, int, torch.Generator], tuple[torch.Tensor, ...]]


def _draw_bd3(
    x0: torch.Tensor, block_size: int, generator: torch.Generator
) -> tuple[torch.Tensor, ...]:
    """Run the bd3 sampler with the t range used in the examples."""
    return sample_block_noise(x0, block_size, MASK_ID, 0.2, 0.8, generator)


def _draw_complementary(
    x0: torch.Tensor, block_size: int, generator: torch.Generator
) -> tuple[torch.Tensor, ...]:
    """Run the complementary sampler with the t range used in the examples."""
    return sample_complementary_noise(x0, block_size, MASK_ID, 0.2, 0.8, generator)


def _draw_validation(
    x0: torch.Tensor, block_size: int, generator: torch.Generator
) -> tuple[torch.Tensor, ...]:
    """Run the validation sampler, which takes no t range."""
    return sample_validation_noise(x0, block_size, MASK_ID, generator)


SAMPLERS: tuple[Draw, ...] = (_draw_bd3, _draw_complementary, _draw_validation)
SAMPLER_IDS = ("bd3", "complementary", "validation")


def _generator(seed: int) -> torch.Generator:
    """Return a CPU generator seeded with ``seed``."""
    return torch.Generator(device="cpu").manual_seed(seed)


def _clean_tokens(seed: int, batch: int = BATCH, length: int = SEQ_LEN) -> torch.Tensor:
    """Return random clean tokens in [0, 100) that never equal ``MASK_ID``."""
    tokens = torch.randint(0, 100, (batch, length), generator=_generator(seed))
    assert not bool((tokens == MASK_ID).any())
    return tokens


def _block_counts(masked: torch.Tensor, block_size: int) -> torch.Tensor:
    """Count the masked positions of each block.

    The result has shape ``[rows, K]``.
    """
    rows, length = masked.shape
    return masked.reshape(rows, length // block_size, block_size).sum(dim=-1)


@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize(("t_min", "t_max"), T_RANGES)
def test_block_noise_gives_every_block_a_mask(block_size: int, t_min: float, t_max: float) -> None:
    """I4: each block has a mask, p_block is the masked fraction, and xt follows the mask."""
    num_blocks = SEQ_LEN // block_size
    for seed in SEEDS:
        x0 = _clean_tokens(seed)
        xt, masked, p_block = sample_block_noise(
            x0, block_size, MASK_ID, t_min, t_max, _generator(seed)
        )
        assert xt.dtype == torch.long
        assert masked.dtype == torch.bool
        assert p_block.dtype == torch.float32
        assert xt.shape == (BATCH, SEQ_LEN)
        assert masked.shape == (BATCH, SEQ_LEN)
        assert p_block.shape == (BATCH, num_blocks)

        counts = _block_counts(masked, block_size)
        assert bool((counts >= 1).all())
        assert torch.equal(p_block, counts.to(torch.float32) / block_size)
        assert bool((xt[masked] == MASK_ID).all())
        assert torch.equal(xt[~masked], x0[~masked])
        if (t_min, t_max) == (0.0, 0.0):
            assert bool((counts == 1).all())
        if (t_min, t_max) == (1.0, 1.0):
            assert bool(masked.all())


@pytest.mark.parametrize("block_size", COMPLEMENTARY_BLOCK_SIZES)
@pytest.mark.parametrize(("t_min", "t_max"), T_RANGES)
def test_complementary_copies_are_disjoint_and_cover_each_block(
    block_size: int, t_min: float, t_max: float
) -> None:
    """I4: the two copies partition every position, and copy 1 keeps 1..B-1 masks per block."""
    num_blocks = SEQ_LEN // block_size
    for seed in SEEDS:
        x0 = _clean_tokens(seed)
        xt, masked, p_block = sample_complementary_noise(
            x0, block_size, MASK_ID, t_min, t_max, _generator(seed)
        )
        assert xt.dtype == torch.long
        assert xt.shape == (2 * BATCH, SEQ_LEN)
        assert masked.shape == (2 * BATCH, SEQ_LEN)
        assert p_block.shape == (2 * BATCH, num_blocks)

        copy1, copy2 = masked[:BATCH], masked[BATCH:]
        assert not bool((copy1 & copy2).any())
        assert bool((copy1 | copy2).all())

        counts1 = _block_counts(copy1, block_size)
        counts2 = _block_counts(copy2, block_size)
        assert torch.equal(counts1 + counts2, torch.full_like(counts1, block_size))
        assert bool(((counts1 >= 1) & (counts1 <= block_size - 1)).all())
        assert bool(((counts2 >= 1) & (counts2 <= block_size - 1)).all())

        counts = _block_counts(masked, block_size)
        assert torch.equal(p_block, counts.to(torch.float32) / block_size)
        if (t_min, t_max) == (1.0, 1.0):
            assert bool((counts1 == block_size - 1).all())
        if (t_min, t_max) == (0.0, 0.0):
            assert bool((counts1 == 1).all())

        x0_stacked = torch.cat([x0, x0], dim=0)
        assert bool((xt[masked] == MASK_ID).all())
        assert torch.equal(xt[~masked], x0_stacked[~masked])


def test_complementary_noise_rejects_block_size_one() -> None:
    """Complementary masking needs at least two positions per block."""
    with pytest.raises(ValueError, match="block_size >= 2"):
        sample_complementary_noise(_clean_tokens(0), 1, MASK_ID, 0.0, 1.0, _generator(0))


@pytest.mark.parametrize("draw", SAMPLERS, ids=SAMPLER_IDS)
def test_same_seed_gives_identical_outputs(draw: Draw) -> None:
    """Identical inputs and the same seed reproduce every output exactly."""
    x0 = _clean_tokens(7)
    first = draw(x0, 8, _generator(42))
    second = draw(x0, 8, _generator(42))
    for left, right in zip(first, second, strict=True):
        assert torch.equal(left, right)


@pytest.mark.parametrize("draw", SAMPLERS, ids=SAMPLER_IDS)
def test_different_seed_gives_different_masks(draw: Draw) -> None:
    """A different seed changes the mask pattern (L = 64, B = 8)."""
    x0 = _clean_tokens(7)
    masked_a = draw(x0, 8, _generator(1))[1]
    masked_b = draw(x0, 8, _generator(2))[1]
    assert not torch.equal(masked_a, masked_b)


@pytest.mark.parametrize("block_size", BLOCK_SIZES)
def test_validation_noise_masks_l_positions_per_block(block_size: int) -> None:
    """I6: l_k lies in 1..B, exactly l_k positions are masked, and block_weight is B / l_k."""
    num_blocks = SEQ_LEN // block_size
    for seed in SEEDS:
        x0 = _clean_tokens(seed)
        xt, masked, block_weight = sample_validation_noise(
            x0, block_size, MASK_ID, _generator(seed)
        )
        assert xt.dtype == torch.long
        assert masked.dtype == torch.bool
        assert block_weight.dtype == torch.float32
        assert block_weight.shape == (BATCH, num_blocks)

        counts = _block_counts(masked, block_size)
        assert bool(((counts >= 1) & (counts <= block_size)).all())
        assert torch.equal(block_weight, block_size / counts.to(torch.float32))
        assert bool((xt[masked] == MASK_ID).all())
        assert torch.equal(xt[~masked], x0[~masked])


def _validation_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return fixed clean tokens, per-token ce values, and a scorable mask for
    the estimator."""
    x0 = _clean_tokens(3)
    ce = torch.rand(BATCH, SEQ_LEN, generator=_generator(11))
    scorable = torch.ones(BATCH, SEQ_LEN, dtype=torch.bool)
    scorable[:, 0] = False  # position 0 has no left context and is never scored
    return x0, ce, scorable


def _validation_estimate(seed: int) -> tuple[torch.Tensor, ...]:
    """Run the validation sampler and the estimator with the given seed."""
    x0, ce, scorable = _validation_inputs()
    xt, masked, block_weight = sample_validation_noise(x0, 4, MASK_ID, _generator(seed))
    nelbo_sum, scored = validation_nelbo(ce, masked, block_weight, scorable)
    return xt, masked, block_weight, nelbo_sum, scored


def test_validation_estimate_is_identical_for_identical_inputs_and_seed() -> None:
    """I6: the sampler and the estimator return identical values for identical inputs."""
    first = _validation_estimate(5)
    second = _validation_estimate(5)
    for left, right in zip(first, second, strict=True):
        assert torch.equal(left, right)


def _reference_nelbo(
    ce: torch.Tensor,
    masked: torch.Tensor,
    block_weight: torch.Tensor,
    scorable: torch.Tensor,
    block_size: int,
) -> float:
    """Sum (B / l_k) * ce over the masked, scorable positions with plain Python
    loops."""
    total = 0.0
    for row in range(ce.shape[0]):
        for pos in range(ce.shape[1]):
            if bool(masked[row, pos]) and bool(scorable[row, pos]):
                total += float(block_weight[row, pos // block_size]) * float(ce[row, pos])
    return total


def test_validation_nelbo_matches_python_reference() -> None:
    """The vectorized estimator equals a loop over the definition in spec
    section 9."""
    block_size = 4
    x0, ce, scorable = _validation_inputs()
    _, masked, block_weight = sample_validation_noise(x0, block_size, MASK_ID, _generator(5))
    nelbo_sum, scored = validation_nelbo(ce, masked, block_weight, scorable)
    expected = _reference_nelbo(ce, masked, block_weight, scorable, block_size)
    assert nelbo_sum.item() == pytest.approx(expected, rel=1e-5)
    assert scored.item() == float(scorable.sum())


def test_validation_nelbo_hand_case() -> None:
    """B=1, L=4, B=2: position 0 is not scorable, so nelbo_sum = 1 * (3 + 4)
    and scored = 3."""
    ce = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    masked = torch.tensor([[True, False, True, True]])
    block_weight = torch.tensor([[2.0, 1.0]])
    scorable = torch.tensor([[False, True, True, True]])
    nelbo_sum, scored = validation_nelbo(ce, masked, block_weight, scorable)
    assert nelbo_sum.dtype == torch.float32
    assert nelbo_sum.dim() == 0
    assert scored.dtype == torch.float32
    assert scored.dim() == 0
    assert nelbo_sum.item() == 7.0
    assert scored.item() == 3.0


def test_validation_nelbo_ignores_non_finite_values_outside_the_targets() -> None:
    """Positions that are not targets cannot leak into the sum, even when their
    ce is infinite."""
    ce = torch.tensor([[float("inf"), float("inf"), 3.0, 4.0]])
    masked = torch.tensor([[True, False, True, True]])
    block_weight = torch.tensor([[2.0, 1.0]])
    scorable = torch.tensor([[False, True, True, True]])
    nelbo_sum, scored = validation_nelbo(ce, masked, block_weight, scorable)
    assert nelbo_sum.item() == 7.0
    assert scored.item() == 3.0


def test_expand_block_values_repeats_each_block_value() -> None:
    """Each block value is repeated over the positions of its block, in
    order."""
    expanded = expand_block_values(torch.tensor([[2.0, 1.0]]), 3)
    assert expanded.tolist() == [[2.0, 2.0, 2.0, 1.0, 1.0, 1.0]]


@pytest.mark.parametrize("draw", SAMPLERS, ids=SAMPLER_IDS)
def test_samplers_reject_length_not_divisible_by_block_size(draw: Draw) -> None:
    """L % block_size != 0 raises ValueError."""
    x0 = torch.zeros(2, 10, dtype=torch.long)
    with pytest.raises(ValueError, match="multiple of block_size"):
        draw(x0, 4, _generator(0))


@pytest.mark.parametrize("draw", SAMPLERS, ids=SAMPLER_IDS)
def test_samplers_reject_block_size_below_one(draw: Draw) -> None:
    """A block size below 1 raises ValueError."""
    with pytest.raises(ValueError, match="block_size"):
        draw(_clean_tokens(0), 0, _generator(0))


@pytest.mark.parametrize("draw", SAMPLERS, ids=SAMPLER_IDS)
def test_samplers_reject_floating_point_tokens(draw: Draw) -> None:
    """A floating-point x0 raises ValueError."""
    x0 = torch.zeros(2, 8)
    with pytest.raises(ValueError, match="integer dtype"):
        draw(x0, 4, _generator(0))


@pytest.mark.parametrize("draw", SAMPLERS, ids=SAMPLER_IDS)
def test_samplers_accept_any_integer_dtype_and_return_long(draw: Draw) -> None:
    """Every integer dtype is accepted, and xt is always torch.long."""
    x0 = _clean_tokens(0).to(torch.int32)
    xt = draw(x0, 4, _generator(0))[0]
    assert xt.dtype == torch.long


def test_samplers_reject_negative_mask_id() -> None:
    """The mask id must be a non-negative token id."""
    with pytest.raises(ValueError, match="mask_id"):
        sample_validation_noise(_clean_tokens(0), 4, -1, _generator(0))


@pytest.mark.parametrize(
    "sampler", [sample_block_noise, sample_complementary_noise], ids=["bd3", "complementary"]
)
def test_t_range_must_be_ordered_and_inside_the_unit_interval(
    sampler: Callable[..., object],
) -> None:
    """t_min > t_max, or a bound outside [0, 1], raises ValueError."""
    x0 = _clean_tokens(0)
    with pytest.raises(ValueError, match="t_min"):
        sampler(x0, 4, MASK_ID, 0.8, 0.2, _generator(0))
    with pytest.raises(ValueError, match="t_min"):
        sampler(x0, 4, MASK_ID, -0.1, 0.5, _generator(0))
    with pytest.raises(ValueError, match="t_min"):
        sampler(x0, 4, MASK_ID, 0.0, 1.5, _generator(0))
