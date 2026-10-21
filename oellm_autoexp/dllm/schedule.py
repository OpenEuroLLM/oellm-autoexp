"""Block-size schedule for block diffusion (spec section 8).

The schedule maps the training progress ``iteration / train_iters`` to a block size B. The
module has no dependency on the GPU stack, so it runs on the login node and in CPU-only CI.
"""

from __future__ import annotations


def parse_block_schedule(spec: str) -> list[tuple[float, int]]:
    """Parse a ``--bd-block-schedule`` string into a list of entries.

    The string is a comma-separated list of ``fraction:block_size`` items, for example
    ``"0:4,0.15:8"``. Spaces around items and around ``:`` are ignored. An empty string or a
    string of whitespace means that no schedule is set.

    Args:
        spec: The schedule string. The first fraction must be exactly 0, and the fractions must
            strictly increase.

    Returns:
        The entries in the order they were written. The list is empty if ``spec`` is empty or
        contains only whitespace.

    Raises:
        ValueError: If an item does not contain exactly one ``:``, a fraction is not a float in
            ``[0, 1)``, a block size is not an integer ``>= 1``, the first fraction is not
            exactly 0, or the fractions do not strictly increase.
    """
    if not spec.strip():
        return []
    entries: list[tuple[float, int]] = []
    for position, item in enumerate(spec.split(","), start=1):
        entries.append(_parse_item(item, position))
    _check_entries(entries)
    return entries


def block_size_at(
    iteration: int,
    train_iters: int,
    schedule: list[tuple[float, int]],
    default_block_size: int,
) -> int:
    """Return the block size B that is in effect at a training iteration.

    With a non-empty schedule, the fraction is ``iteration / train_iters``. The block size is the
    one of the last entry whose fraction is less than or equal to that value.

    Args:
        iteration: The current training iteration.
        train_iters: The total number of training iterations.
        schedule: Entries from ``parse_block_schedule``. An empty list means a constant block
            size.
        default_block_size: The block size that is returned when the schedule is empty.

    Returns:
        The block size for ``iteration``.

    Raises:
        ValueError: If the schedule is not empty and ``train_iters`` is not positive, if the
            schedule breaks the rules of ``parse_block_schedule``, or if no entry covers the
            fraction (for example for a negative iteration).
    """
    if not schedule:
        return default_block_size
    if train_iters <= 0:
        raise ValueError(
            f"train_iters must be positive when a block schedule is set, got {train_iters}"
        )
    _check_entries(schedule)
    fraction = iteration / train_iters
    chosen: int | None = None
    for entry_fraction, block_size in schedule:
        if entry_fraction > fraction:
            break
        chosen = block_size
    if chosen is None:
        raise ValueError(
            f"no block schedule entry covers iteration {iteration} of {train_iters} "
            f"(fraction {fraction!r})"
        )
    return chosen


def validate_block_sizes(
    seq_len_effective: int,
    default_block_size: int,
    schedule: list[tuple[float, int]],
) -> None:
    """Check that every block size divides the effective sequence length.

    Spec section 8 requires ``L % B == 0`` for every block size B in the schedule. The default
    block size is checked as well, because it is used when the schedule is empty. Call this
    once at startup.

    Args:
        seq_len_effective: The effective length L, which is ``seq_length - 1``.
        default_block_size: The constant block size (``--bd-block-size``).
        schedule: Entries from ``parse_block_schedule``.

    Raises:
        ValueError: If ``seq_len_effective`` is below 1, or if the default block size or a
            scheduled block size is below 1 or does not divide ``seq_len_effective``.
    """
    if seq_len_effective < 1:
        raise ValueError(f"the effective length must be at least 1, got {seq_len_effective}")
    _check_block_size(default_block_size, seq_len_effective, "default block size")
    for position, (fraction, block_size) in enumerate(schedule, start=1):
        label = f"block size of schedule entry {position} (fraction {fraction!r})"
        _check_block_size(block_size, seq_len_effective, label)


def _parse_item(item: str, position: int) -> tuple[float, int]:
    """Parse one ``fraction:block_size`` item of a schedule string.

    Args:
        item: The text between two commas, untrimmed.
        position: The 1-based position of the item, used in error messages.

    Returns:
        The fraction and the block size. The range checks happen in ``_check_entries``.

    Raises:
        ValueError: If the item does not contain exactly one ``:``, or if the fraction or the
            block size is not a number.
    """
    shown = item.strip()
    parts = item.split(":")
    if len(parts) != 2:
        raise ValueError(
            f"block schedule item {position} ({shown!r}) must contain exactly one ':' "
            "between the fraction and the block size, for example '0.15:8'"
        )
    fraction_text, block_text = parts[0].strip(), parts[1].strip()
    try:
        fraction = float(fraction_text)
    except ValueError:
        raise ValueError(
            f"block schedule item {position} ({shown!r}): fraction {fraction_text!r} "
            "is not a number"
        ) from None
    try:
        block_size = int(block_text)
    except ValueError:
        raise ValueError(
            f"block schedule item {position} ({shown!r}): block size {block_text!r} "
            "is not an integer"
        ) from None
    return fraction, block_size


def _check_entries(entries: list[tuple[float, int]]) -> None:
    """Check the rules that every block schedule must follow.

    Args:
        entries: The ``(fraction, block_size)`` pairs to check. An empty list passes.

    Raises:
        ValueError: If the first fraction is not exactly 0, a fraction is not in ``[0, 1)``, a
            block size is not an integer ``>= 1``, or the fractions do not strictly increase.
    """
    if not entries:
        return
    first_fraction = entries[0][0]
    if first_fraction != 0.0:
        raise ValueError(
            f"the first block schedule fraction must be exactly 0, got {first_fraction!r}"
        )
    previous: float | None = None
    for position, (fraction, block_size) in enumerate(entries, start=1):
        if not 0.0 <= fraction < 1.0:
            raise ValueError(
                f"block schedule entry {position}: fraction {fraction!r} must be in [0, 1)"
            )
        if not isinstance(block_size, int) or block_size < 1:
            raise ValueError(
                f"block schedule entry {position}: block size {block_size!r} must be an "
                "integer >= 1"
            )
        if previous is not None and fraction <= previous:
            raise ValueError(
                f"block schedule entry {position}: fraction {fraction!r} must be greater "
                f"than the previous fraction {previous!r}"
            )
        previous = fraction


def _check_block_size(block_size: int, seq_len_effective: int, label: str) -> None:
    """Raise if a block size is below 1 or does not divide the length.

    Args:
        block_size: The block size to check.
        seq_len_effective: The effective length L.
        label: How the block size is named in the error message.

    Raises:
        ValueError: If the block size is below 1, or if it does not divide
            ``seq_len_effective``.
    """
    if block_size < 1:
        raise ValueError(f"{label} must be at least 1, got {block_size}")
    if seq_len_effective % block_size != 0:
        raise ValueError(
            f"{label} is {block_size}, which does not divide the effective length "
            f"{seq_len_effective}"
        )
