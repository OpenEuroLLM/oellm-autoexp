"""Fork a pool of workers over independent chunks of a sweep.

Resolving a sweep point and rendering its job script are both pure CPU work on
data that is already in memory, so ``fork`` is the right tool: the children
inherit the loaded modules, the registered resolvers and the warm config
caches, and start doing useful work immediately.

``HYDRA_STAGED_SWEEP_WORKERS`` controls the pool: unset or ``0`` picks one
worker per available CPU, ``1`` keeps everything in-process.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

LOGGER = logging.getLogger(__name__)

__all__ = ["run_chunks", "split_evenly", "worker_count"]

T = TypeVar("T")

WORKERS_ENV = "HYDRA_STAGED_SWEEP_WORKERS"


def worker_count(chunks: int, items: int, *, requested: int | None = None) -> int:
    """How many processes to use; 1 means stay in-process."""
    if requested is None:
        configured = os.environ.get(WORKERS_ENV)
        requested = int(configured) if configured else 0
    if requested < 0:
        raise ValueError(f"{WORKERS_ENV} must not be negative: {requested}")
    if requested == 0:
        requested = (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else (os.cpu_count() or 1)
        )
    if "fork" not in multiprocessing.get_all_start_methods():
        # Without fork a worker would have to re-import and re-register
        # everything, which costs more than a sweep this size saves.
        return 1
    # Below roughly two items per worker the process overhead dominates.
    if chunks < 2 or items < 8:
        return 1
    return max(1, min(requested, chunks))


def run_chunks(func: Callable[[Any], T], chunks: Sequence[Any], workers: int) -> list[T]:
    """Apply ``func`` to each chunk, in a fork pool when ``workers > 1``.

    ``func`` and the data it closes over are inherited through the fork, so
    only each chunk and its result cross the process boundary.
    """
    if workers <= 1:
        return [func(chunk) for chunk in chunks]
    context = multiprocessing.get_context("fork")
    with context.Pool(processes=workers) as pool:
        return pool.map(func, chunks, chunksize=1)


def split_evenly(items: Sequence[T], groups: int) -> list[list[T]]:
    """Deal ``items`` round-robin into ``groups`` lists, preserving order."""
    if groups <= 1:
        return [list(items)]
    buckets: list[list[T]] = [[] for _ in range(groups)]
    for position, item in enumerate(items):
        buckets[position % groups].append(item)
    return [bucket for bucket in buckets if bucket]
