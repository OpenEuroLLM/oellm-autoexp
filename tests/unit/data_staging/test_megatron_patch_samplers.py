"""The data-staging patches on Megatron's samplers / argument parsing (no GPU, no data).

The cyclic sampler must read the windowed-shuffled dataset in order, as the single sampler
does within an epoch, and repeat it at the end; the prefetch wrapper must derive the
prefetcher's data-parallel layout from the Megatron arguments.
"""

from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
MEGATRON = REPO / "submodules" / "Megatron-LM"

samplers = pytest.importorskip("megatron.training.datasets.data_samplers")


def _patched(monkeypatch):
    monkeypatch.setenv("OELLM_SHUFFLE_LANES", "4")
    from oellm_autoexp.data_staging import megatron_patch

    orig = samplers.MegatronPretrainingRandomSampler.__iter__
    monkeypatch.setattr(samplers.MegatronPretrainingRandomSampler, "__iter__", orig)
    monkeypatch.delattr(samplers.MegatronPretrainingRandomSampler, "_oellm_sequential", raising=False)
    assert megatron_patch.apply_cyclic_sequential()


def _cyclic(total, consumed, mbs, rank, dp):
    return samplers.MegatronPretrainingRandomSampler(
        list(range(total)), total_samples=total, consumed_samples=consumed, micro_batch_size=mbs,
        data_parallel_rank=rank, data_parallel_size=dp, data_sharding=True)


def _single(total, consumed, mbs, rank, dp):
    return samplers.MegatronPretrainingSampler(
        total_samples=total, consumed_samples=consumed, micro_batch_size=mbs,
        data_parallel_rank=rank, data_parallel_size=dp)


@pytest.mark.parametrize("consumed", [0, 24])
def test_cyclic_matches_single_within_an_epoch(monkeypatch, consumed):
    _patched(monkeypatch)
    for rank in range(3):
        got = list(_cyclic(100, consumed, 4, rank, 3))
        want = list(_single(100, consumed, 4, rank, 3))
        assert got == want[: len(got)] and len(got) == len(want)


def test_cyclic_repeats_at_the_end(monkeypatch):
    _patched(monkeypatch)
    s = _cyclic(26, 0, 2, 1, 2)  # 6 whole windows of 4; the last 2 samples are dropped
    first = list(s)
    assert first == [[2 + 4 * w, 3 + 4 * w] for w in range(6)]
    assert s.consumed_samples == 24
    assert list(s) == first  # epoch 2 (cyclic_iter re-enters __iter__): same order again
    # resuming mid-epoch-2 continues where it left off
    assert list(_cyclic(26, 24 + 8, 2, 1, 2)) == first[2:]


def test_cyclic_unpatched_without_lanes(monkeypatch):
    monkeypatch.delenv("OELLM_SHUFFLE_LANES", raising=False)
    from oellm_autoexp.data_staging import megatron_patch

    assert not megatron_patch.apply_cyclic_sequential()


def _wrapper(monkeypatch, argv):
    monkeypatch.delenv("OELLM_PREFETCH_AUTOSTART", raising=False)
    monkeypatch.setattr(sys, "argv", ["pretrain_hybrid_prefetch.py", *argv])
    spec = importlib.util.spec_from_file_location(
        "pretrain_hybrid_prefetch", REPO / "scripts" / "pretrain_hybrid_prefetch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # no __main__: patches only, no training
    return mod


def test_wrapper_dp_layout_and_prefix(monkeypatch):
    m = _wrapper(monkeypatch, ["--pipeline-model-parallel-size", "4", "--micro-batch-size", "4",
                               "--per-split-data-args-path", "/x/split.json"])
    # PP4 on 8 nodes x 4 GPUs: dp 8, node n holds dp ranks 4n%8 .. 4n%8+3
    assert m._node_dp_ranks(0, 4, 32) == ([0, 1, 2, 3], 8)
    assert m._node_dp_ranks(1, 4, 32) == ([4, 5, 6, 7], 8)
    assert m._node_dp_ranks(2, 4, 32) == ([0, 1, 2, 3], 8)
    assert m._data_prefix() == "json:/x/split.json"
    m = _wrapper(monkeypatch, ["--tensor-model-parallel-size", "2", "--data-path", "0.3", "/a",
                               "0.7", "/b", "--micro-batch-size", "8"])
    assert m._node_dp_ranks(1, 4, 8) == ([2, 3], 4)
    assert m._data_prefix() == "/a,/b"
