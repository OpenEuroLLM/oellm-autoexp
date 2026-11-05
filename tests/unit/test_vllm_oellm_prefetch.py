"""submodules/vllm .../model_loader/oellm_prefetch.py: the parallel page-cache read of a
checkpoint before vLLM loads it (standard library only, loaded here by path: no vLLM install)."""

from __future__ import annotations

import importlib.util
import multiprocessing as mp
import os
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "submodules/vllm/vllm/model_executor/model_loader/oellm_prefetch.py"

pytestmark = pytest.mark.skipif(not SRC.exists(), reason="vllm submodule not checked out")


def _module():
    spec = importlib.util.spec_from_file_location("oellm_prefetch", SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _files(tmp_path: Path, sizes: list[int]) -> list[str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, size in enumerate(sizes):
        p = tmp_path / f"model-{i:05d}.safetensors"
        p.write_bytes(os.urandom(size))
        paths.append(str(p))
    return paths


def test_reads_every_byte_once_and_then_trusts_the_marker(tmp_path):
    m = _module()
    # sizes off the range size: the last range of each file is short
    files = _files(tmp_path / "ckpt", [0, 1, 5000, 3 * 4096 + 7])
    lock_dir = str(tmp_path / "locks")
    assert m.prefetch_checkpoint_parallel(files, num_threads=3, chunk_size=4096, lock_dir=lock_dir)
    markers = list(Path(lock_dir).glob("oellm_prefetch_*.done"))
    assert len(markers) == 1
    import json

    info = json.loads(markers[0].read_text())
    assert info["bytes"] == sum(os.path.getsize(f) for f in files)
    # a second process of the node (another data-parallel engine): already read
    assert not m.prefetch_checkpoint_parallel(files, num_threads=3, chunk_size=4096, lock_dir=lock_dir)


def test_a_stale_marker_or_a_changed_file_reads_again(tmp_path):
    m = _module()
    files = _files(tmp_path / "ckpt", [10000, 20000])
    lock_dir = str(tmp_path / "locks")
    assert m.prefetch_checkpoint_parallel(files, num_threads=2, chunk_size=4096, lock_dir=lock_dir)
    marker = next(Path(lock_dir).glob("oellm_prefetch_*.done"))
    old = time.time() - m.MARKER_MAX_AGE_S - 1
    os.utime(marker, (old, old))
    assert m.prefetch_checkpoint_parallel(files, num_threads=2, chunk_size=4096, lock_dir=lock_dir)
    # a re-exported checkpoint (other size / mtime) is another key
    Path(files[1]).write_bytes(os.urandom(30000))
    assert m.prefetch_checkpoint_parallel(files, num_threads=2, chunk_size=4096, lock_dir=lock_dir)


def _worker(args):
    src, files, lock_dir = args
    spec = importlib.util.spec_from_file_location("oellm_prefetch", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.prefetch_checkpoint_parallel(files, num_threads=4, chunk_size=1 << 16, lock_dir=lock_dir)


def test_concurrent_engines_of_a_node_read_it_once(tmp_path):
    files = _files(tmp_path / "ckpt", [3 << 20, 2 << 20, 1 << 20])
    lock_dir = str(tmp_path / "locks")
    with mp.get_context("spawn").Pool(4) as pool:
        results = pool.map(_worker, [(str(SRC), files, lock_dir)] * 4)
    assert sorted(results) == [False, False, False, True], results


def test_enabled_auto_on_gpfs_only_and_the_switches(monkeypatch):
    m = _module()
    monkeypatch.delenv("OELLM_PREFETCH", raising=False)
    assert m.enabled("gpfs") and not m.enabled("ext4") and not m.enabled("nfs4")
    monkeypatch.setenv("OELLM_PREFETCH", "0")
    assert not m.enabled("gpfs")
    monkeypatch.setenv("OELLM_PREFETCH", "1")
    assert m.enabled("ext4")
