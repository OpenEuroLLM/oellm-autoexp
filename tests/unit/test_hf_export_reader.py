"""Exact CPU/DCP round trips for the opt-in export reader."""
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner

from oellm_autoexp.hf_export.fast_fsdp_reader import ExportFileSystemReader
from oellm_autoexp.hf_export.fsdp_dtensor import iter_layers


def save(path, offset=0):
    data = {
        'model.module.embedding.word_embeddings.weight': torch.arange(24).reshape(6, 4).float() + offset,
        'model.module.decoder.layers.0.weight': torch.arange(33).reshape(3, 11).to(torch.bfloat16) + offset,
        'model.module.decoder.layers.1.weight': torch.arange(10).reshape(5, 2).float() + offset,
        'optimizer.state': torch.ones(5),
    }
    dcp.save(data, storage_writer=dcp.FileSystemWriter(path, single_file_per_rank=False))
    return data


def test_parallel_reader_matches_legacy(tmp_path, monkeypatch):
    save(tmp_path / 'ckpt')
    monkeypatch.setenv('HF_EXPORT_READER', 'legacy')
    old = list(iter_layers(tmp_path / 'ckpt'))
    monkeypatch.setenv('HF_EXPORT_READER', 'fast')
    monkeypatch.setenv('HF_EXPORT_READ_THREADS', '4')
    new = list(iter_layers(tmp_path / 'ckpt'))
    assert [i for i, _ in old] == [i for i, _ in new]
    for (_, a), (_, b) in zip(old, new):
        assert a.keys() == b.keys()
        for k in a:
            assert a[k].dtype == b[k].dtype
            assert torch.equal(a[k].view(torch.uint8), b[k].view(torch.uint8))


def test_cache_reset_and_multiple_files(tmp_path):
    first, second = tmp_path / 'one', tmp_path / 'two'
    a, b = save(first), save(second, 7)
    r = ExportFileSystemReader(first, threads=4)
    for _ in range(2):
        out = {k: torch.empty_like(v) for k, v in a.items()}
        dcp.load(out, storage_reader=r, planner=DefaultLoadPlanner(flatten_state_dict=False))
        assert all(torch.equal(out[k], v) for k, v in a.items())
    assert r.metadata_disk_reads == 1
    r.reset(second)
    out = {k: torch.empty_like(v) for k, v in b.items()}
    dcp.load(out, storage_reader=r, planner=DefaultLoadPlanner(flatten_state_dict=False))
    assert r.metadata_disk_reads == 2
    assert all(torch.equal(out[k], v) for k, v in b.items())
