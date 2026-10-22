"""End-to-end test of the flag decontamination on a synthetic Megatron dataset (CPU; GPU if any).

    python -m pytest scripts/flag_decontam/test_flag_decontam.py -q
    python scripts/flag_decontam/test_flag_decontam.py        # without pytest (the MN5 image)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_table import make_table  # noqa: E402
from common import SHORT_BELOW, Index, filter_index, ngram_hashes, write_index  # noqa: E402
import scan  # noqa: E402

V = 1000
WS = 3            # a whitespace-only piece


def make_lut():
    lut = np.full(V, -1, np.int64)
    lut[4:501] = np.arange(4, 501)            # '▁word'
    lut[501:] = np.arange(501, V) - 497       # the same words without '▁' -> same canonical id
    return lut


def variant(rng, canon):
    """A token sequence with canonical stream `canon`: random '▁' twins, whitespace pieces."""
    out = []
    for c in canon:
        if rng.random() < 0.3:
            out.append(WS)
        out.append(c + 497 if (c <= 502 and rng.random() < 0.5) else c)
    return out


def write_dataset(path, docs):
    lengths = np.array([len(d) for d in docs], np.int32)
    pointers = np.concatenate([[0], np.cumsum(lengths.astype(np.int64) * 4)[:-1]])
    np.concatenate(docs).astype(np.int32).tofile(path + ".bin")
    write_index(path + ".idx", 4, lengths, pointers, np.arange(len(docs) + 1))
    info = {"idx": {"size": os.path.getsize(path + ".idx")}, "bin": {"size": os.path.getsize(path + ".bin")}}
    json.dump(info, open(path.removesuffix("_text_document") + ".info.json", "w"))


def test_hash_cpu_gpu_agree():
    c = torch.randint(0, 200_000, (100_000,), dtype=torch.int64)
    for n in (8, 13):
        h = ngram_hashes(c, n)
        assert h.unique().numel() == h.numel()          # no collisions on random windows
        if torch.cuda.is_available():
            assert torch.equal(h, ngram_hashes(c.cuda(), n).cpu())


def make_setup(tmp_path):
    rng = np.random.default_rng(1)
    lut = make_lut()
    seg_a = rng.integers(4, 501, 40)          # long segment -> n = 13
    seg_b = rng.integers(4, 501, 12)          # short segment -> n = 8
    seg_c = rng.integers(4, 501, 30)          # placed across a document boundary: must not match
    docs = [rng.integers(4, V, int(rng.integers(50, 3000))).astype(np.int64) for _ in range(200)]

    def plant(d, toks, at=None):
        at = len(docs[d]) // 2 if at is None else at
        docs[d] = np.concatenate([docs[d][:at], np.array(toks, np.int64), docs[d][at:]])

    plant(17, variant(rng, seg_a))
    plant(150, variant(rng, seg_a[:14]))       # 2 of 28 n-grams: below 50 %
    plant(99, variant(rng, seg_b))
    plant(50, seg_c[:15], at=len(docs[50]))
    plant(51, seg_c[15:], at=0)
    for d in range(len(docs)):
        docs[d] = np.concatenate([docs[d], [2]])   # <eos>, a control token

    root = tmp_path / "flag"
    (root / "src").mkdir(parents=True)
    prefix = str(root / "src" / "shard_0000_text_document")
    write_dataset(prefix, docs)
    (root / "datamix.txt").write_text(f"1.0 src/shard_0000_text_document\n")

    work = tmp_path / "work"
    (work / "table").mkdir(parents=True)
    np.save(work / "table" / "lut.npy", lut)
    hashes, segids, ns = [], [], []
    for i, s in enumerate([seg_a, seg_b, seg_c]):
        n = 13 if len(s) >= SHORT_BELOW else 8
        h = np.unique(ngram_hashes(torch.from_numpy(s.astype(np.int64)), n).numpy())
        hashes.append(h)
        segids.append(np.full(len(h), i, np.int32))
        ns.append(n)
    uniq, csr_off, seg, n_grams, _ = make_table(np.concatenate(hashes), np.concatenate(segids), 3, 10)
    np.savez(work / "table" / "table.npz", hash=uniq, csr_off=csr_off, csr_seg=seg)
    np.savez(work / "table" / "segments.npz", src=np.array([0, 1, 2], np.int32), split=np.zeros(3, np.int16),
             n=np.array(ns, np.int8), n_tokens=np.array([40, 12, 30]), n_grams=n_grams, n_grams_built=n_grams)
    import gzip
    refs = tmp_path / "refs.jsonl.gz"
    with gzip.open(refs, "wt") as f:
        for i in range(3):
            f.write(json.dumps({"src": f"bench{i}", "split": "test", "text": f"segment {i}"}) + "\n")
    json.dump({"sources": ["bench0", "bench1", "bench2"], "splits": ["test"], "refs": str(refs)},
              open(work / "table" / "sources.json", "w"))
    return root, work, prefix, tmp_path


def test_end_to_end(tmp_path):
    root, work, prefix, tmp = make_setup(tmp_path)
    (tmp / "shards.txt").write_text(prefix + "\n")
    units = scan.plan_units([prefix], 5000, 0)
    assert len(units) > 10 and units[-1][2] == Index(prefix + ".idx").seq_count
    args = argparse.Namespace(mode="scan", work=str(work), root=str(root), part=0, min_hits=3)
    scan.worker(0, 1, units, args)
    out = tmp / "flag-decon"
    subprocess.run([sys.executable, os.path.join(HERE, "finalize.py"), "--work", str(work), "--root", str(root),
                    "--out", str(out)], check=True)
    removed = np.load(str(out / "src" / "shard_0000_text_document.removed.npz"))
    assert sorted(set(removed["doc"].tolist())) == [17, 99]
    old, new = Index(prefix + ".idx"), Index(str(out / "src" / "shard_0000_text_document.idx"))
    assert new.doc_count - 1 == 198 and os.path.islink(str(out / "src" / "shard_0000_text_document.bin"))
    kept = [d for d in range(200) if d not in (17, 99)]
    for i in (0, 16, 17, 97, 98, 197):
        o, n = kept[i], i
        a = np.fromfile(prefix + ".bin", np.int32, count=int(old.lengths[o]), offset=int(old.pointers[o]))
        b = np.fromfile(str(out / "src" / "shard_0000_text_document.bin"), np.int32, count=int(new.lengths[n]), offset=int(new.pointers[n]))
        assert np.array_equal(a, b)
    info = json.load(open(out / "src" / "shard_0000.info.json"))
    assert info["idx"]["size"] == os.path.getsize(out / "src" / "shard_0000_text_document.idx")
    assert info["decontamination"]["dropped_documents"] == 2

    # Megatron reads the new index against the old .bin
    sys.path.insert(0, os.path.join(HERE, "..", "..", "submodules", "Megatron-LM"))
    from megatron.core.datasets.indexed_dataset import IndexedDataset
    ds = IndexedDataset(str(out / "src" / "shard_0000_text_document"))
    assert len(ds.document_indices) - 1 == 198
    assert np.array_equal(ds.get(17), np.fromfile(prefix + ".bin", np.int32, count=int(old.lengths[18]), offset=int(old.pointers[18])))


def test_resume_and_count(tmp_path):
    root, work, prefix, tmp = make_setup(tmp_path)
    units = scan.plan_units([prefix], 5000, 20000)
    assert sum(u[3] for u in units) >= 20000 and sum(u[3] for u in units[:-1]) < 20000
    args = argparse.Namespace(mode="count", work=str(work), root=str(root), part=0, min_hits=3)
    scan.worker(0, 1, units, args)
    counts = np.load(work / "counts" / "part000_gpu0.npy")
    assert counts.shape[0] == len(np.load(work / "table" / "table.npz")["hash"])


def test_filter_index_multi_sequence_docs():
    class I:  # 3 documents of 2, 1, 2 sequences
        lengths = np.array([5, 6, 7, 8, 9], np.int32)
        pointers = np.array([0, 20, 44, 72, 104], np.int64)
        doc_idx = np.array([0, 2, 3, 5], np.int64)
    lengths, pointers, doc_idx = filter_index(I, np.array([True, False, True]))
    assert lengths.tolist() == [5, 6, 8, 9] and pointers.tolist() == [0, 20, 72, 104]
    assert doc_idx.tolist() == [0, 2, 4]


if __name__ == "__main__":
    import inspect
    import tempfile
    from pathlib import Path
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            with tempfile.TemporaryDirectory() as d:
                fn(*([Path(d)] if inspect.signature(fn).parameters else []))
            print(f"PASSED {name}", flush=True)
    print("all passed")
