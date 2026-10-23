"""End-to-end test of the flag decontamination on a synthetic Megatron dataset (CPU; GPU if any).

    python -m pytest scripts/flag_decontam/test_flag_decontam.py -q
    python scripts/flag_decontam/test_flag_decontam.py        # without pytest (the MN5 image)
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_table import make_table, variants  # noqa: E402
from common import Index, filter_index, ngram_hashes, runs, write_index  # noqa: E402
import scan  # noqa: E402

V = 1000
WS = 3            # a whitespace-only piece
N = 13


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
    for n in (13, 24, 40):
        h = ngram_hashes(c, n)
        assert h.unique().numel() == h.numel()          # no collisions on random windows
        if torch.cuda.is_available():
            assert torch.equal(h, ngram_hashes(c.cuda(), n).cpu())


def test_runs():
    # segment 0 (n = 13): windows 0..9 contiguous, a gap of 3 (bridged), 13..15; then 40..41 (new run)
    pos = torch.tensor([*range(10), 13, 14, 15, 40, 41, 5])
    seg = torch.tensor([0] * 15 + [1])
    seq = torch.zeros(16, dtype=torch.int64)
    n = torch.tensor([13, 13])
    r_seq, r_seg, a, b, w = runs(pos, seg, seq, n)
    got = sorted(zip(r_seg.tolist(), a.tolist(), b.tolist(), w.tolist()))
    assert got == [(0, 0, 28, 13), (0, 40, 54, 2), (1, 5, 18, 1)]


def test_variants():
    assert variants('He said "hi"\nok') == ['He said "hi"\nok', 'He said \\"hi\\"\\nok']
    assert variants("Janet’s ducks") == ["Janet’s ducks", "Janet\\u2019s ducks"]
    assert variants("Цена уговора је била") == ["Цена уговора је била"]      # no \\uXXXX form for non-ASCII text


def make_setup(tmp_path):
    rng = np.random.default_rng(1)
    lut = make_lut()
    seg_a = rng.integers(4, 501, 40)          # a question
    seg_b = rng.integers(4, 501, 300)         # a passage
    seg_c = rng.integers(4, 501, 30)          # placed across a document boundary: must not match
    docs = [rng.integers(4, V, int(rng.integers(50, 3000))).astype(np.int64) for _ in range(200)]

    def plant(d, toks, at=None):
        at = len(docs[d]) // 2 if at is None else at
        docs[d] = np.concatenate([docs[d][:at], np.array(toks, np.int64), docs[d][at:]])

    plant(17, variant(rng, seg_a))                                    # verbatim copy: drop
    modified = seg_a.copy()
    modified[20] = 4 if seg_a[20] != 4 else 5
    plant(30, variant(rng, modified))                                 # one word changed: keep
    scattered = np.concatenate([np.concatenate([seg_a[i:i + N], rng.integers(4, 501, 40)]) for i in (0, 13, 26)])
    plant(60, scattered)                                              # its n-grams, scattered: keep
    plant(99, variant(rng, seg_b[40:260]))                            # 220 of 300 tokens verbatim: drop (abs_run)
    plant(50, seg_c[:15], at=len(docs[50]))
    plant(51, seg_c[15:], at=0)
    for d in range(len(docs)):
        docs[d] = np.concatenate([docs[d], [2]])   # <eos>, a control token
    docs[120][5] = V + 7                           # an id outside the vocabulary: dropped, reported

    root = tmp_path / "flag"
    (root / "src").mkdir(parents=True)
    prefix = str(root / "src" / "shard_0000_text_document")
    write_dataset(prefix, docs)
    (root / "datamix.txt").write_text("1.0 src/shard_0000_text_document\n")

    work = tmp_path / "work"
    (work / "table").mkdir(parents=True)
    np.save(work / "table" / "lut.npy", lut)
    hashes, segids = [], []
    for i, s in enumerate([seg_a, seg_b, seg_c]):
        h = np.unique(ngram_hashes(torch.from_numpy(s.astype(np.int64)), N).numpy())
        hashes.append(h)
        segids.append(np.full(len(h), i, np.int32))
    uniq, csr_off, seg, n_grams, _ = make_table(np.concatenate(hashes), np.concatenate(segids), 3, 10)
    np.savez(work / "table" / "table.npz", hash=uniq, csr_off=csr_off, csr_seg=seg)
    np.savez(work / "table" / "segments.npz", src=np.array([0, 1, 2], np.int32), split=np.zeros(3, np.int16),
             n=np.array([N, N, N], np.int8), n_tokens=np.array([40, 300, 30]), n_grams=n_grams, n_grams_built=n_grams)
    refs = tmp_path / "refs.jsonl.gz"
    with gzip.open(refs, "wt") as f:
        for i in range(3):
            f.write(json.dumps({"src": f"bench{i}", "split": "test", "text": f"segment {i}"}) + "\n")
    with gzip.open(str(refs).replace(".jsonl.gz", ".attrib.jsonl.gz"), "wt") as f:   # segment 0 is also in "alias"
        for i in range(3):
            f.write(json.dumps({"also_in": [[f"bench{i}", "test", "x", 1]] + ([["alias", "test", "y", 1]] if i == 0 else [])
                                + [["trainonly", "train", "z", 0]]}) + "\n")
    json.dump({"sources": ["bench0", "bench1", "bench2"], "splits": ["test"], "refs": str(refs)},
              open(work / "table" / "sources.json", "w"))
    return root, work, prefix, tmp_path


def test_end_to_end(tmp_path):
    root, work, prefix, tmp = make_setup(tmp_path)
    units = scan.plan_units([prefix], 5000, 0)
    assert len(units) > 10 and units[-1][2] == Index(prefix + ".idx").seq_count
    args = argparse.Namespace(mode="scan", work=str(work), root=str(root), part=0, min_run=20, readers=2)
    scan.worker(0, 1, units, args)
    hits = np.concatenate([np.load(f)["hits"] for f in sorted((work / "hits").glob("*/*.npz"))])
    best = {(int(s), int(g)): int(r) for s, g, r, _, _ in hits}
    assert best[(17, 0)] == 40 and best[(99, 1)] == 220 and (60, 0) not in best
    assert best.get((30, 0), 0) < 30 and (50, 2) not in best and (51, 2) not in best
    anomalies = json.load(open(work / "anomalies" / "scan_part000_gpu0.json"))
    assert len(anomalies) == 1 and anomalies[0]["max"] == V + 7
    out = tmp / "flag-decon"
    fin = [sys.executable, os.path.join(HERE, "finalize.py"), "--work", str(work), "--root", str(root), "--out", str(out)]
    subprocess.run(fin + ["--dry-run", "--frac", "0.5,0.9"], check=True)
    assert not out.exists()
    alltsv = open(work / "_report" / "frac_0.9" / "benchmarks_all.tsv").read()
    assert "alias\t1\t1\t100.00\t1" in alltsv and "trainonly" not in alltsv
    for fr in ("0.5", "0.9"):
        assert json.load(open(work / "_report" / f"frac_{fr}" / "report.json"))["docs_dropped"] == 2
    subprocess.run(fin, check=True)
    removed = np.load(str(out / "src" / "shard_0000_text_document.removed.npz"))
    assert sorted(set(removed["doc"].tolist())) == [17, 99]
    old, new = Index(prefix + ".idx"), Index(str(out / "src" / "shard_0000_text_document.idx"))
    assert new.doc_count - 1 == 198 and os.path.islink(str(out / "src" / "shard_0000_text_document.bin"))
    kept = [d for d in range(200) if d not in (17, 99)]
    newbin = str(out / "src" / "shard_0000_text_document.bin")
    for i in (0, 16, 17, 97, 98, 197):
        o = kept[i]
        a = np.fromfile(prefix + ".bin", np.int32, count=int(old.lengths[o]), offset=int(old.pointers[o]))
        b = np.fromfile(newbin, np.int32, count=int(new.lengths[i]), offset=int(new.pointers[i]))
        assert np.array_equal(a, b)
    info = json.load(open(out / "src" / "shard_0000.info.json"))
    assert info["idx"]["size"] == os.path.getsize(out / "src" / "shard_0000_text_document.idx")
    assert info["decontamination"]["dropped_documents"] == 2

    # Megatron reads the new index against the old .bin
    sys.path.insert(0, os.path.join(HERE, "..", "..", "submodules", "Megatron-LM"))
    from megatron.core.datasets.indexed_dataset import IndexedDataset
    ds = IndexedDataset(str(out / "src" / "shard_0000_text_document"))
    assert len(ds.document_indices) - 1 == 198
    assert np.array_equal(ds.get(17), np.fromfile(prefix + ".bin", np.int32, count=int(old.lengths[18]),
                                                  offset=int(old.pointers[18])))


def test_resume_and_count(tmp_path):
    root, work, prefix, tmp = make_setup(tmp_path)
    units = scan.plan_units([prefix], 5000, 20000)
    assert 20000 <= sum(u[3] for u in units) < 23000          # stops at the first sequence past the limit
    args = argparse.Namespace(mode="count", work=str(work), root=str(root), part=0, min_run=20, readers=2)
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
