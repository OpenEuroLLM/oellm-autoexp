"""End-to-end check of the cross-document packing pipeline (docs/cross_doc_seqs/README.md) on a tiny real
Megatron blend: the offline stream replay reproduces Megatron's samples, the plan keeps every token of
its window exactly once and never puts a copy next to its source, and Megatron's PackingPlanDataset
serves the plan (all boundary modes) while every other sample stays as it was.

Needs torch + the Megatron submodule (CPU only), e.g. ~/.miniconda3/envs/torch/bin/python -m pytest.
"""

from __future__ import annotations

import collections
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "submodules", "Megatron-LM"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "korbi", "cross_doc_seqs"))

pytest.importorskip("megatron.core.datasets.gpt_dataset")
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder  # noqa: E402
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig  # noqa: E402
from megatron.core.datasets.indexed_dataset import DType, IndexedDatasetBuilder  # noqa: E402
from megatron.core.datasets.packing_plan import PackingPlanDataset  # noqa: E402
from megatron.core.datasets.utils import compile_helpers  # noqa: E402

import build_plan  # noqa: E402
import extract_doc_stream  # noqa: E402
import check_plan  # noqa: E402
import verify_stream  # noqa: E402
from cds_data import (BlendStream, SamplerGeometry, StepSampler, find_cache_entry, load_stream,  # noqa: E402
                      read_blend)

BOS, EOD, VOCAB = 1, 2, 5000
SEQ, GBS, PHASE, TRAIN_ITERS, REF_DP = 32, 16, 10, 40, 8
KEY_SHIFT = 40


class _Tok:
    bos, eod, pad, vocab_size = BOS, EOD, 0, VOCAB
    unique_identifiers = {"class": "stub"}


@pytest.fixture(scope="module")
def blend(tmp_path_factory):
    compile_helpers()
    d = tmp_path_factory.mktemp("cds")
    rng = np.random.default_rng(0)
    prefixes = []
    # dataset 2 is tiny and heavily weighted: the window reads its documents several times (epochs)
    for i, (ndoc, maxlen) in enumerate([(300, 40), (120, 90), (60, 12)]):
        prefix = str(d / f"ds{i}")
        b = IndexedDatasetBuilder(prefix + ".bin", dtype=DType.optimal_dtype(VOCAB))
        for j in range(ndoc):
            n = int(rng.integers(2, maxlen))
            doc = list(rng.integers(3, VOCAB, n - 1)) + [EOD]
            b.add_document(np.array(doc), [len(doc)])
        b.finalize(prefix + ".idx")
        prefixes.append(prefix)
    weights = [0.5, 0.3, 0.2]
    mix = d / "mix.txt"
    mix.write_text("\n".join(f"{w} {p}" for w, p in zip(weights, prefixes)) + "\n")
    config = GPTDatasetConfig(
        random_seed=1234, sequence_length=SEQ, blend=(prefixes, weights), split="98,2,0",
        tokenizer=_Tok(), reset_position_ids=False, reset_attention_mask=False, eod_mask_loss=False,
        create_attention_mask=False, inter_document_masking=True, add_bos=True,
        path_to_cache=str(d / "cache"))
    n_train = (TRAIN_ITERS - PHASE) * GBS
    train, _, _ = BlendedMegatronDatasetBuilder(GPTDataset, [n_train, 8, 8], lambda: True, config).build()
    return dict(dir=d, mix=str(mix), prefixes=prefixes, cache=str(d / "cache"), train=train)


def _extract(blend, out, from_ckpt, steps, chunk):
    argv = ["x", "--data-args-path", blend["mix"], "--data-cache-path", blend["cache"],
            "--seq-length", str(SEQ), "--global-batch-size", str(GBS), "--train-iters", str(TRAIN_ITERS),
            "--phase-start", str(PHASE), "--ref-dp", str(REF_DP), "--ref-mbs", "1",
            "--from-ckpt", str(from_ckpt), "--steps", str(steps), "--chunk", str(chunk), "--out", str(out),
            "--check-megatron"]
    old = sys.argv
    sys.argv = argv
    try:
        extract_doc_stream.main()
    finally:
        sys.argv = old


def test_stream_reproduces_megatron_samples(blend, tmp_path):
    _extract(blend, tmp_path, from_ckpt=14, steps=6, chunk=4)  # not tied to any start: 14 here
    _, prefixes = read_blend(blend["mix"])
    entry = find_cache_entry(blend["cache"], prefixes, SEQ)
    assert entry.blend_md5 == blend["train"].unique_description_hash
    stream = BlendStream(entry)
    n = 0
    for name in sorted(os.listdir(tmp_path / "stream")):
        arr, meta = load_stream(str(tmp_path / "stream" / name))
        for s in range(len(arr["slot_bidx"])):
            a, b = arr["seg_ptr"][s], arr["seg_ptr"][s + 1]
            toks = np.concatenate([stream.read(int(arr["seg_ds"][k]), int(arr["seg_doc"][k]),
                                               int(arr["seg_off"][k]), int(arr["seg_len"][k]), BOS)
                                   for k in range(a, b)])
            ref = blend["train"][int(arr["slot_bidx"][s])]["tokens"].numpy()
            assert np.array_equal(toks, ref)
            n += 1
    assert n == 6 * GBS
    # the sampler replay equals Megatron's own sampler (also checked by --check-megatron)
    geo = SamplerGeometry(total_samples=entry.size, global_batch_size=GBS, phase_start=PHASE, ref_dp=REF_DP, ref_mbs=1)
    for ck in (PHASE, 17, TRAIN_ITERS - 1):
        StepSampler(geo).check_against_megatron(ck)


def test_verify_stream_against_gptdataset(blend, tmp_path):
    _extract(blend, tmp_path, from_ckpt=20, steps=4, chunk=4)
    old = sys.argv
    sys.argv = ["x", "--out", str(tmp_path), "--data-cache-path", blend["cache"], "--data-args-path", blend["mix"],
                "--bos-id", str(BOS), "--eod-id", str(EOD), "--vocab-size", str(VOCAB), "--slots", "40"]
    try:
        with pytest.raises(SystemExit) as e:
            verify_stream.main()
    finally:
        sys.argv = old
    assert e.value.code == 0


def _fake_embeddings(out, emb_name, dup_pairs):
    """Random unit vectors per document; documents in dup_pairs get (almost) the same vector."""
    keys = set()
    for name in os.listdir(out / "stream"):
        arr, _ = load_stream(str(out / "stream" / name))
        keys |= set(((np.asarray(arr["seg_ds"]).astype(np.int64) << KEY_SHIFT) | np.asarray(arr["seg_doc"])).tolist())
    keys = np.array(sorted(keys), np.int64)
    rng = np.random.default_rng(1)
    v = rng.standard_normal((len(keys), 16)).astype(np.float32)
    pos = {int(k): i for i, k in enumerate(keys)}
    for a, b in dup_pairs:
        if a in pos and b in pos:
            v[pos[b]] = v[pos[a]] + 1e-3 * rng.standard_normal(16)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    p = out / "emb" / emb_name / "all"
    p.mkdir(parents=True)
    np.savez(p / "part_0000_of_0001.npz", keys=keys, vec=v.astype(np.float16))


def _plan(out, from_ckpt, steps, window):
    argv = ["x", "--out", str(out), "--emb", "fake", "--from-ckpt", str(from_ckpt), "--steps", str(steps),
            "--window", str(window), "--coarse", "16", "--leaf-tokens", "48", "--iters", "4",
            "--device", "cpu", "--dup-cos", "0.97", "--stat-seqs", "50"]
    old = sys.argv
    sys.argv = argv
    try:
        build_plan.main()
    finally:
        sys.argv = old


def _owned(segs):
    """Multiset of (dataset, document, virtual offset) of the owned tokens of a segment list."""
    c = collections.Counter()
    for ds, doc, off, n in segs:
        for t in range(off, off + n):
            c[(ds, doc, t)] += 1
    return c


@pytest.fixture(scope="module")
def planned(blend, tmp_path_factory):
    out = tmp_path_factory.mktemp("pipe")
    _extract(blend, out, from_ckpt=12, steps=20, chunk=5)
    dups = [((0 << KEY_SHIFT) | 3, (0 << KEY_SHIFT) | 4), ((1 << KEY_SHIFT) | 7, (1 << KEY_SHIFT) | 8)]
    _fake_embeddings(out, "fake", dups)
    _plan(out, from_ckpt=12, steps=20, window=10)
    return out, dups


def test_plan_conserves_tokens_and_separates_copies(planned):
    out, dups = planned
    plans = sorted((out / "plan" / "fake").iterdir())
    assert [p.name for p in plans] == ["ck0000012_0000022", "ck0000022_0000032"]
    for p in plans:
        w0, w1 = (int(x) for x in p.name[2:].split("_"))
        win = build_plan.load_window(str(out), w0, w1)
        solo = win["nseg"] == 1
        seg_slot = np.repeat(np.arange(len(win["nseg"])), win["nseg"])
        k = ~solo[seg_slot]
        before = _owned(zip(win["seg_ds"][k], win["seg_doc"][k], win["seg_off"][k], win["seg_len"][k]))
        z = {n: np.load(p / f"{n}.npy") for n in ("slot_bidx", "slot_seq", "seq_ptr", "seg_ds", "seg_doc",
                                                   "seg_off", "seg_len", "seg_group")}
        assert np.array_equal(np.sort(z["slot_bidx"]), np.sort(win["slot_bidx"][~solo]))
        assert sorted(z["slot_seq"].tolist()) == list(range(len(z["seq_ptr"]) - 1))
        after = collections.Counter()
        for s in range(len(z["seq_ptr"]) - 1):
            a, b = z["seq_ptr"][s], z["seq_ptr"][s + 1]
            segs = list(zip(z["seg_ds"][a:b], z["seg_doc"][a:b], z["seg_off"][a:b], z["seg_len"][a:b]))
            assert sum(x[3] for x in segs) == SEQ + 1
            ds_, doc_, off_, n_ = segs[-1]  # drop the label-only token
            segs[-1] = (ds_, doc_, off_, n_ - 1)
            owned = _owned(segs)
            after += owned
            # a document occurrence never meets another copy of itself (same tokens twice in one
            # sequence) and an injected near duplicate never meets its source
            assert max(owned.values()) == 1
            keys = {(int(d) << KEY_SHIFT) | int(c) for d, c, _, _ in segs}
            for x, y in dups:
                assert not (x in keys and y in keys), f"near duplicates {x}, {y} in one sequence"
        assert after == before
        stats = __import__("json").load(open(p / "stats.json"))
        assert stats["units"] > 0 and stats["slots"] == (w1 - w0) * GBS


@pytest.mark.parametrize("mode", ["doc", "group", "none"])
def test_megatron_serves_plan(blend, planned, mode):
    out, _ = planned
    train = blend["train"]
    ds = PackingPlanDataset(train, [str(p) for p in sorted((out / "plan" / "fake").iterdir())], mode)
    assert len(ds) == len(train)
    k, _, _ = ds._lookup_tables()
    planned_idx = set(k.tolist())
    assert planned_idx
    for idx in range(len(train)):
        got = ds[idx]
        if idx not in planned_idx:
            ref = train[idx]
            for key in ref:
                assert np.array_equal(np.asarray(got[key]), np.asarray(ref[key])), (idx, key)
            continue
        plan, slot = ds.plan_slot(idx)
        s = int(plan.slot_seq[slot])
        a, b = int(plan.seq_ptr[s]), int(plan.seq_ptr[s + 1])
        text = np.concatenate([train.datasets[int(plan.seg_ds[j])]._get_document(
            int(plan.seg_doc[j]), offset=int(plan.seg_off[j]), length=int(plan.seg_len[j])) for j in range(a, b)])
        assert np.array_equal(got["tokens"].numpy(), text[:-1])
        assert np.array_equal(got["labels"].numpy(), text[1:])
        cu = got["cu_seqlens"].numpy()
        cu = cu[: int(np.argmax(cu == SEQ)) + 1]
        lens = plan.seg_len[a:b].astype(np.int64).copy()
        lens[-1] -= 1
        if mode == "none":
            assert cu.tolist() == [0, SEQ]
        elif mode == "doc":
            assert cu.tolist() == np.r_[0, np.cumsum(lens[lens > 0])].tolist()
        else:
            grp = plan.seg_group[a:b]
            starts = np.nonzero(np.r_[True, grp[1:] != grp[:-1]])[0]
            g = np.add.reduceat(plan.seg_len[a:b].astype(np.int64), starts)
            g[-1] -= 1
            assert cu.tolist() == np.r_[0, np.cumsum(g[g > 0])].tolist()
        # positions restart per segment
        pos = got["position_ids"].numpy()
        for x, y in zip(cu[:-1], cu[1:]):
            assert pos[x:y].tolist() == list(range(y - x))


def test_check_plan_through_megatron(blend, planned):
    out, _ = planned
    old = sys.argv
    sys.argv = ["x", "--out", str(out), "--plan-name", "fake", "--data-cache-path", blend["cache"],
                "--data-args-path", blend["mix"], "--bos-id", str(BOS), "--eod-id", str(EOD),
                "--vocab-size", str(VOCAB), "--samples", "16"]
    try:
        with pytest.raises(SystemExit) as e:
            check_plan.main()
    finally:
        sys.argv = old
    assert e.value.code == 0


def test_place_copies_separates_a_giant_class():
    """Production windows hold one class of tens of thousands of near-identical documents: spread over a
    third of the sequences, none of them may share a sequence (one pass, no repair rounds)."""
    rng = np.random.default_rng(3)
    S = 65536
    n_base = 20000
    utok_base = rng.integers(50, 2400, n_base)
    giant = 300
    small = rng.integers(2, 6, 40)
    n_copies = giant + int(small.sum())
    utok = np.r_[utok_base, rng.integers(50, 2400, n_copies)].astype(np.int64)
    n = len(utok)
    cls = np.arange(n, dtype=np.int64)
    copies = np.arange(n_base, n)
    cls[copies[:giant]] = 7  # base unit 7 is the giant class's kept member
    o = giant
    for i, k in enumerate(small):
        cls[copies[o:o + k]] = 100 + i  # base unit 100+i is that class's kept member
        o += k
    base = rng.permutation(n_base)
    order, forced = build_plan.place_copies(base, copies, cls, utok, S, rng)
    assert sorted(order.tolist()) == list(range(n))
    assert forced == 0
    assert build_plan.count_conflicts(order, utok, cls, S) == 0
    n_seq = int(utok.sum()) // S
    assert giant > n_seq // 2  # the class really is large against the sequences


def test_global_single_window_plan(blend, planned):
    """build_plan_global.py: one window over the whole extracted range, sharded (2 assign / 3 order processes,
    run in turn here), checked like any plan by check_plan through Megatron's loader."""
    import build_plan_global

    out, _ = planned

    def run(*args):
        old = sys.argv
        sys.argv = ["x", "--out", str(out), "--emb", "fake", "--name", "gfake", "--from-ckpt", "12", "--steps", "20",
                    "--coarse", "40", "--sample", "100000", "--batch", "64", "--epochs", "3", "--leaf-tokens", "48",
                    "--iters", "4", "--device", "cpu", *args]
        try:
            build_plan_global.main()
        finally:
            sys.argv = old

    run("--stage", "centroids", "--buckets", "3")
    for r in range(2):
        run("--stage", "assign", "--task-rank", str(r), "--task-world", "2")
    for r in range(3):
        run("--stage", "order", "--task-rank", str(r), "--task-world", "3")
    run("--stage", "assemble")
    plans = sorted((out / "plan" / "gfake").iterdir())
    assert [p.name for p in plans] == ["ck0000012_0000032"]
    stats = __import__("json").load(open(plans[0] / "stats.json"))
    assert stats["units_sharing_a_sequence_with_a_copy"] == 0 and stats["copies_forced"] == 0
    old = sys.argv
    sys.argv = ["x", "--out", str(out), "--plan-name", "gfake", "--data-cache-path", blend["cache"],
                "--data-args-path", blend["mix"], "--bos-id", str(BOS), "--eod-id", str(EOD),
                "--vocab-size", str(VOCAB), "--samples", "16"]
    try:
        with pytest.raises(SystemExit) as e:
            check_plan.main()
    finally:
        sys.argv = old
    assert e.value.code == 0
