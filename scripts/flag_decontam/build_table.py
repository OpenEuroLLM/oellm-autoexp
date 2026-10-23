"""Reference n-gram table for the flag decontamination (docs/flag_post_decontamination.md).

    python scripts/flag_decontam/build_table.py --refs refs.jsonl.gz --work $WORK            # build
    python scripts/flag_decontam/build_table.py --work $WORK --prune --max-count 50         # prune

build: tokenise every reference segment with the training tokenizer, map it to the canonical
stream (common.py), and hash its distinct n-grams, with n the smallest of 13 / 24 / 40 whose
n-grams span >= 30 characters (median; byte-fallback scripts get longer n). The JSON-escaped
forms of the segment are hashed too, under the same segment id. Segments shorter than 20
canonical tokens or 80 characters are not matched (counted as too short). N-grams shared by
more than --max-segments distinct segments are prompt / template text and dropped.
Writes $WORK/table/{lut.npy, table.npz, segments.npz, sources.json, build_report.json}:
  table.npz:    hash (sorted unique int64), csr_off / csr_seg (the segments of each hash)
  segments.npz: per segment its source id, split id, n, its number of n-grams in the table
                (n_grams); a segment's id is its line number in refs

prune: drop the n-grams that `scan.py --mode count` found more than --max-count times in the
sample of the training data (common phrases, licence headers, boilerplate), recompute n_grams,
and keep the previous table as table_unpruned.npz. Prune once per build: a second prune would
overwrite table_unpruned.npz with the already pruned table.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
from collections import Counter
from multiprocessing import Pool

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MIN_GRAM_CHARS, MIN_SEG_CHARS, MIN_SEG_TOKENS, NGRAM_SIZES, build_lut, key_chars, ngram_hashes  # noqa: E402

TOKENIZER = "/gpfs/projects/ehpc533/tokenizers/tokenizer-256k/tokenizer.model"
_sp = _lut = _chars = None


def _init(tok_path, lut, chars):
    global _sp, _lut, _chars
    import sentencepiece as spm
    _sp = spm.SentencePieceProcessor(model_file=tok_path)
    _lut, _chars = lut, chars
    torch.set_num_threads(1)


def _canon(ids):
    c = _lut[np.asarray(ids, dtype=np.int64)] if ids else np.zeros(0, np.int64)
    return c[c >= 0]


def choose_n(c, chars):
    """Smallest n whose n-grams span >= MIN_GRAM_CHARS characters (median), else the largest."""
    cs = np.concatenate([[0.0], np.cumsum(chars[c], dtype=np.float64)])
    for n in NGRAM_SIZES:
        if len(c) >= n and np.median(cs[n:] - cs[:-n]) >= MIN_GRAM_CHARS:
            return n
    return NGRAM_SIZES[-1]


def variants(text):
    """The text and its JSON-escaped forms (copies inside agent trajectories / JSON dumps)."""
    out = [text]
    for v in (json.dumps(text, ensure_ascii=False)[1:-1], json.dumps(text)[1:-1]):
        if v not in out:
            out.append(v)
    return out


def _encode(texts):
    try:   # one call per batch, single-threaded: the pool already uses every core
        return _sp.encode(texts, num_threads=1)
    except TypeError:  # older sentencepiece without num_threads
        return _sp.encode(texts)


def _hash_batch(texts):
    vss = [variants(t) for t in texts]
    flat = _encode([v for vs in vss for v in vs])
    out, k = [], 0
    for text, vs in zip(texts, vss):
        cs = [_canon(ids) for ids in flat[k:k + len(vs)]]
        k += len(vs)
        c = cs[0]
        n = choose_n(c, _chars) if len(c) else NGRAM_SIZES[0]
        if len(c) < max(MIN_SEG_TOKENS, n + 2) or len(text) < MIN_SEG_CHARS:
            out.append((len(c), 0, None))
            continue
        h = np.unique(np.concatenate([ngram_hashes(torch.from_numpy(v), n).numpy() for v in cs if len(v) >= n]))
        out.append((len(c), n, h))
    return out


def make_table(hashes, seg_ids, n_segments, max_segments):
    """Sorted unique hashes with their segments (CSR), dropping hashes of > max_segments segments."""
    order = np.lexsort((seg_ids, hashes))
    hashes, seg_ids = hashes[order], seg_ids[order]
    first = np.ones(len(hashes), bool)                  # one entry per (hash, segment)
    first[1:] = (hashes[1:] != hashes[:-1]) | (seg_ids[1:] != seg_ids[:-1])
    hashes, seg_ids = hashes[first], seg_ids[first]
    uniq, nseg = np.unique(hashes, return_counts=True)
    common = nseg > max_segments
    seg_ids = seg_ids[np.repeat(~common, nseg)]
    uniq, nseg = uniq[~common], nseg[~common]
    csr_off = np.concatenate([[0], np.cumsum(nseg)]).astype(np.int64)
    n_grams = np.bincount(seg_ids, minlength=n_segments).astype(np.int32)
    return uniq, csr_off, seg_ids, n_grams, common


def build(args):
    import sentencepiece as spm
    tdir = os.path.join(args.work, "table")
    os.makedirs(tdir, exist_ok=True)
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    lut, keys = build_lut(sp)
    chars = key_chars(keys)
    np.save(os.path.join(tdir, "lut.npy"), lut)
    np.save(os.path.join(tdir, "canon_chars.npy"), chars)
    print(f"lut: {len(lut)} ids -> {len(keys)} canonical, {(lut < 0).sum()} dropped", flush=True)

    texts, src_of, split_of = [], [], []
    srcs, splits = {}, {}
    with gzip.open(args.refs, "rt", encoding="utf-8") as f:
        for line in f:                      # segment id = line number in refs
            r = json.loads(line)
            texts.append(r["text"])
            src_of.append(srcs.setdefault(r["src"], len(srcs)))
            split_of.append(splits.setdefault(r["split"], len(splits)))
    print(f"{len(texts)} segments from {len(srcs)} sources", flush=True)

    B = 2000
    batches = [texts[i:i + B] for i in range(0, len(texts), B)]
    with Pool(args.procs, initializer=_init, initargs=(args.tokenizer, lut, chars)) as pool:
        res = [r for batch in pool.imap(_hash_batch, batches, chunksize=4) for r in batch]
    ntok = np.array([r[0] for r in res], dtype=np.int64)
    nsz = np.array([r[1] for r in res], dtype=np.int8)
    seg_ids = np.concatenate([np.full(len(r[2]), i, np.int32) for i, r in enumerate(res) if r[2] is not None])
    hashes = np.concatenate([r[2] for r in res if r[2] is not None])
    print(f"{len(hashes)} (segment, n-gram) pairs; {int((nsz == 0).sum())} segments too short", flush=True)

    uniq, csr_off, seg_ids, n_grams, common = make_table(hashes, seg_ids, len(texts), args.max_segments)
    np.savez(os.path.join(tdir, "table.npz"), hash=uniq, csr_off=csr_off, csr_seg=seg_ids)
    np.savez(os.path.join(tdir, "segments.npz"), src=np.array(src_of, np.int32), split=np.array(split_of, np.int16),
             n=nsz, n_tokens=ntok, n_grams=n_grams, n_grams_built=n_grams.copy())
    with open(os.path.join(tdir, "sources.json"), "w") as f:
        json.dump({"sources": list(srcs), "splits": list(splits), "refs": os.path.abspath(args.refs)}, f, indent=1)
    report = {
        "segments": len(texts), "too_short": int((nsz == 0).sum()),
        **{f"n{n}_segments": int((nsz == n).sum()) for n in NGRAM_SIZES},
        "min_seg_tokens": MIN_SEG_TOKENS, "min_seg_chars": MIN_SEG_CHARS, "min_gram_chars": MIN_GRAM_CHARS,
        "unique_ngrams": int(len(uniq)), "template_ngrams_dropped": int(common.sum()),
        "max_segments": args.max_segments,
        "too_short_by_source": dict(Counter(list(srcs)[s] for s, n in zip(src_of, nsz) if n == 0).most_common()),
    }
    with open(os.path.join(tdir, "build_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "too_short_by_source"}), flush=True)


def prune(args):
    tdir = os.path.join(args.work, "table")
    files = sorted(glob.glob(os.path.join(args.work, "counts", "*.npy")))
    assert files, "no counts: run scan.py --mode count first"
    tab = dict(np.load(os.path.join(tdir, "table.npz")))
    counts = sum(np.load(f) for f in files)
    assert len(counts) == len(tab["hash"]), "counts were made with another table"
    drop = counts > args.max_count
    nseg = np.diff(tab["csr_off"])
    keep_pair = np.repeat(~drop, nseg)
    seg = tab["csr_seg"][keep_pair]
    os.replace(os.path.join(tdir, "table.npz"), os.path.join(tdir, "table_unpruned.npz"))
    np.save(os.path.join(tdir, "counts_unpruned.npy"), counts)
    np.savez(os.path.join(tdir, "table.npz"), hash=tab["hash"][~drop],
             csr_off=np.concatenate([[0], np.cumsum(nseg[~drop])]).astype(np.int64), csr_seg=seg)
    segs = dict(np.load(os.path.join(tdir, "segments.npz")))
    segs["n_grams"] = np.bincount(seg, minlength=len(segs["n"])).astype(np.int32)
    np.savez(os.path.join(tdir, "segments.npz"), **segs)
    # matchable (>= 3 n-grams, finalize's --min-hits) before pruning, not after
    lost = (segs["n_grams_built"] > 0) & (segs["n_grams"] == 0)
    hist = {f">{t}": int((counts > t).sum()) for t in (0, 1, 5, 10, 20, 50, 100, 1000, 10000)}
    seen_seg = np.bincount(tab["csr_seg"][np.repeat(counts > 0, nseg)], minlength=len(segs["n"]))
    rep = {"count_files": len(files), "ngrams_with_count": hist,
           "segments_with_any_ngram_seen": int((seen_seg > 0).sum()),
           "max_count": args.max_count, "ngrams_dropped": int(drop.sum()),
           "ngrams_kept": int((~drop).sum()), "segments_left_unmatchable": int(lost.sum())}
    with open(os.path.join(tdir, "prune_report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(rep), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--refs")
    ap.add_argument("--tokenizer", default=TOKENIZER)
    ap.add_argument("--procs", type=int, default=min(os.cpu_count() or 1, 64))
    ap.add_argument("--max-segments", type=int, default=10)
    ap.add_argument("--prune", action="store_true")
    ap.add_argument("--max-count", type=int, default=50)
    args = ap.parse_args()
    prune(args) if args.prune else build(args)


if __name__ == "__main__":
    main()
