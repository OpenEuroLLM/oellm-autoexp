"""Re-tokenize the flag mix's chat / agent delimiters with tokenizer-256k-agent (docs/flag_retokenization.md).

    python scripts/flag_retok/retok.py find   --root flag --shards shards.txt --work W [--part i --nparts n] [--procs p]
    python scripts/flag_retok/retok.py write  --root flag --out flag-agenttok --work W [--part i --nparts n]
    python scripts/flag_retok/retok.py decon  --out flag-agenttok --decon flag-decon-v2 --decon-out flag-decon-v2-agenttok
    python scripts/flag_retok/retok.py verify --root flag --out flag-agenttok --work W [--part i --nparts n]

The agent tokenizer equals the base one except for renamed <unused_N> slots, so only sequences whose text
holds a delimiter change. find: decode every sequence (base model), count the delimiters, keep the ids of the
sequences that hold one ($W/find/<shard key>.npz). write: per shard with hits, a new .bin / .idx in which
those sequences are encode_agent(decode_base(sequence)) (trailing <bos> / <eos> kept) and every other
sequence is copied byte for byte; document boundaries and order are unchanged. Shards without hits are
symlinked (.bin, .idx -> flag/). decon: the decontaminated tree on top -- every document flag-decon-v2
dropped is dropped again (same document ids). verify: decoded text identical, untouched sequences
token-identical, counts equal. Nothing under --root is ever written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "flag_decontam"))
from common import Index, filter_index, write_index  # noqa: E402
from delimiters import AGENT_TOKENS  # noqa: E402

BASE_TOK = "/gpfs/projects/ehpc533/tokenizers/tokenizer-256k/tokenizer.model"
AGENT_TOK = "/gpfs/projects/ehpc533/tokenizers/tokenizer-256k-agent/tokenizer.model"
CONTROL = (1, 2)                       # <bos>, <eos>: decode drops them; re-appended where they were
UNIT_TOKENS = 1 << 26
DELIMS = [t for _, t in AGENT_TOKENS]
RX = re.compile("|".join(re.escape(d) for d in sorted(DELIMS, key=len, reverse=True)))

_G = {}


def shard_key(rel):
    return rel.replace("/", "__")


def _init(base_tok, agent_tok=None):
    import sentencepiece as spm
    _G["base"] = spm.SentencePieceProcessor(model_file=base_tok)
    _G["V"] = _G["base"].get_piece_size()
    if agent_tok:
        _G["agent"] = spm.SentencePieceProcessor(model_file=agent_tok)


def units_of(root, rels, unit_tokens=UNIT_TOKENS):
    out = []
    for rel in rels:
        idx = Index(os.path.join(root, rel) + ".idx")
        cum = np.concatenate([[0], np.cumsum(idx.lengths.astype(np.int64))])
        s = 0
        while s < idx.seq_count:
            e = int(np.searchsorted(cum, cum[s] + unit_tokens, "right"))
            e = max(e, s + 1)
            e = min(e, idx.seq_count)
            out.append((rel, s, e))
            s = e
    return out


def read_seqs(path, idx, s, e):
    """Token arrays of sequences s..e-1 (one contiguous read)."""
    a, b = int(idx.pointers[s]), int(idx.pointers[e - 1]) + 4 * int(idx.lengths[e - 1])
    buf = np.fromfile(path + ".bin", np.int32, count=(b - a) // 4, offset=a)
    off = (idx.pointers[s:e] - a) // 4
    return [buf[o:o + n] for o, n in zip(off, idx.lengths[s:e])]


def split_body(t):
    """(body, trailing control ids): decode drops <bos> / <eos>, they are put back after re-encoding."""
    k = len(t)
    while k and t[k - 1] in CONTROL:
        k -= 1
    return t[:k], t[k:]


def decode(t):
    V = _G["V"]
    body, _ = split_body(t)
    return _G["base"].decode(body[(body >= 0) & (body < V)].tolist()), bool(((body < 0) | (body >= V)).any())


def _find_unit(u):
    rel, s, e = u
    p = os.path.join(_G["root"], rel)
    idx = Index(p + ".idx")
    seqs, counts = [], np.zeros(len(DELIMS), np.int64)
    pos = {d: i for i, d in enumerate(DELIMS)}
    for k, t in enumerate(read_seqs(p, idx, s, e)):
        text, _ = decode(t)
        m = RX.findall(text)
        if m:
            seqs.append(s + k)
            for d in m:
                counts[pos[d]] += 1
    return rel, s, e, np.array(seqs, np.int64), counts, int(idx.lengths[s:e].astype(np.int64).sum())


def cmd_find(a):
    rels = [l.strip()[len(a.root) + 1:] if l.startswith(a.root) else l.strip()
            for l in open(a.shards) if l.strip()]
    units = units_of(a.root, rels)[a.part::a.nparts]
    os.makedirs(os.path.join(a.work, "find"), exist_ok=True)
    per = {}
    t0, tok = time.time(), 0
    _G["root"] = a.root
    with Pool(a.procs, initializer=_init, initargs=(a.base_tok,)) as pool:
        for rel, s, e, seqs, counts, n in pool.imap_unordered(_find_unit, units, chunksize=1):
            r = per.setdefault(rel, {"seqs": [], "counts": np.zeros(len(DELIMS), np.int64), "units": 0, "tokens": 0})
            r["seqs"].append(seqs)
            r["counts"] += counts
            r["units"] += 1
            r["tokens"] += n
            tok += n
            print(f"[find {a.part}] {rel} {s}-{e}: {len(seqs)} seqs with delimiters | {tok / 1e9:.1f}B tok, "
                  f"{tok / (time.time() - t0) / 1e6:.0f} Mtok/s", flush=True)
    for rel, r in per.items():               # one file per (shard, part): parts may split a shard
        np.savez(os.path.join(a.work, "find", f"{shard_key(rel)}.part{a.part:03d}.npz"),
                 seqs=np.sort(np.concatenate(r["seqs"])), counts=r["counts"], tokens=r["tokens"])
    print(f"[find {a.part}] done: {tok / 1e9:.1f}B tokens in {time.time() - t0:.0f}s", flush=True)


def load_find(work, rel):
    import glob
    fs = glob.glob(os.path.join(work, "find", f"{glob.escape(shard_key(rel))}.part*.npz"))
    if not fs:
        return None
    zs = [np.load(f) for f in fs]
    return np.unique(np.concatenate([z["seqs"] for z in zs])), sum(z["counts"] for z in zs)


def md5(path):
    m = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            m.update(b)
    return m.hexdigest()


def link(src, dst):
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(os.path.realpath(src), dst)


def info_path(prefix):
    return prefix.removesuffix("_text_document") + ".info.json"


def _write_shard(job):
    rel, root, out, work = job
    src, dst = os.path.join(root, rel), os.path.join(out, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    found = load_find(work, rel)
    info = json.load(open(info_path(src))) if os.path.exists(info_path(src)) else {"idx": {}, "bin": {}}
    if found is None or not len(found[0]):
        link(src + ".bin", dst + ".bin")
        link(src + ".idx", dst + ".idx")
        json.dump(info, open(info_path(dst), "w"), indent=2)
        return rel, 0, 0, 0, 0
    hit = found[0]
    idx = Index(src + ".idx")
    assert idx.dtype == np.int32, idx.dtype
    lengths = idx.lengths.astype(np.int64).copy()
    changed = not_identity = skipped = 0
    tmp = dst + ".bin.tmp"
    is_hit = np.zeros(idx.seq_count, bool)
    is_hit[hit] = True
    with open(tmp, "wb") as fo:
        for rel_, s, e in units_of(root, [rel]):
            seqs = read_seqs(src, idx, s, e)
            if not is_hit[s:e].any():           # whole unit unchanged: one write
                for t in seqs:
                    fo.write(t.tobytes())
                continue
            for k, t in enumerate(seqs):
                if not is_hit[s + k]:
                    fo.write(t.tobytes())
                    continue
                body, tail = split_body(t)
                text, odd = decode(t)
                if odd:                         # ids outside the vocabulary (<pad>): leave the sequence as it is
                    skipped += 1
                    fo.write(t.tobytes())
                    continue
                if _G["base"].encode(text) != body.tolist():
                    not_identity += 1           # the original was not plain encode(text); re-encoded anyway
                new = np.array(_G["agent"].encode(text) + tail.tolist(), np.int32)
                fo.write(new.tobytes())
                lengths[s + k] = len(new)
                changed += 1
    pointers = np.concatenate([[0], np.cumsum(lengths[:-1] * 4)]).astype(np.int64)
    os.replace(tmp, dst + ".bin")
    assert os.path.getsize(dst + ".bin") == 4 * int(lengths.sum())
    write_index(dst + ".idx", idx.dtype_code, lengths.astype(np.int32), pointers, np.asarray(idx.doc_idx))
    info.setdefault("idx", {}).update(size=os.path.getsize(dst + ".idx"), md5sum=md5(dst + ".idx"),
                                      sequence_count=int(idx.seq_count), document_count=int(idx.doc_count),
                                      total_tokens=int(lengths.sum()))
    info.setdefault("bin", {}).update(size=os.path.getsize(dst + ".bin"), md5sum=md5(dst + ".bin"))
    info["retokenization"] = {"tokenizer": "tokenizer-256k-agent", "source_prefix": src,
                              "sequences_reencoded": changed, "sequences_skipped_pad": skipped,
                              "reencoded_not_identity_under_base": not_identity,
                              "tokens_before": int(idx.lengths.astype(np.int64).sum()),
                              "tokens_after": int(lengths.sum()),
                              "delimiter_counts": {d: int(c) for d, c in zip(DELIMS, found[1]) if c}}
    json.dump(info, open(info_path(dst), "w"), indent=2)
    return rel, changed, not_identity, int(idx.lengths.astype(np.int64).sum()), int(lengths.sum())


def mix_rels(root):
    return [l.split()[1] for l in open(os.path.join(root, "datamix.txt")) if l.strip()]


def cmd_write(a):
    assert os.path.realpath(a.out) != os.path.realpath(a.root)
    rels = mix_rels(a.root)[a.part::a.nparts]
    with Pool(a.procs, initializer=_init, initargs=(a.base_tok, a.agent_tok)) as pool:
        for rel, ch, ni, t0, t1 in pool.imap_unordered(_write_shard, [(r, a.root, a.out, a.work) for r in rels]):
            print(f"[write {a.part}] {'rewrote' if ch else 'linked '} {rel}: {ch} seqs re-encoded"
                  f"{f' ({ni} not identity)' if ni else ''}, tokens {t0} -> {t1}", flush=True)
    if a.part == 0:
        shutil.copy(os.path.join(a.root, "datamix.txt"), os.path.join(a.out, "datamix.txt"))
    print(f"[write {a.part}] done", flush=True)


def cmd_decon(a):
    """flag-decon-v2-agenttok: flag-agenttok with flag-decon-v2's dropped documents (same document ids)."""
    for rel in mix_rels(a.out):
        tok, dec, dst = (os.path.join(x, rel) for x in (a.out, a.decon, a.decon_out))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        link(tok + ".bin", dst + ".bin")
        info = json.load(open(info_path(tok)))
        rem = dec + ".removed.npz"
        if os.path.exists(rem):
            idx = Index(tok + ".idx")
            old = Index(dec + ".idx")
            drop = np.zeros(idx.doc_count - 1, bool)
            drop[np.unique(np.load(rem)["doc"])] = True
            lengths, pointers, doc_idx = filter_index(idx, ~drop)
            assert len(doc_idx) == old.doc_count and len(lengths) == old.seq_count, rel   # same documents kept
            if os.path.lexists(dst + ".idx"):
                os.remove(dst + ".idx")
            write_index(dst + ".idx", idx.dtype_code, lengths, pointers, doc_idx)
            shutil.copy(rem, dst + ".removed.npz")
            info["idx"].update(size=os.path.getsize(dst + ".idx"), md5sum=md5(dst + ".idx"),
                               sequence_count=int(len(lengths)), document_count=int(len(doc_idx)),
                               total_tokens=int(lengths.astype(np.int64).sum()))
            info["decontamination"] = json.load(open(info_path(dec))).get("decontamination", {})
        else:
            link(tok + ".idx", dst + ".idx")
        json.dump(info, open(info_path(dst), "w"), indent=2)
    shutil.copy(os.path.join(a.out, "datamix.txt"), os.path.join(a.decon_out, "datamix.txt"))
    print("[decon] done", flush=True)


def _verify_shard(job):
    rel, root, out, work, n = job
    src, dst = os.path.join(root, rel), os.path.join(out, rel)
    a, b = Index(src + ".idx"), Index(dst + ".idx")
    assert a.doc_count == b.doc_count and a.seq_count == b.seq_count and np.array_equal(a.doc_idx, b.doc_idx), rel
    assert os.path.getsize(dst + ".bin") == 4 * int(b.lengths.astype(np.int64).sum()), rel
    found = load_find(work, rel)
    hit = set(found[0].tolist()) if found is not None else set()
    rng = np.random.default_rng(0)
    pick = list(rng.choice(a.seq_count, size=min(n, a.seq_count), replace=False)) + sorted(hit)[:n]
    bad = 0
    for s in pick:
        x = read_seqs(src, a, s, s + 1)[0]
        y = read_seqs(dst, b, s, s + 1)[0]
        if s in hit:
            bx, tx = split_body(x)
            by, ty = split_body(y)
            V = _G["V"]
            ok = (np.array_equal(tx, ty) and
                  _G["base"].decode(bx[(bx >= 0) & (bx < V)].tolist()) == _G["agent"].decode(by.tolist()))
            ok = ok or (bx.size and ((bx < 0) | (bx >= V)).any() and np.array_equal(x, y))   # skipped (<pad>)
        else:
            ok = np.array_equal(x, y)
        bad += not ok
    return rel, len(pick), bad


def cmd_verify(a):
    rels = mix_rels(a.root)[a.part::a.nparts]
    tot = bad = 0
    with Pool(a.procs, initializer=_init, initargs=(a.base_tok, a.agent_tok)) as pool:
        for rel, n, b in pool.imap_unordered(_verify_shard, [(r, a.root, a.out, a.work, a.samples) for r in rels]):
            tot += n
            bad += b
            if b:
                print(f"[verify {a.part}] MISMATCH {rel}: {b} of {n}", flush=True)
    print(f"[verify {a.part}] {len(rels)} shards, {tot} sequences checked, {bad} mismatches", flush=True)
    sys.exit(1 if bad else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["find", "write", "decon", "verify"])
    ap.add_argument("--root", default="/gpfs/projects/ehpc533/training-data-collection/flag")
    ap.add_argument("--out", default="/gpfs/projects/ehpc533/training-data-collection/flag-agenttok")
    ap.add_argument("--decon", default="/gpfs/projects/ehpc533/training-data-collection/flag-decon-v2")
    ap.add_argument("--decon-out", default="/gpfs/projects/ehpc533/training-data-collection/flag-decon-v2-agenttok")
    ap.add_argument("--work", default="/gpfs/scratch/ehpc533/users/korbip/flag_retok")
    ap.add_argument("--shards", default="")
    ap.add_argument("--base-tok", default=BASE_TOK)
    ap.add_argument("--agent-tok", default=AGENT_TOK)
    ap.add_argument("--part", type=int, default=int(os.environ.get("SLURM_NODEID", 0)))
    ap.add_argument("--nparts", type=int, default=int(os.environ.get("SLURM_NNODES", 1)))
    ap.add_argument("--procs", type=int, default=min(os.cpu_count() or 1, 72))
    ap.add_argument("--samples", type=int, default=200)
    a = ap.parse_args()
    if a.mode != "find":
        _init(a.base_tok, a.agent_tok)
    {"find": cmd_find, "write": cmd_write, "decon": cmd_decon, "verify": cmd_verify}[a.mode](a)


if __name__ == "__main__":
    main()
