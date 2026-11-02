"""Re-tokenize the v8ml data (the datamix shards and the fixed validation sets) from tokenizer-256k-agent to
tokenizer-256k-v2-agent (scripts/tok_v2/make_tokenizer_v2.py), into a NEW tree -- the existing data is only read.

Every sequence is decoded with the old tokenizer and encoded with the new one; documents and their sequences
stay 1:1 (same document order, same number of sequences per document, the same doc_idx), so everything defined
on document / sequence indices carries over unchanged: the decontamination (the shards are the decontaminated
views), split_stages.py's stage of a document (hash of the shard path relative to the data root and the
document index -- run it with --data-root <out root>), and the validation documents held out by --split
998,2,0 (sequences [round(0.998 S), S)).

  decode   tokenizer-256k-agent: identity normalizer, no dummy prefix, so a sequence's text is the concatenation
           of its pieces ('▁' -> ' ', <0xNN> -> the byte; <bos> / <eos> are kept as ids, not text). A
           multi-byte character split between two sequences of one document is carried to the next sequence.
  encode   GigaToken (exactly SentencePiece's ids) with tokenizer-256k-v2-agent; <eos> (and <bos>) ids at the
           same places as before (ids 1 / 2 in both tokenizers).

Work: shards are split into units at document boundaries (~--unit-tokens tokens); a node runs one ordered pool
over all units of its shards and writes each shard sequentially to <out>/<rel>.bin.tmp, then .idx / .bin and
<out>/<rel>.tok2.json (counts). A finished shard (its .tok2.json) is skipped on a rerun; an unfinished one
starts over.

    python retok_v2.py plan  --list shards.txt --data-root <root> --out <out root> --parts 4 > plan.json
    python retok_v2.py run   --plan plan.json --part <i>            # one node; SLURM_NODEID picks from --parts-of
    python retok_v2.py check --plan plan.json [--sample 200]       # counts + decode(new) == decode(old) samples
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
import time
from multiprocessing import Pool

import numpy as np

_HEADER = b"MMIDIDX\x00\x00"
EOS, BOS = 2, 1
OLD_TOK = os.environ.get("TOK2_OLD", "/gpfs/projects/ehpc533/users/korbip/data/tokenizers/tokenizer-256k-agent")
NEW_TOK = os.environ.get("TOK2_NEW", "/gpfs/projects/ehpc533/users/korbip/data/tokenizers/tokenizer-256k-v2-agent")


# ---------------------------------------------------------------- Megatron .idx (as scripts/flag_decontam/common.py)
class Index:
    def __init__(self, path):
        with open(path, "rb") as f:
            assert f.read(9) == _HEADER, path
            assert struct.unpack("<Q", f.read(8))[0] == 1
            self.dtype_code = struct.unpack("<B", f.read(1))[0]
            self.seq_count = struct.unpack("<Q", f.read(8))[0]
            self.doc_count = struct.unpack("<Q", f.read(8))[0]
            off = f.tell()
        assert self.dtype_code == 4, (path, self.dtype_code)      # int32 tokens
        mm = np.memmap(path, mode="r", dtype=np.uint8)
        self.lengths = np.frombuffer(mm, np.int32, self.seq_count, off)
        off += self.lengths.nbytes
        self.pointers = np.frombuffer(mm, np.int64, self.seq_count, off)
        off += self.pointers.nbytes
        self.doc_idx = np.frombuffer(mm, np.int64, self.doc_count, off)


def write_index(path, lengths, doc_idx):
    lengths = np.ascontiguousarray(lengths, np.int32)
    pointers = np.concatenate([[0], np.cumsum(lengths[:-1].astype(np.int64) * 4)]) if len(lengths) else \
        np.zeros(0, np.int64)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(_HEADER + struct.pack("<Q", 1) + struct.pack("<B", 4) + struct.pack("<Q", len(lengths))
                + struct.pack("<Q", len(doc_idx)))
        f.write(lengths.tobytes())
        f.write(np.ascontiguousarray(pointers, np.int64).tobytes())
        f.write(np.ascontiguousarray(doc_idx, np.int64).tobytes())
    os.replace(tmp, path)


# ---------------------------------------------------------------- old tokenizer: ids -> bytes
class Decoder:
    """tokenizer-256k-agent ids -> UTF-8 bytes, exactly SentencePiece's decode (identity normalizer, no dummy
    prefix): pieces with '▁' -> ' ', byte pieces raw, <unk> ' ⁇ ', control pieces empty -- except that added
    (delimiter) pieces are kept verbatim."""

    def __init__(self, tok_dir):
        j = json.load(open(os.path.join(tok_dir, "tokenizer.json")))
        V = max(max(j["model"]["vocab"].values()), max(a["id"] for a in j["added_tokens"])) + 1
        piece = [b""] * V
        for p, i in j["model"]["vocab"].items():
            m = re.fullmatch(r"<0x([0-9A-Fa-f]{2})>", p)
            piece[i] = bytes([int(m.group(1), 16)]) if m else p.replace("▁", " ").encode()
        for a in j["added_tokens"]:                # verbatim: the DeepSeek delimiters hold a real U+2581, and
            piece[a["id"]] = a["content"].encode()  # GigaToken matches a delimiter only in that form (SentencePiece
            #                                         decodes it to spaces and matches both)
        unk = j["model"].get("unk_id", 0)
        piece[unk] = " ⁇ ".encode()
        for i in (BOS, EOS):
            piece[i] = b""
        self.lens = np.array([len(b) for b in piece], np.int64)
        self.offs = np.concatenate([[0], np.cumsum(self.lens)])
        self.blob = np.frombuffer(b"".join(piece), np.uint8)

    def __call__(self, ids):
        if len(ids) == 0:
            return b""
        ln = self.lens[ids]
        start = self.offs[ids] - np.concatenate([[0], np.cumsum(ln)[:-1]])
        return self.blob[np.repeat(start, ln) + np.arange(int(ln.sum()))].tobytes()


def split_incomplete(b: bytes):
    """(complete part, trailing incomplete UTF-8 character)."""
    for k in range(1, min(4, len(b)) + 1):
        c = b[-k]
        if c & 0xC0 == 0x80:                # continuation byte: keep looking for the lead byte
            continue
        need = 1 if c < 0x80 else 2 if c >> 5 == 6 else 3 if c >> 4 == 14 else 4 if c >> 3 == 30 else 1
        return (b, b"") if need <= k else (b[:-k], b[-k:])
    return b, b""


# ---------------------------------------------------------------- one unit: documents [d0, d1) of a shard
_W = {}


def _worker_init():
    import gigatoken as gt
    _W["dec"] = Decoder(OLD_TOK)
    _W["enc"] = gt.Tokenizer.from_sentencepiece(os.path.join(NEW_TOK, "tokenizer.model"))


def _shard(prefix):
    if _W.get("prefix") != prefix:
        _W["prefix"], _W["idx"] = prefix, Index(prefix + ".idx")
        _W["bin"] = np.memmap(prefix + ".bin", mode="r", dtype=np.int32)
    return _W["idx"], _W["bin"]


def retok_unit(task):
    prefix, d0, d1 = task
    idx, binf = _shard(prefix)
    dec, enc = _W["dec"], _W["enc"]
    texts, layout = [], []                   # per sequence: list of ("t", text index) / ("c", control id)
    di = idx.doc_idx
    for d in range(d0, d1):
        carry = b""
        s0, s1 = int(di[d]), int(di[d + 1])
        for s in range(s0, s1):
            p, n = int(idx.pointers[s]) // 4, int(idx.lengths[s])
            ids = np.asarray(binf[p:p + n])
            parts = []
            ctl = np.flatnonzero((ids == EOS) | (ids == BOS))
            lo = 0
            for k, c in enumerate(list(ctl) + [n]):
                seg = ids[lo:c]
                b = carry + dec(seg)
                last_text = k == len(ctl) and s < s1 - 1      # text at the end of a non-final sequence
                if last_text:
                    b, carry = split_incomplete(b)
                else:
                    carry = b""
                if b:
                    parts.append(("t", len(texts)))
                    texts.append(b.decode("utf-8", "replace"))
                if c < n:
                    parts.append(("c", int(ids[c])))
                lo = c + 1
            layout.append(parts)
    enc_ids = enc.encode_batch(texts, parallel=False) if texts else []
    out, lens = [], np.empty(len(layout), np.int32)
    for i, parts in enumerate(layout):                # (a running count: re-summing `out` was quadratic)
        n = 0
        for kind, v in parts:
            x = np.asarray(enc_ids[v], np.int32) if kind == "t" else np.array([v], np.int32)
            out.append(x)
            n += len(x)
        lens[i] = n
    toks = np.concatenate(out) if out else np.zeros(0, np.int32)
    return prefix, d0, d1, lens, toks, int(idx.lengths[int(di[d0]):int(di[d1])].astype(np.int64).sum())


# ---------------------------------------------------------------- plan / run / check
def units_of(prefix, unit_tokens):
    idx = Index(prefix + ".idx")
    doc_tok = np.add.reduceat(idx.lengths.astype(np.int64), idx.doc_idx[:-1]) if idx.seq_count else \
        np.zeros(idx.doc_count - 1, np.int64)
    doc_tok[idx.doc_idx[:-1] == idx.doc_idx[1:]] = 0                # documents without sequences
    cum = np.cumsum(doc_tok)
    cuts = np.searchsorted(cum, np.arange(unit_tokens, cum[-1] if len(cum) else 0, unit_tokens), side="right")
    bounds = np.unique(np.concatenate([[0], cuts, [len(doc_tok)]]))
    return [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if b > a], int(cum[-1]) if len(cum) else 0


def cmd_plan(a):
    shards = []
    for line in open(a.list):
        line = line.split("#")[0].strip()
        if not line:
            continue
        src, rel = line.split()[:2] if len(line.split()) > 1 else (line, None)
        rel = rel or os.path.relpath(src, a.data_root)
        assert not rel.startswith(".."), src
        shards.append((src, rel))
    jobs = []
    for src, rel in shards:
        units, tok = units_of(src, a.unit_tokens)
        jobs.append({"src": src, "out": os.path.join(a.out, rel), "tokens": tok, "units": units})
    load = [0] * a.parts
    for j in sorted(jobs, key=lambda j: -j["tokens"]):               # largest first to the emptiest part
        k = load.index(min(load))
        j["part"] = k
        load[k] += j["tokens"]
    json.dump({"out": a.out, "old": OLD_TOK, "new": NEW_TOK, "parts": a.parts, "shards": jobs}, sys.stdout)
    print(f"{len(jobs)} shards, {sum(len(j['units']) for j in jobs)} units, "
          f"{sum(load) / 1e9:.1f}B tokens; per part {', '.join(f'{x / 1e9:.0f}B' for x in load)}", file=sys.stderr)


def cmd_run(a):
    plan = json.load(open(a.plan))
    parts = [int(x) for x in a.parts_of.split(",")] if a.parts_of else None
    part = parts[int(os.environ.get("SLURM_NODEID", 0))] if parts else a.part
    mine = [j for j in plan["shards"] if j["part"] == part and not os.path.exists(j["out"] + ".tok2.json")]
    if a.only:
        mine = [j for j in mine if any(s in j["src"] for s in a.only.split(","))]
    tasks = [(j["src"], d0, d1) for j in mine for d0, d1 in j["units"]]
    by_src = {j["src"]: j for j in mine}
    print(f"[tok2 part {part}] {len(mine)} shards, {len(tasks)} units, "
          f"{sum(j['tokens'] for j in mine) / 1e9:.1f}B tokens, {a.workers} workers", flush=True)
    t0, done_tok, cur = time.time(), 0, None
    with Pool(a.workers, initializer=_worker_init) as pool:
        for src, d0, d1, lens, toks, n_in in pool.imap(retok_unit, tasks, chunksize=1):
            j = by_src[src]
            if cur is None or cur["src"] != src:
                cur = {"src": src, "f": None, "lens": [], "n_in": 0, "n_out": 0, "t": time.time()}
                os.makedirs(os.path.dirname(j["out"]), exist_ok=True)
                cur["f"] = open(j["out"] + ".bin.tmp", "wb")
            cur["f"].write(toks.tobytes())
            cur["lens"].append(lens)
            cur["n_in"] += n_in
            cur["n_out"] += len(toks)
            done_tok += n_in
            if d1 == j["units"][-1][1]:                               # the shard's last unit: finish it
                cur["f"].close()
                src_idx = Index(src + ".idx")
                lens_all = np.concatenate(cur["lens"]) if cur["lens"] else np.zeros(0, np.int32)
                assert len(lens_all) == src_idx.seq_count, (src, len(lens_all), src_idx.seq_count)
                write_index(j["out"] + ".idx", lens_all, np.asarray(src_idx.doc_idx))
                os.replace(j["out"] + ".bin.tmp", j["out"] + ".bin")
                rec = {"source": src, "old_tokenizer": plan["old"], "new_tokenizer": plan["new"],
                       "sequences": int(src_idx.seq_count), "documents": int(src_idx.doc_count - 1),
                       "tokens_old": cur["n_in"], "tokens_new": cur["n_out"],
                       "seconds": round(time.time() - cur["t"], 1)}
                json.dump(rec, open(j["out"] + ".tok2.json", "w"), indent=1)
                el = time.time() - t0
                print(f"[tok2 part {part}] {os.path.relpath(j['out'], plan['out'])}: {cur['n_in'] / 1e9:.2f}B -> "
                      f"{cur['n_out'] / 1e9:.2f}B ({cur['n_out'] / max(cur['n_in'], 1):.3f}x) | total "
                      f"{done_tok / 1e9:.1f}B in {el / 60:.1f} min = {done_tok / el / 1e6:.1f}M tok/s", flush=True)
                cur = None
    print(f"[tok2 part {part}] done: {done_tok / 1e9:.1f}B tokens in {(time.time() - t0) / 60:.1f} min", flush=True)


def cmd_check(a):
    import gigatoken  # noqa: F401  (the same environment as run)
    plan = json.load(open(a.plan))
    _worker_init()
    dec = _W["dec"]
    newdec = Decoder(plan["new"])
    rng = np.random.default_rng(0)
    bad = n_done = 0
    for j in plan["shards"]:
        if a.only and not any(s in j["src"] for s in a.only.split(",")):
            continue
        if not os.path.exists(j["out"] + ".tok2.json"):
            print("MISSING", j["out"])
            bad += 1
            continue
        n_done += 1
        o, n = Index(j["src"] + ".idx"), Index(j["out"] + ".idx")
        ob, nb = np.memmap(j["src"] + ".bin", mode="r", dtype=np.int32), np.memmap(j["out"] + ".bin", mode="r",
                                                                                    dtype=np.int32)
        ok = (o.seq_count == n.seq_count and np.array_equal(o.doc_idx, n.doc_idx)
              and os.path.getsize(j["out"] + ".bin") == 4 * int(n.lengths.astype(np.int64).sum()))
        for d in rng.integers(0, max(o.doc_count - 1, 1), min(a.sample, o.doc_count - 1)):
            s0, s1 = int(o.doc_idx[d]), int(o.doc_idx[d + 1])
            old = b"".join(dec(np.asarray(ob[o.pointers[s] // 4:o.pointers[s] // 4 + o.lengths[s]]))
                           for s in range(s0, s1))
            new = b"".join(newdec(np.asarray(nb[n.pointers[s] // 4:n.pointers[s] // 4 + n.lengths[s]]))
                           for s in range(s0, s1))
            eo = [int(ob[o.pointers[s] // 4 + o.lengths[s] - 1]) == EOS for s in range(s0, s1) if o.lengths[s]]
            en = [int(nb[n.pointers[s] // 4 + n.lengths[s] - 1]) == EOS for s in range(s0, s1) if n.lengths[s]]
            ok &= (old.decode("utf-8", "replace") == new.decode("utf-8", "replace")) and eo == en
        if not ok:
            bad += 1
            print("BAD", j["out"])
    print(f"check: {n_done} shards, {bad} bad")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["plan", "run", "check"])
    ap.add_argument("--list", help="plan: shard prefixes ('prefix [rel]' per line)")
    ap.add_argument("--data-root", default="/gpfs/projects/ehpc533/users/korbip/data")
    ap.add_argument("--out", help="plan: the new tree's root")
    ap.add_argument("--parts", type=int, default=4)
    ap.add_argument("--unit-tokens", type=int, default=16 << 20)
    ap.add_argument("--plan")
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts-of", default="", help="run: comma-separated plan parts, indexed by SLURM_NODEID")
    ap.add_argument("--only", default="", help="run / check: shards whose source path contains one of these")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--sample", type=int, default=50)
    a = ap.parse_args()
    sys.exit({"plan": cmd_plan, "run": cmd_run, "check": cmd_check}[a.mode](a) or 0)


if __name__ == "__main__":
    main()
