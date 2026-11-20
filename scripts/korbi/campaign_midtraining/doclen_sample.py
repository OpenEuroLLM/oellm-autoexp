#!/usr/bin/env python3
"""Document-length layout of packed training sequences for a Megatron data mix (stdlib only: runs on a login
node). Mimics the blended GPT dataset: a sequence comes from one dataset (chosen by the mix weights) and is a
contiguous cut through its documents in shuffled order -- so it starts inside a document (length-biased) and its
last document is cut at the sequence end. Writes the per-sequence segment lengths (each segment one document piece,
attended with document masking).

    doclen_sample.py --mix <data_args file: weight prefix ...> --seq 65536 --n 2000 --out seqs_64k.json
"""

from __future__ import annotations

import argparse
import bisect
import json
import random
import struct


def idx_lengths(prefix: str, k: int, rng: random.Random) -> list[int]:
    """k document lengths drawn uniformly (by document) from an indexed dataset's .idx (seek per sample)."""
    with open(prefix + ".idx", "rb") as f:
        f.read(9)  # magic
        f.read(8)  # version
        f.read(1)  # dtype code
        n_seq = struct.unpack("<Q", f.read(8))[0]
        f.read(8)  # number of documents
        base = f.tell()
        out = []
        for _ in range(k):
            i = rng.randrange(n_seq)
            f.seek(base + 4 * i)
            out.append(struct.unpack("<i", f.read(4))[0])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mix", required=True)
    ap.add_argument("--seq", type=int, default=65536)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--docs-per-dataset", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    toks = open(a.mix).read().split()
    pairs = [(float(toks[i]), toks[i + 1]) for i in range(0, len(toks) - 1, 2)]
    tot = sum(w for w, _ in pairs)
    cum, acc = [], 0.0
    for w, _ in pairs:
        acc += w / tot
        cum.append(acc)
    pools = {}
    seqs = []
    for _ in range(a.n):
        di = min(bisect.bisect_left(cum, rng.random()), len(pairs) - 1)
        prefix = pairs[di][1]
        if prefix not in pools:
            pools[prefix] = [x for x in idx_lengths(prefix, a.docs_per_dataset, rng) if x > 0] or [1]
        pool = pools[prefix]
        # start inside a document chosen by length (a token's document), at a uniform offset
        w = [float(x) for x in pool]
        first = rng.choices(pool, weights=w)[0]
        segs, left = [], a.seq
        piece = min(left, first - rng.randrange(first))
        segs.append(piece)
        left -= piece
        while left > 0:
            d = rng.choice(pool)
            segs.append(min(d, left))
            left -= segs[-1]
        seqs.append(segs)
    longest = sorted(max(s) for s in seqs)
    print(f"{len(seqs)} sequences of {a.seq}: documents per sequence median "
          f"{sorted(len(s) for s in seqs)[len(seqs) // 2]}, longest piece median {longest[len(seqs) // 2]}, "
          f"p90 {longest[int(0.9 * len(seqs))]}, max {longest[-1]}; {len(pools)} datasets sampled")
    with open(a.out, "w") as f:
        json.dump({"mix": a.mix, "seq": a.seq, "seqs": seqs}, f)


if __name__ == "__main__":
    main()
