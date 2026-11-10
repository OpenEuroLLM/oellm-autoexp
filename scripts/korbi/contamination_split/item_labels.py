#!/usr/bin/env python3
"""Per test item: was it in the VANILLA flag mix (what the oellm_32b trained on)? From the existing v6
decontamination scan (docs/flag_post_decontamination.md, no new matching).

    python scripts/korbi/contamination_split/item_labels.py --work /gpfs/scratch/ehpc533/users/korbip/flag_decon_v6 \
        --out <dir> [--procs 64] [--src-regex ...]

1. Over the 452 vanilla shards (shards_all.txt; openthoughts-3 was added later, not in the vanilla mix) every
   hits/<shard>/*.npz row (seq, seg, longest run, windows, runs): per reference segment the longest verbatim run
   and the number of sequences whose pair passes finalize.py's rule (run >= max(min_run, min(ceil(frac * n_tok),
   abs_run)); frac 0.7, abs 200, min 20 -- the flag-decon-v2 drop rule). Every flagged document was dropped from
   flag-decon-v2 (our training data) except the 2 generic segments of _report/frac_0.7/generic_segments.tsv.
2. The items of the selected benchmark sources (refs_v6.attrib.items.jsonl.gz: src, split, file, row, segs
   [[seg, field]], short fields) with per segment: matchable (segments.npz n > 0), n_tokens, max run, flagged
   sequences, and its text (refs_v6.jsonl.gz line = segment id) for aligning with lm-eval samples.
Writes <out>/seg_stats.npz (maxrun, nflag over all segments) and <out>/items.jsonl.gz.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import time
from multiprocessing import Pool

import numpy as np

ROOT = "/gpfs/projects/ehpc533/training-data-collection/flag"
SRC_REGEX = (r"^(EleutherAI/coqa|lighteval/squad_v2|rajpurkar/squad_v2|rajpurkar/squad|aps/super_glue|winograd_wsc|"
             r"allenai/winogrande|facebook/belebele|Rowan/hellaswag|baber/piqa|allenai/social_i_qa|allenai/openbookqa|"
             r"EleutherAI/lambada_openai|allenai/ai2_arc)$")


def shard_key(prefix: str) -> str:
    return os.path.relpath(prefix, ROOT).replace("/", "__")


_need = None
_S = 0


def _init(need):
    global _need, _S
    _need, _S = need, len(need)


def _scan(hdir):
    maxrun = np.zeros(_S, np.int32)
    nflag = np.zeros(_S, np.int32)
    nfiles = nrows = 0
    if os.path.isdir(hdir):
        for fn in os.listdir(hdir):
            if not fn.endswith(".npz") or ".tmp" in fn:
                continue
            h = np.load(os.path.join(hdir, fn))["hits"]
            nfiles += 1
            if not len(h):
                continue
            nrows += len(h)
            seg, run = h[:, 1].astype(np.int64), h[:, 2].astype(np.int64)
            np.maximum.at(maxrun, seg, run.astype(np.int32))
            sel = run >= _need[seg]
            if sel.any():
                np.add.at(nflag, seg[sel], 1)
    return maxrun, nflag, nfiles, nrows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=48)
    ap.add_argument("--frac", type=float, default=0.7)
    ap.add_argument("--abs-run", type=int, default=200)
    ap.add_argument("--min-run", type=int, default=20)
    ap.add_argument("--src-regex", default=SRC_REGEX)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    segs = np.load(os.path.join(a.work, "table", "segments.npz"))
    n_tok = segs["n_tokens"].astype(np.int64)
    need = np.maximum(a.min_run, np.minimum(np.ceil(a.frac * n_tok), a.abs_run)).astype(np.int64)
    need[segs["n"] == 0] = np.iinfo(np.int64).max
    S = len(need)
    shards = [l.strip() for l in open(os.path.join(a.work, "shards_all.txt")) if l.strip()]
    hdirs = [os.path.join(a.work, "hits", shard_key(p)) for p in shards]
    missing = [h for h in hdirs if not os.path.isdir(h)]
    print(f"{S} segments; {len(shards)} vanilla shards, {len(missing)} without a hits dir", flush=True)
    maxrun = np.zeros(S, np.int32)
    nflag = np.zeros(S, np.int64)
    tf = tr = 0
    with Pool(a.procs, initializer=_init, initargs=(need,)) as pool:
        for i, (m, f, nf, nr) in enumerate(pool.imap_unordered(_scan, hdirs)):
            np.maximum(maxrun, m, out=maxrun)
            nflag += f
            tf += nf
            tr += nr
            if i % 50 == 0:
                print(f"  {i + 1}/{len(hdirs)} shards, {tf} files, {tr} rows, {time.time() - t0:.0f} s", flush=True)
    np.savez(os.path.join(a.out, "seg_stats.npz"), maxrun=maxrun, nflag=nflag, need=need, n=segs["n"],
             n_tokens=n_tok)
    print(f"scan: {tf} files, {tr} rows, {int((nflag > 0).sum())} segments flagged, {time.time() - t0:.0f} s",
          flush=True)

    rx = re.compile(a.src_regex)
    items, want = [], set()
    with gzip.open(os.path.join(a.work, "refs", "refs_v6.attrib.items.jsonl.gz"), "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if rx.match(r["src"]):
                items.append(r)
                want.update(s for s, _ in r["segs"])
    print(f"{len(items)} items of the selected sources, {len(want)} segments, {time.time() - t0:.0f} s", flush=True)
    text = {}
    with gzip.open(os.path.join(a.work, "refs", "refs_v6.jsonl.gz"), "rt", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i in want:
                text[i] = json.loads(line)["text"]
    print(f"texts: {len(text)}, {time.time() - t0:.0f} s", flush=True)
    with gzip.open(os.path.join(a.out, "items.jsonl.gz"), "wt", encoding="utf-8") as out:
        for r in items:
            r["seg_info"] = [{"seg": s, "field": f, "matchable": bool(segs["n"][s] > 0), "n_tokens": int(n_tok[s]),
                              "maxrun": int(maxrun[s]), "need": int(min(need[s], 10 ** 9)), "nflag": int(nflag[s]),
                              "text": text.get(s, "")} for s, f in r["segs"]]
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"done {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
