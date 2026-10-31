"""datamix v8: one Megatron tree (`--out`, default data/v8_raw) over the sources that are not in flag/, in the
flag convention (no <bos>, <eos> per document), for the decontamination scan (run_step.sh with ROOT=<out>).

    mv1sub/<subset>/megatron-lm/shard_0000_text_document   links to v8_sources/jsc_mv1_subsets (JUPITER flag tree)
    vtok/<name>/megatron-lm/shard_0000_text_document       LUMI vitiugin/data_tok: `<bos> doc <eos>` -> a new .idx
                                                            that starts every document one token later (the .bin
                                                            is linked, never copied); checked: every document
                                                            starts with <bos> (1) and ends with <eos> (2)
    hftok/<source>/megatron-lm/shard_NNNN_text_document    links to the GigaToken shards (v8_sources/jsc_tok)
    datamix.txt                                             "1.0 <prefix>" per shard (the scan's shard list)

    python scripts/datamix_v8/build_raw_tree.py [--only vtok]   # resumable: existing outputs are kept
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "flag_decontam"))
from common import Index, write_index  # noqa: E402

DATA = "/gpfs/projects/ehpc533/users/korbip/data"
SRC = f"{DATA}/v8_sources"
BOS, EOS = 1, 2


def link(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        if os.path.realpath(dst) == os.path.realpath(src):
            return
        os.remove(dst)
    os.symlink(src, dst)


def link_shard(src_prefix, dst_prefix):
    for ext in (".bin", ".idx"):
        link(src_prefix + ext, dst_prefix + ext)
    info = src_prefix.removesuffix("_text_document") + ".info.json"
    if os.path.exists(info):
        link(info, dst_prefix.removesuffix("_text_document") + ".info.json")


def strip_bos(src_prefix, dst_prefix):
    info_p = dst_prefix.removesuffix("_text_document") + ".info.json"
    if os.path.exists(info_p) and os.path.exists(dst_prefix + ".idx"):
        return json.load(open(info_p))["bos_strip"]
    idx = Index(src_prefix + ".idx")
    assert idx.multimodal_tail == 0 and idx.dtype_code == 4, src_prefix
    mm = np.memmap(src_prefix + ".bin", np.int32, mode="r")
    n = idx.seq_count
    first_seq = np.zeros(n, bool)
    first_seq[idx.doc_idx[:-1]] = True                 # a document's first sequence holds its <bos>
    last_seq = np.zeros(n, bool)
    last_seq[idx.doc_idx[1:] - 1] = True
    starts = idx.pointers // 4
    first_tok = np.empty(n, np.int32)
    last_tok = np.empty(n, np.int32)
    for a in range(0, n, 1 << 20):
        b = min(n, a + (1 << 20))
        first_tok[a:b] = mm[starts[a:b]]
        last_tok[a:b] = mm[starts[a:b] + idx.lengths[a:b].astype(np.int64) - 1]
    is_bos = first_seq & (first_tok == BOS)
    stats = {"documents": int(idx.doc_count - 1), "sequences": int(n),
             "docs_with_bos": int(is_bos.sum()), "docs_without_bos": int((first_seq & ~is_bos).sum()),
             "docs_ending_eos": int((last_seq & (last_tok == EOS)).sum()),
             "empty_after_strip": int((is_bos & (idx.lengths == 1)).sum())}
    lengths = idx.lengths - is_bos.astype(np.int32)
    pointers = idx.pointers + 4 * is_bos.astype(np.int64)
    keep_doc = np.ones(idx.doc_count - 1, bool)
    if stats["empty_after_strip"]:                     # a document that was only <bos>: dropped
        empty = is_bos & (lengths == 0)
        keep_doc[np.searchsorted(idx.doc_idx, np.flatnonzero(empty), "right") - 1] = False
    keep_seq = np.repeat(keep_doc, np.diff(idx.doc_idx))
    lengths, pointers = lengths[keep_seq], pointers[keep_seq]
    doc_idx = np.concatenate([[0], np.cumsum(np.diff(idx.doc_idx)[keep_doc])]).astype(np.int64)
    os.makedirs(os.path.dirname(dst_prefix), exist_ok=True)
    link(src_prefix + ".bin", dst_prefix + ".bin")
    write_index(dst_prefix + ".idx", 4, lengths, pointers, doc_idx)
    info = {"idx": {"size": os.path.getsize(dst_prefix + ".idx"), "dtype": "int32", "sequence_count": int(len(lengths)),
                    "document_count": int(len(doc_idx)), "total_tokens": int(lengths.astype(np.int64).sum())},
            "bin": {"size": os.path.getsize(src_prefix + ".bin")},
            "bos_strip": stats, "source": src_prefix}
    json.dump(info, open(info_p, "w"), indent=1)
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=f"{DATA}/v8_raw")
    ap.add_argument("--only", default="", help="comma list of mv1sub,vtok,hftok")
    a = ap.parse_args()
    only = set(a.only.split(",")) if a.only else {"mv1sub", "vtok", "hftok"}
    rows = []
    if "mv1sub" in only:
        for p in sorted(glob.glob(f"{SRC}/jsc_mv1_subsets/*/megatron-lm/*_text_document.idx")):
            p = p[:-4]
            sub = p.split("/jsc_mv1_subsets/")[1].split("/")[0]
            rel = f"mv1sub/{sub}/megatron-lm/{os.path.basename(p)}"
            link_shard(p, os.path.join(a.out, rel))
            rows.append(rel)
    if "vtok" in only:
        for p in sorted(glob.glob(f"{SRC}/lumi_vtok/*.idx")):
            p = p[:-4]
            name = os.path.basename(p).removesuffix("_text_document").removesuffix(".dedup.clean")
            rel = f"vtok/{name}/megatron-lm/shard_0000_text_document"
            s = strip_bos(p, os.path.join(a.out, rel))
            print(f"vtok {name}: {s}", flush=True)
            bad = s["docs_without_bos"] + (s["documents"] - s["docs_ending_eos"])
            if bad:
                print(f"  WARNING {name}: {s['docs_without_bos']} documents without <bos>, "
                      f"{s['documents'] - s['docs_ending_eos']} not ending in <eos>", flush=True)
            rows.append(rel)
    if "hftok" in only:
        for p in sorted(glob.glob(f"{SRC}/jsc_tok/*/megatron-lm/*_text_document.idx")):
            p = p[:-4]
            src = p.split("/jsc_tok/")[1].split("/")[0]
            rel = f"hftok/{src}/megatron-lm/{os.path.basename(p)}"
            link_shard(p, os.path.join(a.out, rel))
            rows.append(rel)
    # datamix.txt lists every shard in the tree (re-runs with --only keep the others)
    dm = os.path.join(a.out, "datamix.txt")
    old = [l.split()[1] for l in open(dm)] if os.path.exists(dm) else []
    allrows = sorted(set(old) | set(rows))
    with open(dm + ".tmp", "w") as f:
        f.write("".join(f"1.0 {r}\n" for r in allrows))
    os.replace(dm + ".tmp", dm)
    print(f"{len(rows)} shards this run, {len(allrows)} in {dm}")


if __name__ == "__main__":
    main()
