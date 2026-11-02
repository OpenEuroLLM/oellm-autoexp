#!/usr/bin/env python3
"""Continue a run on the decontaminated agenttrove WITHOUT changing its data order.

WHY
---
agenttrove-0.0-pure is agenttrove-0.0 minus 3712 distinct documents (4192 copies), every other
document token-identical, but in a different FILE order. Swapping the path in the mix would give
agenttrove a new shuffle (Megatron shuffles by file position) and a different 99.8/0.2 train/valid
split. Instead we keep the original agenttrove dataset and edit only its train shuffle_index:

    new[:J0]  = old[:J0]                               (samples already drawn: untouched)
    new[J0:]  = [w for w in old[J0:] if not bad[w]]    (same windows, same order, minus every
                                                        window touching a removed document)
                + padding from the same filtered list to the old length (never reached)

J0 = number of agenttrove samples the blend has drawn before the resume iteration (the blend draws
agenttrove samples 0, 1, 2, ... in order). Every other dataset, the blend tables and the mix file are
untouched: the cache directory is a symlink tree of the current read-only cache with this ONE file
replaced, so Megatron hits the cache everywhere (a hit = the 4 files exist, gpt_dataset.py) and the
run continues with exactly the samples it would have trained, minus the removed documents.

A window w covers document_index positions sample_index[w,0] .. sample_index[w+1,0] INCLUSIVE: the
end position always contributes at least the extra (label) token, even at offset 0.

USAGE (on a JUPITER login node; numpy only, except `verify --megatron`)
-----
    S=scripts/korbi/agenttrove_pure_shuffle.py
    python3 $S setup                      # symlink tree of the current cache
    python3 $S mask                       # removed doc ids + bad-window mask (run unaffected)
    python3 $S build --iteration 330125   # at the switch: iteration of the resume checkpoint
    python3 $S verify --iteration 330125  # exhaustive offline checks
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

FLAG = "/e/fscratch/e-sta-openeurollm/training_v2/training/collection/flag"
ORIG = f"{FLAG}/agenttrove-0.0/megatron-lm/data/shard_0000_text_document"
PURE = f"{FLAG}/agenttrove-0.0-pure/megatron-lm/openeurollm-256k-without-bos/data/shard_0000"
OLD_CACHE = "/e/fscratch/e-sta-openeurollm/jj1/data_cache_link"
NEW_CACHE = "/e/scratch/e-ext-2025e02-108/oellm_32b_dense_b1/data_cache_agenttrove_pure"
WORK = "/e/scratch/e-ext-2025e02-108/poeppel1/agenttrove_pure_check"
GBS = 4096


def load_idx(prefix):
    with open(prefix + ".idx", "rb") as f:
        assert f.read(9) == b"MMIDIDX\x00\x00", prefix
        f.read(8)
        f.read(1)
        n = struct.unpack("<Q", f.read(8))[0]
        f.read(8)
        lens = np.frombuffer(f.read(4 * n), dtype=np.int32).astype(np.int64)
        ptr = np.frombuffer(f.read(8 * n), dtype=np.int64)
    return lens, ptr // 4, np.memmap(prefix + ".bin", dtype=np.int32, mode="r")


def doc_md5s(prefix):
    lens, off, tok = load_idx(prefix)
    out = np.empty((len(lens), 16), np.uint8)
    for i in range(len(lens)):
        out[i] = np.frombuffer(hashlib.md5(tok[off[i] : off[i] + lens[i]].tobytes()).digest(), np.uint8)
    return out


def find_cache_base(cache, cls, split, match):
    """Hash prefix of the one cache entry of class/split whose description satisfies match()."""
    hits = []
    for p in Path(cache).glob(f"*-{cls}-{split}-description.txt"):
        d = json.loads(p.read_text())
        if match(d):
            hits.append(p.name[: -len("-description.txt")])
    return hits


def blend_base(cache):
    hits = find_cache_base(
        cache, "BlendedDataset", "train",
        lambda d: any(ORIG in json.dumps(x) for x in d.get("datasets", [])),
    )
    assert len(hits) == 1, f"expected one train blend over agenttrove-0.0, found {hits}"
    return hits[0]


def agent_base(cache, num_samples):
    hits = find_cache_base(
        cache, "GPTDataset", "train",
        lambda d: d.get("dataset_path") == ORIG and d.get("num_samples") == num_samples,
    )
    assert len(hits) == 1, f"expected one agenttrove train index with {num_samples} samples, found {hits}"
    return hits[0]


def blend_info(cache):
    base = blend_base(cache)
    d = json.loads((Path(cache) / f"{base}-description.txt").read_text())
    ids = [i for i, x in enumerate(d["datasets"]) if ORIG in json.dumps(x)]
    assert len(ids) == 1, ids
    k = ids[0]
    num_samples = d["datasets"][k]["num_samples"]
    return base, k, num_samples


def paths(cache, base):
    return {a: Path(cache) / f"{base}-{a}"
            for a in ("description.txt", "document_index.npy", "sample_index.npy", "shuffle_index.npy")}


def j0_for_iteration(cache, blend, k, iteration):
    di = np.load(Path(cache) / f"{blend}-dataset_index.npy", mmap_mode="r")
    si = np.load(Path(cache) / f"{blend}-dataset_sample_index.npy", mmap_mode="r")
    n = iteration * GBS
    count, last = 0, -1
    step = 1 << 27
    for a in range(0, n, step):
        sel = np.nonzero(np.asarray(di[a : min(n, a + step)]) == k)[0]
        if len(sel):
            count += len(sel)
            last = int(si[a + sel[-1]])
    assert last == count - 1, f"blend does not draw agenttrove samples in order ({last} vs {count - 1})"
    return count


def cmd_setup(args):
    new = Path(NEW_CACHE)
    new.mkdir(parents=True, exist_ok=True)
    made = 0
    for p in Path(OLD_CACHE).iterdir():
        dst = new / p.name
        if not dst.exists() and not dst.is_symlink():
            dst.symlink_to(os.path.realpath(p))
            made += 1
    (new / "README.txt").write_text(
        "Symlink tree of " + OLD_CACHE + " with ONE real file: the agenttrove-0.0 train\n"
        "shuffle_index, filtered by scripts/korbi/agenttrove_pure_shuffle.py so that, from the\n"
        "switch iteration on, no window touching a document absent from agenttrove-0.0-pure is\n"
        "drawn. The mix file still names agenttrove-0.0; the trained tokens are the pure set.\n"
        "See build.json for the switch iteration and counts.\n"
    )
    print(f"[setup] {new}: {made} new symlink(s), {sum(1 for _ in new.iterdir())} entries")


def cmd_mask(args):
    Path(WORK).mkdir(parents=True, exist_ok=True)
    om, pm = Path(WORK) / "orig_md5.npy", Path(WORK) / "pure_md5.npy"
    ha = np.load(om) if om.exists() else doc_md5s(ORIG)
    hb = np.load(pm) if pm.exists() else doc_md5s(PURE)
    np.save(om, ha)
    np.save(pm, hb)
    keep = {x.tobytes() for x in hb}
    removed = np.array([i for i in range(len(ha)) if ha[i].tobytes() not in keep], dtype=np.int64)
    np.save(Path(WORK) / "removed_doc_ids.npy", removed)
    print(f"[mask] removed original documents: {len(removed)} of {len(ha)}")

    blend, k, ns = blend_info(OLD_CACHE)
    base = agent_base(OLD_CACHE, ns)
    p = paths(OLD_CACHE, base)
    doc = np.load(p["document_index.npy"], mmap_mode="r")
    smp = np.load(p["sample_index.npy"], mmap_mode="r")
    bad_pos = np.isin(np.asarray(doc), removed)
    cs = np.concatenate([[0], np.cumsum(bad_pos, dtype=np.int64)])
    beg, end = np.asarray(smp[:-1, 0], np.int64), np.asarray(smp[1:, 0], np.int64)
    bad = (cs[end + 1] - cs[beg]) > 0
    np.save(Path(WORK) / "bad_windows.npy", bad)
    json.dump(
        {"blend": blend, "dataset_id": k, "num_samples": ns, "gpt_base": base,
         "windows": int(len(bad)), "bad_windows": int(bad.sum()),
         "removed_docs": int(len(removed)), "bad_doc_positions": int(bad_pos.sum())},
        open(Path(WORK) / "mask.json", "w"), indent=2,
    )
    print(f"[mask] agenttrove train index {base}: {len(bad)} windows, {int(bad.sum())} touch a removed "
          f"document ({bad.mean():.4%}); blend {blend}, dataset id {k}, num_samples {ns}")


def filtered(old, bad, j0):
    tail = np.asarray(old[j0:])
    keep = tail[~bad[tail]]
    need = len(old) - j0
    pad = np.resize(keep, need - len(keep)) if len(keep) < need else keep[:0]
    return np.concatenate([np.asarray(old[:j0]), keep, pad]).astype(old.dtype), len(keep)


def cmd_build(args):
    done = Path(NEW_CACHE) / "build.json"
    if done.exists() and not args.force:
        sys.exit(f"[build] {done} exists (switch at iteration {json.load(open(done))['iteration']}); a run "
                 "already trains on this index -- rebuilding moves its switch point. --force to override.")
    m = json.load(open(Path(WORK) / "mask.json"))
    bad = np.load(Path(WORK) / "bad_windows.npy")
    j0 = j0_for_iteration(OLD_CACHE, m["blend"], m["dataset_id"], args.iteration)
    old_p = paths(OLD_CACHE, m["gpt_base"])["shuffle_index.npy"]
    new_p = paths(NEW_CACHE, m["gpt_base"])["shuffle_index.npy"]
    old = np.load(old_p, mmap_mode="r")
    new, kept = filtered(old, bad, j0)
    used = m["num_samples"]  # agenttrove samples the blend will ever request (upper bound)
    assert kept >= used - j0, f"filtered list too short: {kept} clean windows for {used - j0} draws"
    tmp = new_p.with_name(new_p.name + ".tmp.npy")
    np.save(tmp, new, allow_pickle=True)
    if new_p.is_symlink() or new_p.exists():
        new_p.unlink()
    os.replace(tmp, new_p)
    json.dump(
        {"iteration": args.iteration, "gbs": GBS, "j0": int(j0), "kept_clean": int(kept),
         "dropped_in_tail": int(len(old) - j0 - kept), "file": str(new_p), **m},
        open(Path(NEW_CACHE) / "build.json", "w"), indent=2,
    )
    print(f"[build] iteration {args.iteration}: J0 = {j0} agenttrove samples drawn; "
          f"{len(old) - j0 - kept} windows dropped from the tail; wrote {new_p}")


def cmd_verify(args):
    m = json.load(open(Path(WORK) / "mask.json"))
    bad = np.load(Path(WORK) / "bad_windows.npy")
    removed = np.load(Path(WORK) / "removed_doc_ids.npy")
    j0 = j0_for_iteration(OLD_CACHE, m["blend"], m["dataset_id"], args.iteration)
    op, np_ = paths(OLD_CACHE, m["gpt_base"]), paths(NEW_CACHE, m["gpt_base"])
    old = np.load(op["shuffle_index.npy"], mmap_mode="r")
    new = np.load(np_["shuffle_index.npy"], mmap_mode="r")
    fails = 0

    def check(ok, what):
        nonlocal fails
        fails += not ok
        print(("  OK   " if ok else "  FAIL ") + what)

    check(not np_["shuffle_index.npy"].is_symlink(), "new shuffle_index is a real file")
    for a in ("description.txt", "document_index.npy", "sample_index.npy"):
        check(os.path.realpath(np_[a]) == os.path.realpath(op[a]), f"{a} is the old file")
    check(new.dtype == old.dtype and new.shape == old.shape, f"dtype/shape unchanged {old.dtype} {old.shape}")
    check(bool(np.array_equal(new[:j0], old[:j0])), f"first {j0} entries (drawn before the switch) identical")
    used = m["num_samples"]
    t = np.asarray(new[j0:used])
    check(not bool(bad[t].any()), f"no window touching a removed document among draws {j0}..{used}")
    tail = np.asarray(old[j0:])
    check(bool(np.array_equal(t, tail[~bad[tail]][: used - j0])), "draws after the switch = old order minus bad windows")
    # independent token-level check: decode some windows and look for removed documents by id
    doc = np.load(op["document_index.npy"], mmap_mode="r")
    smp = np.load(op["sample_index.npy"], mmap_mode="r")
    rs = np.random.default_rng(0).choice(np.arange(j0, used), size=min(20000, used - j0), replace=False)
    rem = set(removed.tolist())
    hit = sum(any(int(doc[i]) in rem for i in range(int(smp[w, 0]), int(smp[w + 1, 0]) + 1)) for w in new[rs])
    check(hit == 0, f"20000 random draws after the switch, decoded by document ids: {hit} contain a removed doc")
    blend = m["blend"]
    for a in ("dataset_index.npy", "dataset_sample_index.npy", "description.txt"):
        f = Path(NEW_CACHE) / f"{blend}-{a}"
        check(os.path.realpath(f) == os.path.realpath(Path(OLD_CACHE) / f.name), f"blend {a} is the old file")
    zero = [p.name for p in Path(NEW_CACHE).iterdir() if p.is_file() and p.stat().st_size == 0]
    check(not zero, f"no 0-byte files in the cache ({len(zero)})")
    if args.megatron:
        fails += megatron_check(m, j0, used, removed)
    print(f"[verify] {'PASS' if not fails else f'{fails} FAILED'}")
    sys.exit(1 if fails else 0)


def megatron_check(m, j0, used, removed):
    """Through Megatron's own GPTDataset: same cache hash (= cache hit, nothing rebuilt) and the
    document ids it returns for draws after the switch contain no removed document."""
    from megatron.core.datasets.gpt_dataset import GPTDataset
    from megatron.core.datasets.indexed_dataset import IndexedDataset

    desc = json.loads(paths(OLD_CACHE, m["gpt_base"])["description.txt"].read_text())
    print(f"  [megatron] description: {json.dumps({k: desc[k] for k in desc if k != 'tokenizer'})}")
    # The class's own index walk on the NEW cache files (a full config would need the tokenizer).
    ds = GPTDataset.__new__(GPTDataset)
    ds.dataset = IndexedDataset(ORIG, multimodal=False, mmap=True)
    ds.shuffle_index = None
    p = paths(NEW_CACHE, m["gpt_base"])
    ds.path_to_shuffle_index = str(p["shuffle_index.npy"])
    ds.path_to_sample_index = str(p["sample_index.npy"])
    ds.path_to_document_index = str(p["document_index.npy"])

    class _C:
        add_extra_token_to_sequence = True
        sequence_length = desc["sequence_length"]

    ds.config = _C()
    ds._pad_token_id = -1
    rem = set(removed.tolist())
    rs = np.random.default_rng(1).choice(np.arange(j0, used), size=min(2000, used - j0), replace=False)
    hit = 0
    for j in rs:
        tokens, doc_ids, _ = ds._query_document_sample_shuffle_indices(int(j))
        assert len(tokens) == desc["sequence_length"] + 1
        hit += any(int(d) in rem for d in doc_ids)
    ok = hit == 0
    print(("  OK   " if ok else "  FAIL ") + f"[megatron] 2000 draws via GPTDataset._query_document_sample_shuffle_indices: {hit} hit a removed doc")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup")
    sub.add_parser("mask")
    b = sub.add_parser("build")
    b.add_argument("--iteration", type=int, required=True, help="iteration of the resume checkpoint")
    b.add_argument("--force", action="store_true", help="rebuild although build.json exists")
    v = sub.add_parser("verify")
    v.add_argument("--iteration", type=int, required=True)
    v.add_argument("--megatron", action="store_true", help="also query through Megatron's GPTDataset")
    args = ap.parse_args()
    {"setup": cmd_setup, "mask": cmd_mask, "build": cmd_build, "verify": cmd_verify}[args.cmd](args)


if __name__ == "__main__":
    main()
