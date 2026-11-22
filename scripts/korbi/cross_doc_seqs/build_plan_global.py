#!/usr/bin/env python3
"""One packing plan over a long step range (docs/cross_doc_seqs/README.md): every document of e.g. 10,000 steps
(~540M documents) clustered together, sharded over many GPUs. The windowed build_plan.py holds a whole window in
one process; this is the same plan (same format, same guarantees) for a window that does not fit one.

    stages (each a pure function of files under <out>/global/<name>/, run by run_pipeline.sh):
      centroids  one process: e-means coarse centroids (Bessa et al. 2026, mini-batch k-means with EMA centroid
                 updates) on a ~--sample document sample, chained greedily by similarity; the chain is cut into one
                 contiguous range per bucket (= per order process), balanced by the sample's cluster sizes
      assign     every GPU, its share of the window's stream chunks: each document -> nearest centroid -> bucket;
                 written sorted by bucket (one slice per bucket and file)
      order      every GPU, one bucket: units, then per coarse cluster in chain order near duplicates, fine leaves,
                 leaf chains (build_plan.order_clusters)
      assemble   one process: the buckets concatenated in chain order, copies (repeats, near duplicates) placed
                 over the WHOLE window (build_plan.place_copies: no sequence holds two of a class), cut into
                 sequences, shuffled over the window's non-solo slots -> <out>/plan/<name>/ck<w0>_<w1>/

All pieces and repeats of a document share its embedding, so they land in the same coarse cluster and bucket:
units (occurrence layers, merged pieces) and near-duplicate classes are complete inside a bucket.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_plan as bp  # noqa: E402
from cds_data import load_stream  # noqa: E402


def gdir(a) -> str:
    return os.path.join(a.out, "global", a.name)


def window_chunks(a) -> list:
    """The stream chunks exactly covering checkpoints [w0, w1)."""
    out = []
    for p in sorted(glob.glob(os.path.join(a.out, "stream", "ck*_*"))):
        if ".tmp" in os.path.basename(p):
            continue
        c0, c1 = (int(x) for x in os.path.basename(p)[2:].split("_"))
        if c0 >= a.w0 and c1 <= a.w1:
            out.append((c0, c1, p))
    assert out and out[0][0] == a.w0 and out[-1][1] == a.w1, f"stream chunks do not cover [{a.w0}, {a.w1})"
    assert all(x[1] == y[0] for x, y in zip(out, out[1:])), "gap in the stream chunks"
    return out


def chunk_pieces(path: str):
    """Pieces of a chunk's non-solo slots: (ds, doc, off, len)."""
    arr, meta = load_stream(path)
    nseg = np.diff(np.asarray(arr["seg_ptr"]))
    keep = np.repeat(nseg > 1, nseg)
    return (np.asarray(arr["seg_ds"])[keep], np.asarray(arr["seg_doc"])[keep],
            np.asarray(arr["seg_off"])[keep].astype(np.int64), np.asarray(arr["seg_len"])[keep].astype(np.int64)), meta


def norm_dim(v: np.ndarray, dim: int | None) -> np.ndarray:
    """Matryoshka truncation to dim (then re-normalized), float32."""
    v = np.asarray(v, np.float32)
    if dim and v.shape[1] > dim:
        v = v[:, :dim]
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)


def chunk_vectors(a, c0: int, c1: int, keys: np.ndarray, rng):
    """Embeddings of the given (sorted unique) keys of a chunk; a key without one gets a random direction."""
    k, v = bp.load_embeddings(os.path.join(a.out, "emb", a.emb), keys, c0, c1)
    out = None
    r = np.searchsorted(k, keys) if len(k) else np.zeros(len(keys), np.int64)
    have = (r < len(k)) & (k[np.minimum(r, max(len(k) - 1, 0))] == keys) if len(k) else np.zeros(len(keys), bool)
    dim = (a.dim or v.shape[1]) if v is not None else (a.dim or 256)
    out = np.empty((len(keys), dim), np.float32)
    if have.any():
        out[have] = norm_dim(v[r[have]], a.dim)
    if (~have).any():
        out[~have] = norm_dim(rng.standard_normal((int((~have).sum()), dim)), None)
    return out, int((~have).sum())


# ---------------------------------------------------------------------------------------------- e-means
def emeans(x, k: int, epochs: float, batch: int, alpha0: float, seed: int):
    """Spherical mini-batch k-means with EMA centroid statistics (e-means, Bessa et al. 2026, arXiv 2611.09501):
    per batch N_j <- g N_j + (1-g) n_j, S_j <- g S_j + (1-g) s_j, c_j = S_j / |S_j|, step 1-g on a cosine schedule
    from alpha0 to alpha0 / 10; empty clusters re-seeded from the largest every 5 batches. x: torch [n, d] unit rows."""
    import math

    import torch

    n, d = x.shape
    g = torch.Generator(device="cpu").manual_seed(seed)
    c = x[torch.randperm(n, generator=g)[:k].to(x.device)].float().clone()
    N = torch.full((k,), 1e-10, device=x.device)
    S = c * N[:, None]
    steps = max(1, int(math.ceil(epochs * n / batch)))
    for t in range(steps):
        alpha = alpha0 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t / max(steps - 1, 1))))
        idx = torch.randint(0, n, (min(batch, n),), generator=g).to(x.device)
        xb = x[idx].float()
        lab = torch.cat([(xb[i:i + 65536] @ c.T).argmax(1) for i in range(0, len(xb), 65536)])
        nb = torch.bincount(lab, minlength=k).float()
        sb = torch.zeros_like(S).index_add_(0, lab, xb)
        N = (1 - alpha) * N + alpha * nb
        S = (1 - alpha) * S + alpha * sb
        if t % 5 == 4:  # revive dead clusters next to the largest ones
            dead = torch.nonzero(N < 1e-3 * N.mean()).flatten()
            if len(dead):
                donors = torch.topk(N, len(dead)).indices
                S[dead] = S[donors] + 1e-3 * torch.randn(len(dead), d, generator=g).to(x.device) * S[donors].norm(
                    dim=1, keepdim=True)
                N[dead] = N[donors] / 2
                N[donors] = N[donors] / 2
                S[donors] = S[donors] / 2
        c = torch.nn.functional.normalize(S, dim=1)
    return c


def assign_rows(v: np.ndarray, c, dev, block: int = 262144) -> np.ndarray:
    import torch

    out = np.empty(len(v), np.int64)
    for i in range(0, len(v), block):
        xb = torch.as_tensor(v[i:i + block], device=dev, dtype=c.dtype)
        out[i:i + block] = (xb @ c.T).argmax(1).cpu().numpy()
    return out


# ------------------------------------------------------------------------------------------------- stages
def stage_centroids(a, dev) -> None:
    import torch

    dst = os.path.join(gdir(a), "centroids.npz")
    if os.path.exists(dst):
        print(f"{dst}: exists", flush=True)
        return
    t0 = time.time()
    chunks = window_chunks(a)
    n_est = 0
    for _, _, p in chunks:
        arr, _ = load_stream(p)
        nseg = np.diff(np.asarray(arr["seg_ptr"]))
        n_est += int(nseg[nseg > 1].sum())
    k = max(1, int(round(n_est / a.coarse)))
    files = []  # the window's chunk directories (and any non-chunk directory, like load_embeddings)
    for d in sorted(glob.glob(os.path.join(a.out, "emb", a.emb, "*"))):
        b = os.path.basename(d)
        if b.startswith("ck") and "_" in b:
            c0, c1 = (int(x) for x in b[2:].split("_"))
            if c0 < a.w0 or c1 > a.w1:
                continue
        files += sorted(f for f in glob.glob(os.path.join(d, "part_*.npz")) if not f.endswith(".tmp.npz"))
    rng = np.random.default_rng(a.seed)
    rng.shuffle(files)
    vs, n = [], 0
    for f in files:
        v = np.load(f)["vec"]
        vs.append(norm_dim(v, a.dim).astype(np.float16))
        n += len(v)
        if n >= a.sample:
            break
    x = np.concatenate(vs)
    if len(x) > a.sample:
        x = x[rng.choice(len(x), a.sample, replace=False)]
    xt = torch.as_tensor(x, device=dev)
    c = emeans(xt, min(k, len(x)), a.epochs, a.batch, a.alpha0, a.seed)
    lab = assign_rows(x, c.to(torch.float16) if dev.type == "cuda" else c, dev)
    fit = float(np.mean([(xt[i:i + 65536].float() @ c.T).max(1).values.mean().item() for i in range(0, len(x), 65536)]))
    cn = c.float().cpu().numpy()
    k = len(cn)
    chain = bp.greedy_chain(cn @ cn.T) if k > 1 else np.zeros(1, np.int64)
    pos_of = np.empty(k, np.int64)
    pos_of[chain] = np.arange(k)
    # contiguous chain ranges of ~equal (sampled) size, one per bucket
    size = np.bincount(pos_of[lab], minlength=k).astype(np.float64)
    cum = np.cumsum(size) / max(size.sum(), 1)
    bucket_of_pos = np.minimum((cum * a.buckets - 1e-9).astype(np.int64), a.buckets - 1)
    bucket_of = bucket_of_pos[pos_of]
    os.makedirs(gdir(a), exist_ok=True)
    tmp = dst[:-4] + ".tmp.npz"
    np.savez(tmp, centroids=cn.astype(np.float16), chain=chain, pos_of=pos_of, bucket_of=bucket_of,
             buckets=a.buckets, n_est=n_est)
    os.replace(tmp, dst)
    print(f"centroids: {k} coarse clusters for ~{n_est} documents (sample {len(x)}, mean cosine to centroid "
          f"{fit:.3f}), {a.buckets} buckets, {time.time() - t0:.0f} s", flush=True)


def stage_assign(a, dev, rank: int, world: int) -> None:
    import torch

    dst = os.path.join(gdir(a), "assign", f"r{rank:04d}")
    if os.path.exists(os.path.join(dst, "done")):
        return
    t0 = time.time()
    z = np.load(os.path.join(gdir(a), "centroids.npz"))
    c = torch.as_tensor(z["centroids"], device=dev)
    c = c if dev.type == "cuda" else c.float()
    bucket_of, P = z["bucket_of"], int(z["buckets"])
    rng = np.random.default_rng(a.seed + rank)
    P_ds, P_doc, P_off, P_len, P_b, K_key, K_cl, K_vec, K_b = [], [], [], [], [], [], [], [], []
    miss = 0
    for c0, c1, p in window_chunks(a)[rank::world]:
        (ds, doc, off, ln), _ = chunk_pieces(p)
        key = bp.doc_key(ds, doc)
        ukey, inv = np.unique(key, return_inverse=True)
        v, m = chunk_vectors(a, c0, c1, ukey, rng)
        miss += m
        cl = assign_rows(v.astype(np.float16) if dev.type == "cuda" else v, c, dev)
        b = bucket_of[cl]
        P_ds.append(ds); P_doc.append(doc); P_off.append(off); P_len.append(ln); P_b.append(b[inv])
        K_key.append(ukey); K_cl.append(cl); K_vec.append(v.astype(np.float16)); K_b.append(b)
    os.makedirs(dst + ".tmp", exist_ok=True)

    def save_sorted(names_arrays, b):
        o = np.argsort(b, kind="stable")
        for name, arr in names_arrays:
            np.save(os.path.join(dst + ".tmp", name + ".npy"), arr[o])
        np.save(os.path.join(dst + ".tmp", names_arrays[0][0].split("_")[0] + "_ptr.npy"),
                np.searchsorted(b[o], np.arange(P + 1)))

    cat = (lambda xs, dt: np.concatenate(xs).astype(dt) if xs else np.zeros(0, dt))
    pb = cat(P_b, np.int64)
    save_sorted([("piece_ds", cat(P_ds, np.int16)), ("piece_doc", cat(P_doc, np.int64)),
                 ("piece_off", cat(P_off, np.int64)), ("piece_len", cat(P_len, np.int64))], pb)
    kb = cat(K_b, np.int64)
    dim = K_vec[0].shape[1] if K_vec else (a.dim or 256)
    save_sorted([("key_key", cat(K_key, np.int64)), ("key_cluster", cat(K_cl, np.int64)),
                 ("key_vec", np.concatenate(K_vec) if K_vec else np.zeros((0, dim), np.float16))], kb)
    open(os.path.join(dst + ".tmp", "done"), "w").write(json.dumps({"pieces": int(len(pb)), "missing": miss}))
    os.replace(dst + ".tmp", dst)
    print(f"assign r{rank}: {len(pb)} pieces, {miss} keys without embedding, {time.time() - t0:.0f} s", flush=True)


def stage_order(a, dev, bucket: int) -> None:
    import torch

    dst = os.path.join(gdir(a), "order", f"b{bucket:04d}")
    if os.path.exists(os.path.join(dst, "done")):
        return
    t0 = time.time()
    z = np.load(os.path.join(gdir(a), "centroids.npz"))
    pos_of = z["pos_of"]
    parts = {k: [] for k in ("piece_ds", "piece_doc", "piece_off", "piece_len", "key_key", "key_cluster", "key_vec")}
    for r in sorted(glob.glob(os.path.join(gdir(a), "assign", "r*"))):
        if r.endswith(".tmp"):
            continue
        for kind in ("piece", "key"):
            ptr = np.load(os.path.join(r, f"{kind}_ptr.npy"))
            lo, hi = int(ptr[bucket]), int(ptr[bucket + 1])
            for name in parts:
                if name.startswith(kind):
                    parts[name].append(np.asarray(np.load(os.path.join(r, name + ".npy"), mmap_mode="r")[lo:hi]))
    cat = {k: np.concatenate(v) for k, v in parts.items()}
    kk, first = np.unique(cat["key_key"], return_index=True)
    kcl, kvec = cat["key_cluster"][first], cat["key_vec"][first]
    u = bp.units_from_pieces(cat["piece_ds"], cat["piece_doc"], cat["piece_off"], cat["piece_len"])
    us = u["unit_start"]
    ukey, ulayer = u["key"][us], u["layer"][us]
    utok = np.add.reduceat(u["len"], us) if len(us) else np.zeros(0, np.int64)
    r = np.searchsorted(kk, ukey)
    assert len(kk) == 0 or np.array_equal(kk[r], ukey), "a unit without its key's embedding"
    ucl = kcl[r]
    uvec = kvec[r]
    l0 = np.nonzero(ulayer == 0)[0]
    o = np.lexsort((l0, pos_of[ucl[l0]]))  # layer-0 units by chain position of their coarse cluster
    l0s = l0[o]
    cpos = pos_of[ucl[l0s]]
    starts = np.r_[0, np.nonzero(cpos[1:] != cpos[:-1])[0] + 1, len(l0s)]
    clusters = [np.searchsorted(l0, l0s[s:e]) for s, e in zip(starts[:-1], starts[1:])]  # indices into l0
    x = torch.as_tensor(uvec[l0], device=dev, dtype=torch.float16 if dev.type == "cuda" else torch.float32)
    order, leaf, deferred, pairs, _ = bp.order_clusters(x, utok[l0], None, clusters, a, a.seed + bucket, dev)
    base = l0[order]
    unit_leaf = np.full(len(us), -1, np.int64)
    unit_leaf[l0] = leaf
    copies = np.r_[l0[deferred], np.nonzero(ulayer > 0)[0]].astype(np.int64)
    pa = np.array([[l0[x_], l0[y_]] for x_, y_ in pairs], np.int64).reshape(-1, 2)
    os.makedirs(dst + ".tmp", exist_ok=True)
    for name, arr in (("ds", u["ds"].astype(np.int16)), ("doc", u["doc"]), ("off", u["off"]), ("len", u["len"]),
                      ("unit_start", us), ("ukey", ukey), ("utok", utok), ("base", base), ("copies", copies),
                      ("pairs", pa), ("leaf", unit_leaf), ("uvec", uvec.astype(np.float16))):
        np.save(os.path.join(dst + ".tmp", name + ".npy"), arr)
    open(os.path.join(dst + ".tmp", "done"), "w").write(json.dumps(
        {"units": int(len(us)), "base": int(len(base)), "copies": int(len(copies)), "clusters": len(clusters)}))
    os.replace(dst + ".tmp", dst)
    print(f"order b{bucket}: {len(us)} units in {len(clusters)} coarse clusters, {len(copies)} copies, "
          f"{time.time() - t0:.0f} s", flush=True)


def stage_assemble(a) -> None:
    t0 = time.time()
    dst = os.path.join(a.out, "plan", a.name, f"ck{a.w0:07d}_{a.w1:07d}")
    if os.path.isdir(dst):
        print(f"{dst}: exists", flush=True)
        return
    z = np.load(os.path.join(gdir(a), "centroids.npz"))
    P = int(z["buckets"])
    bdirs = [os.path.join(gdir(a), "order", f"b{p:04d}") for p in range(P)]
    load = (lambda d, n: np.load(os.path.join(d, n + ".npy")))
    ds, doc, off, ln, ustart, ukey, utok, base, copies, pairs, leaf = ([] for _ in range(11))
    uoff, poff, loff, U_of = 0, 0, 0, [0]
    for d in bdirs:
        us = load(d, "unit_start")
        ds.append(load(d, "ds")); doc.append(load(d, "doc")); off.append(load(d, "off")); ln.append(load(d, "len"))
        ustart.append(us + poff); ukey.append(load(d, "ukey")); utok.append(load(d, "utok"))
        base.append(load(d, "base") + uoff); copies.append(load(d, "copies") + uoff); pairs.append(load(d, "pairs") + uoff)
        lf = load(d, "leaf")
        leaf.append(np.where(lf >= 0, lf + loff, -1))
        loff += int(lf.max()) + 1 if (lf >= 0).any() else 0
        uoff += len(us)
        poff += len(load(d, "ds"))
        U_of.append(uoff)
    cat = (lambda xs: np.concatenate(xs) if xs else np.zeros(0, np.int64))
    ds, doc, off, ln = cat(ds), cat(doc), cat(off), cat(ln)
    ustart, ukey, utok = cat(ustart), cat(ukey), cat(utok)
    base, copies, leaf = cat(base), cat(copies), cat(leaf)
    pairs = np.concatenate(pairs) if pairs else np.zeros((0, 2), np.int64)
    n_units = len(ukey)
    S = json.load(open(os.path.join(a.out, "blend.json")))["seq_length"]
    rng = np.random.default_rng(a.seed + a.w0)
    cls = bp.conflict_classes(ukey, [tuple(p) for p in pairs.tolist()])
    uorder, forced = bp.place_copies(base, copies, cls, utok, S, rng)
    assert len(uorder) == n_units and len(np.unique(uorder)) == n_units
    unseparated = bp.count_conflicts(uorder, utok, cls, S)
    leaf[copies] = loff + np.arange(len(copies))  # each copy its own group
    ulen = np.diff(np.r_[ustart, len(ds)])
    pidx = bp.ranges(ustart[uorder], ulen[uorder])
    p_grp = np.repeat(leaf[uorder], ulen[uorder])
    seq_ptr, seg = bp.cut_sequences(ds[pidx], doc[pidx], off[pidx], ln[pidx], p_grp, S)
    n_seq = len(seq_ptr) - 1
    # the window's non-solo slots (the indices the plan replaces), shuffled over the sequences
    bidx, o_ptr, o_ds, o_doc = [], [0], [], []
    blend_md5 = None
    for c0, c1, p in window_chunks(a):
        arr, meta = load_stream(p)
        blend_md5 = meta["blend_md5"]
        nseg = np.diff(np.asarray(arr["seg_ptr"]))
        bidx.append(np.asarray(arr["slot_bidx"])[nseg > 1])
    bidx = np.concatenate(bidx)
    assert len(bidx) == n_seq, f"{n_seq} sequences for {len(bidx)} non-solo slots"
    perm = rng.permutation(n_seq)
    o = np.argsort(bidx)
    stats = {"window": [a.w0, a.w1], "global": True, "coarse_clusters": int(len(z["chain"])), "buckets": P,
             "units": int(n_units), "copies": int(len(copies)), "near_duplicate_pairs": int(len(pairs)),
             "copies_forced": int(forced), "units_sharing_a_sequence_with_a_copy": int(unseparated),
             "sequences": int(n_seq), "segments_per_seq_after": float(len(seg["ds"]) / max(n_seq, 1))}
    stats.update(coherence(a, bdirs, U_of, ukey, seq_ptr, seg))
    stats["seconds"] = round(time.time() - t0, 1)
    meta = bp.plan_meta(a.out, blend_md5, S, a.w0, a.w1, a.emb,
                        {k: getattr(a, k) for k in ("coarse", "leaf_tokens", "iters", "dup_cos", "seed", "sample",
                                                    "epochs", "batch", "alpha0", "dim")} | {"global": True})
    bp.write_plan(dst, bidx[o], perm[o], seq_ptr, seg, meta, stats)
    print(f"{dst}: {json.dumps(stats)}", flush=True)


def coherence(a, bdirs, U_of, ukey, seq_ptr, seg, n_sample: int = 2000) -> dict:
    """Mean cosine between the distinct documents of a sequence, before (the original slots) and after (the plan),
    over sampled sequences; vectors read from the buckets' unit tables."""
    order = np.argsort(ukey, kind="stable")
    skey = ukey[order]
    uvecs = [np.load(os.path.join(d, "uvec.npy"), mmap_mode="r") for d in bdirs]
    U_of = np.asarray(U_of)

    def vecs(keys):
        """Vectors of many keys: one sorted fancy-index read per bucket file (single-row reads from GPFS memory
        maps cost a page fault each)."""
        r = np.searchsorted(skey, keys)
        ok = (r < len(skey)) & (skey[np.minimum(r, len(skey) - 1)] == keys)
        unit = order[r[ok]]
        out = np.zeros((len(keys), uvecs[0].shape[1]), np.float32)
        b = np.searchsorted(U_of, unit, side="right") - 1
        idx = np.nonzero(ok)[0]
        for bi in np.unique(b):
            m = b == bi
            rows = unit[m] - U_of[bi]
            o = np.argsort(rows)
            out[idx[m][o]] = np.asarray(uvecs[bi][rows[o]], np.float32)
        return out, ok

    def mean_cos(ptr, ds_, doc_, rng):
        n = len(ptr) - 1
        seqs = rng.choice(n, size=min(n_sample, n), replace=False)
        keys = [np.unique(bp.doc_key(ds_[ptr[s]:ptr[s + 1]], doc_[ptr[s]:ptr[s + 1]])) for s in seqs]
        allk = np.unique(np.concatenate(keys)) if keys else np.zeros(0, np.int64)
        V, ok = vecs(allk)
        vals = []
        for k in keys:
            i = np.searchsorted(allk, k)
            i = i[ok[i]]
            if len(i) < 2:
                continue
            v = V[i]
            g = v @ v.T
            vals.append((g.sum() - np.trace(g)) / (len(v) * (len(v) - 1)))
        return float(np.mean(vals)) if vals else float("nan")

    rng = np.random.default_rng(0)
    after = mean_cos(seq_ptr, seg["ds"], seg["doc"], rng)
    o_ptr, o_ds, o_doc = [0], [], []
    for c0, c1, p in window_chunks(a)[:: max(1, len(window_chunks(a)) // 4)]:
        (ds, doc, _, _), _ = chunk_pieces(p)
        arr, _ = load_stream(p)
        nseg = np.diff(np.asarray(arr["seg_ptr"]))
        lens = nseg[nseg > 1]
        o_ptr += list(o_ptr[-1] + np.cumsum(lens))
        o_ds.append(ds)
        o_doc.append(doc)
    before = mean_cos(np.asarray(o_ptr), np.concatenate(o_ds), np.concatenate(o_doc), rng)
    return {"cos_within_seq_before": before, "cos_within_seq_after": after}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["centroids", "assign", "order", "assemble"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--name", required=True, help="plan name (<out>/plan/<name>, work dir <out>/global/<name>)")
    ap.add_argument("--from-ckpt", dest="w0", type=int, required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--dim", type=int, default=None, help="Matryoshka truncation of the stored vectors")
    ap.add_argument("--coarse", type=int, default=32768, help="documents per coarse cluster")
    ap.add_argument("--buckets", type=int, default=1, help="order processes (centroids stage)")
    ap.add_argument("--sample", type=int, default=16_000_000, help="documents to train the centroids on")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch", type=int, default=1_048_576)
    ap.add_argument("--alpha0", type=float, default=0.3)
    ap.add_argument("--leaf-tokens", type=int, default=32768)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--dup-cos", type=float, default=0.97)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--task-rank", type=int, default=0)
    ap.add_argument("--task-world", type=int, default=1)
    a = ap.parse_args()
    a.w1 = a.w0 + a.steps
    dev = bp._device(a.device)
    if a.stage == "centroids":
        stage_centroids(a, dev)
    elif a.stage == "assign":
        stage_assign(a, dev, a.task_rank, a.task_world)
    elif a.stage == "order":
        z = np.load(os.path.join(gdir(a), "centroids.npz"))
        for b in range(a.task_rank, int(z["buckets"]), a.task_world):
            stage_order(a, dev, b)
    else:
        stage_assemble(a)


if __name__ == "__main__":
    main()
