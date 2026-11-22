#!/usr/bin/env python3
"""Step 4 of the cross-document packing pipeline (docs/cross_doc_seqs/README.md): a packing plan per
window of training steps, read by Megatron's --packing-plan-path (megatron/core/datasets/packing_plan.py).

    build_plan.py --out <pipeline dir> --emb <embedding name> --from-ckpt 49000 --steps 4000 --window 500

For every window of --window steps (steps taken from checkpoints [w0, w1)):

1. Take the document pieces of every slot of the window (step 1's stream). Slots that are one piece of
   one document ("solo": a long document filling the whole sequence) are left alone.
2. Units: the pieces of one occurrence of a document (pieces of the same document are merged in
   offset order -- a document cut at a sequence boundary comes back together). A document that the
   window reads twice (another epoch of a small dataset) is two units.
3. Order the units semantically (embedding of the document's start, step 2):
   coarse spherical k-means (~--coarse units per cluster) -> per coarse cluster a fine k-means into
   leaves of ~--leaf-tokens tokens -> leaves chained greedily by centroid similarity inside the
   coarse cluster, coarse clusters chained greedily (in-context pretraining's nearest-neighbour
   traversal, made hierarchical so it scales to tens of millions of documents).
   Exact repeats and near duplicates (cosine >= --dup-cos inside a coarse cluster) are taken out of the
   semantic order and spread over it so that no sequence holds two units of one conflict class
   (place_copies; a copy in context is a free lunch for the loss and teaches copying, not language).
   Optional metadata groups (--group-keys) are kept together inside a leaf.
4. Concatenate the units' pieces in that order and cut the stream into sequences of seq_length
   (+ 1 label token), exactly like Megatron's own packing -- so the window's tokens are conserved
   exactly (every token of the original slots appears once).
5. Shuffle the sequences over the window's (non-solo) slots: each global batch stays a random mix of
   the window, only the inside of a sequence is coherent.

Output <out>/plan/<emb>/ck<w0>_<w1>/: slot_bidx (sorted), slot_seq, seq_ptr, seg_{ds,doc,off,len,group}
+ meta.json + stats.json. Megatron serves a plan sequence in place of the blended sample slot_bidx;
every other index (and every step outside the plans) is served as before.
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
from cds_data import load_stream  # noqa: E402

KEY_SHIFT = 40  # unit key = dataset << 40 | document


def doc_key(ds: np.ndarray, doc: np.ndarray) -> np.ndarray:
    return (ds.astype(np.int64) << KEY_SHIFT) | doc.astype(np.int64)


def ranges(starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """concatenate(arange(s, s + l) for s, l in zip(starts, lengths)), vectorized."""
    lengths = np.asarray(lengths, np.int64)
    if len(lengths) == 0:
        return np.zeros(0, np.int64)
    csum = np.cumsum(lengths)
    return np.repeat(np.asarray(starts, np.int64) - (csum - lengths), lengths) + np.arange(csum[-1])


# ------------------------------------------------------------------------------------- similarity
def _torch():
    import torch

    return torch


def _device(prefer: str = "auto"):
    torch = _torch()
    if prefer == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda" if prefer == "auto" else prefer)


def spherical_kmeans(x, k: int, iters: int, seed: int, train_max: int = 1_000_000, block: int = 65536):
    """(labels [n] long, centroids [k, d]) of unit-norm rows ``x`` (torch, any device)."""
    torch = _torch()
    n = x.shape[0]
    k = max(1, min(k, n))
    g = torch.Generator(device="cpu").manual_seed(seed)
    if k == 1:
        c = torch.nn.functional.normalize(x.float().mean(0, keepdim=True), dim=1)
        return torch.zeros(n, dtype=torch.long, device=x.device), c
    tr = x if n <= train_max else x[torch.randperm(n, generator=g)[:train_max].to(x.device)]
    c = tr[torch.randperm(tr.shape[0], generator=g)[:k].to(x.device)].float().clone()

    def assign(rows):
        out = torch.empty(rows.shape[0], dtype=torch.long, device=x.device)
        for i in range(0, rows.shape[0], block):
            out[i:i + block] = (rows[i:i + block].float() @ c.T).argmax(1)
        return out

    for _ in range(iters):
        lab = assign(tr)
        s = torch.zeros_like(c).index_add_(0, lab, tr.float())
        cnt = torch.bincount(lab, minlength=k)
        empty = cnt == 0
        if empty.any():  # re-seed empty clusters with random points
            ne = int(empty.sum())
            s[empty] = tr[torch.randint(0, tr.shape[0], (ne,), generator=g).to(x.device)].float()
        c = torch.nn.functional.normalize(s, dim=1)
    return assign(x), c


def greedy_chain(sim: np.ndarray, start: int = 0) -> np.ndarray:
    """Nearest-neighbour traversal of a similarity matrix: from ``start`` always to the most similar
    unvisited node (in-context pretraining, Shi et al. 2023, on centroids)."""
    m = sim.shape[0]
    order = np.empty(m, dtype=np.int64)
    seen = np.zeros(m, dtype=bool)
    cur = start
    for i in range(m):
        order[i] = cur
        seen[cur] = True
        if i + 1 < m:
            row = np.where(seen, -np.inf, sim[cur])
            cur = int(row.argmax())
    return order


def near_duplicates(x, thr: float, block: int = 8192) -> np.ndarray:
    """For every row of ``x`` (unit norm, torch): the first EARLIER row at cosine >= thr, else -1
    (rows with a partner are deferred)."""
    torch = _torch()
    n = x.shape[0]
    partner = torch.full((n,), -1, dtype=torch.long, device=x.device)
    xf = x.float()
    for i in range(0, n, block):
        s = xf[i:i + block] @ xf[: i + block].T  # rows i.., columns 0..i+block
        r = torch.arange(s.shape[0], device=x.device)
        s[r[:, None] + i <= torch.arange(s.shape[1], device=x.device)[None, :]] = -2  # only earlier rows
        hit = s >= thr
        first = hit.to(torch.int8).argmax(1)
        partner[i:i + block] = torch.where(hit.any(1), first, torch.full_like(first, -1))
    return partner.cpu().numpy()


def conflict_classes(ukey: np.ndarray, pairs: list) -> np.ndarray:
    """Class id per unit: units of the same document, or linked by a near-duplicate pair, share a class
    (union-find over the few units that are not alone)."""
    cls = np.arange(len(ukey), dtype=np.int64)
    o = np.argsort(ukey, kind="stable")
    same = np.nonzero(ukey[o][1:] == ukey[o][:-1])[0]
    edges = [(int(o[i]), int(o[i + 1])) for i in same] + [(int(a), int(b)) for a, b in pairs]
    parent: dict = {}

    def find(v):
        while parent.get(v, v) != v:
            parent[v] = parent.get(parent[v], parent[v])
            v = parent[v]
        return v

    for a_, b_ in edges:
        ra, rb = find(a_), find(b_)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    for v in parent:
        cls[v] = find(v)
    return cls


def count_conflicts(uorder: np.ndarray, utok: np.ndarray, cls: np.ndarray, S: int) -> int:
    """Units that share a sequence with another unit of their conflict class (0 = every copy separated)."""
    multi = np.bincount(cls, minlength=len(cls))[cls] > 1
    if not multi.any():
        return 0
    start = np.r_[0, np.cumsum(utok[uorder])[:-1]]
    pos = np.empty(len(uorder), np.int64)
    pos[uorder] = np.arange(len(uorder))
    cand = np.nonzero(multi)[0]
    s0 = start[pos[cand]]
    c0, c1 = s0 // S, (s0 + utok[cand] - 1) // S
    n = c1 - c0 + 1
    unit, chunk = np.repeat(cand, n), ranges(c0, n)
    o = np.lexsort((pos[unit], chunk, cls[unit]))
    unit, chunk = unit[o], chunk[o]
    clash = np.r_[False, (cls[unit][1:] == cls[unit][:-1]) & (chunk[1:] == chunk[:-1])]
    return int(len(np.unique(unit[clash])))


def place_copies(base: np.ndarray, copies: np.ndarray, cls: np.ndarray, utok: np.ndarray, S: int, rng):
    """Insert the copies (repeated documents, near duplicates) into the semantic base order so that no
    sequence of S tokens holds two units of one conflict class (a class has at most one unit in the base).

    Every class is spread evenly over the whole base (copy r of m aims at (r + offset) / m of it); the
    copies, taken in the order of their aims, go in at the next insertion point where the sequence holds no
    other unit of their class. Insertion points only move forward, so positions already decided never shift
    relative to each other: one pass, no repair rounds. Returns (order, copies placed without separation)."""
    from collections import deque
    import bisect

    nB, M = len(base), len(copies)
    if M == 0:
        return base, 0
    base_cum = np.r_[0, np.cumsum(utok[base])]
    in_base = np.full(len(cls), -1, np.int64)
    in_base[base] = np.arange(nB)
    copy_cls = set(cls[copies].tolist())
    kept = {int(cls[u]): int(in_base[u]) for u in base[np.isin(cls[base], list(copy_cls))]}
    # every class spread evenly over the whole base: copy r of a class with m copies aims at base position
    # (r + offset) / m (a random offset per class), all copies taken in the order of their aims
    perm = rng.permutation(M)
    c = copies[perm]
    o = np.argsort(cls[c], kind="stable")
    c = c[o]
    first = np.r_[True, cls[c][1:] != cls[c][:-1]]
    grp = np.cumsum(first) - 1
    grp_start = np.nonzero(first)[0]
    size = np.diff(np.r_[grp_start, M])
    rank = np.arange(M) - grp_start[grp]
    aim = (rank + rng.random(len(size))[grp]) / size[grp]
    o = np.argsort(aim, kind="stable")
    c = c[o]
    targets = np.minimum((aim[o] * nB).astype(np.int64), nB)
    ins_idx, ins_cum, out_j, out_u = [], [], [], []
    last_end: dict = {}
    off = 0

    def try_place(u: int, j: int, force: bool = False) -> bool:
        nonlocal off
        k = int(cls[u])
        L = int(utok[u])
        P = int(base_cum[j]) + off
        s0, s1 = P // S, (P + L - 1) // S
        ok = s0 > last_end.get(k, -1)
        q = kept.get(k, -1)
        if ok and 0 <= q < j:  # the class's base unit lies before: its (final) end sequence
            n_before = bisect.bisect_right(ins_idx, q)
            offq = ins_cum[n_before - 1] if n_before else 0
            ok = s0 > (int(base_cum[q + 1]) + offq - 1) // S
        if ok and q >= j:  # it lies after: it starts at least here (later insertions only push it on)
            ok = s1 < (int(base_cum[q]) + off + L) // S
        if not ok and not force:
            return False
        ins_idx.append(j)
        off += L
        ins_cum.append(off)
        last_end[k] = max(last_end.get(k, -1), s1)
        out_j.append(j)
        out_u.append(u)
        return True

    pending: deque = deque()
    for idx in range(M):
        j = int(targets[idx])
        for _ in range(min(len(pending), 64)):  # earlier copies that did not fit yet
            u = pending.popleft()
            if not try_place(u, j):
                pending.append(u)
        u = int(c[idx])
        if not try_place(u, j):
            pending.append(u)
    forced = 0
    while pending:  # the rest goes after the base, separated where the tail allows
        progress = False
        for _ in range(len(pending)):
            u = pending.popleft()
            if try_place(u, nB):
                progress = True
            else:
                pending.append(u)
        if not progress:
            u = pending.popleft()
            try_place(u, nB, force=True)
            forced += 1
    order = np.insert(base, np.asarray(out_j, np.int64), np.asarray(out_u, np.int64))
    return order, forced


# ------------------------------------------------------------------------------------------ order
def order_clusters(x, tokens: np.ndarray, groups: np.ndarray | None, clusters, a, seed: int, dev,
                   leaf_base: int = 0):
    """Semantic order inside given coarse clusters (lists of unit indices, in chain order): per cluster,
    near duplicates leave (deferred, with their earlier partner), a fine k-means into leaves of
    ~--leaf-tokens tokens, leaves chained greedily from next to the previous cluster's last leaf.
    Returns (order, leaf id per unit, deferred mask, pairs, next leaf id)."""
    torch = _torch()
    n = x.shape[0]
    leaf = np.zeros(n, dtype=np.int64)
    deferred = np.zeros(n, dtype=bool)
    pairs = []  # (unit, earlier near duplicate)
    out, nleaf, prev_c = [], leaf_base, None
    for ci, idx in enumerate(clusters):
        if len(idx) == 0:
            continue
        xc = x[torch.as_tensor(idx, device=dev)]
        if a.dup_cos < 1.0:
            partner = near_duplicates(xc, a.dup_cos)
            d = partner >= 0
            pairs.extend(zip(idx[d].tolist(), idx[partner[d]].tolist()))
            deferred[idx[d]] = True
            idx, xc = idx[~d], xc[torch.as_tensor(~d, device=dev)]
            if len(idx) == 0:
                continue
        k2 = max(1, int(np.ceil(tokens[idx].sum() / a.leaf_tokens)))
        lab2, c2 = spherical_kmeans(xc, k2, a.iters, seed + 1 + ci)
        lab2 = lab2.cpu().numpy()
        c2n = c2.cpu().numpy()
        # chain the leaves, starting next to the previous coarse cluster's last leaf
        start = int((c2n @ prev_c).argmax()) if prev_c is not None else 0
        lorder = greedy_chain(c2n @ c2n.T, start) if k2 > 1 else np.zeros(1, np.int64)
        for li in lorder:
            sel = idx[lab2 == li]
            if len(sel) == 0:
                continue
            if groups is not None:  # metadata groups (same repo / site) stay contiguous in a leaf
                sel = sel[np.argsort(groups[sel], kind="stable")]
            out.append(sel)
            leaf[sel] = nleaf
            nleaf += 1
        prev_c = c2n[lorder[-1]]
    order = np.concatenate(out) if out else np.zeros(0, np.int64)
    return order, leaf, deferred, pairs, nleaf


def order_pool(emb, tokens: np.ndarray, groups: np.ndarray | None, a, seed: int, dev):
    """Semantic order of one pool of units (coarse spherical k-means, clusters chained greedily, then
    order_clusters). Returns (order, leaf id per unit, deferred mask, pairs)."""
    n = emb.shape[0]
    if n == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, bool), []
    x = emb.to(dev)
    k1 = max(1, int(round(n / a.coarse)))
    lab1, c1 = spherical_kmeans(x, k1, a.iters, seed)
    lab1 = lab1.cpu().numpy()
    members = np.argsort(lab1, kind="stable")
    bounds = np.searchsorted(lab1[members], np.arange(k1 + 1))
    c1n = c1.cpu().numpy()
    corder = greedy_chain(c1n @ c1n.T) if k1 > 1 else np.zeros(1, np.int64)
    clusters = [members[bounds[ci]:bounds[ci + 1]] for ci in corder]
    order, leaf, deferred, pairs, _ = order_clusters(x, tokens, groups, clusters, a, seed, dev)
    return order, leaf, deferred, pairs


# ---------------------------------------------------------------------------------------- window
def load_window(out: str, w0: int, w1: int) -> dict:
    """Slots with steps from checkpoints [w0, w1) (logged iterations w0+1 .. w1) and their pieces."""
    parts = []
    for p in sorted(glob.glob(os.path.join(out, "stream", "ck*_*"))):
        if ".tmp" in os.path.basename(p):
            continue
        arr, meta = load_stream(p)
        if meta["to_ckpt"] <= w0 or meta["from_ckpt"] >= w1:
            continue
        sel = np.nonzero((arr["slot_iter"] > w0) & (arr["slot_iter"] <= w1))[0]
        ptr = np.asarray(arr["seg_ptr"])
        lo, hi = ptr[sel], ptr[sel + 1]
        segsel = ranges(lo, hi - lo)
        parts.append({
            "slot_iter": np.asarray(arr["slot_iter"])[sel], "slot_pos": np.asarray(arr["slot_pos"])[sel],
            "slot_bidx": np.asarray(arr["slot_bidx"])[sel], "nseg": hi - lo,
            "seg_ds": np.asarray(arr["seg_ds"])[segsel], "seg_doc": np.asarray(arr["seg_doc"])[segsel],
            "seg_off": np.asarray(arr["seg_off"])[segsel], "seg_len": np.asarray(arr["seg_len"])[segsel],
            "meta": meta,
        })
    assert parts, f"no extracted stream covers checkpoints [{w0}, {w1})"
    win = {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if k != "meta"}
    have = np.unique(win["slot_iter"])
    assert len(have) == w1 - w0 and have[0] == w0 + 1, f"stream covers {len(have)} of {w1 - w0} steps of the window"
    win["seq_length"] = parts[0]["meta"]["seq_length"]
    win["blend_md5"] = parts[0]["meta"]["blend_md5"]
    return win


def load_embeddings(emb_root: str, keys_needed: np.ndarray, w0: int | None = None, w1: int | None = None):
    """(sorted keys, float16 vectors) for the needed keys from the part files below emb_root (only the
    stream chunks overlapping checkpoints [w0, w1) when given, plus any non-chunk directory)."""
    ks, vs = [], []
    files = []
    for d in sorted(glob.glob(os.path.join(emb_root, "*"))):
        name = os.path.basename(d)
        if w0 is not None and name.startswith("ck") and "_" in name:
            a_, b_ = (int(x) for x in name[2:].split("_"))
            if b_ <= w0 or a_ >= w1:
                continue
        files += sorted(f for f in glob.glob(os.path.join(d, "part_*.npz")) if not f.endswith(".tmp.npz"))
    for f in files:  # every part of the window's chunks: concatenate, filter once (np.isin per file
        z = np.load(f)  # against tens of millions of needed keys re-sorted them for every file)
        ks.append(z["keys"])
        vs.append(z["vec"])
    if not ks:
        return np.zeros(0, np.int64), None
    k = np.concatenate(ks)
    v = np.concatenate(vs)
    k, first = np.unique(k, return_index=True)
    need = np.unique(keys_needed)
    r = np.searchsorted(need, k)
    keep = (r < len(need)) & (need[np.minimum(r, max(len(need) - 1, 0))] == k)
    return k[keep], v[first[keep]]


def units_from_pieces(ds, doc, off, ln) -> dict:
    """Document pieces -> units (one occurrence of a document, its pieces in offset order): a document read
    twice (another epoch) is two units (occurrence layer = number of earlier overlapping pieces).
    Returns piece arrays sorted by (key, layer, offset), unit start pointers and the layers."""
    off, ln = np.asarray(off).astype(np.int64), np.asarray(ln).astype(np.int64)
    key = doc_key(np.asarray(ds), np.asarray(doc))
    o = np.lexsort((off, key))
    ds, doc, off, ln, key = np.asarray(ds)[o], np.asarray(doc)[o], off[o], ln[o], key[o]
    layer = np.zeros(len(key), dtype=np.int64)
    newkey = np.r_[True, key[1:] != key[:-1]]
    multi = np.nonzero(~newkey)[0]
    if len(multi):
        starts = np.nonzero(newkey)[0]
        grp = np.searchsorted(starts, np.arange(len(key)), side="right") - 1
        for g in np.unique(grp[multi]):
            a_, b_ = starts[g], (starts[g + 1] if g + 1 < len(starts) else len(key))
            ends = []  # end offset per layer
            for i in range(a_, b_):
                for L, e in enumerate(ends):
                    if off[i] >= e:
                        ends[L] = off[i] + ln[i]
                        layer[i] = L
                        break
                else:
                    layer[i] = len(ends)
                    ends.append(off[i] + ln[i])
    o = np.lexsort((off, layer, key))
    ds, doc, off, ln, key, layer = ds[o], doc[o], off[o], ln[o], key[o], layer[o]
    unit_start = np.nonzero(np.r_[True, (key[1:] != key[:-1]) | (layer[1:] != layer[:-1])])[0]
    return dict(ds=ds, doc=doc, off=off, len=ln, key=key, layer=layer, unit_start=unit_start)


def build_units(win: dict):
    """Pieces of the window's non-solo slots -> units (units_from_pieces)."""
    S = win["seq_length"]
    solo_slot = (win["nseg"] == 1)
    seg_slot = np.repeat(np.arange(len(win["nseg"])), win["nseg"])
    keep = ~solo_slot[seg_slot]
    u = units_from_pieces(win["seg_ds"][keep], win["seg_doc"][keep], win["seg_off"][keep], win["seg_len"][keep])
    assert u["len"].sum() == (~solo_slot).sum() * S
    u["n_slots"] = int((~solo_slot).sum())
    return u


def write_plan(dst: str, slot_bidx, slot_seq, seq_ptr, seg: dict, meta: dict, stats: dict) -> None:
    """One plan directory (the format Megatron's packing_plan.py reads), written atomically."""
    tmp = dst + ".tmp"
    os.makedirs(tmp, exist_ok=True)
    np.save(os.path.join(tmp, "slot_bidx.npy"), np.asarray(slot_bidx, np.int64))
    np.save(os.path.join(tmp, "slot_seq.npy"), np.asarray(slot_seq, np.int64))
    np.save(os.path.join(tmp, "seq_ptr.npy"), np.asarray(seq_ptr, np.int64))
    np.save(os.path.join(tmp, "seg_ds.npy"), seg["ds"].astype(np.int16))
    np.save(os.path.join(tmp, "seg_doc.npy"), seg["doc"].astype(np.int64))
    np.save(os.path.join(tmp, "seg_off.npy"), seg["off"].astype(np.int64))
    np.save(os.path.join(tmp, "seg_len.npy"), seg["len"].astype(np.int32))
    np.save(os.path.join(tmp, "seg_group.npy"), seg["grp"].astype(np.int64))
    json.dump(meta, open(os.path.join(tmp, "meta.json"), "w"), indent=1)
    json.dump(stats, open(os.path.join(tmp, "stats.json"), "w"), indent=1)
    os.replace(tmp, dst)


def plan_meta(out: str, blend_md5: str, S: int, w0: int, w1: int, emb: str, params: dict) -> dict:
    """meta.json of a plan: the blend, the window and the sampler geometry the replaced indices belong to
    (Megatron refuses a plan for another geometry)."""
    geometry = json.load(open(os.path.join(out, "blend.json")))["geometry"]
    return {"format": 1, "blend_md5": blend_md5, "seq_length": S, "from_ckpt": w0, "to_ckpt": w1,
            "geometry": {k: geometry.get(k) for k in ("global_batch_size", "ref_dp", "ref_mbs", "ref_gbs",
                                                      "phase_start", "train_iters", "total_samples")},
            "embedding": emb, "params": params}


def cut_sequences(p_ds, p_doc, p_off, p_len, p_grp, S: int):
    """Concatenate pieces and cut into sequences of S owned tokens + 1 label token (the next token of
    the stream; the last sequence takes the stream's first token). Returns seq_ptr + segment arrays."""
    total = int(p_len.sum())
    assert total % S == 0
    n = total // S
    start = np.r_[0, np.cumsum(p_len)[:-1]]
    cuts = np.arange(1, n) * S
    bounds = np.union1d(np.r_[start, total], np.r_[0, cuts, total])
    pid = np.searchsorted(start, bounds[:-1], side="right") - 1  # piece of each new segment
    s_off = p_off[pid] + (bounds[:-1] - start[pid])
    s_len = np.diff(bounds)
    s_ds, s_doc, s_grp = p_ds[pid], p_doc[pid], p_grp[pid]
    s_seq = bounds[:-1] // S
    ptr = np.searchsorted(s_seq, np.arange(n + 1))
    # the label token of sequence k = the first token of sequence k + 1 (k = n-1: of sequence 0)
    first = ptr[:-1]
    nxt = np.r_[first[1:], first[0]]
    last = ptr[1:] - 1
    contig = (s_ds[nxt] == s_ds[last]) & (s_doc[nxt] == s_doc[last]) & (s_off[nxt] == s_off[last] + s_len[last])
    out = {k: [] for k in ("ds", "doc", "off", "len", "grp")}
    seq_ptr = [0]
    for k in range(n):
        a_, b_ = ptr[k], ptr[k + 1]
        ds_, doc_, off_, len_, grp_ = (s_ds[a_:b_], s_doc[a_:b_], s_off[a_:b_], s_len[a_:b_].copy(), s_grp[a_:b_])
        if contig[k]:
            len_[-1] += 1
            parts = (ds_, doc_, off_, len_, grp_)
        else:
            j = nxt[k]
            parts = (np.r_[ds_, s_ds[j]], np.r_[doc_, s_doc[j]], np.r_[off_, s_off[j]], np.r_[len_, 1],
                     np.r_[grp_, s_grp[j]])
        for name, v in zip(("ds", "doc", "off", "len", "grp"), parts):
            out[name].append(v)
        seq_ptr.append(seq_ptr[-1] + len(parts[0]))
    seg = {k: np.concatenate(v) for k, v in out.items()}
    return np.asarray(seq_ptr, np.int64), seg


def seq_similarity(ptr, ds, doc, key_index, vecs, n_sample: int, rng) -> float:
    """Mean cosine between the distinct documents of a sequence (pairs inside a sequence, averaged over
    sampled sequences)."""
    n = len(ptr) - 1
    vals = []
    for s in rng.choice(n, size=min(n_sample, n), replace=False):
        k = np.unique(doc_key(ds[ptr[s]:ptr[s + 1]], doc[ptr[s]:ptr[s + 1]]))
        r = np.searchsorted(key_index, k)
        ok = (r < len(key_index)) & (key_index[np.minimum(r, len(key_index) - 1)] == k)
        v = vecs[r[ok]].astype(np.float32)
        if len(v) < 2:
            continue
        g = v @ v.T
        m = len(v)
        vals.append((g.sum() - np.trace(g)) / (m * (m - 1)))
    return float(np.mean(vals)) if vals else float("nan")


def build_window(a, w0: int, w1: int, dev) -> None:
    torch = _torch()
    t0 = time.time()
    dst = os.path.join(a.out, "plan", a.name, f"ck{w0:07d}_{w1:07d}")
    if os.path.isdir(dst):
        print(f"{dst}: exists, skipped", flush=True)
        return
    win = load_window(a.out, w0, w1)
    S = win["seq_length"]
    u = build_units(win)
    n_units = len(u["unit_start"])
    ukey = u["key"][u["unit_start"]]
    ulayer = u["layer"][u["unit_start"]]
    utok = np.add.reduceat(u["len"], u["unit_start"])
    keys, vecs = load_embeddings(os.path.join(a.out, "emb", a.emb), np.unique(ukey), w0, w1)
    r = np.searchsorted(keys, ukey) if len(keys) else np.zeros(n_units, np.int64)
    have = (r < len(keys)) & (keys[np.minimum(r, max(len(keys) - 1, 0))] == ukey) if len(keys) else np.zeros(n_units, bool)
    miss = int((~have).sum())
    if miss / max(n_units, 1) > a.max_missing:
        raise SystemExit(f"window {w0}-{w1}: {miss} of {n_units} units have no embedding (--max-missing {a.max_missing})")
    dim = vecs.shape[1] if vecs is not None else 8
    g = np.random.default_rng(a.seed + w0)
    E = np.empty((n_units, dim), np.float32)
    if have.any():
        E[have] = vecs[r[have]].astype(np.float32)
    if miss:  # no embedding: a random direction (placed like an unrelated document)
        E[~have] = g.standard_normal((miss, dim)).astype(np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
    emb = torch.from_numpy(E).to(dev, torch.float16 if dev.type == "cuda" else torch.float32)
    groups = None
    if a.group_keys:
        gk = np.load(a.group_keys)  # {"keys": unit doc keys (sorted), "group": int64 group id}
        gi = np.searchsorted(gk["keys"], ukey)
        okg = (gi < len(gk["keys"])) & (gk["keys"][np.minimum(gi, len(gk["keys"]) - 1)] == ukey)
        groups = np.where(okg, gk["group"][np.minimum(gi, len(gk["keys"]) - 1)], -1 - np.arange(n_units))

    # semantic order of the first occurrences; near duplicates found inside a coarse cluster leave it
    l0 = np.nonzero(ulayer == 0)[0]
    order, leaf, deferred, pairs = order_pool(emb[torch.as_tensor(l0, device=dev)], utok[l0],
                                              None if groups is None else groups[l0], a, a.seed, dev)
    base = l0[order]
    leaf_all = np.zeros(n_units, np.int64)
    leaf_all[base] = leaf[order]
    dup_pairs = [(int(l0[x]), int(l0[y])) for x, y in pairs]
    # copies: near duplicates and repeated occurrences; spread so that no sequence holds two of a class
    copies = np.r_[l0[deferred], np.nonzero(ulayer > 0)[0]].astype(np.int64)
    cls = conflict_classes(ukey, dup_pairs)
    uorder, forced = place_copies(base, copies, cls, utok, S, g)
    leaf_all[copies] = int(leaf_all.max()) + 1 + np.arange(len(copies))  # each copy its own group
    assert len(uorder) == n_units and len(np.unique(uorder)) == n_units
    unseparated = count_conflicts(uorder, utok, cls, S)
    pool_stats = [{"base": int(len(base)), "near_duplicates": int(deferred.sum()),
                   "repeats": int((ulayer > 0).sum()), "largest_class": int(np.bincount(cls).max())}]

    # pieces in unit order
    ustart = u["unit_start"]
    ulen = np.diff(np.r_[ustart, len(u["key"])])
    pidx = ranges(ustart[uorder], ulen[uorder])
    p_grp = np.repeat(leaf_all[uorder], ulen[uorder])
    seq_ptr, seg = cut_sequences(u["ds"][pidx], u["doc"][pidx], u["off"][pidx], u["len"][pidx], p_grp, S)
    n_seq = len(seq_ptr) - 1
    assert n_seq == u["n_slots"]

    # shuffle the sequences over the window's non-solo slots
    solo = win["nseg"] == 1
    slots = np.nonzero(~solo)[0]
    perm = g.permutation(n_seq)
    bidx = win["slot_bidx"][slots]
    o = np.argsort(bidx)
    slot_bidx, slot_seq = bidx[o], perm[o]

    # stats: coherence of sequences before (the original slots) and after
    seg_slot = np.repeat(np.arange(len(win["nseg"])), win["nseg"])
    optr = np.r_[0, np.cumsum(win["nseg"][slots])]
    okeep = ~solo[seg_slot]
    o_ds, o_doc = win["seg_ds"][okeep], win["seg_doc"][okeep]
    stats = {
        "window": [w0, w1], "slots": int(len(solo)), "solo_slots": int(solo.sum()), "units": n_units,
        "units_missing_embedding": miss, "copies": pool_stats, "near_duplicate_pairs": len(dup_pairs),
        "copies_forced": int(forced), "units_sharing_a_sequence_with_a_copy": int(unseparated),
        "pieces_per_seq_before": float(okeep.sum() / max(len(slots), 1)),
        "segments_per_seq_after": float(len(seg["ds"]) / max(n_seq, 1)),
        "datasets_per_seq_before": float(np.mean([len(np.unique(o_ds[optr[i]:optr[i + 1]]))
                                                  for i in range(0, len(slots), max(1, len(slots) // 2000))])),
        "datasets_per_seq_after": float(np.mean([len(np.unique(seg["ds"][seq_ptr[i]:seq_ptr[i + 1]]))
                                                 for i in range(0, n_seq, max(1, n_seq // 2000))])),
    }
    if have.any():
        rng = np.random.default_rng(0)
        stats["cos_within_seq_before"] = seq_similarity(optr, o_ds, o_doc, keys, vecs, a.stat_seqs, rng)
        stats["cos_within_seq_after"] = seq_similarity(seq_ptr, seg["ds"], seg["doc"], keys, vecs, a.stat_seqs, rng)
    stats["seconds"] = round(time.time() - t0, 1)

    meta = plan_meta(a.out, win["blend_md5"], S, w0, w1, a.emb,
                     {k: getattr(a, k) for k in ("coarse", "leaf_tokens", "iters", "dup_cos", "seed", "group_keys")})
    write_plan(dst, slot_bidx, slot_seq, seq_ptr, seg, meta, stats)
    print(f"{dst}: {json.dumps(stats)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="pipeline directory (stream/, emb/, plan/)")
    ap.add_argument("--emb", required=True, help="embedding run name under <out>/emb/")
    ap.add_argument("--name", default=None, help="plan name under <out>/plan/ (default: --emb)")
    ap.add_argument("--from-ckpt", type=int, required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--window", type=int, default=500, help="steps per plan window (documents move only inside it)")
    ap.add_argument("--coarse", type=int, default=4096, help="units per coarse cluster")
    ap.add_argument("--leaf-tokens", type=int, default=32768, help="tokens per leaf (fine cluster)")
    ap.add_argument("--iters", type=int, default=10, help="k-means iterations")
    ap.add_argument("--dup-cos", type=float, default=0.97, help="near-duplicate cosine (>= : deferred)")
    ap.add_argument("--max-missing", type=float, default=0.001, help="tolerated fraction of units without embedding")
    ap.add_argument("--group-keys", default=None, help="npz {keys, group}: metadata groups kept together")
    ap.add_argument("--stat-seqs", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="auto", help="auto / cpu / cuda / cuda:N")
    ap.add_argument("--task-rank", type=int, default=0, help="this task builds windows rank, rank + world, ...")
    ap.add_argument("--task-world", type=int, default=1)
    a = ap.parse_args()
    a.name = a.name or a.emb
    dev = _device(a.device)
    assert a.steps % a.window == 0 or a.steps < a.window, "--steps should be a multiple of --window"
    windows = list(range(a.from_ckpt, a.from_ckpt + a.steps, a.window))
    for w0 in windows[a.task_rank::a.task_world]:
        build_window(a, w0, min(w0 + a.window, a.from_ckpt + a.steps), dev)


if __name__ == "__main__":
    main()
