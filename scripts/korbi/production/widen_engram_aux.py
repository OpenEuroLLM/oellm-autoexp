#!/usr/bin/env python3
"""Add auxiliary Engram lookups (Megatron engram_aux_*) to an fsdp_dtensor checkpoint, function-preserving.

The auxiliary lookups (user 2026-11-06): a second set of n-gram lookups over another token map -- the
accent-preserving compression (NFKC + lowercase, no accent stripping: engram_token_map.py --keep-accents),
because stripping accents merges distinct letters (Finnish a / ä, Czech c / č, Cyrillic и / й, ...).
Megatron appends their table rows after the main ones and their rows to the memory vector after the main
ones; the main lookups do not change. Per Engram layer this tool

  table              [R_main, d] -> [R_main + R_aux, d]: the old chunks stay where they are (the merged
                     .metadata points to them); the new rows N(0, init_std) in one new chunk
  table optimizer    (optimizer.1.param_state.<i>:engram_table, exp_avg / exp_avg_sq): zero rows appended
  key_proj, value_proj [h, C] -> [h, C + L_aux * d]: zero columns -- the layer computes exactly what it
                     did (tests/unit_tests/ssm/test_engram.py TestEngramAux); the new columns get
                     gradients from the random new rows. Muown classical states: g unchanged (row norms),
                     m_u zero columns, and the spectral Muon scale grows sqrt(max(h, C + ...)) /
                     sqrt(max(h, C)): v_norm x that ratio, m_u / it (the rows' rotation rate stays)

Run single-threaded on a login node for small tables (no Megatron needed; CUDA_VISIBLE_DEVICES= keeps torch off the GPUs),
else in a job (the production tables add ~160 GB):

    python widen_engram_aux.py --src <ckpt>/iter_N --dst <new>/iter_N --aux-orders 2,3 --aux-heads 8 \\
        --aux-table-size 811682 --main-table-size 1623365 --main-orders 2,3,4 --main-heads 8
"""

from __future__ import annotations

import argparse
import copy
import io
import math
import os
import pickle
import re
import sys
import time
from types import SimpleNamespace

import torch

OPT0 = "optimizer.0.state.module.module.module.decoder.layers.{l}.{n}.{s}"
MUOWN = ("g", "m_u", "m_g", "v_g", "v_norm")


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    f = 3
    while f * f <= n:
        if n % f == 0:
            return False
        f += 2
    return True


def engram_table_sizes(cfg) -> list:
    """megatron.core.ssm.engram.engram_table_sizes without importing Megatron (its import, with
    Transformer Engine, hangs for minutes on a busy GPFS login node); the selftest checks they agree."""
    sizes, candidate = [], cfg.engram_table_size
    for _ in range(len(cfg.engram_ngram_orders) * cfg.engram_num_heads):
        while not _is_prime(candidate):
            candidate += 1
        sizes.append(candidate)
        candidate += 1
    aux = len(cfg.engram_aux_ngram_orders) * (cfg.engram_aux_num_heads or cfg.engram_num_heads)
    candidate, used = cfg.engram_aux_table_size or cfg.engram_table_size, set(sizes)
    for _ in range(aux):
        while not _is_prime(candidate) or candidate in used:
            candidate += 1
        sizes.append(candidate)
        used.add(candidate)
        candidate += 1
    return sizes


def read_bytes_item(src, md, fqn):
    info = [v for k, v in md.storage_data.items() if k.fqn == fqn]
    assert len(info) == 1, (fqn, len(info))
    with open(os.path.join(src, info[0].relative_path), "rb") as f:
        f.seek(info[0].offset)
        raw = f.read(info[0].length)
    return torch.load(io.BytesIO(raw), weights_only=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--main-orders", default="2,3,4")
    ap.add_argument("--main-heads", type=int, default=8)
    ap.add_argument("--main-table-size", type=int, required=True)
    ap.add_argument("--aux-orders", default="2,3")
    ap.add_argument("--aux-heads", type=int, default=8)
    ap.add_argument("--aux-table-size", type=int, required=True)
    ap.add_argument("--init-std", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=4321)
    a = ap.parse_args()

    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
    from torch.distributed.checkpoint.metadata import ChunkStorageMetadata, MetadataIndex

    t0 = time.time()
    log = lambda *x: print(f"[{time.time() - t0:7.1f}s]", *x, flush=True)  # noqa: E731
    cfg = SimpleNamespace(
        engram_ngram_orders=[int(x) for x in a.main_orders.split(",")], engram_num_heads=a.main_heads,
        engram_table_size=a.main_table_size, engram_aux_ngram_orders=[int(x) for x in a.aux_orders.split(",")],
        engram_aux_num_heads=a.aux_heads, engram_aux_table_size=a.aux_table_size,
    )
    sizes = engram_table_sizes(cfg)
    n_main = len(cfg.engram_ngram_orders) * cfg.engram_num_heads
    r_main, r_aux = sum(sizes[:n_main]), sum(sizes[n_main:])
    l_aux = len(sizes) - n_main

    reader = FileSystemReader(a.src)
    md = reader.read_metadata()
    smd = md.state_dict_metadata
    tables = sorted(int(m.group(1)) for k in smd for m in [re.fullmatch(r"model\.module\.decoder\.layers\.(\d+)\.table", k)] if m)
    opt_tables = sorted(k for k in smd if re.fullmatch(r"optimizer\.1\.param_state\.\d+:engram_table\.exp_avg(_sq)?", k))
    log(f"Engram layers {tables}; table rows main {r_main:,} (+ aux {r_aux:,} in {l_aux} lookups); {len(opt_tables)} table moments")
    for l in tables:
        got = smd[f"model.module.decoder.layers.{l}.table"].size[0]
        assert got == r_main, f"layer {l}: checkpoint table has {got} rows, the main config gives {r_main}"
    for k in opt_tables:
        assert smd[k].size[0] == r_main, (k, smd[k].size)

    # small tensors read whole: key / value projections + their Muown states
    proj_keys = {}
    for l in tables:
        for n in ("key_proj.weight", "value_proj.weight"):
            proj_keys[(l, n, "w")] = f"model.module.decoder.layers.{l}.{n}"
            for s in MUOWN:
                proj_keys[(l, n, s)] = OPT0.format(l=l, n=n, s=s)
    want = {k: torch.empty(smd[k].size, dtype=smd[k].properties.dtype) for k in proj_keys.values()}
    dcp.load(want, storage_reader=reader, no_dist=True)
    log("read the key / value projections")

    gen = torch.Generator().manual_seed(a.seed)
    full, rows = {}, {}  # full: rewritten whole; rows: the appended row chunk of a table-shaped key
    for l in tables:
        rows[f"model.module.decoder.layers.{l}.table"] = torch.randn(r_aux, smd[f"model.module.decoder.layers.{l}.table"].size[1], generator=gen) * a.init_std
        for n in ("key_proj.weight", "value_proj.weight"):
            w = want[proj_keys[(l, n, "w")]]
            h, c = w.shape
            add = l_aux * smd[f"model.module.decoder.layers.{l}.table"].size[1]
            ratio = math.sqrt(max(h, c + add)) / math.sqrt(max(h, c))
            full[proj_keys[(l, n, "w")]] = torch.cat([w, torch.zeros(h, add, dtype=w.dtype)], 1)
            mu = want[proj_keys[(l, n, "m_u")]]
            full[proj_keys[(l, n, "m_u")]] = torch.cat([mu / ratio, torch.zeros(h, add, dtype=mu.dtype)], 1)
            full[proj_keys[(l, n, "v_norm")]] = want[proj_keys[(l, n, "v_norm")]] * ratio
            for s in ("g", "m_g", "v_g"):  # per row: unchanged (zero columns keep the row norms)
                full[proj_keys[(l, n, s)]] = want[proj_keys[(l, n, s)]].clone()
            log(f"layer {l} {n}: [{h}, {c}] -> [{h}, {c + add}], spectral ratio {ratio:.4f}")
    for k in opt_tables:
        rows[k] = torch.zeros((r_aux,) + tuple(smd[k].size[1:]), dtype=smd[k].properties.dtype)
    # the stored args stay as they are (not read: unpickling them imports Megatron, which initializes
    # CUDA, and that hangs on a login node whose driver check times out); loading never uses them

    os.makedirs(a.dst, exist_ok=False)
    tmp = os.path.join(a.dst, ".engaux_tmp")
    dcp.save({**full, **{f"__rows__.{k}": v for k, v in rows.items()}}, storage_writer=FileSystemWriter(tmp), no_dist=True)
    small = FileSystemReader(tmp).read_metadata()
    log(f"wrote {len(full)} rewritten keys + {len(rows)} appended row chunks")

    out = copy.copy(md)
    out.state_dict_metadata = dict(smd)
    storage = {k: v for k, v in md.storage_data.items() if k.fqn not in full}
    files = {}

    def target_name(rel):
        if rel not in files:
            i = len(files)
            while os.path.exists(os.path.join(a.src, f"engaux_{i}{rel}")):
                i += 1
            files[rel] = f"engaux_{i}{rel}"
        return files[rel]

    for k in full:  # replaced whole
        out.state_dict_metadata[k] = small.state_dict_metadata[k]
    for k, v in small.storage_data.items():
        if k.fqn in full:
            storage[k] = copy.copy(v)
            storage[k].relative_path = target_name(v.relative_path)
    for k in rows:  # old chunks + one appended chunk at row r_main
        old = smd[k]
        new_size = torch.Size((r_main + r_aux,) + tuple(old.size[1:]))
        chunk_off = torch.Size((r_main,) + (0,) * (len(old.size) - 1))
        chunk = ChunkStorageMetadata(offsets=chunk_off, sizes=torch.Size((r_aux,) + tuple(old.size[1:])))
        meta = copy.copy(old)
        meta.size = new_size
        meta.chunks = list(old.chunks) + [chunk]
        out.state_dict_metadata[k] = meta
        (src_idx, src_info), = [(i, v) for i, v in small.storage_data.items() if i.fqn == f"__rows__.{k}"]
        info = copy.copy(src_info)
        info.relative_path = target_name(src_info.relative_path)
        storage[MetadataIndex(fqn=k, offset=chunk_off, index=len(old.chunks))] = info
    out.storage_data = storage
    for old, name in files.items():
        os.rename(os.path.join(tmp, old), os.path.join(a.dst, name))
    os.remove(os.path.join(tmp, ".metadata"))
    os.rmdir(tmp)
    n_link = 0
    for f in os.listdir(a.src):
        if f.endswith(".distcp"):
            os.link(os.path.join(a.src, f), os.path.join(a.dst, f))
            n_link += 1
    with open(os.path.join(a.dst, ".metadata"), "wb") as f:
        pickle.dump(out, f)
    log(f"done: {n_link} hard links, {len(files)} new file(s), {len(full)} keys rewritten, {len(rows)} keys extended")

    # read-back: a rewritten projection, and the boundary rows of one table + its moments
    chk = FileSystemReader(a.dst)
    k0 = proj_keys[(tables[0], "key_proj.weight", "w")]
    t = {k0: torch.empty(full[k0].shape, dtype=full[k0].dtype)}
    dcp.load(t, storage_reader=chk, no_dist=True)
    assert torch.equal(t[k0], full[k0])
    log(f"read-back check passed: {k0.split('decoder.')[-1]} {tuple(t[k0].shape)}; tables: "
        + ", ".join(f"{k.split('.')[-1] if 'optimizer' not in k else k.split('param_state.')[-1]} {tuple(out.state_dict_metadata[k].size)}" for k in rows))


if __name__ == "__main__":
    sys.exit(main())
