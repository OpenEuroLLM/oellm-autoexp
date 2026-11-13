#!/usr/bin/env python3
"""Write a rewritten fsdp_dtensor checkpoint (torch DCP) next to its source, sharded like the source.

The checkpoint surgery scripts (widen_attention_head_dim.py, widen_engram_aux.py) used to save the
rewritten tensors with ``dcp.save(no_dist=True)``: one chunk per tensor. On resume every rank reads
the WHOLE chunk of each such tensor and narrows it to its shard: 1024 ranks x the full tensors from
one file. That was most of the 23 min load of the 11-05 head-dim widening.

Here every rewritten tensor is stored as row chunks of about the size of the source's chunks (one
per rank of the saving run), so each loading rank reads ~1-2 chunks per tensor, as from a normal
checkpoint. The new tree hard-links every .distcp file of the source (written once, never
modified), adds a few files with the rewritten items and writes a merged .metadata.

    python dcp_surgery.py selftest      # 2 gloo ranks on CPU: chunk layout + sharded read-back
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import pickle

import torch

DEFAULT_CHUNK_BYTES = 64 << 20


def row_chunks(shape, itemsize, target_bytes):
    """[(first row, rows)] splitting dim 0 into pieces of <= ~target_bytes (at least one row each)."""
    if len(shape) == 0:
        return [(None, None)]
    row = itemsize * math.prod(shape[1:])
    per = max(1, int(target_bytes) // max(row, 1))
    return [(r, min(per, shape[0] - r)) for r in range(0, shape[0], per)] or [(0, 0)]


def source_chunk_bytes(meta):
    """Median chunk size (bytes) of a source tensor entry; None for non-tensor entries."""
    chunks = getattr(meta, "chunks", None)
    if not chunks:
        return None
    itemsize = torch.empty((), dtype=meta.properties.dtype).element_size()
    sizes = sorted(math.prod(c.sizes) * itemsize for c in chunks)
    return max(sizes[len(sizes) // 2], 1)


def write_surgery(src, dst, md, replace, append_rows=None, *, tag, log=print, chunk_bytes=None,
                  thread_count=8):
    """Write ``dst`` = ``src`` with ``replace`` (key -> tensor or picklable object, rewritten whole)
    and ``append_rows`` (key -> tensor of rows appended after the source's rows of that key).

    Tensors are stored in row chunks of ``chunk_bytes`` (default: the median chunk of the same key
    in the source, else of all source tensors, else 64 MB). ``md`` is the source's metadata.
    Returns the merged metadata (also written to ``dst/.metadata``).
    """
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
    from torch.distributed.checkpoint.metadata import ChunkStorageMetadata, MetadataIndex, TensorStorageMetadata

    append_rows = append_rows or {}
    smd = md.state_dict_metadata
    overlap = set(replace) & set(append_rows)
    assert not overlap, overlap
    all_sizes = [b for b in (source_chunk_bytes(m) for m in smd.values()) if b]
    fallback = sorted(all_sizes)[len(all_sizes) // 2] if all_sizes else DEFAULT_CHUNK_BYTES

    def target(key):
        if chunk_bytes is not None:
            return chunk_bytes
        return source_chunk_bytes(smd.get(key)) or fallback

    # the items to save: chunks under synthetic names, objects under their key
    items, plan = {}, []  # plan: (key, item name, row offset, chunk, base row)
    for key, value in list(replace.items()) + [(k, (v, "append")) for k, v in append_rows.items()]:
        if isinstance(value, tuple) and len(value) == 2 and value[1] == "append":
            value, base = value[0], smd[key].size[0]
        else:
            base = 0
        if not isinstance(value, torch.Tensor):
            items[key] = value
            continue
        value = value.contiguous()
        for r0, n in row_chunks(tuple(value.shape), value.element_size(), target(key)):
            name = f"__surgery_chunk_{len(plan)}"
            piece = value if r0 is None else value[r0:r0 + n]
            items[name] = piece.clone()  # a view would save its whole storage
            plan.append((key, name, r0, piece, base))

    os.makedirs(dst, exist_ok=False)
    tmp = os.path.join(dst, f".{tag}_tmp")
    dcp.save(items, storage_writer=FileSystemWriter(tmp, thread_count=thread_count), no_dist=True)
    small = FileSystemReader(tmp).read_metadata()
    by_fqn = {}
    for idx, info in small.storage_data.items():
        by_fqn.setdefault(idx.fqn, []).append(info)

    files = {}

    def target_name(rel):
        if rel not in files:
            i = len(files)
            while os.path.exists(os.path.join(src, f"{tag}_{i}{rel}")):
                i += 1
            files[rel] = f"{tag}_{i}{rel}"
        return files[rel]

    def moved(info):
        info = copy.copy(info)
        info.relative_path = target_name(info.relative_path)
        return info

    out = copy.copy(md)
    out.state_dict_metadata = dict(smd)
    storage = {k: v for k, v in md.storage_data.items() if k.fqn not in replace}
    for key, value in replace.items():  # objects: one item each
        if not isinstance(value, torch.Tensor):
            out.state_dict_metadata[key] = small.state_dict_metadata[key]
            (info,) = by_fqn[key]
            storage[MetadataIndex(fqn=key)] = moved(info)
    tensor_keys = [k for k, v in replace.items() if isinstance(v, torch.Tensor)] + list(append_rows)
    for key in tensor_keys:
        mine = [p for p in plan if p[0] == key]
        if key in append_rows:
            old = smd[key]
            meta = copy.copy(old)
            meta.size = torch.Size((old.size[0] + append_rows[key].shape[0],) + tuple(old.size[1:]))
            chunks = list(old.chunks)
        else:
            full = replace[key]
            meta = TensorStorageMetadata(
                properties=small.state_dict_metadata[mine[0][1]].properties, size=torch.Size(full.shape), chunks=[]
            )
            chunks = []
        for _, name, r0, piece, base in mine:
            off = torch.Size([]) if r0 is None else torch.Size((base + r0,) + (0,) * (piece.dim() - 1))
            chunks.append(ChunkStorageMetadata(offsets=off, sizes=torch.Size(piece.shape)))
            (info,) = by_fqn[name]
            storage[MetadataIndex(fqn=key, offset=off, index=len(chunks) - 1)] = moved(info)
        meta.chunks = chunks
        out.state_dict_metadata[key] = meta
    out.storage_data = storage

    for old_name, name in files.items():
        os.rename(os.path.join(tmp, old_name), os.path.join(dst, name))
    os.remove(os.path.join(tmp, ".metadata"))
    # a few small tensors leave some of the writer's thread files empty and unreferenced (layer_swap zero arms)
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))
    os.rmdir(tmp)
    n_link = 0
    for f in os.listdir(src):
        if f.endswith(".distcp"):
            os.link(os.path.join(src, f), os.path.join(dst, f))
            n_link += 1
    with open(os.path.join(dst, ".metadata"), "wb") as f:
        pickle.dump(out, f)
    n_chunks = len(plan)
    log(f"surgery: {n_link} hard links, {len(files)} new file(s), {len(replace)} keys rewritten, "
        f"{len(append_rows)} keys extended, {n_chunks} tensor chunks")
    return out


# ---------------------------------------------------------------------------------------------
# selftest: a 2-rank sharded source, rewritten + extended, read back as 2 / 3 rank shards
# ---------------------------------------------------------------------------------------------


def _selftest_rank(rank, world, port, root, sizes):
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    mesh = init_device_mesh("cpu", (world,))
    g = torch.Generator().manual_seed(7)
    a, t, v = torch.randn(8, 4, generator=g), torch.randn(10, 3, generator=g), torch.randn(5, generator=g)
    src, dst = os.path.join(root, "src"), os.path.join(root, "dst")
    if sizes == "save":
        sd = {k: distribute_tensor(x, mesh, [Shard(0)]) for k, x in (("a", a), ("t", t), ("v", v))}
        sd["args"] = argparse.Namespace(kv_channels=128)
        dcp.save(sd, storage_writer=FileSystemWriter(src))
    else:  # load the rewritten checkpoint as DTensor shards on `world` ranks
        new_a = torch.randn(16, 4, generator=torch.Generator().manual_seed(8))
        extra = torch.randn(7, 3, generator=torch.Generator().manual_seed(9))
        want = {"a": new_a, "t": torch.cat([t, extra]), "v": v}
        sd = {k: distribute_tensor(torch.zeros_like(x), mesh, [Shard(0)]) for k, x in want.items()}
        sd["args"] = None
        reads = []
        orig = FileSystemReader.read_data

        def counting(self, plan, planner):
            reads.extend(r.storage_index for r in plan.items if r.storage_index.fqn in ("a", "t"))
            return orig(self, plan, planner)

        FileSystemReader.read_data = counting
        try:
            dcp.load(sd, storage_reader=FileSystemReader(dst))
        finally:
            FileSystemReader.read_data = orig
        for k, x in want.items():
            assert torch.equal(sd[k].full_tensor(), x), k
        assert sd["args"].kv_channels == 256, sd["args"]
        # this rank read only the chunks overlapping its rows, not the whole rewritten tensor
        md = FileSystemReader(dst).read_metadata()
        n_a = len(md.state_dict_metadata["a"].chunks)
        got_a = len({r.offset for r in reads if r.fqn == "a"})
        assert n_a >= 4 and got_a < n_a, (n_a, got_a)
    dist.barrier()
    dist.destroy_process_group()


def _free_port():
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def selftest():
    import tempfile

    import torch.multiprocessing as mp
    from torch.distributed.checkpoint import FileSystemReader

    with tempfile.TemporaryDirectory() as root:
        mp.spawn(_selftest_rank, args=(2, _free_port(), root, "save"), nprocs=2, join=True)
        src, dst = os.path.join(root, "src"), os.path.join(root, "dst")
        md = FileSystemReader(src).read_metadata()
        new_a = torch.randn(16, 4, generator=torch.Generator().manual_seed(8))
        extra = torch.randn(7, 3, generator=torch.Generator().manual_seed(9))
        out = write_surgery(src, dst, md, {"a": new_a, "args": argparse.Namespace(kv_channels=256)}, {"t": extra},
                            tag="selftest", chunk_bytes=2 * 4 * 4)  # 2 rows of a per chunk
        assert len(out.state_dict_metadata["a"].chunks) == 8
        assert out.state_dict_metadata["t"].size[0] == 17
        for world in (2, 3):
            mp.spawn(_selftest_rank, args=(world, _free_port(), root, "load"), nprocs=world, join=True)
    print("dcp_surgery selftest passed: chunked rewrite + append read back as 2 / 3 rank shards")


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["selftest"]:
        selftest()
    else:
        sys.exit(__doc__)
