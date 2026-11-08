"""Parallel HF export of the ambitious main runs: the arithmetic of :mod:`export_ambitious`
(``convert_nonlayer`` / ``convert_layer``), run by a pool of worker processes.

The sequential exporter reads one layer at a time through DCP and writes through one serial
safetensors writer: ~8 min for v4 (654 GB fp32 model tensors in, 248 GB out). Here:

- the checkpoint metadata is read once, in the parent; each work unit gets its list of
  (file, offset, length) chunk reads, so the workers never see DCP's metadata;
- a work unit is one decoder layer (or the non-layer tensors), or a row range of an Engram table
  (whole checkpoint chunks); the units run largest first on ``--workers`` forked processes, each
  reading its chunks with ``--read-threads`` threads;
- every layer unit writes its own safetensors file; an Engram table gets one preallocated file
  (header written up front) that its row-range units fill with ``pwrite``.

The output is HF's sharded layout (``model-0000k-of-0000N.safetensors`` + index); the shard
partition differs from the sequential exporter's, the tensors are byte-identical
(``compare_exports``).

    python -m oellm_autoexp.hf_export.export_parallel --ckpt-dir <run>/ckpt_eval/iter_0005900 \\
        --config <run>/config-<job>.yaml --out-dir <exports>/ambitious_v4_2ph_fp8_5900 \\
        --tokenizer <tok> --expert-dtype fp8
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import logging
import multiprocessing as mp
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .export_ambitious import convert_layer, convert_nonlayer, export_hf_config, finish_export
from .export_q35 import megatron_args
from .fsdp_dtensor import PREFIX

LOGGER = logging.getLogger(__name__)
_LAYER = re.compile(r"^decoder\.layers\.(\d+)\.")
_ST_DTYPES = {
    torch.float32: "F32", torch.bfloat16: "BF16", torch.float16: "F16", torch.float64: "F64",
    torch.uint8: "U8", torch.int8: "I8", torch.int32: "I32", torch.int64: "I64", torch.bool: "BOOL",
    torch.float8_e4m3fn: "F8_E4M3",
}


@dataclass
class Read:
    """One checkpoint chunk: ``nbytes`` at ``offset`` of ``path`` (a torch.save'd tensor) -> rows /
    columns ``dest`` of the destination tensor ``name``."""
    path: str
    offset: int
    nbytes: int
    name: str
    dest: tuple[int, ...]
    sizes: tuple[int, ...]


@dataclass
class Unit:
    kind: str                      # "nonlayer" | "layer" | "table"
    layer: int | None
    file: str                      # the output safetensors file
    tensors: dict[str, tuple[tuple[int, ...], torch.dtype]]  # name -> (shape, dtype) to read
    reads: list[Read] = field(default_factory=list)
    rows: tuple[int, int] = (0, 0)  # table: checkpoint rows [a, b) of this unit
    out_rows: int = 0               # table: the exported rows (the table without its padding)
    data_offset: int = 0            # table: the table's data start in ``file``

    @property
    def nbytes(self) -> int:
        return sum(r.nbytes for r in self.reads)


def _safetensors_header(entries: list[tuple[str, torch.dtype, tuple[int, ...]]]) -> tuple[bytes, list[int]]:
    """The safetensors header of ``entries`` laid out in order; (header bytes, data offsets)."""
    meta, offsets, pos = {}, [], 0
    for name, dtype, shape in entries:
        n = torch.Size(shape).numel() * dtype.itemsize
        meta[name] = {"dtype": _ST_DTYPES[dtype], "shape": list(shape), "data_offsets": [pos, pos + n]}
        offsets.append(pos)
        pos += n
    meta["__metadata__"] = {"format": "pt"}
    h = json.dumps(meta, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    return len(h).to_bytes(8, "little") + h, offsets


def _bytes(t: torch.Tensor) -> memoryview:
    return memoryview(t.contiguous().reshape(-1).view(torch.uint8).numpy())


def write_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    header, _ = _safetensors_header([(k, v.dtype, tuple(v.shape)) for k, v in tensors.items()])
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "wb", buffering=0) as f:
        f.write(header)
        for v in tensors.values():
            f.write(_bytes(v))
    tmp.rename(path)


def _check_finite(name: str, t: torch.Tensor) -> None:
    if t.is_floating_point() and not bool(torch.isfinite(t).all()):
        raise ValueError(f"{name}: non-finite values")


def plan(ckpt_dir: Path, cfg_fn, table_unit_bytes: int) -> tuple[dict, list[Unit], list[str]]:
    """Read the checkpoint metadata once: (hf config, work units, output files in index order)."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.metadata import MetadataIndex, TensorStorageMetadata

    started = time.perf_counter()
    md = dcp.FileSystemReader(str(ckpt_dir)).read_metadata()
    LOGGER.info("metadata: %.1fs", time.perf_counter() - started)
    keys: dict[str, TensorStorageMetadata] = {}
    for key, entry in md.state_dict_metadata.items():
        if key.startswith(PREFIX) and isinstance(entry, TensorStorageMetadata) and "_extra_state" not in key:
            keys[key[len(PREFIX):]] = entry
    if not keys:
        raise ValueError(f"{ckpt_dir}: no '{PREFIX}' model tensors (not an fsdp_dtensor checkpoint?)")
    cfg = cfg_fn(keys["embedding.word_embeddings.weight"].size[0])

    def reads(name: str, row0: int = 0, chunks=None) -> list[Read]:
        entry, out = keys[name], []
        for c in entry.chunks if chunks is None else chunks:
            si = md.storage_data[MetadataIndex(PREFIX + name, c.offsets)]
            if si.transform_descriptors:
                raise ValueError(f"{name}: transformed (compressed?) checkpoint chunk {si.transform_descriptors}")
            dest = (c.offsets[0] - row0, *c.offsets[1:])
            out.append(Read(str(Path(ckpt_dir) / si.relative_path), si.offset, si.length, name, dest, tuple(c.sizes)))
        return out

    by_layer: dict[int | None, list[str]] = {}
    for name in keys:
        mm = _LAYER.match(name)
        by_layer.setdefault(int(mm.group(1)) if mm else None, []).append(name)
    units: list[Unit] = []
    files: list[str] = []

    def new_file() -> str:
        files.append(f"model-{len(files) + 1:05d}-of-NNNNN.safetensors")
        return files[-1]

    for i in [None] + sorted(k for k in by_layer if k is not None):
        names = by_layer[i]
        table = f"decoder.layers.{i}.table" if i is not None and cfg["block_types"][i] == "engram" else None
        small = [n for n in names if n != table]
        u = Unit("nonlayer" if i is None else "layer", i, new_file(),
                 {n: (tuple(keys[n].size), keys[n].properties.dtype) for n in small})
        for n in small:
            u.reads += reads(n)
        units.append(u)
        if table is None:
            continue
        entry = keys[table]
        rows = sum(cfg["engram_table_sizes"])
        if entry.size[0] < rows:
            raise ValueError(f"layer {i}: table has {entry.size[0]} rows < {rows}")
        chunks = sorted(entry.chunks, key=lambda c: c.offsets[0])
        if any(c.offsets[1:] != (0,) * (len(entry.size) - 1) or tuple(c.sizes[1:]) != tuple(entry.size[1:]) for c in chunks):
            raise ValueError(f"{table}: expects row-sharded chunks")
        live = [c for c in chunks if c.offsets[0] < rows]  # the rest is alignment padding
        ends = [c.offsets[0] + c.sizes[0] for c in live]
        if not live or live[0].offsets[0] != 0 or ends[-1] < rows or any(
                c.offsets[0] != e for c, e in zip(live[1:], ends)):
            raise ValueError(f"{table}: chunks do not tile rows [0, {rows})")
        f = new_file()
        row_bytes = torch.Size(entry.size[1:]).numel() * entry.properties.dtype.itemsize
        groups, acc = [[]], 0
        for c in live:
            if acc >= table_unit_bytes:
                groups.append([])
                acc = 0
            groups[-1].append(c)
            acc += c.sizes[0] * row_bytes
        for g in groups:
            ga, gb = g[0].offsets[0], g[-1].offsets[0] + g[-1].sizes[0]
            units.append(Unit("table", i, f, {table: ((gb - ga, *entry.size[1:]), entry.properties.dtype)},
                              reads(table, ga, g), rows=(ga, gb), out_rows=rows))
    n = len(files)
    rename = {f: f.replace("NNNNN", f"{n:05d}") for f in files}
    for u in units:
        u.file = rename[u.file]
    del md
    return cfg, units, [rename[f] for f in files]


# worker state (set before the fork)
_W: dict = {}


def _read_unit(u: Unit) -> dict[str, torch.Tensor]:
    t = {name: torch.empty(shape, dtype=dtype) for name, (shape, dtype) in u.tensors.items()}
    per_file: dict[str, list[Read]] = {}
    for r in u.reads:
        per_file.setdefault(r.path, []).append(r)

    def read_file(rs: list[Read]) -> None:
        rs.sort(key=lambda r: r.offset)
        with open(rs[0].path, "rb", buffering=0) as f:
            for r in rs:
                buf = bytearray(r.nbytes)
                view, got = memoryview(buf), 0
                f.seek(r.offset)
                while got < r.nbytes:
                    k = f.readinto(view[got:])
                    if not k:
                        raise EOFError(f"{r.path}: short read at {r.offset + got}")
                    got += k
                src = torch.load(io.BytesIO(buf), map_location="cpu", weights_only=True)
                if tuple(src.shape) != r.sizes:
                    raise AssertionError(f"{r.name}: chunk {tuple(src.shape)} != {r.sizes}")
                dst = t[r.name]
                for d, (o, s) in enumerate(zip(r.dest, r.sizes)):
                    dst = dst.narrow(d, o, s)
                dst.copy_(src)

    with ThreadPoolExecutor(max_workers=_W["read_threads"]) as ex:
        list(ex.map(read_file, per_file.values()))
    return t


def _run(u: Unit) -> tuple[str, dict[str, int], float, float, int]:
    started = time.perf_counter()
    t = _read_unit(u)
    read_s = time.perf_counter() - started
    out_dir: Path = _W["out_dir"]
    if u.kind == "table":
        (name, src), = t.items()
        a, b = u.rows
        rows = min(b, u.out_rows) - a
        cast = src[:rows].to(_W["tcast"])
        del src, t
        _check_finite(name, cast)
        fd = os.open(out_dir / u.file, os.O_WRONLY)
        try:
            data, pos = _bytes(cast), u.data_offset + a * cast[0].numel() * cast.element_size()
            done = 0
            while done < len(data):
                done += os.pwrite(fd, data[done:], pos + done)
        finally:
            os.close(fd)
        return u.file, {}, read_s, time.perf_counter() - started, u.nbytes
    out: dict[str, torch.Tensor] = {}

    def add(name: str, tensor: torch.Tensor) -> None:
        if name in out:
            raise KeyError(f"duplicate tensor {name}")
        _check_finite(name, tensor)
        out[name] = tensor.contiguous()

    if u.kind == "nonlayer":
        convert_nonlayer(t, _W["cfg"], add, _W["cast"])
    else:
        convert_layer(u.layer, t, _W["cfg"], _W["m"], add, _W["cast"], _W["tcast"], _W["expert_dtype"], table=False)
    write_safetensors(out_dir / u.file, out)
    sizes = {k: v.numel() * v.element_size() for k, v in out.items()}
    return u.file, sizes, read_s, time.perf_counter() - started, u.nbytes


def _init_worker(threads: int) -> None:
    torch.set_num_threads(threads)


def convert(ckpt_dir: Path, config_yaml: Path, out_dir: Path, tokenizer: Path | None, dtype: str = "bf16",
            engram_table_dtype: str = "bf16", expert_dtype: str = "bf16", workers: int = 32,
            read_threads: int = 4, table_unit_gib: float = 1.0) -> Path:
    started = time.perf_counter()
    m = megatron_args(config_yaml)
    cast = {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
    tcast = {"bf16": torch.bfloat16, "fp32": torch.float32}[engram_table_dtype]
    cfg, units, files = plan(
        Path(ckpt_dir), lambda vocab: export_hf_config(m, vocab, dtype, engram_table_dtype, expert_dtype),
        int(table_unit_gib * 2**30))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # the Engram tables' files: header up front, then the units' pwrites
    sizes: dict[str, dict[str, int]] = {}
    for i in sorted({u.layer for u in units if u.kind == "table"}):
        tus = [u for u in units if u.kind == "table" and u.layer == i]
        (shape, _), = tus[0].tensors.values()
        name = f"model.layers.{i}.mixer.table"
        tshape = (tus[0].out_rows, *shape[1:])
        header, _ = _safetensors_header([(name, tcast, tshape)])
        nbytes = torch.Size(tshape).numel() * tcast.itemsize
        with open(out_dir / tus[0].file, "wb") as f:
            f.write(header)
            f.truncate(len(header) + nbytes)
        for u in tus:
            u.data_offset = len(header)
        sizes[tus[0].file] = {name: nbytes}
    total_in = sum(u.nbytes for u in units)
    LOGGER.info("plan: %d units (%d table ranges), %d files, %.1f GB to read, %.1fs",
                len(units), sum(u.kind == "table" for u in units), len(files), total_in / 1e9,
                time.perf_counter() - started)
    _W.update(out_dir=out_dir, cfg=cfg, m=m, cast=cast, tcast=tcast, expert_dtype=expert_dtype,
              read_threads=read_threads)
    order = sorted(units, key=lambda u: -u.nbytes)  # largest first
    gc.collect()
    gc.freeze()  # the forked workers share the parent's objects without touching their pages
    threads = max(1, (os.cpu_count() or workers) // workers)
    done_in = 0
    t0 = time.perf_counter()
    with mp.get_context("fork").Pool(workers, initializer=_init_worker, initargs=(threads,)) as pool:
        for k, (f, s, read_s, wall_s, nin) in enumerate(pool.imap_unordered(_run, order), 1):
            sizes.setdefault(f, {}).update(s)
            done_in += nin
            if k % 25 == 0 or k == len(order):
                el = time.perf_counter() - t0
                LOGGER.info("%d/%d units, %.0f GB read, %.1f GB/s (last: %s read %.1fs, total %.1fs)",
                            k, len(order), done_in / 1e9, done_in / 1e9 / el, f, read_s, wall_s)
    gc.unfreeze()
    weight_map = {name: f for f, s in sizes.items() for name in s}
    total = sum(n for s in sizes.values() for n in s.values())
    missing = [f for f in files if f not in sizes]
    if missing:
        raise RuntimeError(f"no tensors written to {missing}")
    index = {"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))}
    (out_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
    finish_export(out_dir, cfg, tokenizer, ckpt_dir, config_yaml)
    LOGGER.info("%d files, %d tensors, %.1f GB in %.1fs", len(files), len(weight_map), total / 1e9,
                time.perf_counter() - started)
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt-dir", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True, help="the run's resolved config-<job>.yaml")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--engram-table-dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--expert-dtype", default="bf16", choices=["bf16", "fp8"])
    p.add_argument("--workers", type=int, default=int(os.environ.get("HF_EXPORT_WORKERS", "32")))
    p.add_argument("--read-threads", type=int, default=int(os.environ.get("HF_EXPORT_READ_THREADS", "4")))
    p.add_argument("--table-unit-gib", type=float, default=1.0, help="checkpoint bytes per Engram table work unit")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    convert(a.ckpt_dir, a.config, a.out_dir, a.tokenizer, a.dtype, a.engram_table_dtype, a.expert_dtype,
            a.workers, a.read_threads, a.table_unit_gib)


if __name__ == "__main__":
    main()
