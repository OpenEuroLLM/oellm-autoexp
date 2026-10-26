"""Stream the model tensors of a Megatron-FSDP ``fsdp_dtensor`` checkpoint, layer by layer.

The main runs (ambitious hybrid, conservative q35) save with ``ckpt_format: fsdp_dtensor``: a
PyTorch DCP checkpoint whose model keys carry the FSDP module prefix (``model.module.``) and whose
SwiGLU ``linear_fc1`` weights are stored as their two halves (``…weight_w`` = gate, ``…weight_v`` =
up; per expert ``…weight{E}_w`` / ``…weight{E}_v``, fc2 per expert ``…weight{E}``), next to the
optimizer state and the rng state. A 35-91B checkpoint does not fit in memory as fp32, so
:func:`iter_layers` reads one layer (and the non-layer tensors) at a time, the way the HF exporters
consume it, and :class:`ShardedSafetensorsWriter` writes the export in HF's sharded layout.

Names are normalised to the training naming without the prefix (``decoder.layers.3.mlp.router.weight``);
the SwiGLU halves and per-expert tensors keep their suffixes, so the exporter decides how to fuse them.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import torch

LOGGER = logging.getLogger(__name__)

PREFIX = "model.module."
_LAYER = re.compile(r"^decoder\.layers\.(\d+)\.")


def model_keys(ckpt_dir: Path) -> tuple[object, dict[str, tuple[str, tuple[int, ...], torch.dtype]]]:
    """(reader, {normalised name: (checkpoint key, shape, dtype)}) of the model tensors."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    reader = dcp.FileSystemReader(str(ckpt_dir))
    md = reader.read_metadata().state_dict_metadata
    out = {}
    for key, entry in md.items():
        if not key.startswith(PREFIX) or not isinstance(entry, TensorStorageMetadata):
            continue
        if "_extra_state" in key:
            continue
        out[key[len(PREFIX):]] = (key, tuple(entry.size), entry.properties.dtype)
    if not out:
        raise ValueError(f"{ckpt_dir}: no '{PREFIX}' model tensors (not an fsdp_dtensor checkpoint?)")
    return reader, out


def load(reader, entries: dict[str, tuple[str, tuple[int, ...], torch.dtype]]) -> dict[str, torch.Tensor]:
    """Read the given tensors (normalised name -> (key, shape, dtype)) in one DCP pass."""
    import torch.distributed.checkpoint as dcp

    template = {key: torch.empty(shape, dtype=dtype) for key, shape, dtype in entries.values()}
    dcp.load(template, storage_reader=reader, no_dist=True)
    return {name: template[key] for name, (key, _, _) in entries.items()}


def iter_layers(ckpt_dir: Path) -> Iterator[tuple[int | None, dict[str, torch.Tensor]]]:
    """Yield ``(None, non-layer tensors)`` first, then ``(i, tensors of decoder layer i)`` in order."""
    reader, keys = model_keys(ckpt_dir)
    by_layer: dict[int | None, dict] = defaultdict(dict)
    for name, entry in keys.items():
        m = _LAYER.match(name)
        by_layer[int(m.group(1)) if m else None][name] = entry
    LOGGER.info("%s: %d model tensors, %d layers", ckpt_dir, len(keys), len(by_layer) - (None in by_layer))
    yield None, load(reader, by_layer.pop(None, {}))
    for i in sorted(by_layer):
        yield i, load(reader, by_layer[i])


def stack_experts(tensors: dict[str, torch.Tensor], prefix: str, suffix: str = "") -> torch.Tensor:
    """Stack ``{prefix}{e}{suffix}`` for e = 0..E-1 into one [E, ...] tensor (and drop them)."""
    pat = re.compile(re.escape(prefix) + r"(\d+)" + re.escape(suffix) + "$")
    found = sorted((int(m.group(1)), k) for k in list(tensors) if (m := pat.match(k)))
    if not found or [e for e, _ in found] != list(range(len(found))):
        raise KeyError(f"experts {prefix}N{suffix}: not a contiguous 0..E-1 set ({len(found)} found)")
    return torch.stack([tensors.pop(k) for _, k in found])


class ShardedSafetensorsWriter:
    """HF's sharded safetensors layout: model-0000N-of-0000M.safetensors + model.safetensors.index.json.

    Tensors are buffered and flushed into a shard when the buffer exceeds ``shard_bytes``; the shard
    count is only known at the end, so shards are named in a temporary scheme and renamed by
    :meth:`close`."""

    def __init__(self, out_dir: Path, shard_bytes: int = 5 * 2**30):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_bytes = shard_bytes
        self.buf: dict[str, torch.Tensor] = {}
        self.buf_bytes = 0
        self.shards: list[list[str]] = []
        self.total = 0

    def add(self, name: str, tensor: torch.Tensor) -> None:
        if name in self.buf or any(name in s for s in self.shards):
            raise KeyError(f"duplicate tensor {name}")
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name}: non-finite values")
        t = tensor.contiguous()
        self.buf[name] = t
        n = t.numel() * t.element_size()
        self.buf_bytes += n
        self.total += n
        if self.buf_bytes >= self.shard_bytes:
            self.flush()

    def flush(self) -> None:
        if not self.buf:
            return
        from safetensors.torch import save_file

        idx = len(self.shards)
        save_file(self.buf, str(self.out_dir / f"model-part{idx:05d}.safetensors"), metadata={"format": "pt"})
        self.shards.append(list(self.buf))
        LOGGER.info("wrote shard %d (%d tensors, %.1f GB)", idx, len(self.buf), self.buf_bytes / 1e9)
        self.buf, self.buf_bytes = {}, 0

    def close(self) -> None:
        self.flush()
        n = len(self.shards)
        weight_map = {}
        for i, names in enumerate(self.shards):
            final = f"model-{i + 1:05d}-of-{n:05d}.safetensors"
            (self.out_dir / f"model-part{i:05d}.safetensors").rename(self.out_dir / final)
            weight_map.update({name: final for name in names})
        index = {"metadata": {"total_size": self.total}, "weight_map": dict(sorted(weight_map.items()))}
        (self.out_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
        LOGGER.info("%d shards, %.1f GB", n, self.total / 1e9)
