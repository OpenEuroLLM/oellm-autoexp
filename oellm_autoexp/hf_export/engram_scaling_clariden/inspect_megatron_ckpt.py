"""Inspect a Megatron ``torch_dist`` checkpoint of a HybridModel run, read-only, before writing an exporter.

A ``torch_dist`` checkpoint is a PyTorch DCP directory: ``.metadata`` indexes every saved entry (global
shape, dtype, and the byte ranges of its shards in the ``__<rank>_0.distcp`` files), so the model can be
listed and read on a CPU without Megatron. This prints

1. the saved entries by category (model weights, optimizer state, TE ``_extra_state``, the rest),
2. the run's layer pattern, one Megatron layer per symbol,
3. the model tensors of every layer, each distinct (symbol, tensor names / shapes / dtypes) signature once,
   with the layers that share it; a symbol with more than one signature is flagged,
4. the non-layer tensors, and the Engram table rows against the prime table sizes of the config,
5. with ``--stats``: per-tensor statistics of the weights (loads the model tensors, not the optimizer).

    python -m oellm_autoexp.hf_export.engram_scaling_clariden.inspect_megatron_ckpt \\
        --ckpt-dir <run>/ckpt/iter_0001000 --config <run>/config-<job>.yaml [--stats]
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch

from ..export_ambitious import engram_table_sizes
from ..export_q35 import megatron_args

_MODEL_PREFIXES = ("embedding.", "decoder.", "output_layer.")
_LAYER = re.compile(r"^decoder\.layers\.(\d+)\.(.+)$")
_KINDS = {"*": "attention", "L": "linear mixer", "E": "MoE", "N": "Engram", "-": "MLP",
          "M": "Mamba", "G": "GDN", "C": "short conv"}


def _category(key: str, entry) -> str:
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    if "_extra_state" in key:
        return "te_extra_state"
    if "optimizer" in key:
        return "optimizer"
    if key.startswith(_MODEL_PREFIXES) and isinstance(entry, TensorStorageMetadata):
        return "model"
    return "other"


def _nbytes(entry) -> int:
    size = getattr(entry, "size", None)
    dtype = getattr(getattr(entry, "properties", None), "dtype", None)
    if size is None or dtype is None:
        return 0
    return int(torch.Size(size).numel()) * torch.empty((), dtype=dtype).element_size()


def _dtype(entry) -> str:
    return str(entry.properties.dtype).replace("torch.", "")


def read_metadata(ckpt_dir: Path):
    import torch.distributed.checkpoint as dcp

    reader = dcp.FileSystemReader(str(ckpt_dir))
    return reader, reader.read_metadata().state_dict_metadata


def summarize_entries(md: dict) -> dict[str, dict]:
    by_cat: dict[str, dict] = defaultdict(dict)
    for key, entry in md.items():
        by_cat[_category(key, entry)][key] = entry
    print("== saved entries")
    for cat in ("model", "optimizer", "te_extra_state", "other"):
        entries = by_cat.get(cat, {})
        gb = sum(_nbytes(e) for e in entries.values()) / 1e9
        print(f"  {cat:15s} {len(entries):6d} entries  {gb:8.2f} GB")
    dtypes = Counter(_dtype(e) for e in by_cat["model"].values())
    print(f"  model dtypes: {dict(dtypes)}")
    return by_cat


def print_layers(model: dict, pattern: str) -> None:
    symbols = [c for c in pattern if c != "|"]
    print(f"\n== layer pattern {pattern!r}: {len(symbols)} Megatron layers")
    print("  " + " ".join(f"{i}:{s}" for i, s in enumerate(symbols)))

    per_layer: dict[int, dict[str, tuple]] = defaultdict(dict)
    for key, entry in model.items():
        m = _LAYER.match(key)
        if m:
            per_layer[int(m.group(1))][m.group(2)] = (tuple(entry.size), _dtype(entry))

    found = sorted(per_layer)
    if found != list(range(len(symbols))):
        missing = sorted(set(range(len(symbols))) - set(found))
        extra = sorted(set(found) - set(range(len(symbols))))
        print(f"  !! layer indices do not match the pattern: missing {missing}, extra {extra}")

    groups: dict[tuple, list[int]] = defaultdict(list)
    for i in found:
        sym = symbols[i] if i < len(symbols) else "?"
        groups[(sym, tuple(sorted(per_layer[i].items())))].append(i)

    signatures_per_symbol = Counter(sym for sym, _ in groups)
    for (sym, sig), layers in sorted(groups.items(), key=lambda kv: kv[1][0]):
        flag = "  !! more than one signature for this symbol" if signatures_per_symbol[sym] > 1 else ""
        print(f"\n-- '{sym}' {_KINDS.get(sym, '?')}: layers {layers}{flag}")
        for name, (shape, dtype) in sig:
            print(f"     {name:60s} {str(list(shape)):22s} {dtype}")


def print_nonlayer(model: dict) -> None:
    print("\n== non-layer tensors")
    for key, entry in sorted(model.items()):
        if not _LAYER.match(key):
            print(f"  {key:62s} {str(list(entry.size)):22s} {_dtype(entry)}")


def check_engram(model: dict, m: dict) -> None:
    symbols = [c for c in m["hybrid_layer_pattern"] if c != "|"]
    engram = [i for i, s in enumerate(symbols) if s == "N"]
    if not engram:
        return
    sizes = engram_table_sizes(m)
    print(f"\n== Engram: layers {engram}, {len(sizes)} tables "
          f"({len(m['engram_ngram_orders'])} orders x {m['engram_num_heads']} heads), "
          f"sizes {sizes[0]} .. {sizes[-1]}, sum {sum(sizes)}")
    for i in engram:
        rows = model[f"decoder.layers.{i}.table"].size[0]
        pad = rows - sum(sizes)
        note = "no padding" if pad == 0 else f"{pad} padding rows (Megatron pads to 4096 outside dp table parallel)"
        print(f"  layer {i}: table rows {rows} ({note}); "
              f"hash seed offset 10007 * layer_number = 10007 * {i + 1}"
              f"{'' if m.get('engram_hash_per_layer') else ' (unused: engram_hash_per_layer off)'}")


def print_stats(reader, model: dict) -> None:
    import torch.distributed.checkpoint as dcp

    template = {k: torch.empty(tuple(e.size), dtype=e.properties.dtype) for k, e in model.items()}
    dcp.load(template, storage_reader=reader, no_dist=True)
    print("\n== weight statistics (fp32)")
    for key, t in sorted(template.items(), key=lambda kv: (_LAYER.match(kv[0]) is not None, kv[0])):
        x = t.float()
        finite = bool(torch.isfinite(x).all())
        print(f"  {key:66s} mean {x.mean():+.4e} std {x.std() if x.numel() > 1 else 0:.4e} "
              f"min {x.min():+.3e} max {x.max():+.3e}{'' if finite else '  !! non-finite'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt-dir", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True, help="the run's resolved config-<job>.yaml")
    p.add_argument("--stats", action="store_true", help="load the model tensors and print their statistics")
    a = p.parse_args()

    m = megatron_args(a.config)
    reader, md = read_metadata(a.ckpt_dir)
    model = summarize_entries(md)["model"]
    print_layers(model, m["hybrid_layer_pattern"])
    print_nonlayer(model)
    check_engram(model, m)
    if a.stats:
        print_stats(reader, model)


if __name__ == "__main__":
    main()
