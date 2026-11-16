#!/usr/bin/env python3
"""Reuse completed, content-addressed dataset indices without copying their bytes.

Megatron publishes indices with atomic rename and the description last. A
builder replacing a linked destination therefore cannot modify the old cache.
Never link lock files or temporary files; never overwrite an existing entry.
"""
from pathlib import Path
import os

source = Path('/gpfs/scratch/ehpc533/hybrid_nucleus/production/kvshare_main/runs/data_cache_shared')
dest = Path('/gpfs/scratch/ehpc533/hybrid_nucleus/production/kvshare_k_fresh35900_20261116/runs/data_cache_shared')
dest.mkdir(parents=True, exist_ok=True)
linked = 0
logical_bytes = 0
for description in sorted(source.glob('*-description.txt')):
    base = description.name.removesuffix('-description.txt')
    target_description = dest / description.name
    if target_description.exists():
        if description.read_bytes() != target_description.read_bytes():
            raise RuntimeError(f'cache description mismatch: {description.name}')
    indices = sorted(source.glob(base + '-*.npy'))
    if not indices:
        continue
    for path in [*indices, description]:
        try:
            os.link(path, dest / path.name)
        except FileExistsError:
            continue
        linked += 1
        logical_bytes += path.stat().st_size
print(f'linked {linked} completed cache files, {logical_bytes / 1e9:.1f} GB logical; no data copies', flush=True)
