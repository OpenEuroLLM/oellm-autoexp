"""Asynchronous data-staging helpers for Megatron training on stall-prone
shared filesystems (e.g. GPFS on MareNostrum).

The access order of a Megatron GPTDataset is fully deterministic and precomputed
in the cached index files. `prefetch_order` reconstructs the exact sequence of
`.bin` byte ranges a run will touch, so a sidecar prefetcher can warm those bytes
(page cache, MVP) or stage them to node-local disk ahead of the GPU cursor.
"""
