#!/usr/bin/env python3
"""CPU check of the actual Megatron sampler classes at the v4 migration geometry.

Load class definitions without Megatron's GPU/distributed training imports. Compare
each step's sample multiset with the stock 128-node cyclic sampler, from scratch
and from a checkpoint cursor, in both curriculum phases. No data or jobs needed.
"""

import ast
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def main():
    source = Path(__file__).resolve().parents[3] / "submodules/Megatron-LM/megatron/training/datasets/data_samplers.py"
    tree = ast.parse(source.read_text())
    classes = ast.Module(body=[n for n in tree.body if isinstance(n, ast.ClassDef)], type_ignores=[])
    namespace = dict(torch=torch, np=np, random=random, Dataset=Dataset)
    exec(compile(classes, str(source), "exec"), namespace)
    stock = namespace["MegatronPretrainingRandomSampler"]
    reference = namespace["MegatronPretrainingReferenceLayoutSampler"]
    results = []
    for phase, gbs, mbs, actual_mbs in [(1, 8192, 8, 8), (1, 8192, 8, 4), (2, 1024, 1, 1)]:
        total = 7 * gbs + 123  # includes epoch crossings and an incomplete tail
        for start_step in [0, 4000]:
            consumed = start_step * gbs
            old = [iter(stock(None, total, consumed, mbs, rank, 512, True)) for rank in range(512)]
            new = [iter(reference(None, total, consumed, actual_mbs, gbs, rank, 1024, True, 512, mbs))
                   for rank in range(1024)]
            counters = [consumed] * 512
            for step in range(12):
                want, got = [], []
                for rank in range(512):
                    for _ in range(gbs // (512 * mbs)):
                        try:
                            batch = next(old[rank])
                        except StopIteration:
                            old[rank] = iter(stock(None, total, counters[rank], mbs, rank, 512, True))
                            batch = next(old[rank])
                        counters[rank] += mbs * 512
                        want.extend(batch)
                for it in new:
                    for _ in range(gbs // (1024 * actual_mbs)):
                        got.extend(next(it))
                assert len(got) == gbs
                assert sorted(got) == sorted(want), (phase, start_step, step)
            results.append(dict(phase=phase, micro_batch_size=actual_mbs, start_step=start_step, steps=12, same_samples=True))
    print(json.dumps(dict(reference_dp=512, actual_dp=1024, checks=results), indent=2))


if __name__ == "__main__":
    main()
