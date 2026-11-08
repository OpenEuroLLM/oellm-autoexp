#!/usr/bin/env python3
"""Exact DTensor checkpoint resharding check for rank-local DCP load plans.

Run with torchrun: 2 ranks `save PATH`, then 4 ranks `load PATH`. Uses CPU/Gloo
by default; --device cuda runs the same check with CUDA/NCCL in the MN5 image.
Checks model/optimizer tensors, uneven/empty shards, and non-tensor resume state.
"""

import argparse
import ast
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader, default_planner
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor


def load_helper():
    source = Path(__file__).resolve().parents[3] / 'submodules/Megatron-LM/megatron/training/checkpointing.py'
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == '_load_fsdp_dtensor_checkpoint')
    namespace = dict(torch=torch, os=os, print_rank_0=lambda *a: None)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace[node.name]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['save', 'load'])
    parser.add_argument('path')
    parser.add_argument('--hsdp', action='store_true', help='Load into two replicated inner shard groups')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    if args.device == 'cuda':
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl' if args.device == 'cuda' else 'gloo')
    shape = (2, dist.get_world_size() // 2) if args.hsdp else (dist.get_world_size(),)
    mesh = init_device_mesh(args.device, shape)
    placements = [Replicate(), Shard(0)] if args.hsdp else [Shard(0)]
    expected = {
        'model': torch.arange(37 * 11, dtype=torch.float32).reshape(37, 11),
        'momentum': torch.arange(37 * 11, dtype=torch.float32).reshape(37, 11) / 8,
        'row_variance': torch.arange(37, dtype=torch.float32).reshape(37, 1),
        'engram_momentum': torch.arange(19 * 7, dtype=torch.bfloat16).reshape(19, 7),
        'empty_rank_shards': torch.arange(3, dtype=torch.float32),
    }

    def state(saving):
        out = {k: distribute_tensor((v if saving else torch.zeros_like(v)).to(args.device),
                                    mesh, placements) for k, v in expected.items()}
        out.update(iteration=4807 if saving else 0,
                   consumed_samples=4807 * 8192 if saving else 0,
                   optimizer_scalars={'step': 4807 if saving else 0,
                                      'groups': {'weight': {'lr': 0.00085 if saving else 0.0}}})
        return out

    if args.mode == 'save':
        dcp.save(state(True), checkpoint_id=args.path)
    else:
        helper = load_helper()
        for local in (False, True):
            os.environ['FSDP_DCP_LOCAL_LOAD_PLAN'] = str(int(local))
            out = state(False)
            gather = dist.gather_object
            if local:
                def forbidden(*a, **kw):
                    raise AssertionError('rank-local load must not gather load-plan objects')
                dist.gather_object = forbidden
            started = time.monotonic()
            try:
                helper(out, FileSystemReader(args.path),
                       default_planner.DefaultLoadPlanner(allow_partial_load=False))
            finally:
                dist.gather_object = gather
            for k, value in expected.items():
                torch.testing.assert_close(out[k].full_tensor().cpu(), value, rtol=0, atol=0)
            assert out['iteration'] == 4807 and out['consumed_samples'] == 4807 * 8192
            assert out['optimizer_scalars'] == {'step': 4807, 'groups': {'weight': {'lr': 0.00085}}}
            if dist.get_rank() == 0:
                print(f'PASS: world={dist.get_world_size()} local_plans={local} '
                      f'exact tensors/scalars; {time.monotonic() - started:.3f}s', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
