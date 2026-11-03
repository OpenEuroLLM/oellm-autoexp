#!/usr/bin/env python3
"""Four-GPU exact-path Muown check: HSDP replicated uneven row shards vs full tensors."""
import os

import torch
import torch.distributed as dist
from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import \
    update_uneven_dtensor_chunk_metadata
from megatron.core.optimizer.emerging_optimizers import \
    TensorParallelMuownClassical
from megatron.core.optimizer.fsdp_row_exchange import local
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard


def main():
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl')
    assert dist.get_world_size() == 4
    os.environ['ANGULAR_MUOWN_COMPACT_CKPT'] = '1'
    mesh = init_device_mesh('cuda', (2, 2), mesh_dim_names=('replica', 'shard'))
    rank = mesh.get_local_rank('shard')
    torch.manual_seed(7)
    layouts = [((12, 8), (5, 7)), ((8, 16), (0, 8)), ((6, 4), (1, 5))]
    fulls = [torch.nn.Parameter(torch.randn(shape, device='cuda') * .2) for shape, _ in layouts]
    shards = []
    for full, (_, counts) in zip(fulls, layouts):
        lo, hi = sum(counts[:rank]), sum(counts[:rank + 1])
        dt = DTensor.from_local(full.detach()[lo:hi].clone(), device_mesh=mesh,
                               placements=[Replicate(), Shard(0)], run_check=False,
                               shape=full.shape, stride=full.stride())
        update_uneven_dtensor_chunk_metadata(dt)
        shards.append(torch.nn.Parameter(dt))
    options = dict(lr=.008, nesterov=False, split_qkv=False, fp32_matmul_prec='highest',
                   scale_mode='spectral', extra_scale_factor=.2)
    reference = TensorParallelMuownClassical(fulls, **options)
    sharded = TensorParallelMuownClassical(shards, **options)
    for step in range(4):
        torch.manual_seed(100 + step)
        for full, shard, (_, counts) in zip(fulls, shards, layouts):
            full.grad = torch.randn_like(full)
            lo, hi = sum(counts[:rank]), sum(counts[:rank + 1])
            shard.grad = DTensor.from_local(full.grad[lo:hi].clone(), device_mesh=mesh,
                                           placements=[Replicate(), Shard(0)], run_check=False,
                                           shape=full.shape, stride=full.stride())
        reference.step()
        sharded.step()
        for full, shard, (_, counts) in zip(fulls, shards, layouts):
            lo, hi = sum(counts[:rank]), sum(counts[:rank + 1])
            torch.testing.assert_close(local(shard), full[lo:hi], rtol=2e-5, atol=2e-6)
    if dist.get_rank() == 0:
        print('PASS: HSDP2 Muown four updates, uneven/empty shards, full-tensor reference', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
