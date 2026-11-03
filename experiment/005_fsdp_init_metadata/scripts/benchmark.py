"""Four-node NCCL A/B benchmark of existing vs batched initialization finalization."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
    _set_uneven_chunk_metadata,
    gather_and_compute_chunk_metadata_batched,
    update_uneven_dtensor_chunk_metadata,
    validate_uneven_dtensor,
    validate_uneven_dtensors,
)

baseline_path = Path(os.environ.get(
    'FSDP_BASELINE_MEGATRON',
    '/gpfs/projects/ehpc533/users/korbip/hybrid_v019/autoexp_rowmeta2/submodules/Megatron-LM',
)) / 'megatron/core/distributed/fsdp/src/megatron_fsdp/uneven_dtensor.py'
spec = importlib.util.spec_from_file_location('fsdp_metadata_baseline', baseline_path)
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)

torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
dist.init_process_group('nccl')
world = dist.get_world_size()
assert world == 16, f'Expected 4 nodes x 4 GPUs, got {world}'
full = init_device_mesh('cuda', (world,))
expert = init_device_mesh('cuda', (4, 4), mesh_dim_names=('outer', 'inner'))
cases = []
for i in range(64):
    if i % 2:
        mesh, placements, rank, size = expert, [Replicate(), Shard(0)], dist.get_rank() % 4, 4
    else:
        mesh, placements, rank, size = full, [Shard(0)], dist.get_rank(), world
    cases.append(DTensor.from_local(
        torch.zeros((rank + 1, 3), device='cuda'), device_mesh=mesh, placements=placements,
        run_check=False, shape=torch.Size((size * (size + 1) // 2, 3)), stride=(3, 1),
    ))

# Warm up every mesh/group and the tensor reductions before timing.
metadata = gather_and_compute_chunk_metadata_batched(cases)
validate_uneven_dtensors(cases, metadata)

def measure(fn):
    dist.barrier(); torch.cuda.synchronize()
    with patch.object(dist, 'all_gather_object', wraps=dist.all_gather_object) as gather, patch.object(
        dist, 'all_reduce', wraps=dist.all_reduce
    ) as reduce:
        started = time.perf_counter(); fn(); torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        counts = {'all_gather_object': gather.call_count, 'all_reduce': reduce.call_count}
    seconds = torch.tensor(elapsed, device='cuda')
    dist.all_reduce(seconds, op=dist.ReduceOp.MAX)
    return {'max_rank_seconds': seconds.item(), **counts}

def baseline():
    for d in cases:
        original.validate_uneven_dtensor(d)
        original.update_uneven_dtensor_chunk_metadata(d)

def batched():
    metadata = gather_and_compute_chunk_metadata_batched(cases)
    validate_uneven_dtensors(cases, metadata)
    for d, m in zip(cases, metadata):
        _set_uneven_chunk_metadata(d, m)

old = measure(baseline)
expected = [d._local_tensor.__create_chunk_list__()[0] for d in cases]
new = measure(batched)
assert [d._local_tensor.__create_chunk_list__()[0] for d in cases] == expected
assert old['all_gather_object'] == 128 and old['all_reduce'] == 64
assert new['all_gather_object'] == 2 and new['all_reduce'] == 2
if dist.get_rank() == 0:
    result = {'ranks': world, 'tensors': len(cases), 'baseline_source': str(baseline_path),
              'baseline_sha256': hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
              'baseline': old, 'batched': new,
              'speedup': old['max_rank_seconds'] / new['max_rank_seconds']}
    path = Path('experiment/005_fsdp_init_metadata/results/benchmark.json')
    path.write_text(json.dumps(result, indent=2) + '\n')
    print('FSDP_METADATA_BENCHMARK ' + json.dumps(result), flush=True)
dist.destroy_process_group()
