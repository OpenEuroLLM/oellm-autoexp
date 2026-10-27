"""FSDP-like communication benchmark of an allocation: the data-parallel collectives Megatron-FSDP
runs at scale, plus a ring of node-to-node transfers to find one slow node / link.

1. per local GPU index g (4 groups of `nodes` ranks: g, g+4, g+8, ...; the FSDP data-parallel
   groups with EP 4 inside a node), all four groups at once: all_gather_into_tensor and
   reduce_scatter_tensor of a 256 MB buffer, timed per group (bus bandwidth);
2. node ring: node i sends 512 MB to node i+1 on all 4 GPUs at once (batched isend/irecv), each
   pair timed by the receiver; rank 0 prints the slowest pairs.

    torchrun --nnodes $NUM_NODES --nproc-per-node 4 --node-rank $SLURM_NODEID \\
        --master-addr $MASTER_ADDR --master-port $MASTER_PORT scripts/korbi/pool/fsdp_comm_bench.py
"""

from __future__ import annotations

import os
import socket
import time

import torch
import torch.distributed as dist


def sync_time(fn, iters: int, group=None) -> float:
    dist.barrier()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters


def main() -> None:
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local))
    rank, world = dist.get_rank(), dist.get_world_size()
    per_node = int(os.environ.get("LOCAL_WORLD_SIZE", 4))
    nodes = world // per_node
    host = socket.gethostname().split(".")[0]
    names = [None] * world
    dist.all_gather_object(names, host)

    # 1. FSDP-like groups: same local GPU index across nodes
    groups = [dist.new_group([n * per_node + g for n in range(nodes)]) for g in range(per_node)]
    mine = groups[local]
    numel = 128 * 2**20  # 256 MB bf16
    shard = torch.ones(numel // nodes, device="cuda", dtype=torch.bfloat16)
    full = torch.empty(numel // nodes * nodes, device="cuda", dtype=torch.bfloat16)
    ag = lambda: dist.all_gather_into_tensor(full, shard, group=mine)  # noqa: E731
    rs = lambda: dist.reduce_scatter_tensor(shard, full, group=mine)  # noqa: E731
    ag()
    rs()
    t_ag = sync_time(ag, 5)
    t_rs = sync_time(rs, 5)
    bw = lambda t: (nodes - 1) / nodes * full.numel() * 2 / t / 1e9  # noqa: E731

    # 2. node ring: node n -> node n+1, all GPUs at once
    buf_s = torch.ones(256 * 2**20, device="cuda", dtype=torch.bfloat16)  # 512 MB
    buf_r = torch.empty_like(buf_s)
    node = rank // per_node
    dst = ((node + 1) % nodes) * per_node + local
    src = ((node - 1) % nodes) * per_node + local

    def ring():
        ops = [dist.P2POp(dist.isend, buf_s, dst), dist.P2POp(dist.irecv, buf_r, src)]
        for req in dist.batch_isend_irecv(ops):
            req.wait()

    ring()
    torch.cuda.synchronize()
    t_ring = sync_time(ring, 3)

    row = torch.tensor([t_ag, t_rs, t_ring], device="cuda", dtype=torch.float64)
    rows = [torch.zeros_like(row) for _ in range(world)]
    dist.all_gather(rows, row)
    if rank == 0:
        for g in range(per_node):
            ags = [float(rows[n * per_node + g][0]) for n in range(nodes)]
            rss = [float(rows[n * per_node + g][1]) for n in range(nodes)]
            print(f"[fsdp_bench] group gpu{g} ({nodes} ranks, 256 MB): all_gather {max(ags) * 1e3:.1f} ms "
                  f"({bw(max(ags)):.1f} GB/s), reduce_scatter {max(rss) * 1e3:.1f} ms ({bw(max(rss)):.1f} GB/s)")
        ring_t = [float(r[2]) for r in rows]
        gbps = [buf_s.numel() * 2 / t / 1e9 for t in ring_t]
        med = sorted(gbps)[len(gbps) // 2]
        print(f"[fsdp_bench] node ring 512 MB per GPU: median {med:.1f} GB/s, min {min(gbps):.1f}, max {max(gbps):.1f}")
        slow = sorted(range(world), key=lambda i: gbps[i])[:10]
        print("[fsdp_bench] slowest receivers (from -> to, GB/s): " + ", ".join(
            f"{names[((i // per_node - 1) % nodes) * per_node]}->{names[i]}:{i % per_node}={gbps[i]:.1f}"
            for i in slow))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
