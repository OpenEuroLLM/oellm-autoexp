"""Health check of every node in an allocation: per-GPU bf16 matmul throughput, then world-wide
all-reduce and all-to-all timings; rank 0 prints the slowest nodes (a 2x slower 128-node run with an
unchanged config points at one slow GPU, NIC or switch path).

    torchrun --nnodes $NUM_NODES --nproc-per-node 4 --node-rank $SLURM_NODEID \\
        --master-addr $MASTER_ADDR --master-port $MASTER_PORT scripts/korbi/pool/node_bench.py
"""

from __future__ import annotations

import os
import socket
import time

import torch
import torch.distributed as dist


def timed(fn, iters: int) -> float:
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
    host = socket.gethostname().split(".")[0]

    n = 8192
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    timed(lambda: a @ b, 5)
    mm_tflops = 2 * n**3 / timed(lambda: a @ b, 30) / 1e12

    x = torch.ones(256 * 2**20, device="cuda", dtype=torch.bfloat16)  # 512 MB
    dist.all_reduce(x)
    t_ar = timed(lambda: dist.all_reduce(x), 5)
    ar_busbw = 2 * (world - 1) / world * x.numel() * 2 / t_ar / 1e9

    per = 2 * 2**20  # elements per peer (4 MB bf16)
    src = torch.ones(per * world, device="cuda", dtype=torch.bfloat16)
    dst = torch.empty_like(src)
    dist.all_to_all_single(dst, src)
    t_a2a = timed(lambda: dist.all_to_all_single(dst, src), 5)

    # a per-rank all-to-all view: each rank's own time to finish a ring of point-to-point
    mine = torch.tensor([mm_tflops, t_ar, t_a2a], device="cuda", dtype=torch.float64)
    rows = [torch.zeros_like(mine) for _ in range(world)]
    dist.all_gather(rows, mine)
    names = [None] * world
    dist.all_gather_object(names, host)
    if rank == 0:
        mm = [float(r[0]) for r in rows]
        print(f"[node_bench] world {world}: matmul bf16 TFLOP/s per GPU min {min(mm):.0f} "
              f"median {sorted(mm)[len(mm) // 2]:.0f} max {max(mm):.0f}")
        print(f"[node_bench] all_reduce 512 MB: {t_ar * 1e3:.1f} ms, bus bw {ar_busbw:.1f} GB/s; "
              f"all_to_all {per * 2 * world / 2**20:.0f} MB per rank: {t_a2a * 1e3:.1f} ms")
        slow = sorted(range(world), key=lambda i: mm[i])[:8]
        print("[node_bench] slowest GPUs (matmul): " + ", ".join(
            f"{names[i]}:{i % 4}={mm[i]:.0f}" for i in slow))
        by_node: dict[str, list[float]] = {}
        for i, h in enumerate(names):
            by_node.setdefault(h, []).append(mm[i])
        worst = sorted(by_node.items(), key=lambda kv: min(kv[1]))[:5]
        print("[node_bench] worst nodes: " + ", ".join(f"{h} {min(v):.0f}" for h, v in worst))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
