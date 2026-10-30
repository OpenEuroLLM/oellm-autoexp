"""Node-health benchmark of an allocation, one process per GPU (torchrun on every node):

1. compute: a bf16 matmul per GPU (TFLOP/s) -- a throttled or broken GPU;
2. FSDP-like collectives: per local GPU index g, the group of the g-th GPU of every node (the
   data-parallel groups of Megatron-FSDP with EP inside a node), all groups at once:
   all_gather_into_tensor and reduce_scatter_tensor, timed per rank -- a degraded NIC / rail
   (MareNostrum 5, 2026-10-27: one rail at 5.7 / 11.9 instead of 22 GB/s halved a 128-node run;
   matmul and all-reduce tests did not see it);
3. node ring: node i sends to node i+1 on all GPUs at once (batched isend / irecv), each pair
   timed by the receiver -- which node is behind a slow group.

Rank 0 prints one machine-readable line (``[node_health] {json}``, read by
oellm_autoexp.node_health.verdict) and a human-readable summary (``[fsdp_bench] ...``)::

    torchrun --nnodes $NUM_NODES --nproc-per-node 4 --node-rank $SLURM_NODEID \\
        --master-addr $MASTER_ADDR --master-port $MASTER_PORT -m oellm_autoexp.node_health.bench

``--device cpu`` runs the same on gloo (tests; a CPU-only allocation).
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import time

PREFIX = "[node_health] "


def _sync(device: str) -> None:
    import torch

    if device == "cuda":
        torch.cuda.synchronize()


def _timed(fn, iters: int, device: str) -> float:
    import torch.distributed as dist

    fn()  # warm-up (NCCL connections, kernels)
    _sync(device)
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync(device)
    return (time.perf_counter() - t0) / iters


def measure(
    *,
    device: str = "cuda",
    coll_mb: int = 256,
    ring_mb: int = 512,
    matmul_n: int = 8192,
    iters: int = 5,
) -> dict | None:
    """Run the three measurements in an initialised process group; the full result on rank 0
    (None elsewhere)."""
    import torch
    import torch.distributed as dist

    rank, world = dist.get_rank(), dist.get_world_size()
    per_node = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    nodes = world // per_node
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    esize = torch.empty((), dtype=dtype).element_size()
    host = os.environ.get("SLURMD_NODENAME") or socket.gethostname().split(".")[0]
    names: list = [None] * world
    dist.all_gather_object(names, host)

    # 1. compute
    a = torch.randn(matmul_n, matmul_n, device=device, dtype=dtype)
    b = torch.randn(matmul_n, matmul_n, device=device, dtype=dtype)
    t_mm = _timed(lambda: a @ b, iters, device)
    tflops = 2 * matmul_n**3 / t_mm / 1e12

    # 2. FSDP-like groups: the same local GPU index across nodes
    groups = [dist.new_group([n * per_node + g for n in range(nodes)]) for g in range(per_node)]
    mine = groups[local]
    numel = coll_mb * 2**20 // esize // nodes * nodes
    shard = torch.ones(numel // nodes, device=device, dtype=dtype)
    full = torch.empty(numel, device=device, dtype=dtype)
    if nodes > 1:
        t_ag = _timed(lambda: dist.all_gather_into_tensor(full, shard, group=mine), iters, device)
        try:
            t_rs = _timed(
                lambda: dist.reduce_scatter_tensor(shard, full, group=mine), iters, device
            )
        except (RuntimeError, ValueError):  # gloo has no reduce_scatter: its traffic as all-reduce
            t_rs = _timed(lambda: dist.all_reduce(full, group=mine), iters, device) / 2
    else:
        t_ag = t_rs = 0.0

    # 3. node ring
    buf_s = torch.ones(ring_mb * 2**20 // esize, device=device, dtype=dtype)
    buf_r = torch.empty_like(buf_s)
    node = rank // per_node
    dst = ((node + 1) % nodes) * per_node + local
    src = ((node - 1) % nodes) * per_node + local

    def ring():
        ops = [dist.P2POp(dist.isend, buf_s, dst), dist.P2POp(dist.irecv, buf_r, src)]
        for req in dist.batch_isend_irecv(ops):
            req.wait()

    t_ring = _timed(ring, max(1, iters // 2), device) if nodes > 1 else 0.0

    row = torch.tensor([tflops, t_ag, t_rs, t_ring], dtype=torch.float64, device=device)
    rows = [torch.zeros_like(row) for _ in range(world)]
    dist.all_gather(rows, row)
    if rank != 0:
        return None
    rows = [[float(x) for x in r.tolist()] for r in rows]
    coll_bytes = numel * esize
    bus = lambda t: (nodes - 1) / nodes * coll_bytes / t / 1e9 if t > 0 else 0.0  # noqa: E731
    node_names = [names[n * per_node] for n in range(nodes)]
    return {
        "nodes": node_names,
        "gpus_per_node": per_node,
        "device": device,
        # per node: TFLOP/s of each of its GPUs
        "compute": {
            node_names[n]: [rows[n * per_node + g][0] for g in range(per_node)]
            for n in range(nodes)
        },
        # per GPU index: bus bandwidth of the slowest rank of the group (that is the group's)
        "groups": {
            str(g): {
                "all_gather": bus(max(rows[n * per_node + g][1] for n in range(nodes))),
                "reduce_scatter": bus(max(rows[n * per_node + g][2] for n in range(nodes))),
            }
            for g in range(per_node)
        },
        # every ring pair: [from node, to node, GPU index, GB/s at the receiver]
        "pairs": [
            [
                node_names[(i // per_node - 1) % nodes],
                node_names[i // per_node],
                i % per_node,
                ring_mb * 2**20 / rows[i][3] / 1e9 if rows[i][3] > 0 else 0.0,
            ]
            for i in range(world)
        ]
        if nodes > 1
        else [],
        "sizes_mb": {"collective": coll_mb, "ring": ring_mb},
    }


def summary(result: dict) -> list[str]:
    """Human-readable lines (the ``[fsdp_bench]`` format of the earlier one-off benchmark)."""
    nodes = len(result["nodes"])
    lines = []
    flops = [x for v in result["compute"].values() for x in v]
    if flops:
        lines.append(
            f"[fsdp_bench] matmul per GPU: median {sorted(flops)[len(flops) // 2]:.1f} TFLOP/s, "
            f"min {min(flops):.1f}, max {max(flops):.1f}"
        )
    mb = result["sizes_mb"]["collective"]
    for g, bw in result["groups"].items():
        lines.append(
            f"[fsdp_bench] group gpu{g} ({nodes} ranks, {mb} MB): "
            f"all_gather ({bw['all_gather']:.1f} GB/s), reduce_scatter ({bw['reduce_scatter']:.1f} GB/s)"
        )
    if result["pairs"]:
        slow = sorted(result["pairs"], key=lambda p: p[3])[:10]
        lines.append(
            "[fsdp_bench] slowest receivers (from -> to, GB/s): "
            + ", ".join(f"{a}->{b}:{g}={bw:.1f}" for a, b, g, bw in slow)
        )
    return lines


def main(argv: list[str] | None = None) -> None:
    import torch
    import torch.distributed as dist

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--coll-mb", type=int, default=256)
    p.add_argument("--ring-mb", type=int, default=512)
    p.add_argument("--matmul-n", type=int, default=8192)
    p.add_argument("--iters", type=int, default=5)
    args = p.parse_args(argv)
    if args.device == "cuda":
        local = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local)
        # device_id binds the communicator (MareNostrum 5 multi-node NCCL errors without it)
        dist.init_process_group("nccl", device_id=torch.device("cuda", local))
    else:
        dist.init_process_group("gloo")
    result = measure(
        device=args.device,
        coll_mb=args.coll_mb,
        ring_mb=args.ring_mb,
        matmul_n=args.matmul_n,
        iters=args.iters,
    )
    if result is not None:
        print("\n".join(summary(result)), flush=True)
        print(PREFIX + json.dumps(result), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
