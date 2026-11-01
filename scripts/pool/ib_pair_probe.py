#!/usr/bin/env python3
"""Interconnect probe for a pool host's pre-run check (pool setting precheck_command).

Started once per node (the pre-run check step has one task per node); it spawns one process per
GPU (POOL_PROBE_GPUS, default 4) and waits for them. Each GPU g of node
i is paired with GPU g of a partner node -- the same rail, so every HCA is exercised -- and the
pairs all-reduce a buffer at the same time. Two rounds with different pairings: round 1 pairs
node 2k with 2k+1, round 2 node 2k+1 with 2k+2 (wrapping). A node's figure is its WORST pair bus
bandwidth over its GPUs in its BEST round: a node behind a bad link or HCA is low in both rounds,
its partners only in one, so only the bad node keeps a low figure. Output, one line per node from its local rank 0, in
scripts/nodecheck.sh's format (scripts/pool/precheck_verdict.py drops the slow outliers):

    [nodecheck] OK <node> ib_gbps=<best round> ib_rounds=<r1>,<r2>
    [nodecheck] FAIL <node>: ib probe: <error>

Rendezvous from SLURM: MASTER_ADDR = first node of the step, MASTER_PORT from the environment.
Run inside the training container, e.g.

    apptainer exec --nv <sif> python3 scripts/pool/ib_pair_probe.py
"""

from __future__ import annotations

import datetime
import os
import re
import socket
import subprocess
import sys
import time


def expand_hostlist(text: str) -> list[str]:
    """SLURM's compressed node list (jpbo-004-[01-03,07],jpbo-005-11) expanded, in order --
    the training container has no scontrol."""
    out: list[str] = []
    for token in re.findall(r"[^,\[]+(?:\[[^\]]*\])?[^,]*", text or ""):
        token = token.strip()
        m = re.fullmatch(r"(.*)\[([^\]]*)\](.*)", token)
        if not m:
            if token:
                out.append(token)
            continue
        pre, body, post = m.groups()
        for part in body.split(","):
            if "-" in part:
                a, b = part.split("-", 1)
                out += [f"{pre}{i:0{len(a)}d}{post}" for i in range(int(a), int(b) + 1)]
            else:
                out.append(f"{pre}{part}{post}")
    return out


def pair_rounds(n_nodes: int) -> list[list[tuple[int, int]]]:
    """Node pairings of the two rounds (indices). An odd node count leaves one node out of
    round 1; round 2 shifts by one, so every node is paired at least once."""
    r1 = [(i, i + 1) for i in range(0, n_nodes - 1, 2)]
    if n_nodes < 3:
        return [r1]
    r2 = [(i, (i + 1) % n_nodes) for i in range(1, n_nodes, 2)]
    return [r1, r2]


def worst_per_node(n_nodes: int, rounds_bw: list[dict[tuple[int, int], float]]) -> dict[int, list]:
    """Per node: its pair bandwidth in each round (None where it sat out)."""
    out = {i: [] for i in range(n_nodes)}
    for bw in rounds_bw:
        seen = {}
        for (a, b), v in bw.items():
            seen[a] = v if a not in seen else min(seen[a], v)
            seen[b] = v if b not in seen else min(seen[b], v)
        for i in range(n_nodes):
            out[i].append(seen.get(i))
    return out


def main():
    gpus = int(os.environ.get("POOL_PROBE_GPUS", "4"))
    me = os.environ.get("SLURMD_NODENAME") or socket.gethostname().split(".")[0]
    if "POOL_PROBE_LOCAL" not in os.environ:  # the node's task: one process per GPU
        procs = [
            subprocess.Popen([sys.executable, __file__], env={**os.environ, "POOL_PROBE_LOCAL": str(g)})
            for g in range(gpus)
        ]
        for proc in procs:
            proc.wait()
        # always 0: the verdict travels in the [nodecheck] line, and a non-zero task would make
        # srun stop the other nodes' probes
        sys.exit(0)
    local = int(os.environ["POOL_PROBE_LOCAL"])
    try:
        nodes = expand_hostlist(
            os.environ.get("SLURM_STEP_NODELIST") or os.environ["SLURM_JOB_NODELIST"]
        )
        node_id = int(os.environ["SLURM_NODEID"])
        rank = node_id * gpus + local
        world = len(nodes) * gpus
        os.environ.setdefault("MASTER_ADDR", nodes[0])
        os.environ.setdefault("MASTER_PORT", os.environ.get("POOL_PROBE_PORT", "29517"))
        import torch
        import torch.distributed as dist

        torch.cuda.set_device(local)
        dist.init_process_group(
            "nccl", rank=rank, world_size=world, timeout=datetime.timedelta(seconds=300)
        )
        size = int(os.environ.get("POOL_PROBE_MB", "256")) * 2**20 // 2  # bf16 elements
        buf = torch.ones(size, dtype=torch.bfloat16, device="cuda")
        rounds_bw = []
        for pairs in pair_rounds(len(nodes)):
            # every rank must create every group (new_group is collective), in the same order
            groups = {}
            for a, b in pairs:
                for g in range(gpus):
                    groups[(a, b, g)] = dist.new_group([a * gpus + g, b * gpus + g])
            mine = next(((a, b) for a, b in pairs if node_id in (a, b)), None)
            bw = {}
            if mine is not None:
                grp = groups[(mine[0], mine[1], local)]
                for _ in range(3):
                    dist.all_reduce(buf, group=grp)
                torch.cuda.synchronize()
                iters = 10
                t0 = time.perf_counter()
                for _ in range(iters):
                    dist.all_reduce(buf, group=grp)
                torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) / iters
                # bus bandwidth of a 2-rank all-reduce = data size / time (2*(n-1)/n = 1)
                bw_local = buf.numel() * buf.element_size() / dt / 1e9
            else:
                bw_local = -1.0
            # gather every rank's figure; keep the worst GPU per pair
            t = torch.tensor([bw_local], device="cuda")
            allv = [torch.zeros(1, device="cuda") for _ in range(world)]
            dist.all_gather(allv, t)
            vals = [float(x.item()) for x in allv]
            for a, b in pairs:
                bw[(a, b)] = min(vals[a * gpus + g] for g in range(gpus))
            rounds_bw.append(bw)
            dist.barrier()
        if local == 0:
            per = worst_per_node(len(nodes), rounds_bw)[node_id]
            got = [v for v in per if v is not None]
            print(
                f"[nodecheck] OK {me} ib_gbps={max(got):.1f} ib_rounds="
                + ",".join("na" if v is None else f"{v:.1f}" for v in per),
                flush=True,
            )
        dist.destroy_process_group()
    except Exception as exc:  # the node's verdict, not the step's
        if local == 0:
            print(f"[nodecheck] FAIL {me}: ib probe: {exc}", flush=True)


if __name__ == "__main__":
    main()
