#!/usr/bin/env python3
"""Per-node GPU speed probe for a pool host's pre-run check (pool setting precheck_command).

Times a bf16 GEMM on every visible GPU (~2-3 s each) and prints one line in
scripts/nodecheck.sh's format, with the node's speed as its SLOWEST GPU:

    [nodecheck] OK <node> gemm_tflops=<min> gemm_per_gpu=<a,b,c,d>
    [nodecheck] FAIL <node>: gemm probe: <error>

A GPU stuck at low clocks or thermally throttled makes every rank of a run wait for it, so the
slowest GPU is what matters. Run inside the training container, one task per node, e.g.

    apptainer exec --nv <sif> python3 scripts/pool/gpu_gemm_probe.py
"""

from __future__ import annotations

import os
import socket
import time


def main():
    node = os.environ.get("SLURMD_NODENAME") or socket.gethostname().split(".")[0]
    try:
        import torch

        n = torch.cuda.device_count()
        if n == 0:
            raise RuntimeError("no CUDA device visible")
        size, iters, per_gpu = 8192, 30, []
        for dev in range(n):
            with torch.cuda.device(dev):
                a = torch.randn(size, size, device="cuda", dtype=torch.bfloat16)
                b = torch.randn(size, size, device="cuda", dtype=torch.bfloat16)
                for _ in range(5):
                    a @ b
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(iters):
                    a @ b
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                per_gpu.append(2 * size**3 * iters / dt / 1e12)
        print(
            f"[nodecheck] OK {node} gemm_tflops={min(per_gpu):.0f}"
            f" gemm_per_gpu={','.join(f'{x:.0f}' for x in per_gpu)}",
            flush=True,
        )
    except Exception as exc:  # any failure here is the node's verdict, not the step's
        print(f"[nodecheck] FAIL {node}: gemm probe: {exc}", flush=True)


if __name__ == "__main__":
    main()
