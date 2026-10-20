"""Host vs GPU time of one LDM forward + backward over all edges (the s1 shape), CuTe workspace
vs the torch one: is the 'cute' path launch (host) bound? From the repo root, one GPU:

    python scripts/korbi/campaign/debug/ldm_launch_bench.py
"""

import os
import sys
import time

import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../..")
sys.path.insert(0, os.path.join(ROOT, "submodules/Megatron-LM"))
sys.path.insert(0, os.path.join(ROOT, "submodules/cute_kernels"))
from cute_ldm.workspace import CuteDepthMemoryWorkspace  # noqa: E402
from megatron.core.transformer.depth_memory_state import TorchDepthMemoryWorkspace  # noqa: E402


def bench(ws, edges, n, g, k, d, reps=10):
    ws.ensure(n, "cuda")
    khat = torch.nn.functional.normalize(torch.randn(n, g, k, device="cuda"), dim=-1)
    query = torch.nn.functional.normalize(torch.randn(n, g, k, device="cuda"), dim=-1)
    value = torch.randn(n, g, d, device="cuda", dtype=torch.bfloat16)
    scale = torch.rand(n, g, device="cuda") + 0.5
    forget = torch.full((g,), 0.95, device="cuda")
    dr = torch.randn(n, g, d, device="cuda", dtype=torch.bfloat16)

    def once():
        for e in range(edges):
            ws.forward(e, khat, query, value, scale, forget)
        for e in reversed(range(edges)):
            ws.backward(e, dr)

    for _ in range(3):
        once()
    torch.cuda.synchronize()
    host = gpu = 0.0
    for _ in range(reps):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        t0 = time.perf_counter()
        once()
        host += time.perf_counter() - t0
        end.record()
        torch.cuda.synchronize()
        gpu += start.elapsed_time(end) / 1e3
    return host / reps * 1e3, gpu / reps * 1e3


if __name__ == "__main__":
    n, g, k = 8192, 16, 8
    for d, edges in ((48, 25), (64, 25), (128, 25)):
        for name, ws in (
            ("cute", CuteDepthMemoryWorkspace(total_edges=edges, groups=g, key_dim=k, value_dim=d)),
            ("torch", TorchDepthMemoryWorkspace(edges, g, k, d, 1.0)),
        ):
            h, t = bench(ws, edges, n, g, k, d)
            print(f"D{d:3d} {name:5s} edges {edges}: host {h:7.2f} ms  wall/GPU {t:7.2f} ms  "
                  f"({h / edges / 2 * 1e3:6.0f} us host per launch-step)", flush=True)
