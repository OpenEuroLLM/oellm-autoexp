#!/usr/bin/env python3
"""One reader process for the GPFS contention benchmark.

Reproduces Megatron's data-path load: small random reads scattered across a large file (the global
shuffle makes every sample a random seek). Many of these run concurrently (srun --ntasks-per-node=P
across N nodes) to recreate the "many workers random-faulting one huge GPFS file" situation, so we
can compare reading DIRECT from the shared GPFS file vs from a bounded NODE-LOCAL window.

Each process does timed random `pread`s and prints one aggregate RESULT line; parse_contention_bench.py
aggregates across all tasks. Pure stdlib (no numpy/torch) so startup is instant and no venv is needed.
"""

import argparse
import os
import random
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--path", required=True, help="file to read (GPFS merged.bin, or node-local window)"
    )
    ap.add_argument(
        "--span-gb",
        type=float,
        default=0.0,
        help="restrict reads to the first N GiB (0 = whole file)",
    )
    ap.add_argument(
        "--read-bytes",
        type=int,
        default=16384,
        help="bytes per random read (~a sample's byte range)",
    )
    ap.add_argument("--duration", type=float, default=20.0, help="measured seconds")
    ap.add_argument("--warmup", type=float, default=3.0, help="unmeasured warmup seconds")
    ap.add_argument(
        "--arm", default="?", help="label: direct-global | direct-window | mirror-window"
    )
    ap.add_argument(
        "--procs", type=int, default=0, help="procs-per-node for this step (for the RESULT line)"
    )
    a = ap.parse_args()

    rank = int(os.environ.get("SLURM_PROCID", 0))
    node = int(os.environ.get("SLURM_NODEID", 0))
    fd = os.open(a.path, os.O_RDONLY)
    size = os.fstat(fd).st_size
    span = int(a.span_gb * (1 << 30)) if a.span_gb > 0 else size
    span = min(span, size)
    hi = max(1, span - a.read_bytes)
    rnd = random.Random(1234 + rank)  # distinct offsets per worker (like distinct dp ranks)

    deadline = time.time() + a.warmup
    while time.time() < deadline:
        off = rnd.randrange(0, hi)
        os.pread(fd, a.read_bytes, off - off % a.read_bytes)

    lat = []
    reads = nbytes = 0
    t0 = time.time()
    deadline = t0 + a.duration
    while time.time() < deadline:
        off = rnd.randrange(0, hi)
        ts = time.perf_counter()
        d = os.pread(fd, a.read_bytes, off - off % a.read_bytes)
        lat.append(time.perf_counter() - ts)
        reads += 1
        nbytes += len(d)
    el = time.time() - t0
    os.close(fd)

    lat.sort()

    def pct(p):
        return lat[min(len(lat) - 1, int(p * len(lat)))] * 1e3 if lat else 0.0

    mean_ms = (sum(lat) / len(lat) * 1e3) if lat else 0.0
    print(
        f"RESULT arm={a.arm} procs={a.procs} node={node} rank={rank} reads={reads} "
        f"bytes={nbytes} elapsed={el:.3f} p50_ms={pct(0.50):.3f} p99_ms={pct(0.99):.3f} mean_ms={mean_ms:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
