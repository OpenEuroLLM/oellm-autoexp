from megatron.core.datasets.indexed_dataset import IndexedDataset, IndexedDatasetBuilder
import datetime
from tqdm import tqdm
import numpy as np
import torch

# --- new imports ---
import argparse
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed


def _u01_from_seed_and_index(seed: int, index: int) -> float:
    """Deterministic U[0,1) from (seed, index), independent of threading."""
    h = hashlib.blake2b(digest_size=8)
    h.update(seed.to_bytes(8, "little", signed=False))
    h.update(index.to_bytes(8, "little", signed=False))
    x = int.from_bytes(h.digest(), "little", signed=False)  # 64-bit
    return x / 2**64


def _slice_bounds(n: int, workers: int, wid: int) -> tuple[int, int]:
    base = n // workers
    rem = n % workers
    start = wid * base + min(wid, rem)
    end = start + base + (1 if wid < rem else 0)
    return start, end

# Exit after iteration if exit_after_iteration is set
def _worker(wid: int, workers: int, path: str, ratio: float, out_prefix: str, seed: int, num_samples: int, exit_after_iteration: int = None) -> tuple[int, int, str]:
    start, end = _slice_bounds(num_samples, workers, wid)

    dataset = IndexedDataset(path)  # per-thread handle
    shard_path = f"{out_prefix}-shard{wid:02d}-of-{workers:02d}"
    builder = IndexedDatasetBuilder(shard_path + ".bin")

    kept = 0
    for i in tqdm(range(start, end), position=wid, leave=False, desc=f"shard {wid}"):
        if exit_after_iteration is not None and i >= exit_after_iteration:
            break
        if _u01_from_seed_and_index(seed, i) < ratio:
            builder.add_item(torch.tensor(dataset[i]))
            kept += 1

    builder.finalize(shard_path + ".idx")
    return wid, kept, shard_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--path", default="/pfs/lustrep4/scratch/project_462000963/preprocessed/gpt-neox-20b/nemotron-cc/1.0/high-synthetic-distill")
    p.add_argument("--ratio", type=float, default=0.1)
    p.add_argument("--out-prefix", default="data_sample_out_shuffled-downsampled")
    p.add_argument("--workers", type=int, default=min(8, (os.cpu_count() or 1)))
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--exit-after-iteration", type=int, default=None)
    args = p.parse_args()

    dataset = IndexedDataset(args.path)
    num_samples = len(dataset)

    out_prefix = f"{args.out_prefix}-{args.ratio:.6f}"
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)

    workers = max(1, args.workers)
    workers = min(workers, num_samples)  # avoid empty shards when tiny

    totals = [0] * workers
    shard_paths: list[str] = [""] * workers
    # print start time 
    print(f"Time at the start of downsampling: \n{datetime.datetime.now()}")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [
            ex.submit(_worker, wid, workers, args.path, args.ratio, out_prefix, args.seed, num_samples, args.exit_after_iteration)
            for wid in range(workers)
        ]
        for fut in as_completed(futs):
            wid, kept, shard_path = fut.result()
            totals[wid] = kept
            shard_paths[wid] = shard_path

    kept_total = sum(totals)
    print(f"num_samples={num_samples} ratio={args.ratio} workers={workers} kept_total={kept_total}")
    for wid in range(workers):
        print(f"  shard {wid:02d}: kept={totals[wid]} -> {shard_paths[wid]}.bin/.idx")

    print(f"Time at the end of downsampling: \n{datetime.datetime.now()}")

if __name__ == "__main__":
    main()