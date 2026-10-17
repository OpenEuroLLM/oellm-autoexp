#!/usr/bin/env python3
"""Sidecar prefetcher: warm the OS page cache (MVP) with the ``.bin`` byte ranges a
Megatron run will read, in consumption order, so the training mmap reader faults
from RAM instead of the stall-prone shared filesystem.

Zero Megatron changes: this reads the SAME inode the trainer mmaps, so a plain
``os.pread`` here populates the page cache the trainer's faults will hit. Within a
bounded look-ahead horizon, reads are issued in file-offset order so even the
backing-store I/O is near-sequential (GPFS-friendly) instead of random 4 KB faults.

Launch once per node (``srun --ntasks-per-node=1 ... python scripts/prefetch_warm.py &``)
BEFORE the training srun; it runs concurrently with training.

NOTE: the rank->sample mapping assumes TP=PP=1 (dp_rank == global_rank). For other
layouts, pass --node-dp-ranks explicitly (ranks sharing a TP/PP group read the same
data, so list each distinct dp rank on the node once).
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oellm_autoexp.data_staging import prefetch_order as po  # noqa: E402

HOST = socket.gethostname()


def log(msg: str) -> None:
    print(f"[prefetch_warm][{HOST}] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prefix", required=True, help="dataset prefix (without .bin/.idx)")
    p.add_argument(
        "--cache-dir", required=True, help="dir with the cached *-GPTDataset-* .npy files"
    )
    p.add_argument(
        "--hash", default=None, help="index description md5 hash (auto-discovered if omitted)"
    )
    p.add_argument("--split", default="train")
    p.add_argument("--dp-size", type=int, required=True, help="data-parallel world size")
    p.add_argument("--micro-batch-size", type=int, required=True)
    p.add_argument(
        "--node-dp-ranks",
        default=None,
        help="comma list of dp ranks this node serves; default derives from SLURM (TP=PP=1)",
    )
    p.add_argument("--gpus-per-node", type=int, default=int(os.environ.get("GPUS_PER_NODE", 4)))
    p.add_argument("--start-sample", type=int, default=0)
    p.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="total samples in the run (default: len(shuffle_index))",
    )
    p.add_argument(
        "--budget-gb", type=float, default=200.0, help="oneshot: stop after warming this many GiB"
    )
    p.add_argument(
        "--horizon-mb",
        type=float,
        default=512.0,
        help="sort/coalesce reads within this many MiB of access stream",
    )
    p.add_argument("--chunk-mb", type=float, default=8.0, help="pread chunk size")
    p.add_argument(
        "--max-mbps", type=float, default=0.0, help="throttle to this MiB/s (0 = unlimited)"
    )
    p.add_argument(
        "--fadvise", action="store_true", help="use posix_fadvise(WILLNEED) instead of pread"
    )
    p.add_argument(
        "--mode",
        choices=["oneshot", "mirror", "lanes"],
        default="oneshot",
        help="oneshot: warm page cache up to --budget-gb then exit. "
        "mirror: fill a node-local mirror by walking the exact consumption order. "
        "lanes: fill a node-local mirror by staging K contiguous lanes sequentially, "
        "round-robin by block (for the locality-preserving windowed shuffle) -- fast sequential reads.",
    )
    p.add_argument(
        "--lanes",
        type=int,
        default=int(os.environ.get("OELLM_SHUFFLE_LANES", "256")),
        help="lanes mode: number of lanes (match OELLM_SHUFFLE_LANES used by the shuffle)",
    )
    p.add_argument(
        "--lane-block-docs",
        type=int,
        default=32768,
        help="lanes mode: documents per lane block (one sequential read)",
    )
    p.add_argument(
        "--train-doc-frac",
        type=float,
        default=float(os.environ.get("OELLM_TRAIN_DOC_FRAC", "1.0")),
        help="lanes mode: fraction of leading documents that form the train split (from --split); "
        "lanes are defined over [0, frac*N) so they align with training's sample lanes",
    )
    p.add_argument(
        "--max-lane-blocks",
        type=int,
        default=int(os.environ.get("OELLM_MAX_LANE_BLOCKS", "0")),
        help="lanes mode: stop after staging this many blocks per lane (0 = whole file). Bounds the "
        "over-fetch for short runs in lieu of a consumption cursor.",
    )
    p.add_argument(
        "--cursor-file",
        default=os.environ.get("OELLM_CURSOR_FILE", ""),
        help="lanes mode: node-local file where training writes consumed_train_samples. When set, "
        "staging/eviction are paced to it (sliding window) instead of one-shot capped staging.",
    )
    p.add_argument(
        "--lookahead-blocks",
        type=int,
        default=int(os.environ.get("OELLM_LOOKAHEAD_BLOCKS", "3")),
        help="cursor mode: rounds to keep staged AHEAD of the consumption cursor",
    )
    p.add_argument(
        "--retain-blocks",
        type=int,
        default=int(os.environ.get("OELLM_RETAIN_BLOCKS", "1")),
        help="cursor mode: rounds to keep staged BEHIND the cursor before evicting",
    )
    p.add_argument(
        "--mirror-dir",
        default=None,
        help="mirror mode: node-local dir for the sparse mirror + bitmap",
    )
    p.add_argument(
        "--passes",
        type=int,
        default=1,
        help="mirror mode: how many times to re-walk the access order (keeps the buffer fresh as training advances)",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=16,
        help="mirror mode: concurrent block-copy threads (scales GPFS read throughput)",
    )
    p.add_argument(
        "--wait-for-cache",
        type=float,
        default=0.0,
        help="poll up to N seconds for the index cache to appear (training builds it at startup)",
    )
    return p.parse_args()


def wait_for_cache(cache_dir: str, split: str, timeout: float) -> None:
    """Block until the split's shuffle_index .npy appears (training builds it),
    or timeout."""
    import glob

    pattern = os.path.join(cache_dir, f"*-GPTDataset-{split}-shuffle_index.npy")
    deadline = time.time() + timeout
    announced = False
    while not glob.glob(pattern):
        if time.time() >= deadline:
            raise TimeoutError(f"index cache not found in {cache_dir} after {timeout}s")
        if not announced:
            log(f"waiting for index cache in {cache_dir} ...")
            announced = True
        time.sleep(2.0)


def resolve_node_dp_ranks(args) -> list[int]:
    if args.node_dp_ranks:
        ranks = [int(x) for x in args.node_dp_ranks.split(",") if x != ""]
    else:
        node_id = int(os.environ.get("SLURM_NODEID", 0))
        gpn = args.gpus_per_node
        ranks = list(range(node_id * gpn, (node_id + 1) * gpn))
    return [r for r in ranks if 0 <= r < args.dp_size]


def coalesce(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Sort by offset and merge overlapping/adjacent [off, off+len) ranges."""
    ranges.sort()
    merged: list[tuple[int, int]] = []
    for off, nbytes in ranges:
        if nbytes <= 0:
            continue
        end = off + nbytes
        if merged and off <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((off, end))
    return [(o, e - o) for o, e in merged]


def warm_ranges(fd: int, ranges: list[tuple[int, int]], chunk: int, fadvise: bool) -> int:
    """Read (or advise) the given byte ranges to populate the page cache.

    Returns bytes touched.
    """
    total = 0
    for off, nbytes in ranges:
        if fadvise:
            os.posix_fadvise(fd, off, nbytes, os.POSIX_FADV_WILLNEED)
            total += nbytes
            continue
        pos, remaining = off, nbytes
        while remaining > 0:
            n = min(chunk, remaining)
            data = os.pread(fd, n, pos)
            if not data:
                break
            pos += len(data)
            remaining -= len(data)
            total += len(data)
    return total


def read_cursor(cursor_file):
    """Read training's global consumed_train_samples from the node-local cursor
    file (None if absent)."""
    if not cursor_file:
        return None
    try:
        with open(cursor_file) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


class _LaneState:
    """Cursor-paced sliding-window staging for ONE dataset file (sub-dataset of
    a blend, or the.

    single dataset). A "round" c stages block c (LBD docs) of every
    lane. The GLOBAL cursor maps to this file via its blend weight
    (file_consumed = global_consumed * weight), then to a round; we keep
    rounds [cursor-retain, cursor+lookahead] resident (block-aligned),
    evict the rest, and stage within this file's budget share. Reads
    fall back to source, so it is correct at any fill state.
    """

    def __init__(self, args, prefix, bounds, epoch_samples, docs_per_sample, weight, budget, pool):
        from oellm_autoexp.data_staging.mirror import BLOCK_SIZE, MirrorWriter, window0_sentinel

        self.writer = MirrorWriter(prefix + ".bin", args.mirror_dir)
        self.sentinel = window0_sentinel(args.mirror_dir, prefix)
        self.bounds = bounds
        self.K = len(bounds) - 1
        self.Me = max(1, int(epoch_samples))
        self.r = float(docs_per_sample)
        self.weight = float(weight)
        self.LBD = max(1, args.lane_block_docs)
        self.budget = int(budget)
        self.lookahead, self.retain = max(0, args.lookahead_blocks), max(0, args.retain_blocks)
        self.cursor_on = bool(args.cursor_file)
        self.maxblocks = max(
            (bounds[j + 1] - bounds[j] + self.LBD - 1) // self.LBD for j in range(self.K)
        )
        if args.max_lane_blocks > 0 and not self.cursor_on:
            self.maxblocks = min(self.maxblocks, args.max_lane_blocks)
        self.round_ranges: dict = {}
        self.resident = 0
        self.fetched = 0
        self.sentinel_written = os.path.exists(self.sentinel)
        self.name = os.path.basename(prefix)
        self._pool = pool
        self._BS = BLOCK_SIZE

    def _do(self, ab):
        a, b = ab
        off, n = self.writer.run_byte_range(a, b)
        aoff = (off // self._BS) * self._BS
        aend = min(((off + n + self._BS - 1) // self._BS) * self._BS, self.writer.bin_size)
        return self.writer.copy_range(aoff, aend - aoff), aoff, aend - aoff

    def _stage_round(self, c):
        batch = []
        for j in range(self.K):
            a = self.bounds[j] + c * self.LBD
            if a >= self.bounds[j + 1]:
                continue
            batch.append((a, min(a + self.LBD, self.bounds[j + 1]) - 1))
        ranges = []
        for nbytes, aoff, an in self._pool.map(self._do, batch) if batch else []:
            if nbytes <= 0:
                continue
            self.writer.mark_range(aoff, an)
            ranges.append((aoff, an))
            self.resident += nbytes
            self.fetched += nbytes
        self.round_ranges[c] = ranges

    def _evict_round(self, c):
        for off, n in self.round_ranges.pop(c, []):
            self.writer.evict_range(off, n)
            self.resident -= n

    def cursor_round(self, global_consumed):
        if global_consumed is None:
            return 0
        file_consumed = int(global_consumed * self.weight)
        return int(((file_consumed % self.Me) // self.K * self.r) // self.LBD)

    def step(self, global_consumed):
        cr = self.cursor_round(global_consumed)
        target = min(self.maxblocks - 1, cr + self.lookahead)
        lo = max(0, cr - self.retain)
        for c in list(self.round_ranges):  # evict outside the window first (frees disk)
            if c < lo or c > target:
                self._evict_round(c)
        for c in range(lo, target + 1):  # stage missing rounds within this file's budget
            if c in self.round_ranges or self.resident >= self.budget:
                continue
            self._stage_round(c)
            if c == 0 and not self.sentinel_written:
                open(self.sentinel, "w").close()  # release this file's pre-stage barrier
                self.sentinel_written = True
        return cr


def run_lanes_multi(args, file_specs) -> int:
    """Cursor-paced sliding-window prefetch over one OR MANY dataset files (a
    Megatron blend).

    Each sub-dataset gets its own node-local mirror + lanes (the read-
    through reader is per IndexedDataset), and the shared budget is
    split across files. file_specs: list of (prefix, bounds, Me, r,
    weight).
    """
    from concurrent.futures import ThreadPoolExecutor

    pool = ThreadPoolExecutor(max_workers=args.threads)
    F = max(1, len(file_specs))
    per_file_budget = int(args.budget_gb * (1 << 30)) // F
    states = [_LaneState(args, *spec, per_file_budget, pool) for spec in file_specs]
    log(
        f"lanes: files={F} budget/file={per_file_budget / (1 << 30):.0f}GiB cursor={'on' if args.cursor_file else 'off'} "
        f"lookahead={args.lookahead_blocks} retain={args.retain_blocks} threads={args.threads} "
        f"weights=[{','.join(f'{s.name}:{s.weight:.2f}' for s in states)}]"
    )

    # Prime: stage round 0 + sentinel of EVERY file first so all per-file barriers release promptly.
    for st in states:
        st.step(0)
    t0 = last_log = time.time()
    while True:
        consumed = read_cursor(args.cursor_file)
        for st in states:
            st.step(consumed)
        now = time.time()
        if now - last_log >= 5.0:
            last_log = now
            tot_res = sum(s.resident for s in states)
            tot_f = sum(s.fetched for s in states)
            detail = " ".join(
                f"{s.name}:r{s.cursor_round(consumed)}/{s.resident / (1 << 30):.1f}G"
                for s in states
            )
            log(
                f"lanes: consumed={consumed} resident={tot_res / (1 << 30):.1f}GiB "
                f"fetched={tot_f / (1 << 30):.1f}GiB ({tot_f / (1 << 20) / max(now - t0, 1e-3):.0f} MiB/s) {detail}"
            )
        time.sleep(0.5 if args.cursor_file else 2.0)


def run_mirror(args, bin_path, make_doc_stream) -> int:
    """Continuously fill a node-local sparse mirror with the exact bytes of
    each document the node consumes, in consumption order, bounded to --budget-
    gb via FIFO eviction.

    Within each horizon, documents are fetched offset-sorted
    (sequential-ish GPFS reads) by a thread pool. Training reads through
    the mirror; documents not yet present are served from the source, so
    it is correct at any fill state. Document granularity avoids the
    ~500x block amplification a globally-shuffled dataset would
    otherwise incur.
    """
    import collections
    from concurrent.futures import ThreadPoolExecutor

    from oellm_autoexp.data_staging.mirror import MirrorWriter

    writer = MirrorWriter(bin_path, args.mirror_dir)
    budget = int(args.budget_gb * (1 << 30))
    rate = args.max_mbps * (1 << 20) if args.max_mbps > 0 else 0
    pool = ThreadPoolExecutor(max_workers=args.threads)
    horizon = max(1, args.threads * 64)  # documents fetched per offset-sorted batch
    log(
        f"mirror: dir={args.mirror_dir} bin_size={writer.bin_size / (1 << 30):.1f}GiB "
        f"docs={writer.n_docs} budget={args.budget_gb}GiB threads={args.threads} passes={args.passes}"
    )

    fifo: collections.deque[tuple[int, int]] = collections.deque()  # (byte_offset, nbytes)
    resident_bytes = 0
    fetched_bytes = 0
    n_fetched = 0
    t0 = time.time()
    last_log = t0

    def flush(docs):
        nonlocal resident_bytes, fetched_bytes, n_fetched, last_log
        if not docs:
            return
        docs.sort()  # doc id == file position -> sort gives increasing offset
        # Coalesce runs of consecutive doc ids into one contiguous (sequential) read each.
        runs = []  # (start_doc, end_doc) inclusive
        for d in docs:
            if runs and d == runs[-1][1] + 1:
                runs[-1] = (runs[-1][0], d)
            else:
                runs.append((d, d))

        def do(r):
            off, nb = writer.run_byte_range(*r)
            return writer.copy_range(off, nb), off, nb

        for n, off, nb in pool.map(do, runs):
            if n <= 0:
                continue
            writer.mark_range(off, nb)  # mark blocks AFTER data written
            fifo.append((off, nb))
            resident_bytes += n
            n_fetched += 1
            while resident_bytes > budget and fifo:
                eo, en = fifo.popleft()
                writer.evict_range(eo, en)
                resident_bytes -= en
            fetched_bytes += n
        if rate:
            target = fetched_bytes / rate
            elapsed = time.time() - t0
            if elapsed < target:
                time.sleep(target - elapsed)
        now = time.time()
        if now - last_log >= 5.0:
            last_log = now
            log(
                f"mirror: resident={resident_bytes / (1 << 30):.2f}GiB fetched={fetched_bytes / (1 << 30):.2f}GiB "
                f"docs={n_fetched} ({fetched_bytes / (1 << 20) / max(now - t0, 1e-3):.0f} MiB/s)"
            )

    for _ in range(args.passes):
        batch: list[int] = []
        seen: set[int] = set()
        for doc in make_doc_stream():
            if doc in seen or writer.bitmap.is_set(doc):
                continue
            seen.add(doc)
            batch.append(doc)
            if len(batch) >= horizon:
                flush(batch)
                batch, seen = [], set()
        flush(batch)
    pool.shutdown(wait=True)
    dt = max(time.time() - t0, 1e-3)
    log(
        f"mirror done: resident={resident_bytes / (1 << 30):.2f}GiB fetched={fetched_bytes / (1 << 30):.2f}GiB "
        f"docs={n_fetched} in {dt:.0f}s ({fetched_bytes / (1 << 20) / dt:.0f} MiB/s)"
    )
    return 0


def main() -> int:
    args = parse_args()
    bin_path = args.prefix + ".bin"
    idx_path = args.prefix + ".idx"

    if args.mode == "lanes":
        # lanes mode stages each shuffle lane's contiguous doc range, with exact sample_index lane
        # boundaries. --prefix may be a comma-separated list of blend sub-datasets; each gets its own
        # mirror + lanes and the cursor maps to it via its blend weight. Training builds every
        # sub-dataset's cache (plus the BlendedDataset weight cache) at startup, so we retry discovery
        # until ALL of them are present (waiting on just the first prefix's cache would race).
        import glob as _glob

        prefixes = [p for p in args.prefix.split(",") if p]
        deadline = time.time() + max(args.wait_for_cache, 0.0)
        announced = False
        while True:
            try:
                if len(prefixes) > 1:
                    if not _glob.glob(
                        os.path.join(
                            args.cache_dir, f"*-BlendedDataset-{args.split}-dataset_index.npy"
                        )
                    ):
                        raise FileNotFoundError("BlendedDataset weight cache not ready")
                    hashes = po.discover_hashes_by_prefix(args.cache_dir, prefixes, args.split)
                    weights = po.load_blend_weights(args.cache_dir, prefixes, args.split)
                else:
                    hashes = {
                        prefixes[0]: args.hash or po.discover_desc_hash(args.cache_dir, args.split)
                    }
                    weights = {prefixes[0]: 1.0}
                specs = []
                for prefix in prefixes:
                    di, si, shi = po.load_cached_indices(args.cache_dir, hashes[prefix], args.split)
                    Me, K, bounds, r = po.compute_lane_bounds(
                        di, si, shi.shape[0], args.lanes, args.lane_block_docs
                    )
                    specs.append((prefix, bounds, Me, r, weights[prefix], K))
                break
            except (ValueError, FileNotFoundError, OSError) as e:
                if time.time() >= deadline:
                    raise
                if not announced:
                    log(f"waiting for blend cache in {args.cache_dir} ({e}) ...")
                    announced = True
                time.sleep(3.0)
        for prefix, bounds, Me, r, w, K in specs:
            log(
                f"lanes file {os.path.basename(prefix)}: epoch_samples={Me} K={K} docs/sample={r:.2f} "
                f"weight={w:.3f} hash={hashes[prefix]}"
            )
        return run_lanes_multi(args, [s[:5] for s in specs])

    if args.wait_for_cache > 0 and not args.hash:
        wait_for_cache(args.cache_dir, args.split, args.wait_for_cache)
    desc_hash = args.hash or po.discover_desc_hash(args.cache_dir, args.split)
    log(f"using index hash {desc_hash} split={args.split}")
    idx_file = po.IdxFile(idx_path)
    document_index, sample_index, shuffle_index = po.load_cached_indices(
        args.cache_dir, desc_hash, args.split
    )
    node_dp_ranks = resolve_node_dp_ranks(args)
    num_samples = args.num_samples or len(shuffle_index)
    log(
        f"dp_size={args.dp_size} node_dp_ranks={node_dp_ranks} mbs={args.micro_batch_size} "
        f"samples={num_samples} budget={args.budget_gb} GiB bin={bin_path}"
    )

    def make_stream():
        return po.iter_access_ranges(
            idx_file,
            document_index,
            sample_index,
            shuffle_index,
            args.dp_size,
            node_dp_ranks,
            args.micro_batch_size,
            args.start_sample,
            num_samples,
        )

    if args.mode == "mirror":

        def make_doc_stream():
            return po.iter_access_docs(
                document_index,
                sample_index,
                shuffle_index,
                args.dp_size,
                node_dp_ranks,
                args.micro_batch_size,
                args.start_sample,
                num_samples,
            )

        return run_mirror(args, bin_path, make_doc_stream)

    fd = os.open(bin_path, os.O_RDONLY)
    budget = int(args.budget_gb * (1 << 30))
    horizon = int(args.horizon_mb * (1 << 20))
    chunk = int(args.chunk_mb * (1 << 20))
    rate = args.max_mbps * (1 << 20) if args.max_mbps > 0 else 0

    stream = po.iter_access_ranges(
        idx_file,
        document_index,
        sample_index,
        shuffle_index,
        args.dp_size,
        node_dp_ranks,
        args.micro_batch_size,
        args.start_sample,
        num_samples,
    )

    t0 = time.time()
    warmed = 0
    pending: list[tuple[int, int]] = []
    pending_bytes = 0

    def flush() -> None:
        nonlocal warmed, pending, pending_bytes
        if not pending:
            return
        merged = coalesce(pending)
        warmed += warm_ranges(fd, merged, chunk, args.fadvise)
        pending = []
        pending_bytes = 0
        if rate:
            target = warmed / rate
            elapsed = time.time() - t0
            if elapsed < target:
                time.sleep(target - elapsed)

    try:
        for off, nbytes in stream:
            pending.append((off, nbytes))
            pending_bytes += nbytes
            if pending_bytes >= horizon:
                flush()
                if warmed >= budget:
                    log(f"reached budget after {warmed / (1 << 30):.1f} GiB")
                    break
        else:
            flush()
            log("warmed entire access stream")
    finally:
        os.close(fd)

    dt = max(time.time() - t0, 1e-3)
    log(
        f"done: warmed {warmed / (1 << 30):.2f} GiB in {dt:.0f}s ({warmed / (1 << 20) / dt:.0f} MiB/s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
