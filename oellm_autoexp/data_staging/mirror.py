"""Node-local sparse mirror of a Megatron ``.bin`` with a per-BLOCK presence
bitmap.

The prefetcher (writer) stages contiguous byte ranges of the .bin into a node-local sparse
mirror; the training bin reader (read-through) serves a read from the local mirror iff every
1 MiB block it spans is present, else falls back to the shared filesystem. Both sides
coordinate through one mmap'd bitmap on node-local disk.

Block granularity (not per-document) is what makes the READER cheap: it maps a read's byte
offset to block ids by integer division -- no .idx lookup, no binary search, no GPFS page
faults on the hot path. It is the right unit here because the locality-preserving windowed
shuffle (see windowed_shuffle.py) makes the prefetcher stage large CONTIGUOUS regions, so
there is no block amplification: whole blocks are filled by big sequential reads.

Correctness invariant (writer): stage the bytes to the mirror, THEN mark the blocks fully
covered by the staged range. A reader that observes a block bit set is guaranteed the block's
bytes are present (pwrite returned). Partial edge blocks of a staged range are left unmarked
(served from source) -- correct, and negligible for large staged ranges. The reader always
has the source open, so it is correct at any fill state.
"""

from __future__ import annotations

import os

import numpy as np

BLOCK_SIZE = 1 << 20  # 1 MiB blocks


def mirror_paths(mirror_dir: str, prefix: str) -> tuple[str, str]:
    name = os.path.basename(prefix)
    return os.path.join(mirror_dir, name + ".bin"), os.path.join(mirror_dir, name + ".bitmap")


def window0_sentinel(mirror_dir: str, prefix: str) -> str:
    """Path of the 'first window staged' sentinel the prefetcher writes after
    staging block 0 of all lanes; the reader blocks on it (pre-stage barrier)
    so training doesn't race ahead into GPFS."""
    return os.path.join(mirror_dir, os.path.basename(prefix) + ".window0_ready")


def num_blocks(bin_size: int) -> int:
    return (bin_size + BLOCK_SIZE - 1) // BLOCK_SIZE


def blocks_for_range(offset: int, nbytes: int) -> tuple[int, int]:
    """Inclusive [first_block, last_block] covering byte range [offset,
    offset+nbytes)."""
    return offset // BLOCK_SIZE, (offset + nbytes - 1) // BLOCK_SIZE


def _load_idx(prefix: str):
    from oellm_autoexp.data_staging.prefetch_order import IdxFile

    return IdxFile(prefix + ".idx")


class Bitmap:
    """Mmap-backed bit-per-block presence map shared across processes on a
    node."""

    def __init__(self, path: str, nbits: int, writable: bool) -> None:
        import mmap as _mmap

        self.nbits = nbits
        nbytes = (nbits + 7) // 8
        flags = os.O_RDWR | os.O_CREAT if writable else os.O_RDONLY
        fd = os.open(path, flags, 0o644)
        if writable and os.fstat(fd).st_size < nbytes:
            os.ftruncate(fd, nbytes)
        prot = (_mmap.PROT_READ | _mmap.PROT_WRITE) if writable else _mmap.PROT_READ
        self._mm = _mmap.mmap(fd, nbytes, prot=prot)
        os.close(fd)
        self._buf = memoryview(self._mm)

    def is_set(self, i: int) -> bool:
        return bool(self._buf[i >> 3] & (1 << (i & 7)))

    def all_set(self, a: int, b: int) -> bool:
        for i in range(a, b + 1):
            if not (self._buf[i >> 3] & (1 << (i & 7))):
                return False
        return True

    def set_range(self, a: int, b: int) -> None:
        """Set bits [a, b] inclusive (fast, byte-wise)."""
        if b < a:
            return
        fb, lb = a >> 3, b >> 3
        if fb == lb:
            self._buf[fb] |= ((0xFF << (a & 7)) & (0xFF >> (7 - (b & 7)))) & 0xFF
            return
        self._buf[fb] |= (0xFF << (a & 7)) & 0xFF
        if lb > fb + 1:
            self._buf[fb + 1 : lb] = b"\xff" * (lb - fb - 1)
        self._buf[lb] |= (0xFF >> (7 - (b & 7))) & 0xFF

    def clear_range(self, a: int, b: int) -> None:
        if b < a:
            return
        fb, lb = a >> 3, b >> 3
        if fb == lb:
            self._buf[fb] &= (~((0xFF << (a & 7)) & (0xFF >> (7 - (b & 7))))) & 0xFF
            return
        self._buf[fb] &= (~(0xFF << (a & 7))) & 0xFF
        if lb > fb + 1:
            self._buf[fb + 1 : lb] = b"\x00" * (lb - fb - 1)
        self._buf[lb] &= (~(0xFF >> (7 - (b & 7)))) & 0xFF

    def count(self) -> int:
        return int(np.unpackbits(np.frombuffer(self._mm, dtype=np.uint8)).sum())


class LocalMirrorBinReader:
    """Drop-in Megatron _BinReader: read from the node-local mirror when every block the read
    spans is present, else from the source (GPFS) .bin. Cheap (offset->block by division, no
    .idx). Correct at any fill state; degrades to source-only when the mirror does not yet exist."""

    def __init__(self, source_bin_path: str, mirror_dir: str) -> None:
        prefix = source_bin_path[:-4]  # strip ".bin"
        self._mirror_bin, self._bitmap_path = mirror_paths(mirror_dir, prefix)
        self._bin_size = os.path.getsize(source_bin_path)
        self._nblocks = num_blocks(self._bin_size)
        self._src_fd = os.open(source_bin_path, os.O_RDONLY)
        self._mirror_fd: int | None = None
        self._bitmap: Bitmap | None = None
        self._tried = False
        self.local_hits = 0
        self.source_reads = 0
        self._log_every = int(os.environ.get("OELLM_MIRROR_LOG_EVERY", "100000"))
        # pre-stage barrier: block the first read until the prefetcher signals window 0 is staged
        self._barrier = os.environ.get("OELLM_PREFETCH_BARRIER") == "1"
        self._barrier_timeout = float(os.environ.get("OELLM_PREFETCH_BARRIER_TIMEOUT", "600"))
        self._sentinel = window0_sentinel(mirror_dir, prefix)
        self._barrier_done = False

    def _try_open_mirror(self) -> None:
        if self._bitmap is not None:
            return
        if os.path.exists(self._mirror_bin) and os.path.exists(self._bitmap_path):
            try:
                self._mirror_fd = os.open(self._mirror_bin, os.O_RDONLY)
                self._bitmap = Bitmap(self._bitmap_path, self._nblocks, writable=False)
            except OSError:
                self._mirror_fd = None
                self._bitmap = None

    def _maybe_log(self) -> None:
        total = self.local_hits + self.source_reads
        if self._log_every and total % self._log_every == 0:
            pct = 100.0 * self.local_hits / total
            print(
                f"[mirror-reader pid={os.getpid()}] reads={total} local_hits={self.local_hits} "
                f"({pct:.1f}% local) source_reads={self.source_reads}",
                flush=True,
            )

    def _barrier_wait(self) -> None:
        """Block the first read until the prefetcher has staged window 0 (block
        0 of all lanes), so training reads the staged mirror instead of racing
        ahead into GPFS.

        Bounded by a timeout so a missing/failed prefetcher degrades to
        source reads rather than deadlocking.
        """
        if self._barrier_done:
            return
        self._barrier_done = True
        if not self._barrier:
            return
        import time

        deadline = time.time() + self._barrier_timeout
        announced = False
        while not os.path.exists(self._sentinel):
            if time.time() >= deadline:
                print(
                    f"[mirror-reader pid={os.getpid()}] barrier timeout; proceeding with source reads",
                    flush=True,
                )
                return
            if not announced:
                print(
                    f"[mirror-reader pid={os.getpid()}] waiting for window-0 prefetch barrier",
                    flush=True,
                )
                announced = True
            time.sleep(0.5)
        self._try_open_mirror()  # window 0 is staged; open the mirror now

    def read(self, dtype, count: int, offset: int) -> np.ndarray:
        self._barrier_wait()
        nbytes = count * np.dtype(dtype).itemsize
        if self._bitmap is None and not self._tried:
            self._try_open_mirror()
            self._tried = True
        if self._bitmap is not None and self._mirror_fd is not None:
            b0, b1 = blocks_for_range(offset, nbytes)
            if self._bitmap.all_set(b0, b1):
                self.local_hits += 1
                self._maybe_log()
                return np.frombuffer(
                    _pread_exact(self._mirror_fd, nbytes, offset), dtype=dtype, count=count
                )
        self.source_reads += 1
        if self._bitmap is None and (self.source_reads & 0x3FF) == 0:
            self._try_open_mirror()
        self._maybe_log()
        return np.frombuffer(_pread_exact(self._src_fd, nbytes, offset), dtype=dtype, count=count)


class MirrorWriter:
    """Stages contiguous byte ranges of the source .bin into the node-local
    sparse mirror and marks the fully-covered 1 MiB blocks present."""

    def __init__(self, source_bin_path: str, mirror_dir: str) -> None:
        os.makedirs(mirror_dir, exist_ok=True)
        prefix = source_bin_path[:-4]
        self._src_fd = os.open(source_bin_path, os.O_RDONLY)
        self.bin_size = os.fstat(self._src_fd).st_size
        self._idx = _load_idx(prefix)
        self.n_docs = self._idx.sequence_count
        self.n_blocks = num_blocks(self.bin_size)
        self._mirror_bin, self._bitmap_path = mirror_paths(mirror_dir, prefix)
        mfd = os.open(self._mirror_bin, os.O_RDWR | os.O_CREAT, 0o644)
        if os.fstat(mfd).st_size < self.bin_size:
            os.ftruncate(mfd, self.bin_size)  # sparse: no physical space until written
        self._mirror_fd = mfd
        self.bitmap = Bitmap(self._bitmap_path, self.n_blocks, writable=True)

    def run_byte_range(self, a: int, b: int) -> tuple[int, int]:
        """Byte (offset, length) spanning documents [a, b] inclusive
        (contiguous in the .bin)."""
        off = int(self._idx.sequence_pointers[a])
        end = (
            int(self._idx.sequence_pointers[b])
            + int(self._idx.sequence_lengths[b]) * self._idx.itemsize
        )
        return off, end - off

    def copy_range(self, offset: int, nbytes: int) -> int:
        """Copy a contiguous byte range source->mirror; no bit set.

        Thread-safe (positional I/O). Plain pread+pwrite (benchmarks
        ~1.6 GiB/s at 8 threads here; os.copy_file_range is both
        unavailable in this Python build and cross-fs EXDEV, so not
        used).
        """
        if nbytes <= 0:
            return 0
        _pwrite_exact(self._mirror_fd, _pread_exact(self._src_fd, nbytes, offset), offset)
        return nbytes

    def mark_range(self, offset: int, nbytes: int) -> None:
        """Mark the blocks FULLY covered by [offset, offset+nbytes).

        Partial edge blocks are left unmarked (served from source) --
        negligible for large ranges.
        """
        first_full = (offset + BLOCK_SIZE - 1) // BLOCK_SIZE
        last_full = (offset + nbytes) // BLOCK_SIZE - 1
        # the final block of the file may be short; treat the file tail as a full block
        if offset + nbytes >= self.bin_size:
            last_full = self.n_blocks - 1
        if last_full >= first_full:
            self.bitmap.set_range(first_full, last_full)

    def evict_range(self, offset: int, nbytes: int) -> None:
        """Clear the fully-covered blocks of a staged range and punch a hole to
        free disk."""
        first_full = (offset + BLOCK_SIZE - 1) // BLOCK_SIZE
        last_full = (offset + nbytes) // BLOCK_SIZE - 1
        if offset + nbytes >= self.bin_size:
            last_full = self.n_blocks - 1
        if last_full >= first_full:
            self.bitmap.clear_range(first_full, last_full)
        _punch_hole(self._mirror_fd, offset, nbytes)


def _pread_exact(fd: int, nbytes: int, offset: int) -> bytes:
    chunks = []
    pos, remaining = offset, nbytes
    while remaining > 0:
        data = os.pread(fd, remaining, pos)
        if not data:
            raise EOFError(f"short read at offset {pos}, {remaining} bytes left")
        chunks.append(data)
        pos += len(data)
        remaining -= len(data)
    return chunks[0] if len(chunks) == 1 else b"".join(chunks)


def _pwrite_exact(fd: int, data: bytes, offset: int) -> None:
    pos, mv = offset, memoryview(data)
    while mv:
        written = os.pwrite(fd, mv, pos)
        pos += written
        mv = mv[written:]


def _punch_hole(fd: int, offset: int, length: int) -> None:
    try:
        os.fallocate(fd, 0x3, offset, length)  # FALLOC_FL_PUNCH_HOLE | KEEP_SIZE
    except (AttributeError, OSError):
        pass
