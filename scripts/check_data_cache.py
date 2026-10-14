#!/usr/bin/env python3
"""Validate (and optionally repair) a Megatron dataset index cache.

WHY THIS EXISTS
---------------
Megatron decides whether a dataset index is cached by FILE PRESENCE ALONE
(megatron/core/datasets/gpt_dataset.py):

    cache_hit = all(map(os.path.isfile, [description, document_index,
                                         sample_index, shuffle_index]))

There is no size, length or checksum check. So any prebuild that is killed
mid-write -- a SLURM timeout, an OOM, an NCCL watchdog abort -- leaves behind
empty or TRUNCATED .npy files that every later run accepts as a hit, skips
rebuilding, and then dies on:

    EOFError: No data left in file

The failure surfaces during dataset construction, i.e. AFTER the allocation is
up, so on a large job it burns the whole allocation to report a corrupt file.
Running two prebuilds concurrently against one cache directory produces the same
damage. Run this between a prebuild and any large job.

WHAT IT CHECKS
--------------
For every "<hash>-<class>-<split>" prefix in the cache:
  * all member files for that CLASS exist (GPTDataset and BlendedDataset cache
    different sets -- see MEMBERS_BY_CLASS);
  * each .npy has a well-formed header AND the file is at least as long as the
    header's own shape/dtype implies. A truncated array passes `os.path.isfile`
    and fails at load time -- this is the case a `-size 0` sweep misses.
The header is parsed directly (npy format spec), so this needs no numpy and
reads only ~128 bytes per file rather than the hundreds of GB on disk.

--repair deletes every member of any prefix that is incomplete or unreadable, so
the next prebuild rebuilds exactly those datasets and nothing else.
"""

from __future__ import annotations

import argparse
import ast
import os
import struct
import sys
from collections import defaultdict

# The member set depends on the dataset CLASS, which is encoded in the prefix as
# "<hash>-<class>-<split>". GPTDataset and BlendedDataset cache different files
# (gpt_dataset.py:329-343 vs blended_dataset.py:114-120); assuming the GPTDataset
# set for everything makes every BlendedDataset prefix look like it is missing
# three files, and --repair would then delete a perfectly good blended index.
MEMBERS_BY_CLASS = {
    "GPTDataset": (
        "description.txt",
        "document_index.npy",
        "sample_index.npy",
        "shuffle_index.npy",
    ),
    "BlendedDataset": ("description.txt", "dataset_index.npy", "dataset_sample_index.npy"),
}
DEFAULT_MEMBERS = MEMBERS_BY_CLASS["GPTDataset"]
ALL_MEMBERS = sorted(
    {m for members in MEMBERS_BY_CLASS.values() for m in members}, key=len, reverse=True
)  # longest first: dataset_sample_index before sample_index

# npy dtype code -> itemsize, enough for the integer index arrays Megatron writes
_ITEMSIZE = {
    "i1": 1,
    "u1": 1,
    "i2": 2,
    "u2": 2,
    "i4": 4,
    "u4": 4,
    "i8": 8,
    "u8": 8,
    "f2": 2,
    "f4": 4,
    "f8": 8,
    "b1": 1,
}


def npy_problem(path: str) -> str | None:
    """Return a reason string if the .npy is unreadable/truncated, else
    None."""
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        return f"stat failed: {exc}"
    if size == 0:
        return "empty file"
    try:
        with open(path, "rb") as fh:
            magic = fh.read(6)
            if magic != b"\x93NUMPY":
                return "bad magic (not a .npy)"
            major, _minor = fh.read(2)
            if major == 1:
                (hlen,) = struct.unpack("<H", fh.read(2))
            else:
                (hlen,) = struct.unpack("<I", fh.read(4))
            raw = fh.read(hlen)
            if len(raw) < hlen:
                return "truncated header"
            header = ast.literal_eval(raw.decode("latin1").strip())
            data_off = fh.tell()
    except Exception as exc:  # noqa: BLE001 - any malformed header counts as corrupt
        return f"unparsable header: {exc}"

    descr = str(header.get("descr", ""))
    shape = header.get("shape", ())
    itemsize = _ITEMSIZE.get(descr.lstrip("<>|="))
    if itemsize is None:
        return None  # unknown dtype: header parsed fine, don't guess at length
    count = 1
    for dim in shape:
        count *= dim
    expected = data_off + count * itemsize
    if size < expected:
        return f"truncated: {size} bytes on disk, header implies {expected}"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("cache_dir")
    ap.add_argument(
        "--repair",
        action="store_true",
        help="delete every member of each bad prefix so it gets rebuilt",
    )
    args = ap.parse_args()

    if not os.path.isdir(args.cache_dir):
        print(f"not a directory: {args.cache_dir}", file=sys.stderr)
        return 2

    # group files by "<hash>-<class>-<split>"
    prefixes: dict[str, set[str]] = defaultdict(set)
    for name in os.listdir(args.cache_dir):
        for member in ALL_MEMBERS:
            if name.endswith("-" + member):
                prefixes[name[: -(len(member) + 1)]].add(member)
                break

    bad: dict[str, list[str]] = {}
    for prefix, present in sorted(prefixes.items()):
        cls = prefix.split("-")[1] if prefix.count("-") >= 2 else ""
        members = MEMBERS_BY_CLASS.get(cls, DEFAULT_MEMBERS)
        reasons = [f"missing {m}" for m in members if m not in present]
        for member in members:
            if member.endswith(".npy") and member in present:
                why = npy_problem(os.path.join(args.cache_dir, f"{prefix}-{member}"))
                if why:
                    reasons.append(f"{member}: {why}")
        if reasons:
            bad[prefix] = reasons

    print(f"cache dir : {args.cache_dir}")
    print(f"prefixes  : {len(prefixes)}")
    print(f"healthy   : {len(prefixes) - len(bad)}")
    print(f"BAD       : {len(bad)}")
    for prefix, reasons in sorted(bad.items()):
        print(f"  {prefix}")
        for reason in reasons:
            print(f"      - {reason}")

    if bad and args.repair:
        removed = 0
        for prefix in bad:
            cls = prefix.split("-")[1] if prefix.count("-") >= 2 else ""
            for member in MEMBERS_BY_CLASS.get(cls, DEFAULT_MEMBERS):
                path = os.path.join(args.cache_dir, f"{prefix}-{member}")
                if os.path.isfile(path):
                    os.remove(path)
                    removed += 1
        print(
            f"\nrepair: removed {removed} files across {len(bad)} prefixes; re-run the "
            f"prebuild to rebuild exactly those datasets"
        )

    return 1 if bad and not args.repair else 0


if __name__ == "__main__":
    raise SystemExit(main())
