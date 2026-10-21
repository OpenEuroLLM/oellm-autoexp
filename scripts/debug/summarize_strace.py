"""Summarize an strace file trace by filesystem.

Usage:
    python scripts/debug/summarize_strace.py strace/strace_<job>_rank0.txt [--ranks 128]

Paths under the shared filesystems (Lustre) are what matters for metadata load;
paths inside the container image (/usr, /opt, ...) are served from the
squashfs .sif and do not hit Lustre per file.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from datetime import datetime

SHARED_PREFIXES = ("/scratch/", "/pfs/", "/flash/", "/users/", "/project/", "/appl/")
LOCAL_PREFIXES = ("/tmp/", "/dev/", "/proc/", "/sys/", "/run/", "/dev/shm/")

LINE_RE = re.compile(r"^(?:\d+\s+)?(\d\d:\d\d:\d\d\.\d+)\s+(?:<\.\.\. )?(\w+)[( ]")
PATH_RE = re.compile(r'"([^"]*)"')
ERROR_RE = re.compile(r"= -1 E\w+")


def classify(path: str) -> str:
    if path == "":
        return "by-fd"  # fstat via newfstatat(fd, "", AT_EMPTY_PATH); filesystem unknown
    if not path.startswith("/"):
        return "shared"  # relative to the job's cwd, which is the repo on /scratch
    if path.startswith(SHARED_PREFIXES):
        return "shared"
    if path.startswith(LOCAL_PREFIXES):
        return "node-local"
    return "container"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace")
    parser.add_argument("--ranks", type=int, default=8, help="ranks per job, for scaling")
    args = parser.parse_args()

    calls: Counter[str] = Counter()
    failed: Counter[str] = Counter()
    syscalls: Counter[str] = Counter()
    unique: dict[str, set[str]] = {}
    top_dirs: Counter[str] = Counter()
    shared_times: list[datetime] = []
    pending: dict[str, str] = {}  # pid -> kind of its "<unfinished ...>" call

    with open(args.trace, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            pid = line.split(maxsplit=1)[0] if line[:1].isdigit() else ""
            if "resumed>" in line:
                # the path was counted on the "<unfinished ...>" line; the result is here
                kind = pending.pop(pid, None)
                if kind and ERROR_RE.search(line):
                    failed[kind] += 1
                continue
            match = LINE_RE.match(line)
            path_match = PATH_RE.search(line)
            if not match or not path_match:
                continue
            timestamp, syscall = match.groups()
            if syscall == "getcwd":
                continue  # answered from the kernel's dentry cache, no FS request
            path = path_match.group(1)
            kind = classify(path)
            calls[kind] += 1
            unique.setdefault(kind, set()).add(path)
            if "<unfinished ...>" in line:
                pending[pid] = kind
            elif ERROR_RE.search(line):
                failed[kind] += 1
            if kind == "shared":
                syscalls[syscall] += 1
                top_dirs["/".join(path.split("/")[:6])] += 1
                shared_times.append(datetime.strptime(timestamp, "%H:%M:%S.%f"))

    print(f"{'filesystem':<12}{'calls':>10}{'failed':>10}{'unique paths':>15}")
    for kind in ("shared", "container", "node-local", "by-fd"):
        print(f"{kind:<12}{calls[kind]:>10}{failed[kind]:>10}{len(unique.get(kind, ())):>15}")

    if shared_times:
        span = (max(shared_times) - min(shared_times)).total_seconds()
        print(f"\nShared-FS calls per rank: {calls['shared']} over {span:.1f}s")
        print(f"Scaled to {args.ranks} ranks: {calls['shared'] * args.ranks:,} calls")
        print("\nShared-FS syscalls:")
        for name, count in syscalls.most_common():
            print(f"  {name:<14}{count:>8}")
        print("\nTop shared-FS directories:")
        for directory, count in top_dirs.most_common(15):
            print(f"  {count:>8}  {directory}")


if __name__ == "__main__":
    main()
