"""Replace the symlinks of finished Megatron trees by hard links to the same files (no second copy; a tree then
keeps working when the tree it links into is moved or removed, e.g. v8_sources after the re-tokenization).

    python scripts/datamix_v8/hardlink_tree.py <tree> [<tree> ...] [--dry-run]

Per symlink (followed to its final target): a regular file on the same filesystem -> hard link, swapped in
atomically (link to <name>.hl.tmp, rename over the symlink), checked with samefile. Symlinks to directories, to
other filesystems, or dangling ones are left as they are and reported. Only run on trees that no pipeline step
still writes: the build steps (finalize.py, retok.py) tell linked from rewritten shards by islink().
"""

from __future__ import annotations

import argparse
import os
import sys


def convert(tree, dry_run=False):
    stats = {"hardlinked": 0, "dir": 0, "dangling": 0, "other_fs": 0, "failed": 0}
    dev = os.stat(tree).st_dev
    for dirpath, dirnames, filenames in os.walk(tree):
        for name in filenames + [d for d in dirnames if os.path.islink(os.path.join(dirpath, d))]:
            p = os.path.join(dirpath, name)
            if not os.path.islink(p):
                continue
            real = os.path.realpath(p)
            if not os.path.exists(real):
                stats["dangling"] += 1
                print(f"dangling: {p} -> {os.readlink(p)}", file=sys.stderr)
                continue
            if os.path.isdir(real):
                stats["dir"] += 1
                continue
            if os.stat(real).st_dev != dev:
                stats["other_fs"] += 1
                continue
            if dry_run:
                stats["hardlinked"] += 1
                continue
            tmp = p + ".hl.tmp"
            try:
                if os.path.lexists(tmp):
                    os.remove(tmp)
                os.link(real, tmp)
                os.replace(tmp, p)
                assert not os.path.islink(p) and os.path.samefile(p, real), p
                stats["hardlinked"] += 1
            except OSError as e:
                stats["failed"] += 1
                print(f"failed: {p}: {e}", file=sys.stderr)
                if os.path.lexists(tmp):
                    os.remove(tmp)
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trees", nargs="+")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    bad = 0
    for t in a.trees:
        s = convert(t, a.dry_run)
        print(f"{t}: {s}{' (dry run)' if a.dry_run else ''}", flush=True)
        bad += s["failed"] + s["dangling"]
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
