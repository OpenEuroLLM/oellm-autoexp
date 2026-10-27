"""Seed the shared data index cache (data_cache_shared) by hard-linking the COMPLETE entries of
existing per-run caches into it (same filesystem: no copy, no extra space). An entry is the files
of one description hash and dataset type and split; it is complete when all of its files exist
(GPTDataset: description + document / sample / shuffle index; BlendedDataset: description +
dataset / dataset-sample index). Existing names are never replaced.

    python3 scripts/korbi/link_data_caches.py <shared dir> <cache dir>...
"""

import os
import re
import sys
from collections import defaultdict

NEED = {"GPTDataset": {"description.txt", "document_index.npy", "sample_index.npy",
                       "shuffle_index.npy"},
        "BlendedDataset": {"description.txt", "dataset_index.npy", "dataset_sample_index.npy"}}
NAME = re.compile(r"^([0-9a-f]{32})-(GPTDataset|BlendedDataset)-(\w+?)-(.+)$")


def main():
    shared, sources = sys.argv[1], sys.argv[2:]
    os.makedirs(shared, exist_ok=True)
    linked = skipped = incomplete = 0
    for src in sources:
        entries = defaultdict(set)
        for f in os.listdir(src):
            m = NAME.match(f)
            if m:
                entries[m.groups()[:3]].add(m.group(4))
        for (h, kind, split), files in entries.items():
            if not NEED[kind] <= files:
                incomplete += 1
                continue
            for suffix in NEED[kind]:
                name = f"{h}-{kind}-{split}-{suffix}"
                dst = os.path.join(shared, name)
                if os.path.exists(dst):
                    skipped += 1
                    continue
                os.link(os.path.join(src, name), dst)
                linked += 1
        print(f"{src}: {len(entries)} entries")
    print(f"linked {linked} files, {skipped} already present, {incomplete} incomplete entries skipped")


if __name__ == "__main__":
    main()
