"""The "flag" training mix on MareNostrum 5 as a Megatron ``data_args_path`` file.

    python scripts/korbi/flag_datamix_mn5.py [--root /gpfs/projects/ehpc533/training-data-collection/flag] \
        [--out $HYBRID_ENV/data/flag_datamix_mn5.txt] [--require-complete]

``<root>/datamix.txt`` lists ``weight prefix`` pairs with prefixes relative to the root (the
same mix as JUPITER's flagship_datamix files). This writes the absolute-path version, keeping
only shards whose ``.bin`` and ``.idx`` both exist with the sizes their ``<shard>.info.json``
records (the collection lands in stages: a shard being copied has a short ``.bin``), with the
weights renormalised to sum to 1, and prints per source how much of the mix weight is present.
``--require-complete`` fails instead when anything is missing (for the reference trainings).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = "/gpfs/projects/ehpc533/training-data-collection/flag"


def complete(root: Path, prefix: str) -> bool:
    """Both files there, with the sizes the shard's info.json records."""
    info = root / (prefix.removesuffix("_text_document") + ".info.json")
    binf, idxf = root / f"{prefix}.bin", root / f"{prefix}.idx"
    if not (info.is_file() and binf.is_file() and idxf.is_file()):
        return False
    meta = json.loads(info.read_text())
    return binf.stat().st_size == meta["bin"]["size"] and idxf.stat().st_size == meta["idx"]["size"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", default=os.path.join(
        os.environ.get("HYBRID_ENV", "/gpfs/projects/ehpc533/users/korbip/hybrid_v019"),
        "data", "flag_datamix_mn5.txt"))
    ap.add_argument("--require-complete", action="store_true")
    ap.add_argument("--exclude", action="append", default=[],
                    help="leave out sources by name (the first path component), e.g. openthoughts-3: "
                         "it is contained in mixture-vitae (docs/flag_retokenization.md)")
    args = ap.parse_args()
    root = Path(args.root)
    rows = []
    for line in (root / "datamix.txt").read_text().splitlines():
        if line.strip():
            w, prefix = line.split()
            if prefix.split("/")[0] in args.exclude:
                continue
            rows.append((float(w), prefix))
    by_source = defaultdict(lambda: [0.0, 0.0, 0, 0])  # weight total, weight present, n, present
    kept = []
    for w, prefix in rows:
        src = prefix.split("/")[0]
        ok = complete(root, prefix)
        s = by_source[src]
        s[0] += w
        s[2] += 1
        if ok:
            s[1] += w
            s[3] += 1
            kept.append((w, str(root / prefix)))
    total = sum(w for w, _ in rows)
    present = sum(w for w, _ in kept)
    print(f"{'source':40s} {'weight':>8s} {'present':>8s} {'shards':>9s}")
    for src, (wt, wp, n, k) in sorted(by_source.items(), key=lambda x: -x[1][0]):
        flag = "" if k == n else "  <- incomplete"
        print(f"{src:40s} {wt / total:8.4f} {wp / total:8.4f} {k:4d}/{n:<4d}{flag}")
    print(f"present: {len(kept)}/{len(rows)} shards, {present / total:.1%} of the mix weight")
    if args.require_complete and len(kept) != len(rows):
        sys.exit("incomplete mix (--require-complete)")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(f"{w / present:.10f} {p}\n" for w, p in kept))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
