#!/usr/bin/env python3
"""Reference segments for benchmarks whose items exist only in our own eval runs.

SWDE, FDA and squad_completion have no public HF dataset, so their reference text is taken
from the eval result JSONL on MN5 (the ``doc`` field of each sample). Output lines use the
same schema as ``extract_refs.py`` so they can be appended to an existing refs file:

    {"src": <benchmark>, "split": "test", "file": <source file>, "text": <segment>}

    python scripts/flag_decontam/refs_from_eval.py \
        --eval-root /gpfs/scratch/ehpc533/hybrid_nucleus/flag_evals_oellm/ambitious_v4b_learned_fp8_36900 \
        --task swde --task fda --task squad_completion --out refs_extra.jsonl.gz

Segments are filtered and deduplicated exactly as ``extract_refs.py`` does, so a passage that
appears in several samples is one reference segment. Append to the pipeline's refs with
``fetch_extra.py --append-to`` or by concatenating the gzip streams.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import os
import re
import sys

# Same rule as extract_refs.is_segment; duplicated because extract_refs imports pyarrow, which
# the orchestration environment does not have.
WORDY = re.compile(r"\S*[^\W\d_]\S*", re.UNICODE)


def is_segment(t: str) -> bool:
    return 25 <= len(t) <= 100_000 and len(WORDY.findall(t[:4000])) >= 4


def flatten_text(value) -> list[str]:
    """Every string inside a nested doc value, depth first."""
    out: list[str] = []
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            out.extend(flatten_text(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            out.extend(flatten_text(item))
    return out


def collect(eval_root: str, task: str) -> list[dict]:
    """Unique reference segments for one benchmark, in extract_refs.py's line schema."""
    pattern = os.path.join(eval_root, "results", task + "*", "**", "samples_*.jsonl")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        print(f"warning: no samples for task {task} under {eval_root}", file=sys.stderr)
        return []

    seen: set[str] = set()
    records: list[dict] = []
    for path in files:
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                for text in flatten_text(row.get("doc", {})):
                    if not is_segment(text):
                        continue
                    digest = hashlib.sha256(text.encode()).hexdigest()
                    if digest in seen:
                        continue
                    seen.add(digest)
                    records.append(
                        {
                            "src": task,
                            "split": "test",
                            "file": os.path.basename(path),
                            "text": text,
                        }
                    )
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-root", required=True, help="one eval dir, e.g. .../ambitious_v4b_learned_fp8_36900")
    ap.add_argument("--task", action="append", required=True, help="benchmark task dir prefix (repeatable)")
    ap.add_argument("--out", required=True, help="output .jsonl.gz")
    args = ap.parse_args()

    records: list[dict] = []
    for task in args.task:
        got = collect(args.eval_root, task)
        print(f"{task}: {len(got)} reference segments")
        records.extend(got)

    if not records:
        print("no reference segments collected", file=sys.stderr)
        return 1

    with gzip.open(args.out, "wt") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} segments to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
