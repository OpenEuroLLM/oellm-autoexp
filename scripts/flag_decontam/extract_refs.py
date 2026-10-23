"""Extract the text of every benchmark we evaluate into one reference file (runs on JUPITER).

JUPITER holds all eval data offline: the FLAG suite's HF cache ($FLAG_WORK/hf_home: the hub
snapshots and the `datasets` arrow cache of lm-eval-harness, lighteval and the Evalchemy loaders)
and the data Evalchemy / human-eval bundle in the vLLM eval image. Run inside that image (its
Python has pyarrow / pandas):

    apptainer exec --cleanenv -B /e $VLLM_SIF python scripts/flag_decontam/extract_refs.py \
        --hf-home /e/scratch/e-sta-openeurollm/$USER/flag-evals/hf_home --out refs.jsonl.gz

Output: one JSON line per unique text segment {"src", "split", "file", "text"} -- every string
field of every row (nested lists / dicts flattened) that looks like natural text or code (>= 4
whitespace-separated words containing letters, 25 chars .. 100k chars). Segments, not rows: a
passage shared by many questions (SQuAD, Belebele, CoQA) is one segment, and an n-gram's segment
count is not inflated by such repetition. Next to it, <out>.manifest.tsv lists every data file
seen with its split and whether it was included.

--attribute REFS: instead of extracting, write for every line of an existing refs file all the
(source, split, file, included) that hold its exact text (deduplication credits a text to the
first source only: MGSM-en *is* GSM8K test, but is stored as MGSM). Same line order as REFS.

Split policy: test / validation / dev / devtest / unknown splits are included; a source's
`train` files are excluded whenever the source has any other split (few-shot pools and training
sets are legitimate pretraining data), and included when train is its only split (GPQA,
multiblimp, ... evaluate on it). The `datasets` cache's cache-*.arrow files (.map() outputs of
unknown split) are skipped; hub files whose blob is missing (partial downloads) are reported in
the manifest and covered by the arrow cache of the same dataset.
"""

from __future__ import annotations

import argparse
import csv
import glob
import gzip
import hashlib
import io
import json
import os
import re
import sys
from collections import defaultdict

import pyarrow as pa
import pyarrow.ipc
import pyarrow.parquet as pq

DATA_EXT = (".parquet", ".arrow", ".json", ".jsonl", ".csv", ".tsv", ".json.gz", ".jsonl.gz")
SPLIT_WORDS = [  # order matters: the first word found in the path decides
    ("auxiliary_train", "train"), ("devtest", "test"), ("validation", "validation"),
    ("valid", "validation"), ("test", "test"), ("dev", "validation"), ("val", "validation"),
    ("train", "train"),
]
# Evalchemy data files that are model outputs / judge material, not benchmark items
EVALCHEMY_SKIP = re.compile(r"response|reference_answer|judge|leaderboard|/docs/")
WORDY = re.compile(r"\S*[^\W\d_]\S*", re.UNICODE)


def split_of(path: str) -> str:
    parts = re.split(r"[-_/.\s]+", path.lower())
    for word, split in SPLIT_WORDS:
        if "_" in word:
            if word in path.lower():
                return split
        elif word in parts:
            return split
    return "unknown"


def texts_of(value, out: list[str]) -> None:
    if isinstance(value, str):
        if 25 <= len(value) <= 100_000 and len(WORDY.findall(value[:4000])) >= 4:
            out.append(value)
    elif isinstance(value, dict):
        for v in value.values():
            texts_of(v, out)
    elif isinstance(value, (list, tuple)):
        for v in value:
            texts_of(v, out)


def rows_of(path: str):
    """Yield rows (dicts) of a data file; unknown formats yield nothing."""
    p = path.lower()
    if p.endswith(".parquet"):
        pf = pq.ParquetFile(path)
        for i in range(pf.num_row_groups):
            yield from pf.read_row_group(i).to_pylist()
    elif p.endswith(".arrow"):
        with open(path, "rb") as f:
            try:
                reader = pa.ipc.open_stream(f)
            except pa.ArrowInvalid:
                f.seek(0)
                reader = pa.ipc.open_file(f)
                for i in range(reader.num_record_batches):
                    yield from reader.get_batch(i).to_pylist()
                return
            for batch in reader:
                yield from batch.to_pylist()
    else:
        opener = gzip.open if p.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8", errors="replace") as f:
            text = f.read()
            if ".json" in p and ".jsonl" not in p:
                try:
                    obj = json.loads(text)
                except json.JSONDecodeError:      # JSON lines named .json (Evalchemy's AIME / AMC)
                    obj = None
            if ".jsonl" in p or (".json" in p and obj is None):
                for line in text.splitlines():
                    line = line.strip()
                    if line:
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            continue
            elif ".json" in p:
                if isinstance(obj, dict):
                    # {"data": [...]} or {id: row}
                    lists = [v for v in obj.values() if isinstance(v, list)]
                    obj = lists[0] if len(lists) == 1 else list(obj.values())
                for row in obj if isinstance(obj, list) else [obj]:
                    yield row if isinstance(row, dict) else {"v": row}
            else:
                csv.field_size_limit(sys.maxsize)
                yield from csv.DictReader(io.StringIO(text), delimiter="\t" if ".tsv" in p else ",")


def hub_files(hf_home: str):
    for repo in sorted(glob.glob(f"{hf_home}/hub/datasets--*")):
        src = os.path.basename(repo).removeprefix("datasets--").replace("--", "/", 1)
        for snap in glob.glob(f"{repo}/snapshots/*"):
            for root, _, files in os.walk(snap, followlinks=True):
                for fn in files:
                    if fn.lower().endswith(DATA_EXT):
                        full = os.path.join(root, fn)
                        yield src, full, os.path.relpath(full, snap)


def arrow_files(hf_home: str):
    for full in sorted(glob.glob(f"{hf_home}/datasets/*/*/*/*/*.arrow")):
        if os.path.basename(full).startswith("cache-"):
            continue      # .map() outputs of any split (train included): the raw split files cover them
        rel = os.path.relpath(full, f"{hf_home}/datasets")
        builder, config = rel.split("/")[:2]
        # the generic parquet/json builders cache under their own name: the hub copy is the source
        src = builder.replace("___", "/") if builder not in ("parquet", "json", "csv") else f"{builder}:{config}"
        yield src, full, rel


def bundled_files(evalchemy: str, human_eval: str):
    for full in sorted(glob.glob(f"{evalchemy}/*/**/data/**/*", recursive=True)):
        if os.path.isfile(full) and full.lower().endswith(DATA_EXT) and not EVALCHEMY_SKIP.search(full):
            rel = os.path.relpath(full, evalchemy)
            yield "evalchemy/" + rel.split("/")[0], full, rel
    for full in sorted(glob.glob(f"{human_eval}/*.jsonl.gz")):
        yield "openai/human-eval", full, os.path.basename(full)


def _key(t):
    return hashlib.blake2b(t.encode("utf-8", "replace"), digest_size=16).digest()


def attribute(args, files, splits_by_src):
    """For every line of an existing refs file: all (src, split, file, included) holding that exact
    text, train splits included (flagged). Keeps the refs' line ids, so tables and scans built on
    them stay valid. Texts of included files missing from the refs (data added since) are counted."""
    ids = {}
    with gzip.open(args.attribute, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            ids[_key(json.loads(line)["text"])] = i
    attrib = [[] for _ in range(len(ids))]
    missing = 0
    for src, full, rel in files:
        split = split_of(rel)
        only_train = splits_by_src[src] <= {"train"}
        generic = src.split(":")[0] in ("parquet", "json", "csv")
        include = split != "train" or (only_train and not generic)
        hit = 0
        try:
            for row in rows_of(full):
                texts: list[str] = []
                texts_of(row, texts)
                for t in texts:
                    i = ids.get(_key(t))
                    if i is None:
                        missing += include
                        continue
                    e = [src, split, rel, int(include)]
                    if e not in attrib[i][-4:]:
                        attrib[i].append(e)
                        hit += 1
        except Exception as e:
            print(f"! {src} {rel}: {type(e).__name__}: {str(e)[:120]}", flush=True)
        print(f"{'+' if include else '-'} {src:50s} {split:10s} attributions={hit:7d} {rel[:80]}", flush=True)
    with gzip.open(args.out, "wt", encoding="utf-8") as out:
        for a in attrib:
            out.write(json.dumps({"also_in": a}, ensure_ascii=False) + "\n")
    multi = sum(len({x[0] for x in a}) > 1 for a in attrib)
    print(f"{len(attrib)} refs lines attributed; {multi} in more than one source; "
          f"{missing} texts of included files not in the refs -> {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf-home", required=True)
    ap.add_argument("--evalchemy", default="/opt/evalchemy/eval/chat_benchmarks")
    ap.add_argument("--human-eval", default="/opt/human-eval/data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--attribute", metavar="REFS", help="attribution mode: every source of each line of an "
                    "existing refs file -> --out (one JSON line per refs line, same order)")
    args = ap.parse_args()

    files = list(hub_files(args.hf_home)) + list(arrow_files(args.hf_home)) + \
        list(bundled_files(args.evalchemy, args.human_eval))
    splits_by_src = defaultdict(set)
    for src, full, rel in files:
        splits_by_src[src].add(split_of(rel))

    if args.attribute:
        return attribute(args, files, splits_by_src)
    seen: set[bytes] = set()
    manifest = []
    n_out = 0
    with gzip.open(args.out, "wt", encoding="utf-8") as out:
        for src, full, rel in files:
            split = split_of(rel)
            only_train = splits_by_src[src] <= {"train"}
            generic = src.split(":")[0] in ("parquet", "json", "csv")
            include = split != "train" or (only_train and not generic)
            rows = segs = new = 0
            err = ""
            if include:
                try:
                    for row in rows_of(full):
                        rows += 1
                        texts: list[str] = []
                        texts_of(row, texts)
                        for t in texts:
                            segs += 1
                            key = hashlib.blake2b(t.encode("utf-8", "replace"), digest_size=16).digest()
                            if key in seen:
                                continue
                            seen.add(key)
                            new += 1
                            out.write(json.dumps({"src": src, "split": split, "file": rel, "text": t},
                                                 ensure_ascii=False) + "\n")
                except Exception as e:  # report, don't stop: one odd file must not lose the rest
                    err = f"{type(e).__name__}: {str(e)[:120]}"
            n_out += new
            manifest.append((src, split, int(include), rows, segs, new, rel, err))
            print(f"{'+' if include else '-'} {src:50s} {split:10s} rows={rows:7d} new={new:7d} {rel[:80]} {err}",
                  flush=True)
    with open(args.out + ".manifest.tsv", "w") as f:
        f.write("src\tsplit\tincluded\trows\tsegments\tnew_segments\tfile\terror\n")
        for m in manifest:
            f.write("\t".join(map(str, m)) + "\n")
    print(f"{n_out} unique segments from {len(files)} files -> {args.out}")


if __name__ == "__main__":
    main()
