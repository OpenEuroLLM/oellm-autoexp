"""Data preparation for the dllm ablations: FineWeb-Edu parquet -> JSONL shards -> checks.

The corpus is split into ``num_shards`` shards by parquet row group, in file order. Shard ``i``
gets the row groups with global index in ``[i * N // num_shards, (i + 1) * N // num_shards)``,
so the split is deterministic and every document lands in exactly one shard. The last shard is
the held-out validation/test split (``data.val_holdout_shards`` in PARAMS).

Commands (run with the dllm venv, which has pyarrow):

    python -m oellm_autoexp.dllm.data_prep write-jsonl --input-dir DIR --num-shards 16 \\
        --shard 3 --output OUT/shard_03.jsonl
    python -m oellm_autoexp.dllm.data_prep inspect --prefix PREFIX [--tokenizer NAME]

Tokenization runs ``submodules/MegaDLMs/tools/preprocess_data.py`` on each JSONL shard (see
``tokenize_fineweb_edu.sbatch``).
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from collections.abc import Sequence
from pathlib import Path

# Megatron indexed dataset (.idx) header, see megatron/core/datasets/indexed_dataset.py.
_IDX_MAGIC = b"MMIDIDX\x00\x00"
_DTYPE_CODES = {
    1: "uint8",
    2: "int8",
    3: "int16",
    4: "int32",
    5: "int64",
    6: "float64",
    7: "float32",
    8: "uint16",
}
_DTYPE_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 8, 7: 4, 8: 2}


def assign_row_groups(
    row_groups_per_file: Sequence[int], num_shards: int
) -> list[list[tuple[int, int]]]:
    """Split the row groups of all files into contiguous, near-equal shards.

    Args:
        row_groups_per_file: Number of row groups of each input file, in file order.
        num_shards: Number of shards.

    Returns:
        One list per shard of ``(file_index, row_group_index)`` pairs, in corpus order.

    Raises:
        ValueError: If ``num_shards`` < 1 or there are fewer row groups than shards.
    """
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    flat = [(f, g) for f, count in enumerate(row_groups_per_file) for g in range(count)]
    if len(flat) < num_shards:
        raise ValueError(f"{len(flat)} row groups cannot fill {num_shards} shards")
    total = len(flat)
    return [
        flat[i * total // num_shards : (i + 1) * total // num_shards] for i in range(num_shards)
    ]


def write_jsonl_shard(input_dir: Path, num_shards: int, shard: int, output: Path) -> int:
    """Write the ``text`` column of one shard as JSONL (``{"text": ...}`` per
    line).

    Args:
        input_dir: Directory with the ``*.parquet`` files (sorted by name).
        num_shards: Number of shards.
        shard: Shard index in ``[0, num_shards)``.
        output: Output JSONL path. Written to ``<output>.tmp`` first, then renamed.

    Returns:
        The number of documents written.
    """
    import pyarrow.parquet as pq

    files = sorted(Path(input_dir).glob("*.parquet"))
    if not files:
        raise ValueError(f"no parquet files in {input_dir}")
    parquet = [pq.ParquetFile(f) for f in files]
    plan = assign_row_groups([p.metadata.num_row_groups for p in parquet], num_shards)
    if not 0 <= shard < num_shards:
        raise ValueError(f"shard {shard} outside [0, {num_shards})")
    tmp = Path(str(output) + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with tmp.open("w", encoding="utf-8") as handle:
        for file_index, group in plan[shard]:
            table = parquet[file_index].read_row_group(group, columns=["text"])
            for text in table.column("text").to_pylist():
                handle.write(json.dumps({"text": text}, ensure_ascii=False))
                handle.write("\n")
                count += 1
    tmp.replace(output)
    return count


def read_idx_header(idx_path: Path) -> dict:
    """Read a Megatron ``.idx`` file and return sequence and token counts.

    Args:
        idx_path: Path of the ``.idx`` file.

    Returns:
        A dict with ``dtype``, ``num_sequences``, ``num_documents``, and ``num_tokens``.
    """
    import numpy as np

    with open(idx_path, "rb") as handle:
        if handle.read(9) != _IDX_MAGIC:
            raise ValueError(f"{idx_path}: not a Megatron index file")
        (version,) = struct.unpack("<Q", handle.read(8))
        (code,) = struct.unpack("<B", handle.read(1))
        (num_sequences,) = struct.unpack("<Q", handle.read(8))
        (num_documents,) = struct.unpack("<Q", handle.read(8))
        offset = handle.tell()
    lengths = np.memmap(idx_path, dtype=np.int32, mode="r", offset=offset, shape=(num_sequences,))
    return {
        "version": version,
        "dtype": _DTYPE_CODES.get(code, f"code{code}"),
        "dtype_size": _DTYPE_SIZES.get(code, 0),
        "num_sequences": int(num_sequences),
        "num_documents": int(num_documents) - 1,
        "num_tokens": int(lengths.astype(np.int64).sum()),
    }


def inspect_prefix(prefix: str, tokenizer_name: str | None, num_docs: int = 3) -> dict:
    """Check a tokenized ``<prefix>.bin/.idx`` pair: counts, max token id,
    first documents.

    Args:
        prefix: Path prefix without ``.bin``/``.idx``.
        tokenizer_name: Hugging Face tokenizer to decode with (None skips decoding).
        num_docs: Number of leading documents to decode.

    Returns:
        A dict with the header fields, ``max_token_id``, ``bin_bytes``, and ``decoded``.
    """
    import numpy as np

    header = read_idx_header(Path(prefix + ".idx"))
    tokens = np.memmap(prefix + ".bin", dtype=header["dtype"], mode="r")
    if tokens.shape[0] != header["num_tokens"]:
        raise ValueError(f"{prefix}: .bin has {tokens.shape[0]} tokens, .idx says {header}")
    max_id = 0
    chunk = 1 << 26
    for start in range(0, tokens.shape[0], chunk):
        max_id = max(max_id, int(tokens[start : start + chunk].max()))
    result = dict(header, max_token_id=max_id, bin_bytes=Path(prefix + ".bin").stat().st_size)
    if tokenizer_name:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        result["len_tokenizer"] = len(tokenizer)
        idx_lengths = _sequence_lengths(Path(prefix + ".idx"), header["num_sequences"])
        decoded, start = [], 0
        for length in idx_lengths[:num_docs]:
            decoded.append(tokenizer.decode(tokens[start : start + length].tolist())[:300])
            start += int(length)
        result["decoded"] = decoded
    return result


def _sequence_lengths(idx_path: Path, num_sequences: int):
    import numpy as np

    return np.memmap(idx_path, dtype=np.int32, mode="r", offset=34, shape=(num_sequences,))


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point.

    Args:
        argv: Arguments (defaults to ``sys.argv[1:]``).

    Returns:
        The exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    write = sub.add_parser("write-jsonl", help="write one JSONL shard")
    write.add_argument("--input-dir", type=Path, required=True)
    write.add_argument("--num-shards", type=int, required=True)
    write.add_argument("--shard", type=int, required=True)
    write.add_argument("--output", type=Path, required=True)
    inspect = sub.add_parser("inspect", help="check tokenized .bin/.idx prefixes")
    inspect.add_argument("--prefix", nargs="+", required=True)
    inspect.add_argument("--tokenizer", default=None)
    args = parser.parse_args(argv)
    if args.command == "write-jsonl":
        count = write_jsonl_shard(args.input_dir, args.num_shards, args.shard, args.output)
        print(json.dumps({"shard": args.shard, "documents": count, "output": str(args.output)}))
        return 0
    total = 0
    for prefix in args.prefix:
        info = inspect_prefix(prefix, args.tokenizer)
        total += info["num_tokens"]
        print(json.dumps({"prefix": prefix, **info}, ensure_ascii=False))
    print(json.dumps({"total_tokens": total}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
