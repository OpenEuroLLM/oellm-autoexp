"""MegaDLMs -> Hugging Face Qwen3 export (spec sections 12 and 13).

The exporter reads the tensor-parallel rank files of one MegaDLMs checkpoint iteration and merges
them with :func:`oellm_autoexp.dllm.convert.qwen3_layout.megatron_to_hf_state`. The checkpoint holds
only the padded vocabulary. When that size is smaller than the Hugging Face ``vocab_size``, the rows
from the padded size up are read from the base model's safetensors, and the rows below it come from
the checkpoint. The exporter writes a Hugging Face checkpoint in bfloat16. The vocabulary is cut
back to the Hugging Face ``vocab_size``, so the mask row M stays and the padding rows are dropped.
The output config.json is the base config plus ``dllm_block_size``, ``dllm_mask_token_id``,
``dllm_token_shift``, and ``torch_dtype``.

Run ``python -m oellm_autoexp.dllm.convert.megadlm_to_hf_qwen3 --help`` for the command line.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor

from oellm_autoexp.dllm.convert.hf_qwen3_to_megadlm import (
    LAYOUTS,
    check_layout,
    find_checkpoint_files,
    load_rank_checkpoint,
    read_hf_config,
    resolve_hf_dir,
)
from oellm_autoexp.dllm.convert.qwen3_layout import (
    Qwen3Dims,
    check_args_against_hf,
    megatron_to_hf_state,
    restore_vocab_tail,
)

# Largest total size in bytes of the tensors in one weight file (4 GiB). A larger export is split
# into shards.
MAX_SHARD_BYTES = 4 * 1024**3
DTYPE = "bfloat16"
WEIGHTS_FILE = "model.safetensors"
INDEX_FILE = "model.safetensors.index.json"
CONFIG_FILE = "config.json"
# Tokenizer and generation files that are copied from the base model when they exist.
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "generation_config.json",
    "special_tokens_map.json",
)
_SHARD_NAME = "model-{index:05d}-of-{total:05d}.safetensors"
_SHARD_PATTERN = re.compile(r"model-\d{5}-of-\d{5}\.safetensors")
# Megatron and Hugging Face names of the vocabulary matrices (spec section 13).
_EMBEDDING = "embedding.word_embeddings.weight"
_HF_EMBEDDING = "model.embed_tokens.weight"
_HF_HEAD = "lm_head.weight"
_MISSING = object()


def _check_settings(block_size: int, token_shift: int, mask_token_id: int, vocab_size: int) -> None:
    """Raise ValueError unless the block-diffusion settings can be written to
    config.json.

    Args:
        block_size: Diffusion block size B. Must be at least 1.
        token_shift: Prediction shift, 0 or 1 (spec section 6).
        mask_token_id: Mask token id M. It must be a row of the exported vocabulary.
        vocab_size: Hugging Face ``vocab_size``.

    Raises:
        ValueError: If B < 1, the shift is not 0 or 1, or M is outside [0, vocab_size).
    """
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")
    if token_shift not in (0, 1):
        raise ValueError(f"token_shift must be 0 or 1, got {token_shift}")
    if not 0 <= mask_token_id < vocab_size:
        raise ValueError(f"mask_token_id must be in [0, {vocab_size}), got {mask_token_id}")


def _tensor_bytes(tensor: Tensor) -> int:
    """Return the number of bytes that a tensor occupies.

    Args:
        tensor: Tensor to measure.

    Returns:
        numel times the element size.
    """
    return tensor.numel() * tensor.element_size()


def _split_into_shards(state: dict[str, Tensor], max_bytes: int) -> list[dict[str, Tensor]]:
    """Group the tensors, in key order, into shards of at most max_bytes each.

    Args:
        state: Tensors to group.
        max_bytes: Size limit of one shard in bytes.

    Returns:
        The shards. A state that fits in max_bytes gives exactly one shard.

    Raises:
        ValueError: If one tensor alone is larger than max_bytes.
    """
    shards: list[dict[str, Tensor]] = []
    current: dict[str, Tensor] = {}
    used = 0
    for key, tensor in state.items():
        size = _tensor_bytes(tensor)
        if size > max_bytes:
            raise ValueError(f"tensor {key} takes {size} bytes, more than the limit {max_bytes}")
        if current and used + size > max_bytes:
            shards.append(current)
            current, used = {}, 0
        current[key] = tensor
        used += size
    if current:
        shards.append(current)
    return shards


def _write_weights(out_path: Path, state: dict[str, Tensor], max_bytes: int) -> list[str]:
    """Write the tensors as one safetensors file, or as indexed shards.

    Args:
        out_path: Existing output directory.
        state: Tensors to write. Each tensor must own its storage.
        max_bytes: Size limit of one weight file in bytes.

    Returns:
        Names of the files written, relative to out_path.

    Raises:
        ValueError: If one tensor alone is larger than max_bytes.
    """
    shards = _split_into_shards(state, max_bytes)
    metadata = {"format": "pt"}
    if len(shards) == 1:
        save_file(shards[0], str(out_path / WEIGHTS_FILE), metadata=metadata)
        return [WEIGHTS_FILE]
    names: list[str] = []
    weight_map: dict[str, str] = {}
    for index, shard in enumerate(shards, start=1):
        name = _SHARD_NAME.format(index=index, total=len(shards))
        save_file(shard, str(out_path / name), metadata=metadata)
        names.append(name)
        for key in shard:
            weight_map[key] = name
    total_size = sum(_tensor_bytes(tensor) for tensor in state.values())
    index_doc = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (out_path / INDEX_FILE).write_text(json.dumps(index_doc, indent=2, sort_keys=True) + "\n")
    return [*names, INDEX_FILE]


def _remove_stale_weights(out_path: Path) -> None:
    """Delete the weight and index files that an earlier export wrote into
    out_path.

    Only the names that this module writes are removed. A stale index file would make Hugging
    Face load the old shards, even after a new single-file export.

    Args:
        out_path: Existing output directory.
    """
    for path in out_path.iterdir():
        stale = path.name in (WEIGHTS_FILE, INDEX_FILE) or _SHARD_PATTERN.fullmatch(path.name)
        if stale and path.is_file():
            path.unlink()


def _copy_tokenizer_files(hf_dir: Path, out_path: Path) -> list[str]:
    """Copy the tokenizer and generation files that the base model directory
    holds.

    Args:
        hf_dir: Base Hugging Face model directory.
        out_path: Output directory.

    Returns:
        Names of the files copied.
    """
    copied: list[str] = []
    for name in TOKENIZER_FILES:
        source = hf_dir / name
        if source.is_file():
            shutil.copy2(source, out_path / name)
            copied.append(name)
    return copied


def _write_config(
    base: dict[str, Any], out_path: Path, block_size: int, token_shift: int, mask_token_id: int
) -> None:
    """Write config.json: the base config plus the block-diffusion fields and
    the dtype.

    Args:
        base: Parsed base config.json. It is not modified.
        out_path: Output directory.
        block_size: Diffusion block size B, written as ``dllm_block_size``.
        token_shift: Prediction shift, written as ``dllm_token_shift``.
        mask_token_id: Mask token id M, written as ``dllm_mask_token_id``.
    """
    config = dict(base)
    config["dllm_block_size"] = block_size
    config["dllm_mask_token_id"] = mask_token_id
    config["dllm_token_shift"] = token_shift
    config["torch_dtype"] = DTYPE
    (out_path / CONFIG_FILE).write_text(json.dumps(config, indent=2) + "\n")


def _check_mask_token(args: Any, mask_token_id: int) -> None:
    """Raise ValueError when the mask token of the checkpoint differs from the
    exported one.

    The model was trained with the mask token of its args. Exporting another id would write a
    ``dllm_mask_token_id`` that the model never used as its mask. The check runs only when args has
    the ``mask_token`` attribute.

    Args:
        args: Arguments of the checkpoint, such as ``checkpoint["args"]``.
        mask_token_id: Mask token id that the exporter was given.

    Raises:
        ValueError: If args has ``mask_token`` and it differs from mask_token_id.
    """
    trained = getattr(args, "mask_token", _MISSING)
    if trained is not _MISSING and trained != mask_token_id:
        raise ValueError(
            f"mask_token_id {mask_token_id} differs from the checkpoint's mask_token {trained!r}"
        )


def _base_weight_file(hf_dir: Path, key: str) -> Path:
    """Return the safetensors file of the base model that holds one tensor.

    Args:
        hf_dir: Base model directory.
        key: Hugging Face tensor name.

    Returns:
        The file that holds key. A sharded model is resolved through model.safetensors.index.json.
        A single-file model uses model.safetensors.

    Raises:
        FileNotFoundError: If the directory holds neither the index nor model.safetensors.
        ValueError: If the index does not list key.
    """
    index_path = hf_dir / INDEX_FILE
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text()).get("weight_map", {})
        if key not in weight_map:
            raise ValueError(f"{key} is not listed in {index_path}")
        return hf_dir / weight_map[key]
    single = hf_dir / WEIGHTS_FILE
    if not single.is_file():
        raise FileNotFoundError(f"neither {INDEX_FILE} nor {WEIGHTS_FILE} exists in {hf_dir}")
    return single


def _read_base_tensor(hf_dir: Path, key: str) -> Tensor:
    """Read one tensor of the base model. No other tensor is loaded.

    Args:
        hf_dir: Base model directory.
        key: Hugging Face tensor name.

    Returns:
        The tensor on the CPU, in the dtype stored in the file.

    Raises:
        FileNotFoundError: If the base weight files are missing.
        ValueError: If the file that the index names does not hold key.
    """
    path = _base_weight_file(hf_dir, key)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        available = handle.keys()
        if key not in available:
            raise ValueError(f"{key} is missing from {path}")
        return handle.get_tensor(key)


def export(
    megatron_dir: str | Path,
    hf_base: str | Path,
    out_dir: str | Path,
    block_size: int,
    token_shift: int,
    mask_token_id: int,
    iteration: str | None = None,
    layout: str = "te",
) -> dict[str, Any]:
    """Export one MegaDLMs Qwen3 checkpoint as a Hugging Face checkpoint in
    bfloat16.

    The base model gives config.json, the architecture sizes, and the tokenizer files. The weights
    come from the tensor-parallel rank files of one iteration. When the padded vocabulary is smaller
    than the Hugging Face vocabulary, the rows from the padded size up are copied from the base
    weights. Weights go into one ``model.safetensors`` file when they fit in MAX_SHARD_BYTES, and
    into indexed shards otherwise.

    Args:
        megatron_dir: MegaDLMs checkpoint root, the ``--save`` or ``--load`` directory.
        hf_base: Hugging Face model directory, or the repo id of a cached snapshot. It supplies
            config.json, the tokenizer files, and the architecture sizes.
        out_dir: Output directory. It must differ from the base model directory. Weight and index
            files of an earlier export in out_dir are removed.
        block_size: Diffusion block size B, written as ``dllm_block_size``. Must be at least 1.
        token_shift: Prediction shift, 0 or 1, written as ``dllm_token_shift``.
        mask_token_id: Mask token id M, written as ``dllm_mask_token_id``. Must be a row of the
            Hugging Face vocabulary. When the checkpoint args hold ``mask_token``, it must equal M.
        iteration: ``"release"`` or an iteration number. None reads the tracker file.
        layout: ``"te"`` or ``"local"``. See ``qwen3_layout.layer_key_table``.

    Returns:
        Dict with the keys ``out_dir`` (the output directory as a string), ``num_tensors`` (the
        number of exported weights), and ``files`` (the names of the files written, relative to
        out_dir).

    Raises:
        FileNotFoundError: If the base config.json or the tracker file is missing.
        CheckpointNotFoundError: If the checkpoint holds no rank files for the iteration.
        TypeError: If a rank file holds a ``model`` entry that is not a dict.
        ValueError: If a setting is unsupported, out_dir is the base directory, the checkpoint args
            do not match the base config, the checkpoint's mask_token differs from mask_token_id, a
            rank file does not match the base model, or one tensor is larger than the shard limit.
    """
    check_layout(layout)
    hf_dir = resolve_hf_dir(str(hf_base))
    base_config = read_hf_config(hf_dir)
    dims = Qwen3Dims.from_hf_config(base_config)
    _check_settings(block_size, token_shift, mask_token_id, dims.vocab_size)
    out_path = Path(out_dir)
    if out_path.resolve() == hf_dir.resolve():
        raise ValueError(f"out_dir must differ from the Hugging Face base directory {hf_dir}")
    _, files = find_checkpoint_files(Path(megatron_dir), iteration)
    shards: list[dict[str, Any]] = []
    for rank, path in enumerate(files):
        # Keep only each rank's model dict, so the rest of a checkpoint is freed on the next load.
        ckpt = load_rank_checkpoint(path)
        if rank == 0 and "args" in ckpt:
            check_args_against_hf(ckpt["args"], base_config, layout)
            _check_mask_token(ckpt["args"], mask_token_id)
        shards.append(ckpt["model"])
    merged = megatron_to_hf_state(shards, dims, layout=layout)
    padded_rows = len(shards) * int(shards[0][_EMBEDDING].shape[0])
    if padded_rows < dims.vocab_size:
        # The checkpoint holds only the padded rows. Rows from padded_rows up come from the base.
        merged[_HF_EMBEDDING] = restore_vocab_tail(
            merged[_HF_EMBEDDING], _read_base_tensor(hf_dir, _HF_EMBEDDING), padded_rows
        )
        if not dims.tie_embeddings:
            merged[_HF_HEAD] = restore_vocab_tail(
                merged[_HF_HEAD], _read_base_tensor(hf_dir, _HF_HEAD), padded_rows
            )
    state = {
        key: value.to(dtype=torch.bfloat16, copy=True).contiguous() for key, value in merged.items()
    }
    out_path.mkdir(parents=True, exist_ok=True)
    _remove_stale_weights(out_path)
    written = _write_weights(out_path, state, MAX_SHARD_BYTES)
    _write_config(base_config, out_path, block_size, token_shift, mask_token_id)
    written.append(CONFIG_FILE)
    written.extend(_copy_tokenizer_files(hf_dir, out_path))
    return {"out_dir": str(out_path), "num_tensors": len(state), "files": written}


def main(argv: Sequence[str] | None = None) -> int:
    """Run the exporter from the command line and print the summary as one JSON
    line.

    Args:
        argv: Arguments without the program name. None reads sys.argv.

    Returns:
        Process exit code, 0 on success.
    """
    parser = argparse.ArgumentParser(
        description="Export a MegaDLMs Qwen3 checkpoint as a Hugging Face checkpoint."
    )
    parser.add_argument("--megatron", required=True, help="MegaDLMs checkpoint root.")
    parser.add_argument(
        "--hf-base",
        required=True,
        help="Base Hugging Face model directory or repo id (config.json and tokenizer files).",
    )
    parser.add_argument("--out", required=True, help="Output Hugging Face directory.")
    parser.add_argument(
        "--block-size", type=int, required=True, help="Diffusion block size B (dllm_block_size)."
    )
    parser.add_argument(
        "--token-shift",
        type=int,
        choices=(0, 1),
        required=True,
        help="Prediction shift, 0 or 1 (dllm_token_shift).",
    )
    parser.add_argument(
        "--mask-token-id", type=int, required=True, help="Mask token id M (dllm_mask_token_id)."
    )
    parser.add_argument(
        "--iteration",
        default=None,
        help="'release' or an iteration number. Default: read the tracker file.",
    )
    parser.add_argument(
        "--layout", choices=LAYOUTS, default="te", help="Megatron layer spec (default: te)."
    )
    args = parser.parse_args(argv)
    summary = export(
        args.megatron,
        args.hf_base,
        args.out,
        block_size=args.block_size,
        token_shift=args.token_shift,
        mask_token_id=args.mask_token_id,
        iteration=args.iteration,
        layout=args.layout,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
