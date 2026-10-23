"""Hugging Face Qwen3 -> MegaDLMs checkpoint conversion by template fill (spec
section 13).

The template is a checkpoint of a one-iteration MegaDLMs run with the target architecture and
tensor-parallel size. Its arguments must match the Hugging Face config, as checked by
:func:`oellm_autoexp.dllm.convert.qwen3_layout.check_args_against_hf`. The converter keeps the
template's key set, shapes, dtypes, and non-model fields, except the optimizer, scheduler, RNG, and
rerun state, and it sets the iteration to 0. It fills every model key from the Hugging Face weights
with :func:`oellm_autoexp.dllm.convert.qwen3_layout.hf_to_megatron_state`, and it writes a release
checkpoint. The output directory must be fresh, so that no stale rank is mixed with the new ones.
Training loads that checkpoint with ``--load <out> --finetune --no-load-optim --no-load-rng``.

Run ``python -m oellm_autoexp.dllm.convert.hf_qwen3_to_megadlm --help`` for the command line.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from torch import Tensor

from oellm_autoexp.dllm.convert.qwen3_layout import (
    Qwen3Dims,
    check_args_against_hf,
    hf_to_megatron_state,
)

LAYOUTS = ("te", "local")
RELEASE = "release"
TRACKER = "latest_checkpointed_iteration.txt"
MODEL_FILE = "model_optim_rng.pt"
_EXTRA_STATE = "_extra_state"
_CHECKPOINT_VERSION = 3.0
# Training-run fields that the converted checkpoint does not keep (spec section 13, step 4).
_DROPPED_FIELDS = ("optimizer", "opt_param_scheduler", "rng_state", "rerun_state_machine")
_EMBEDDING = "embedding.word_embeddings.weight"
_RANK_DIR = re.compile(r"mp_rank_(\d{2})")
_PIPELINE_RANK_DIR = re.compile(r"mp_rank_\d{2}_\d{3}")
_ITERATION = re.compile(r"[0-9]+")


def check_layout(layout: str) -> None:
    """Raise ValueError unless layout names a supported Megatron layer spec.

    Args:
        layout: ``"te"`` for the Transformer Engine spec, or ``"local"`` for the Megatron-core spec.

    Raises:
        ValueError: If layout is not supported.
    """
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")


def resolve_hf_dir(name_or_path: str) -> Path:
    """Return the local directory of a Hugging Face model without using the
    network.

    Args:
        name_or_path: An existing directory, or a Hugging Face repository id whose snapshot is
            already in the local cache.

    Returns:
        The directory that holds config.json and the weight files.

    Raises:
        huggingface_hub.errors.LocalEntryNotFoundError: If name_or_path is neither a directory
            nor a cached snapshot.
    """
    path = Path(name_or_path)
    if path.is_dir():
        return path
    return Path(snapshot_download(name_or_path, local_files_only=True))


def read_hf_config(hf_dir: Path) -> dict[str, Any]:
    """Read the config.json of a Hugging Face model directory.

    Args:
        hf_dir: Model directory.

    Returns:
        The parsed config.json.

    Raises:
        FileNotFoundError: If config.json is missing.
        TypeError: If config.json does not hold a JSON object.
    """
    config_path = Path(hf_dir) / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing Hugging Face config {config_path}")
    config = json.loads(config_path.read_text())
    if not isinstance(config, dict):
        raise TypeError(f"{config_path} does not hold a JSON object")
    return config


def load_hf_state(hf_dir: Path) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """Load the config and every safetensors weight file of a Hugging Face
    model directory.

    Args:
        hf_dir: Model directory with config.json and one or more ``*.safetensors`` files.

    Returns:
        Tuple (state, config). state maps each Hugging Face tensor name to a CPU tensor. config is
        the parsed config.json.

    Raises:
        FileNotFoundError: If config.json is missing.
        ValueError: If the directory has no safetensors file, or if two files hold one tensor name.
    """
    hf_path = Path(hf_dir)
    config = read_hf_config(hf_path)
    files = sorted(hf_path.glob("*.safetensors"))
    if not files:
        raise ValueError(f"no *.safetensors files in {hf_path}")
    state: dict[str, Tensor] = {}
    for path in files:
        for key, tensor in load_file(str(path), device="cpu").items():
            if key in state:
                raise ValueError(f"tensor {key} appears in more than one safetensors file")
            state[key] = tensor
    return state, config


def _iteration_dir_name(iteration: str) -> str:
    """Map an iteration string to its checkpoint directory name.

    Args:
        iteration: ``"release"``, or a non-negative integer as a string.

    Returns:
        ``"release"``, or ``"iter_%07d"`` for an integer iteration.

    Raises:
        ValueError: If iteration is neither ``"release"`` nor a non-negative integer.
    """
    text = str(iteration).strip()
    if text == RELEASE:
        return RELEASE
    if not _ITERATION.fullmatch(text):
        raise ValueError(
            f"iteration must be 'release' or a non-negative integer, got {iteration!r}"
        )
    return f"iter_{int(text):07d}"


class CheckpointNotFoundError(FileNotFoundError, ValueError):
    """No checkpoint exists for the requested iteration.

    It is a FileNotFoundError, because checkpoint files are missing, and
    a ValueError, because the request matches no checkpoint. Callers may
    catch either.
    """


def find_checkpoint_files(ckpt_dir: Path, iteration: str | None = None) -> tuple[str, list[Path]]:
    """Find the model file of every tensor-parallel rank of one MegaDLMs
    checkpoint.

    The layout is ``<ckpt_dir>/<dir>/mp_rank_XX/model_optim_rng.pt``, where ``<dir>`` is
    ``release`` or ``iter_%07d``.

    Args:
        ckpt_dir: Checkpoint root, the directory given to ``--save`` or ``--load``.
        iteration: ``"release"``, or an integer iteration as a string. None reads
            ``latest_checkpointed_iteration.txt`` in ckpt_dir.

    Returns:
        Tuple (dir_name, files). dir_name is ``"release"`` or ``"iter_0000100"``. files holds the
        model file of each rank, in rank order.

    Raises:
        CheckpointNotFoundError: If iteration is None and the tracker file is missing, if the
            checkpoint directory does not exist, or if it holds no rank file.
        ValueError: If iteration is invalid, a pipeline-parallel directory (``mp_rank_XX_YYY``) or
            any other unsupported ``mp_rank`` name exists, a rank file is missing, or the ranks are
            not 0..tp-1.
    """
    root = Path(ckpt_dir)
    if iteration is None:
        tracker = root / TRACKER
        if not tracker.is_file():
            raise CheckpointNotFoundError(f"missing {tracker}; pass the iteration explicitly")
        iteration = tracker.read_text()
    dir_name = _iteration_dir_name(iteration)
    iter_dir = root / dir_name
    if not iter_dir.is_dir():
        raise CheckpointNotFoundError(f"checkpoint directory {iter_dir} does not exist")
    ranks: dict[int, Path] = {}
    for entry in sorted(iter_dir.glob("mp_rank_*")):
        if _PIPELINE_RANK_DIR.fullmatch(entry.name):
            raise ValueError(f"{entry} is a pipeline-parallel checkpoint; only PP = 1 is supported")
        match = _RANK_DIR.fullmatch(entry.name)
        if match is None or not entry.is_dir():
            raise ValueError(f"unsupported checkpoint entry {entry}; expected mp_rank_XX")
        model_file = entry / MODEL_FILE
        if not model_file.is_file():
            raise ValueError(f"missing {model_file}")
        ranks[int(match.group(1))] = model_file
    if not ranks:
        raise CheckpointNotFoundError(f"no mp_rank_XX/{MODEL_FILE} files under {iter_dir}")
    if sorted(ranks) != list(range(len(ranks))):
        raise ValueError(
            f"tensor-parallel ranks under {iter_dir} are {sorted(ranks)}, "
            f"expected 0..{len(ranks) - 1}"
        )
    return dir_name, [ranks[rank] for rank in range(len(ranks))]


def load_rank_checkpoint(path: Path) -> dict[str, Any]:
    """Load one rank file of a MegaDLMs checkpoint on the CPU.

    Args:
        path: Path to ``mp_rank_XX/model_optim_rng.pt``.

    Returns:
        The checkpoint dict, which holds the entry ``model``.

    Raises:
        TypeError: If the file does not hold a dict, or if its ``model`` entry is not a dict.
        ValueError: If the file holds virtual pipeline chunks (``model0``) or has no ``model``
            entry.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise TypeError(f"{path} does not hold a checkpoint dict")
    if "model0" in ckpt:
        raise ValueError(f"{path} holds virtual pipeline model chunks; only one chunk is supported")
    if "model" not in ckpt:
        raise ValueError(f"{path} has no 'model' entry")
    if not isinstance(ckpt["model"], dict):
        raise TypeError(f"{path} has a 'model' entry that is not a dict")
    return ckpt


def _tokenizer_length(hf_dir: Path) -> int:
    """Return len(tokenizer) of a local Hugging Face model directory.

    Args:
        hf_dir: Model directory that holds the tokenizer files.

    Returns:
        The number of tokenizer entries, including added special tokens.
    """
    from transformers import AutoTokenizer  # Deferred: importing transformers is slow.

    return len(AutoTokenizer.from_pretrained(str(hf_dir)))


def _padded_vocab(template_model: dict[str, Any], tp: int) -> int:
    """Return the padded vocabulary size of a template from its embedding
    shard.

    Args:
        template_model: Model dict of rank 0 of the template.
        tp: Tensor-parallel size of the template.

    Returns:
        Rows per rank times tp.

    Raises:
        ValueError: If the template has no 2-D embedding shard.
    """
    embedding = template_model.get(_EMBEDDING)
    if not isinstance(embedding, Tensor) or embedding.dim() != 2:
        raise ValueError(f"template rank 0 has no 2-D {_EMBEDDING} tensor")
    return int(embedding.shape[0]) * tp


def _fill_rank_model(
    template_model: dict[str, Any], filled: dict[str, Tensor], rank: int
) -> dict[str, Any]:
    """Build the model dict of one rank from the template keys and the
    converted tensors.

    Args:
        template_model: Model dict of the template rank.
        filled: Converted tensors of the rank, from hf_to_megatron_state.
        rank: Tensor-parallel rank, used in messages.

    Returns:
        A model dict with the template key order. Tensors take the template shape and dtype.
        ``_extra_state`` entries are copied from the template.

    Raises:
        ValueError: If the template keys are not the converted keys plus the ``_extra_state``
            keys, or if a shape differs.
        TypeError: If a template entry, other than an ``_extra_state`` entry, is not a tensor.
    """
    extra = {key for key in template_model if key.endswith(_EXTRA_STATE)}
    missing = sorted(set(filled) - set(template_model))
    unknown = sorted(set(template_model) - set(filled) - extra)
    if missing or unknown:
        raise ValueError(
            f"rank {rank}: template keys do not match the converted model; "
            f"missing from template: {missing}; unknown in template: {unknown}"
        )
    model: dict[str, Any] = {}
    for key, reference in template_model.items():
        if key in extra:
            model[key] = reference
            continue
        if not isinstance(reference, Tensor):
            raise TypeError(f"rank {rank}: template entry {key} is not a tensor")
        value = filled[key]
        if tuple(value.shape) != tuple(reference.shape):
            raise ValueError(
                f"rank {rank}: {key} has converted shape {tuple(value.shape)} but template "
                f"shape {tuple(reference.shape)}"
            )
        model[key] = value.to(dtype=reference.dtype).contiguous()
    return model


def _check_fresh_release(out_path: Path) -> None:
    """Raise ValueError when out_path/release already holds a rank entry.

    The converter writes ranks 0..tp-1 and never removes other ranks. A release directory with the
    ranks of an earlier conversion would therefore mix old and new ranks. Use a fresh output
    directory instead.

    Args:
        out_path: Output checkpoint root.

    Raises:
        ValueError: If out_path/release holds any ``mp_rank_*`` entry.
    """
    release = out_path / RELEASE
    if not release.is_dir():
        return
    stale = sorted(entry.name for entry in release.glob("mp_rank_*"))
    if stale:
        raise ValueError(
            f"{release} already holds {stale}; use a fresh out_dir so that no stale rank is mixed"
        )


def _checkpoint_fields(template: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    """Build the checkpoint that the converter writes for one rank.

    Every template field is kept, except the optimizer, scheduler, RNG, and rerun state, which
    belong to the training run (spec section 13, step 4). The model is replaced by the filled model,
    and the iteration is set to 0.

    Args:
        template: Checkpoint dict of the template rank.
        model: Model dict filled from the Hugging Face weights.

    Returns:
        A new checkpoint dict. ``checkpoint_version`` is set to 3.0 when the template has none.
    """
    fields = {key: value for key, value in template.items() if key not in _DROPPED_FIELDS}
    fields["model"] = model
    fields["iteration"] = 0
    fields.setdefault("checkpoint_version", _CHECKPOINT_VERSION)
    return fields


def convert(
    hf_dir: str | Path,
    template_dir: str | Path,
    out_dir: str | Path,
    mask_token_id: int | None = None,
    real_vocab: int | None = None,
    layout: str = "te",
) -> dict[str, Any]:
    """Fill a MegaDLMs template with Hugging Face Qwen3 weights and write a
    release checkpoint.

    The template fixes the tensor-parallel size, the padded vocabulary, and the key set. The
    arguments of its rank 0 must match the Hugging Face config (``check_args_against_hf``). The
    template keys that end in ``_extra_state`` are kept with their values. Every other template key
    must be filled from Hugging Face, and every filled tensor must have the template shape. The
    filled tensors are cast to the template dtype. The output keeps the other template fields,
    except the optimizer, scheduler, RNG, and rerun state, and its iteration is 0. Nothing is
    written until every rank has passed the checks.

    Args:
        hf_dir: Hugging Face model directory with config.json and safetensors weights.
        template_dir: Template checkpoint root, for example the ``--save`` directory of a
            one-iteration run. Its tracker file selects the iteration.
        out_dir: Output checkpoint root. It must not already hold ``release/mp_rank_*`` entries, so
            that no stale rank is mixed with the new ones. The converter writes
            ``release/mp_rank_XX`` files and ``latest_checkpointed_iteration.txt`` into it.
        mask_token_id: Mask token id M. None means ``real_vocab``, which is M = len(tokenizer)
            (spec section 12). Otherwise M must satisfy real_vocab <= M < padded vocabulary size. A
            smaller id is rejected, because it would overwrite a real token.
        real_vocab: Number of real tokens, len(tokenizer). None reads it from the tokenizer in
            hf_dir.
        layout: ``"te"`` or ``"local"``. See ``qwen3_layout.layer_key_table``.

    Returns:
        Dict with the keys ``tp``, ``padded_vocab``, ``real_vocab``, ``mask_token_id``, and
        ``num_keys``. num_keys is the number of model keys per rank, including the kept
        ``_extra_state`` keys.

    Raises:
        FileNotFoundError: If the Hugging Face config is missing.
        CheckpointNotFoundError: If the template holds no checkpoint files.
        TypeError: If a template entry, other than an ``_extra_state`` entry, is not a tensor.
        ValueError: If the layout is unsupported, out_dir/release already holds rank entries, the
            Hugging Face config uses an unsupported feature, the template arguments do not match
            the Hugging Face config, the template is invalid, its keys or shapes do not match the
            converted model, or the Hugging Face weights do not fit the model.
    """
    check_layout(layout)
    out_path = Path(out_dir)
    _check_fresh_release(out_path)
    hf_path = Path(hf_dir)
    state, config = load_hf_state(hf_path)
    dims = Qwen3Dims.from_hf_config(config)
    _, files = find_checkpoint_files(Path(template_dir))
    checkpoints = [load_rank_checkpoint(path) for path in files]
    for rank, ckpt in enumerate(checkpoints):
        if "args" not in ckpt:
            raise ValueError(f"template rank {rank} has no 'args' entry")
    template_args = checkpoints[0]["args"]
    check_args_against_hf(template_args, config, layout)
    tp = len(checkpoints)
    template_tp = getattr(template_args, "tensor_model_parallel_size", None)
    if template_tp is not None and int(template_tp) != tp:
        raise ValueError(
            f"the template has {tp} rank file(s) but tensor_model_parallel_size {template_tp}"
        )
    padded_vocab = _padded_vocab(checkpoints[0]["model"], tp)
    if real_vocab is None:
        real_vocab = _tokenizer_length(hf_path)
    if mask_token_id is None:
        mask_token_id = real_vocab
    template_mask = getattr(template_args, "mask_token", None)
    if template_mask is not None and int(template_mask) != mask_token_id:
        # Training masks with args.mask_token, so the mean-initialized row must be that row.
        raise ValueError(
            f"mask_token_id {mask_token_id} differs from the template's mask_token {template_mask}"
        )
    parts = hf_to_megatron_state(
        state,
        dims,
        tp=tp,
        padded_vocab=padded_vocab,
        real_vocab=real_vocab,
        mask_token_id=mask_token_id,
        layout=layout,
    )
    models = [
        _fill_rank_model(ckpt["model"], part, rank)
        for rank, (ckpt, part) in enumerate(zip(checkpoints, parts, strict=True))
    ]
    for rank, (ckpt, model) in enumerate(zip(checkpoints, models, strict=True)):
        rank_path = out_path / RELEASE / f"mp_rank_{rank:02d}" / MODEL_FILE
        rank_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(_checkpoint_fields(ckpt, model), rank_path)
    (out_path / TRACKER).write_text(RELEASE)
    return {
        "tp": tp,
        "padded_vocab": padded_vocab,
        "real_vocab": real_vocab,
        "mask_token_id": mask_token_id,
        "num_keys": len(models[0]),
    }


def _mask_token_arg(value: str) -> int | None:
    """Parse the --mask-token-id argument.

    Args:
        value: An integer, or ``"auto"``.

    Returns:
        The integer, or None for ``"auto"``, which means real_vocab.

    Raises:
        argparse.ArgumentTypeError: If value is neither an integer nor ``"auto"``.
    """
    if value == "auto":
        return None
    try:
        return int(value)
    except ValueError as err:
        raise argparse.ArgumentTypeError(f"expected an integer or 'auto', got {value!r}") from err


def main(argv: Sequence[str] | None = None) -> int:
    """Run the converter from the command line and print the summary as one
    JSON line.

    Args:
        argv: Arguments without the program name. None reads sys.argv.

    Returns:
        Process exit code, 0 on success.
    """
    parser = argparse.ArgumentParser(
        description="Fill a MegaDLMs template with Hugging Face Qwen3 weights."
    )
    parser.add_argument("--hf", required=True, help="Hugging Face model directory or repo id.")
    parser.add_argument("--template", required=True, help="MegaDLMs template checkpoint root.")
    parser.add_argument("--out", required=True, help="Output checkpoint root.")
    parser.add_argument(
        "--mask-token-id",
        type=_mask_token_arg,
        default="auto",
        help="Mask token id, or 'auto' for real_vocab (default: auto).",
    )
    parser.add_argument(
        "--real-vocab",
        type=int,
        default=None,
        help="len(tokenizer). Default: read from the tokenizer in --hf.",
    )
    parser.add_argument(
        "--layout",
        choices=LAYOUTS,
        default="te",
        help="Megatron layer spec (default: te).",
    )
    args = parser.parse_args(argv)
    summary = convert(
        resolve_hf_dir(args.hf),
        args.template,
        args.out,
        mask_token_id=args.mask_token_id,
        real_vocab=args.real_vocab,
        layout=args.layout,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
