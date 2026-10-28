"""Round-trip tests for the Qwen3 converters (spec section 13, invariant I7).

A tiny random Qwen3 model is saved in Hugging Face format. A synthetic MegaDLMs template, with the
key set, shapes, dtypes, and arguments of a converted run, is filled by ``convert``. ``export``
then writes the Hugging Face weights again. For TP 1 and TP 2, for tied and untied embeddings, and
for the "te" and "local" layer specs, the exported weights equal the original ones in bfloat16. The
exception is row 37 of the vocabulary matrices, the mask row, which is the mean of rows 0..36.

When the padded vocabulary is smaller than the Hugging Face vocabulary, the exporter restores the
rows from the padded size up from the base files, which may be sharded. The other tests cover the
stale release check, the argument check of the template, the mask token check of the export, and
the training fields that the converted checkpoint keeps.

Run with:
    pytest tests/unit/dllm/test_convert_roundtrip.py -v
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")

import torch
from safetensors.torch import load_file, save_file

pytest.importorskip("transformers")

from transformers import Qwen3Config, Qwen3ForCausalLM

from oellm_autoexp.dllm.convert import hf_qwen3_to_megadlm, megadlm_to_hf_qwen3
from oellm_autoexp.dllm.convert.hf_qwen3_to_megadlm import (
    CheckpointNotFoundError,
    convert,
    find_checkpoint_files,
)
from oellm_autoexp.dllm.convert.megadlm_to_hf_qwen3 import export
from oellm_autoexp.dllm.convert.qwen3_layout import Qwen3Dims, hf_to_megatron_state

REAL_VOCAB = 37
PADDED_VOCAB = 48
MASK_ID = 37
BLOCK_SIZE = 4
TOKEN_SHIFT = 1
ROPE_THETA = 1000000.0
RMS_EPS = 1e-6
EXTRA_KEY = "decoder.layers.0.self_attention.core_attention._extra_state"
# Training-run fields. The converted checkpoint drops the first four (spec section 13, step 4).
_TRAINING_FIELDS = ("optimizer", "opt_param_scheduler", "rng_state", "rerun_state_machine")
_VOCAB_KEYS = ("model.embed_tokens.weight", "lm_head.weight")
_RANK_FILE = "model_optim_rng.pt"


def _tiny_config(tied: bool, vocab_size: int = 40) -> Qwen3Config:
    """Return the config of the tiny Qwen3 model that the tests convert.

    Args:
        tied: True to tie the input embedding and the output layer.
        vocab_size: Hugging Face vocabulary size.

    Returns:
        A Qwen3 config with RoPE base 1e6 and RMS epsilon 1e-6, as in the Qwen3 models.
    """
    return Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        tie_word_embeddings=tied,
        rope_theta=ROPE_THETA,
        rms_norm_eps=RMS_EPS,
    )


def _save_hf_model(root: Path, tied: bool, vocab_size: int = 40) -> Path:
    """Save a tiny random Qwen3 model in Hugging Face format.

    Args:
        root: Directory that receives the model in the subdirectory ``hf``.
        tied: True to tie the input embedding and the output layer.
        vocab_size: Hugging Face vocabulary size of the model.

    Returns:
        The model directory.
    """
    torch.manual_seed(0)
    hf_dir = root / "hf"
    Qwen3ForCausalLM(_tiny_config(tied, vocab_size)).save_pretrained(hf_dir)
    return hf_dir


def _original_state(hf_dir: Path) -> dict[str, torch.Tensor]:
    """Load every saved Hugging Face weight file, without the rotary buffers.

    The rotary ``inv_freq`` buffers are not weights, and the converter does not export them, so they
    are left out here too. The base model may be sharded, so every ``*.safetensors`` file is read.

    Args:
        hf_dir: Model directory that holds the safetensors files.

    Returns:
        Dict from Hugging Face key to tensor.
    """
    state: dict[str, torch.Tensor] = {}
    for path in sorted(hf_dir.glob("*.safetensors")):
        state.update(load_file(str(path)))
    return {key: tensor for key, tensor in state.items() if not key.endswith("inv_freq")}


def _shard_base_model(hf_dir: Path, state: dict[str, torch.Tensor]) -> None:
    """Replace the single weight file of a base model with two shards and an
    index file.

    Args:
        hf_dir: Model directory that holds model.safetensors.
        state: Tensors of the model, as loaded from model.safetensors.
    """
    keys = sorted(state)
    half = len(keys) // 2
    groups = {
        "model-00001-of-00002.safetensors": keys[:half],
        "model-00002-of-00002.safetensors": keys[half:],
    }
    weight_map: dict[str, str] = {}
    for name, names in groups.items():
        save_file({key: state[key].contiguous() for key in names}, str(hf_dir / name))
        weight_map.update({key: name for key in names})
    (hf_dir / "model.safetensors").unlink()
    index = {"metadata": {}, "weight_map": weight_map}
    (hf_dir / "model.safetensors.index.json").write_text(json.dumps(index))


def _rank_file(root: Path, rank: int) -> Path:
    """Return the model file of one tensor-parallel rank of a checkpoint root.

    Args:
        root: Checkpoint root.
        rank: Tensor-parallel rank.

    Returns:
        The path ``<root>/release/mp_rank_XX/model_optim_rng.pt``.
    """
    return root / "release" / f"mp_rank_{rank:02d}" / _RANK_FILE


def _template_args(dims: Qwen3Dims, tp: int) -> argparse.Namespace:
    """Return the MegaDLMs arguments of a template run for the Qwen3 model of
    dims.

    The values match the Hugging Face config that _tiny_config writes, as a real run for the model
    would. Tests that need a mismatch change one value afterwards.

    Args:
        dims: Qwen3 sizes of the model.
        tp: Tensor-parallel size of the template.

    Returns:
        The arguments of the template run.
    """
    return argparse.Namespace(
        num_layers=dims.num_layers,
        hidden_size=dims.hidden_size,
        ffn_hidden_size=dims.ffn_hidden_size,
        num_attention_heads=dims.num_heads,
        num_query_groups=dims.num_kv_heads,
        group_query_attention=True,
        kv_channels=dims.head_dim,
        rotary_base=ROPE_THETA,
        norm_epsilon=RMS_EPS,
        rotary_interleaved=False,
        untie_embeddings_and_output_weights=not dims.tie_embeddings,
        swiglu=True,
        qk_layernorm=True,
        add_bias_linear=False,
        normalization="RMSNorm",
        position_embedding_type="rope",
        tensor_model_parallel_size=tp,
        mask_token=MASK_ID,
    )


def _write_template(
    root: Path,
    hf_dir: Path,
    tp: int,
    layout: str = "te",
    padded_vocab: int = PADDED_VOCAB,
) -> Path:
    """Write a synthetic MegaDLMs template for the tensor-parallel size tp.

    The template holds the model keys, shapes, and dtypes of spec section 13, with zero values. Each
    rank also has an ``_extra_state`` entry set to None. The checkpoint holds the arguments of a run
    of the model, and the training fields that the converted checkpoint drops, with iteration 1.

    Args:
        root: Directory that receives the template in the subdirectory ``template``.
        hf_dir: Hugging Face model directory that gives the shapes and the arguments.
        tp: Tensor-parallel size.
        layout: Layer spec of the template, ``"te"`` or ``"local"``.
        padded_vocab: Padded vocabulary size of the template.

    Returns:
        The template checkpoint root.
    """
    dims = Qwen3Dims.from_hf_config(json.loads((hf_dir / "config.json").read_text()))
    ranks = hf_to_megatron_state(
        _original_state(hf_dir),
        dims,
        tp=tp,
        padded_vocab=padded_vocab,
        real_vocab=REAL_VOCAB,
        mask_token_id=MASK_ID,
        layout=layout,
    )
    template = root / "template"
    for rank, part in enumerate(ranks):
        model: dict[str, Any] = {key: torch.zeros_like(value) for key, value in part.items()}
        model[EXTRA_KEY] = None
        checkpoint = {
            "args": _template_args(dims, tp),
            "checkpoint_version": 3.0,
            "iteration": 1,
            "model": model,
            "num_floating_point_operations_so_far": 7,
            "optimizer": {"state": "dropped"},
            "opt_param_scheduler": {"num_steps": 1},
            "rng_state": [{"random_rng_state": "dropped"}],
            "rerun_state_machine": {"state": "dropped"},
        }
        path = _rank_file(template, rank)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
    (template / "latest_checkpointed_iteration.txt").write_text("release")
    return template


def _edit_rank0_model(root: Path, edit: Callable[[dict[str, Any]], object]) -> None:
    """Apply an edit to the model dict of rank 0 of a checkpoint, and save the
    file.

    Args:
        root: Checkpoint root.
        edit: Function that changes the model dict in place.
    """
    path = _rank_file(root, 0)
    checkpoint = torch.load(path, weights_only=False)
    edit(checkpoint["model"])
    torch.save(checkpoint, path)


def _set_rank0_args(root: Path, **changes: object) -> None:
    """Set attributes of the args of rank 0 of a checkpoint, and save the file.

    Args:
        root: Checkpoint root.
        **changes: Attribute names and their new values.
    """
    path = _rank_file(root, 0)
    checkpoint = torch.load(path, weights_only=False)
    for name, value in changes.items():
        setattr(checkpoint["args"], name, value)
    torch.save(checkpoint, path)


def _as_trained_run(root: Path, iteration: int = 100, **changes: object) -> None:
    """Make a converted (release) checkpoint look like the checkpoint of a
    training run.

    Args:
        root: Checkpoint root that holds ``release``.
        iteration: Iteration of the training checkpoint.
        **changes: Attributes of the rank-0 args to set first, such as ``bd_objective``.
    """
    _set_rank0_args(root, **changes)
    (root / "release").rename(root / f"iter_{iteration:07d}")
    (root / "latest_checkpointed_iteration.txt").write_text(str(iteration))


def _convert_model(root: Path, tied: bool, tp: int, layout: str = "te") -> tuple[Path, Path]:
    """Save a tiny model and convert it into a MegaDLMs checkpoint.

    Args:
        root: Working directory of the test. It must not hold a model yet.
        tied: True to tie the input embedding and the output layer.
        tp: Tensor-parallel size of the template.
        layout: Layer spec of the template and of the conversion.

    Returns:
        Tuple (hf_dir, megatron): the Hugging Face model directory and the converted checkpoint.
    """
    hf_dir = _save_hf_model(root, tied)
    template = _write_template(root, hf_dir, tp, layout=layout)
    megatron = root / "megatron"
    convert(
        hf_dir,
        template,
        megatron,
        mask_token_id=MASK_ID,
        real_vocab=REAL_VOCAB,
        layout=layout,
    )
    return hf_dir, megatron


def _assert_matches_original(
    original: dict[str, torch.Tensor], exported: dict[str, torch.Tensor]
) -> None:
    """Check that the exported weights equal the original ones in bfloat16,
    except the mask row.

    Every row of a vocabulary matrix must equal the original row, except the mask row. The mask row
    must be the mean of rows 0..36 of the original. The rows from the padded size up are included,
    so they must equal the base model.

    Args:
        original: Hugging Face tensors of the base model, in float32.
        exported: Tensors read back from the exported safetensors file.
    """
    assert set(exported) == set(original)
    for key, reference in original.items():
        expected = reference.to(torch.bfloat16)
        assert exported[key].dtype == torch.bfloat16, key
        if key not in _VOCAB_KEYS:
            assert torch.equal(exported[key], expected), key
            continue
        kept = [row for row in range(reference.shape[0]) if row != MASK_ID]
        assert torch.equal(exported[key][kept], expected[kept]), key
        mask_row = reference[:REAL_VOCAB].mean(dim=0).to(torch.bfloat16)
        torch.testing.assert_close(
            exported[key][MASK_ID].float(), mask_row.float(), rtol=1e-2, atol=1e-5
        )


@pytest.mark.parametrize("layout", ["te", "local"])
@pytest.mark.parametrize("tp", [1, 2])
@pytest.mark.parametrize("tied", [False, True])
def test_convert_then_export_restores_the_hf_weights(
    tmp_path: Path, tied: bool, tp: int, layout: str
) -> None:
    """I7: HF -> template fill -> export restores every weight, except the mask row.

    The exported weights are bfloat16, so the originals are compared after a cast to bfloat16.
    Row 37 of each vocabulary matrix becomes the mean of rows 0..36, and rows 38 and 39 survive.
    The same holds for the "local" layer spec, which is checked as a file-level round trip.
    """
    hf_dir = _save_hf_model(tmp_path, tied)
    original = _original_state(hf_dir)
    template = _write_template(tmp_path, hf_dir, tp, layout=layout)
    megatron = tmp_path / "megatron"

    summary = convert(
        hf_dir,
        template,
        megatron,
        mask_token_id=MASK_ID,
        real_vocab=REAL_VOCAB,
        layout=layout,
    )

    assert summary == {
        "tp": tp,
        "padded_vocab": PADDED_VOCAB,
        "real_vocab": REAL_VOCAB,
        "mask_token_id": MASK_ID,
        "num_keys": 19 if tied else 20,
    }
    dir_name, files = find_checkpoint_files(megatron)
    assert dir_name == "release"
    assert len(files) == tp

    export_dir = tmp_path / "export"
    result = export(
        megatron,
        hf_dir,
        export_dir,
        block_size=BLOCK_SIZE,
        token_shift=TOKEN_SHIFT,
        mask_token_id=MASK_ID,
        layout=layout,
    )

    assert result["num_tensors"] == len(original)
    assert "model.safetensors" in result["files"]
    _assert_matches_original(original, load_file(str(export_dir / "model.safetensors")))

    config = json.loads((export_dir / "config.json").read_text())
    assert config["dllm_block_size"] == BLOCK_SIZE
    assert config["dllm_mask_token_id"] == MASK_ID
    assert config["dllm_token_shift"] == TOKEN_SHIFT
    assert config["dllm_clean_attention"] == "block"  # the default of export
    assert config["torch_dtype"] == "bfloat16"
    assert config["vocab_size"] == 40

    checkpoint = torch.load(_rank_file(megatron, 0), weights_only=False)
    # The converted checkpoint keeps the template fields, except the training-run fields. Its
    # iteration is 0, and its model is the converted one.
    assert set(checkpoint) == {
        "args",
        "checkpoint_version",
        "iteration",
        "model",
        "num_floating_point_operations_so_far",
    }
    for name in _TRAINING_FIELDS:
        assert name not in checkpoint, name
    assert checkpoint["iteration"] == 0
    assert checkpoint["num_floating_point_operations_so_far"] == 7
    assert checkpoint["args"].tensor_model_parallel_size == tp
    assert checkpoint["model"][EXTRA_KEY] is None
    assert (megatron / "latest_checkpointed_iteration.txt").read_text() == "release"


@pytest.mark.parametrize(("tp", "vocab", "padded"), [(1, 40, 38), (2, 44, 40)])
@pytest.mark.parametrize("tied", [False, True])
def test_export_restores_the_rows_above_the_padded_vocabulary_from_the_base(
    tmp_path: Path, tied: bool, tp: int, vocab: int, padded: int
) -> None:
    """With padded < vocab_size, the rows from padded up come from the base
    files.

    The exported matrix then equals the base exactly, except the mask
    row.
    """
    hf_dir = _save_hf_model(tmp_path, tied, vocab_size=vocab)
    original = _original_state(hf_dir)
    template = _write_template(tmp_path, hf_dir, tp, padded_vocab=padded)
    megatron = tmp_path / "megatron"

    summary = convert(hf_dir, template, megatron, mask_token_id=MASK_ID, real_vocab=REAL_VOCAB)
    assert summary["padded_vocab"] == padded

    export_dir = tmp_path / "export"
    export(
        megatron,
        hf_dir,
        export_dir,
        block_size=BLOCK_SIZE,
        token_shift=TOKEN_SHIFT,
        mask_token_id=MASK_ID,
    )
    _assert_matches_original(original, load_file(str(export_dir / "model.safetensors")))
    assert json.loads((export_dir / "config.json").read_text())["vocab_size"] == vocab


def test_export_reads_rows_above_the_padded_size_from_a_sharded_base(tmp_path: Path) -> None:
    """A sharded base model is read through model.safetensors.index.json."""
    hf_dir = _save_hf_model(tmp_path, tied=False, vocab_size=44)
    original = _original_state(hf_dir)
    _shard_base_model(hf_dir, original)
    assert not (hf_dir / "model.safetensors").exists()
    template = _write_template(tmp_path, hf_dir, tp=2, padded_vocab=40)
    megatron = tmp_path / "megatron"
    convert(hf_dir, template, megatron, mask_token_id=MASK_ID, real_vocab=REAL_VOCAB)

    export_dir = tmp_path / "export"
    export(
        megatron,
        hf_dir,
        export_dir,
        block_size=BLOCK_SIZE,
        token_shift=TOKEN_SHIFT,
        mask_token_id=MASK_ID,
    )
    _assert_matches_original(original, load_file(str(export_dir / "model.safetensors")))


def test_convert_refuses_an_out_dir_that_already_holds_release_ranks(tmp_path: Path) -> None:
    """Stale ranks in out_dir/release raise ValueError before anything is
    written."""
    hf_dir = _save_hf_model(tmp_path, tied=False)
    template = _write_template(tmp_path, hf_dir, tp=1)
    out = tmp_path / "megatron"
    (out / "release" / "mp_rank_00").mkdir(parents=True)
    with pytest.raises(ValueError, match="mp_rank"):
        convert(hf_dir, template, out, mask_token_id=MASK_ID, real_vocab=REAL_VOCAB)
    assert not (out / "latest_checkpointed_iteration.txt").exists()
    assert not _rank_file(out, 0).exists()


@pytest.mark.parametrize(
    ("changes", "match"),
    [({"tensor_model_parallel_size": 2}, "rank file"), ({"mask_token": MASK_ID + 1}, "mask_token")],
)
def test_convert_rejects_a_template_whose_tp_or_mask_token_differs(
    tmp_path: Path, changes: dict[str, Any], match: str
) -> None:
    """The rank count must equal the template's TP size, and the mask row must
    be the template's mask_token."""
    hf_dir = _save_hf_model(tmp_path, tied=False)
    template = _write_template(tmp_path, hf_dir, tp=1)
    _set_rank0_args(template, **changes)
    out = tmp_path / "megatron"
    with pytest.raises(ValueError, match=match):
        convert(hf_dir, template, out, mask_token_id=MASK_ID, real_vocab=REAL_VOCAB)
    assert not out.exists()


def test_convert_rejects_template_args_that_do_not_match_the_hf_config(tmp_path: Path) -> None:
    """A template trained with the wrong RoPE base raises ValueError, and
    nothing is written."""
    hf_dir = _save_hf_model(tmp_path, tied=False)
    template = _write_template(tmp_path, hf_dir, tp=1)
    _set_rank0_args(template, rotary_base=10000)
    out = tmp_path / "megatron"
    with pytest.raises(ValueError, match="rotary_base"):
        convert(hf_dir, template, out, mask_token_id=MASK_ID, real_vocab=REAL_VOCAB)
    assert not out.exists()


def test_export_rejects_a_mask_token_id_that_differs_from_the_checkpoint(tmp_path: Path) -> None:
    """An exported mask id other than the checkpoint's mask_token raises
    ValueError."""
    hf_dir, megatron = _convert_model(tmp_path, tied=False, tp=1)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="mask_token"):
        export(
            megatron,
            hf_dir,
            out,
            block_size=BLOCK_SIZE,
            token_shift=TOKEN_SHIFT,
            mask_token_id=MASK_ID + 1,
        )
    assert not out.exists()


def _add_unknown_key(model: dict[str, Any]) -> None:
    """Add a model key that the converter does not produce."""
    model["decoder.layers.1.unknown.weight"] = torch.zeros(3)


def _drop_final_norm(model: dict[str, Any]) -> None:
    """Remove a model key that the converter fills."""
    del model["decoder.final_layernorm.weight"]


def _widen_final_norm(model: dict[str, Any]) -> None:
    """Give a model key a shape that differs from the converted tensor."""
    model["decoder.final_layernorm.weight"] = torch.zeros(17)


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        pytest.param(_add_unknown_key, "unknown in template", id="unknown-key"),
        pytest.param(_drop_final_norm, "missing from template", id="missing-key"),
        pytest.param(_widen_final_norm, "has converted shape", id="wrong-shape"),
    ],
)
def test_convert_rejects_a_bad_template_and_writes_nothing(
    tmp_path: Path, edit: Callable[[dict[str, Any]], object], message: str
) -> None:
    """A missing or extra model key, or a wrong shape, raises ValueError and
    writes nothing."""
    hf_dir = _save_hf_model(tmp_path, tied=False)
    template = _write_template(tmp_path, hf_dir, tp=1)
    _edit_rank0_model(template, edit)
    out = tmp_path / "megatron"
    with pytest.raises(ValueError, match=message):
        convert(hf_dir, template, out, mask_token_id=MASK_ID, real_vocab=REAL_VOCAB)
    assert not out.exists()


def test_export_splits_large_weights_into_indexed_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Weights above the shard limit go to indexed shards, and their tensors
    match one file."""
    hf_dir, megatron = _convert_model(tmp_path, tied=False, tp=2)
    settings = {"block_size": BLOCK_SIZE, "token_shift": TOKEN_SHIFT, "mask_token_id": MASK_ID}
    single_dir = tmp_path / "single"
    export(megatron, hf_dir, single_dir, **settings)
    expected = load_file(str(single_dir / "model.safetensors"))

    monkeypatch.setattr(megadlm_to_hf_qwen3, "MAX_SHARD_BYTES", 4096)
    sharded_dir = tmp_path / "sharded"
    result = export(megatron, hf_dir, sharded_dir, **settings)

    assert not (sharded_dir / "model.safetensors").exists()
    index = json.loads((sharded_dir / "model.safetensors.index.json").read_text())
    names = sorted(set(index["weight_map"].values()))
    assert len(names) >= 2
    assert names == [name for name in result["files"] if name.startswith("model-")]
    assert set(index["weight_map"]) == set(expected)
    merged: dict[str, torch.Tensor] = {}
    for name in names:
        shard = load_file(str(sharded_dir / name))
        shard_bytes = sum(tensor.numel() * tensor.element_size() for tensor in shard.values())
        assert shard_bytes <= 4096, name
        assert set(shard) == {key for key, owner in index["weight_map"].items() if owner == name}
        merged.update(shard)
    assert set(merged) == set(expected)
    for key in expected:
        assert torch.equal(merged[key], expected[key]), key
    total = sum(tensor.numel() * tensor.element_size() for tensor in expected.values())
    assert index["metadata"]["total_size"] == total


@pytest.mark.parametrize(
    ("block_size", "token_shift", "mask_token_id", "message"),
    [
        (0, TOKEN_SHIFT, MASK_ID, "block_size"),
        (BLOCK_SIZE, 2, MASK_ID, "token_shift"),
        (BLOCK_SIZE, TOKEN_SHIFT, 40, "mask_token_id"),
    ],
)
def test_export_rejects_unsupported_block_diffusion_settings(
    tmp_path: Path, block_size: int, token_shift: int, mask_token_id: int, message: str
) -> None:
    """A block size below 1, a shift other than 0 or 1, and a mask id outside
    the vocab raise."""
    hf_dir, megatron = _convert_model(tmp_path, tied=False, tp=1)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match=message):
        export(
            megatron,
            hf_dir,
            out,
            block_size=block_size,
            token_shift=token_shift,
            mask_token_id=mask_token_id,
        )
    assert not out.exists()


@pytest.mark.parametrize("clean_attention", ["block", "causal"])
def test_export_writes_the_clean_attention_next_to_the_block_fields(
    tmp_path: Path, clean_attention: str
) -> None:
    """The exporter writes dllm_clean_attention with the value that export
    receives.

    Spec section 15: an nld model is exported with "causal", and a bd3 model with "block".
    """
    hf_dir, megatron = _convert_model(tmp_path, tied=False, tp=1)
    export_dir = tmp_path / "export"
    export(
        megatron,
        hf_dir,
        export_dir,
        block_size=BLOCK_SIZE,
        token_shift=TOKEN_SHIFT,
        mask_token_id=MASK_ID,
        clean_attention=clean_attention,
    )
    config = json.loads((export_dir / "config.json").read_text())
    assert config["dllm_clean_attention"] == clean_attention
    assert config["dllm_block_size"] == BLOCK_SIZE
    assert config["dllm_mask_token_id"] == MASK_ID
    assert config["dllm_token_shift"] == TOKEN_SHIFT


@pytest.mark.parametrize(
    ("objective", "requested", "written"),
    [
        ("nld", None, "causal"),
        ("nld", "causal", "causal"),
        ("bd3", None, "block"),
        ("bd3", "block", "block"),
        ("ar", None, "block"),
    ],
)
def test_export_takes_the_clean_attention_from_the_checkpoint_objective(
    tmp_path: Path, objective: str, requested: str | None, written: str
) -> None:
    """Without a value the exporter writes the clean attention of bd_objective
    (spec section 15)."""
    hf_dir, megatron = _convert_model(tmp_path, tied=False, tp=1)
    _as_trained_run(megatron, bd_objective=objective)
    export_dir = tmp_path / "export"
    export(
        megatron,
        hf_dir,
        export_dir,
        block_size=BLOCK_SIZE,
        token_shift=TOKEN_SHIFT,
        mask_token_id=MASK_ID,
        clean_attention=requested,
    )
    config = json.loads((export_dir / "config.json").read_text())
    assert config["dllm_clean_attention"] == written


@pytest.mark.parametrize(("objective", "requested"), [("nld", "block"), ("bd3", "causal")])
def test_export_rejects_a_clean_attention_that_the_objective_did_not_train(
    tmp_path: Path, objective: str, requested: str
) -> None:
    """An nld checkpoint exported as block (or bd3 as causal) raises ValueError
    and writes nothing."""
    hf_dir, megatron = _convert_model(tmp_path, tied=False, tp=1)
    _as_trained_run(megatron, bd_objective=objective)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="bd_objective"):
        export(
            megatron,
            hf_dir,
            out,
            block_size=BLOCK_SIZE,
            token_shift=TOKEN_SHIFT,
            mask_token_id=MASK_ID,
            clean_attention=requested,
        )
    assert not out.exists()


@pytest.mark.parametrize("objective", ["bd3", "nld"])
def test_export_rejects_a_token_shift_that_the_checkpoint_did_not_train(
    tmp_path: Path, objective: str
) -> None:
    """A trained bd3 or nld checkpoint keeps its bd_token_shift; another shift
    raises ValueError."""
    hf_dir, megatron = _convert_model(tmp_path, tied=False, tp=1)
    _as_trained_run(megatron, bd_objective=objective, bd_token_shift=1 - TOKEN_SHIFT)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="bd_token_shift"):
        export(
            megatron,
            hf_dir,
            out,
            block_size=BLOCK_SIZE,
            token_shift=TOKEN_SHIFT,
            mask_token_id=MASK_ID,
        )
    assert not out.exists()


def test_export_of_an_ar_checkpoint_ignores_its_token_shift(tmp_path: Path) -> None:
    """The ar objective has no prediction shift, so its bd_token_shift is not
    compared."""
    hf_dir, megatron = _convert_model(tmp_path, tied=False, tp=1)
    _as_trained_run(megatron, bd_objective="ar", bd_token_shift=1 - TOKEN_SHIFT)
    export_dir = tmp_path / "export"
    export(
        megatron,
        hf_dir,
        export_dir,
        block_size=BLOCK_SIZE,
        token_shift=TOKEN_SHIFT,
        mask_token_id=MASK_ID,
    )
    config = json.loads((export_dir / "config.json").read_text())
    assert config["dllm_token_shift"] == TOKEN_SHIFT
    assert config["dllm_clean_attention"] == "block"


def test_export_of_a_release_conversion_ignores_the_template_settings(tmp_path: Path) -> None:
    """A release checkpoint holds the template run's args, which say nothing
    about training.

    The I8 check exports the step-0 conversion with B = 1 and shift 1
    (spec section 14).
    """
    hf_dir, megatron = _convert_model(tmp_path, tied=False, tp=1)
    _set_rank0_args(megatron, bd_objective="nld", bd_token_shift=1 - TOKEN_SHIFT)
    export_dir = tmp_path / "export"
    export(
        megatron,
        hf_dir,
        export_dir,
        block_size=BLOCK_SIZE,
        token_shift=TOKEN_SHIFT,
        mask_token_id=MASK_ID,
    )
    config = json.loads((export_dir / "config.json").read_text())
    assert config["dllm_token_shift"] == TOKEN_SHIFT
    assert config["dllm_clean_attention"] == "block"


@pytest.mark.parametrize(
    ("trained", "iteration", "requested", "accepted"),
    [
        ({"bd_block_size": BLOCK_SIZE, "bd_block_schedule": "none"}, 100, BLOCK_SIZE, True),
        ({"bd_block_size": 2 * BLOCK_SIZE, "bd_block_schedule": "none"}, 100, BLOCK_SIZE, False),
        # Scheduled run: B = 2 for the first half, then B = 4 (the last entry at the end).
        ({"bd_block_size": 2, "bd_block_schedule": "0:2,0.5:4"}, 100, 4, True),
        ({"bd_block_size": 2, "bd_block_schedule": "0:2,0.5:4"}, 100, 2, False),
        ({"bd_block_size": 2, "bd_block_schedule": "0:2,0.5:4"}, 40, 2, True),
    ],
)
def test_export_checks_the_block_size_that_the_checkpoint_trained_with(
    tmp_path: Path, trained: dict[str, Any], iteration: int, requested: int, accepted: bool
) -> None:
    """dllm_block_size must be the block size in effect at the checkpoint's
    iteration (spec section 8)."""
    hf_dir, megatron = _convert_model(tmp_path, tied=False, tp=1)
    _as_trained_run(megatron, iteration, bd_objective="bd3", train_iters=100, **trained)
    out = tmp_path / "out"
    settings = {"token_shift": TOKEN_SHIFT, "mask_token_id": MASK_ID}
    if accepted:
        export(megatron, hf_dir, out, block_size=requested, **settings)
        assert json.loads((out / "config.json").read_text())["dllm_block_size"] == requested
    else:
        with pytest.raises(ValueError, match="block size of the checkpoint"):
            export(megatron, hf_dir, out, block_size=requested, **settings)
        assert not out.exists()


def test_export_of_an_ar_checkpoint_ignores_its_block_size(tmp_path: Path) -> None:
    """The ar objective has no block size; train_then_eval.sh exports ar runs
    with B = 1."""
    hf_dir, megatron = _convert_model(tmp_path, tied=False, tp=1)
    _as_trained_run(megatron, bd_objective="ar", bd_block_size=8, train_iters=100)
    out = tmp_path / "out"
    export(megatron, hf_dir, out, block_size=1, token_shift=1, mask_token_id=MASK_ID)
    assert json.loads((out / "config.json").read_text())["dllm_block_size"] == 1


def test_export_rejects_a_trained_checkpoint_without_args(tmp_path: Path) -> None:
    hf_dir, megatron = _convert_model(tmp_path, tied=False, tp=1)
    _as_trained_run(megatron)
    path = megatron / "iter_0000100" / "mp_rank_00" / _RANK_FILE
    checkpoint = torch.load(path, weights_only=False)
    del checkpoint["args"]
    torch.save(checkpoint, path)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="no 'args' entry"):
        export(
            megatron,
            hf_dir,
            out,
            block_size=BLOCK_SIZE,
            token_shift=TOKEN_SHIFT,
            mask_token_id=MASK_ID,
        )
    assert not out.exists()


def test_export_rejects_an_unsupported_clean_attention(tmp_path: Path) -> None:
    """A clean attention other than block or causal raises ValueError and
    writes nothing."""
    hf_dir, megatron = _convert_model(tmp_path, tied=False, tp=1)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="clean_attention"):
        export(
            megatron,
            hf_dir,
            out,
            block_size=BLOCK_SIZE,
            token_shift=TOKEN_SHIFT,
            mask_token_id=MASK_ID,
            clean_attention="full",
        )
    assert not out.exists()


def test_export_command_line_writes_the_clean_attention(tmp_path: Path) -> None:
    """--clean-attention sets dllm_clean_attention. The default is block, and other values fail."""
    hf_dir, megatron = _convert_model(tmp_path, tied=False, tp=1)
    common = [
        "--megatron",
        str(megatron),
        "--hf-base",
        str(hf_dir),
        "--block-size",
        str(BLOCK_SIZE),
        "--token-shift",
        str(TOKEN_SHIFT),
        "--mask-token-id",
        str(MASK_ID),
    ]
    causal_dir = tmp_path / "causal"
    status = megadlm_to_hf_qwen3.main(
        [*common, "--out", str(causal_dir), "--clean-attention", "causal"]
    )
    assert status == 0
    default_dir = tmp_path / "default"
    assert megadlm_to_hf_qwen3.main([*common, "--out", str(default_dir)]) == 0
    causal = json.loads((causal_dir / "config.json").read_text())
    default = json.loads((default_dir / "config.json").read_text())
    assert causal["dllm_clean_attention"] == "causal"
    assert default["dllm_clean_attention"] == "block"
    refused = tmp_path / "refused"
    with pytest.raises(SystemExit):
        megadlm_to_hf_qwen3.main([*common, "--out", str(refused), "--clean-attention", "full"])
    assert not refused.exists()


def test_export_refuses_to_write_into_the_base_model_directory(tmp_path: Path) -> None:
    """The output directory must differ from the base model, so its weights are
    never replaced."""
    hf_dir, megatron = _convert_model(tmp_path, tied=False, tp=1)
    with pytest.raises(ValueError, match="must differ"):
        export(
            megatron,
            hf_dir,
            hf_dir,
            block_size=BLOCK_SIZE,
            token_shift=TOKEN_SHIFT,
            mask_token_id=MASK_ID,
        )
    assert (hf_dir / "model.safetensors").is_file()


def test_find_checkpoint_files_rejects_missing_and_pipeline_parallel_checkpoints(
    tmp_path: Path,
) -> None:
    """No tracker or no iteration directory is CheckpointNotFoundError, and
    pipeline ranks raise."""
    assert issubclass(CheckpointNotFoundError, FileNotFoundError)
    assert issubclass(CheckpointNotFoundError, ValueError)
    with pytest.raises(FileNotFoundError):
        find_checkpoint_files(tmp_path)
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("100\n")
    with pytest.raises(ValueError, match="does not exist"):
        find_checkpoint_files(tmp_path)
    pipeline_file = tmp_path / "iter_0000100" / "mp_rank_00_000" / _RANK_FILE
    pipeline_file.parent.mkdir(parents=True)
    pipeline_file.write_bytes(b"")
    with pytest.raises(ValueError, match="pipeline-parallel"):
        find_checkpoint_files(tmp_path)


def test_command_line_interfaces_convert_then_export(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Both command lines run end to end and print their summaries as one JSON
    line."""
    hf_dir = _save_hf_model(tmp_path, tied=False)
    template = _write_template(tmp_path, hf_dir, tp=1)
    megatron = tmp_path / "megatron"
    status = hf_qwen3_to_megadlm.main(
        [
            "--hf",
            str(hf_dir),
            "--template",
            str(template),
            "--out",
            str(megatron),
            "--real-vocab",
            str(REAL_VOCAB),
        ]
    )
    assert status == 0
    converted = json.loads(capsys.readouterr().out)
    assert converted["mask_token_id"] == REAL_VOCAB  # the default "auto" is real_vocab

    export_dir = tmp_path / "export"
    status = megadlm_to_hf_qwen3.main(
        [
            "--megatron",
            str(megatron),
            "--hf-base",
            str(hf_dir),
            "--out",
            str(export_dir),
            "--block-size",
            str(BLOCK_SIZE),
            "--token-shift",
            str(TOKEN_SHIFT),
            "--mask-token-id",
            str(MASK_ID),
        ]
    )
    assert status == 0
    exported = json.loads(capsys.readouterr().out)
    assert exported["out_dir"] == str(export_dir)
    assert (export_dir / "model.safetensors").is_file()
