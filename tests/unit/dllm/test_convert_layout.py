"""Invariant I7 and layout tests for the Qwen3 <-> MegaDLMs converter (spec
section 13).

I7: random Hugging Face tensors survive HF -> Megatron -> HF exactly for TP 1 and 2, and for TP 4
with four KV groups. This holds for tied and untied models and for both layer specs ("te" and
"local"). With a mask token, only row 37 of each vocabulary matrix changes, and it becomes the mean
of rows 0..36. When the padded vocabulary is smaller than the Hugging Face vocabulary, the rows
below the padded size still round trip exactly, and the rows at or above it come back as zeros. The
exporter restores those rows from the base model (see test_convert_roundtrip.py). The other tests
check the Megatron attention and fc1 split semantics, the sharding rules, the argument checks
against the Hugging Face config, the supported Hugging Face features, and the settings that must
raise.

Run with:
    pytest tests/unit/dllm/test_convert_layout.py -v
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Any

import pytest

pytest.importorskip("torch")

import torch

from oellm_autoexp.dllm.convert.qwen3_layout import (
    Qwen3Dims,
    check_args_against_hf,
    hf_to_megatron_state,
    layer_key_table,
    megatron_to_hf_state,
    qkv_hf_to_megatron,
    restore_vocab_tail,
    shard_cols,
    shard_fc1,
    shard_qkv,
    shard_rows,
    unshard_cols,
    unshard_fc1,
    unshard_qkv,
    unshard_rows,
    vocab_to_hf,
    vocab_to_megatron,
)

REAL_VOCAB = 37
PADDED_VOCAB = 48
MASK_ID = 37
_DIMS = Qwen3Dims(
    num_layers=2,
    hidden_size=16,
    ffn_hidden_size=24,
    num_heads=4,
    num_kv_heads=2,
    head_dim=8,
    vocab_size=40,
    tie_embeddings=False,
)
_DIMS_KV4 = replace(_DIMS, num_kv_heads=4)
_SHAPES = [(_DIMS, 1), (_DIMS, 2), (_DIMS_KV4, 4)]
# (tensor-parallel size, Hugging Face vocabulary, padded vocabulary). The padded size is below the
# Hugging Face vocabulary, so the converter truncates, and the exporter restores the rows above it.
_TRUNCATED_SHAPES = [(1, 40, 38), (2, 44, 40)]
_VOCAB_KEYS = ("model.embed_tokens.weight", "lm_head.weight")


def _random_hf(dims: Qwen3Dims, seed: int = 0) -> dict[str, torch.Tensor]:
    """Build a random Hugging Face state dict with the keys of
    Qwen3ForCausalLM.

    Args:
        dims: Qwen3 sizes. ``lm_head.weight`` is included only when the model is untied.
        seed: Seed of the random generator.

    Returns:
        Dict from Hugging Face key to float32 tensor.
    """
    gen = torch.Generator().manual_seed(seed)
    hidden, head, ffn = dims.hidden_size, dims.head_dim, dims.ffn_hidden_size

    def rand(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=gen)

    hf = {
        "model.embed_tokens.weight": rand(dims.vocab_size, hidden),
        "model.norm.weight": rand(hidden),
    }
    for layer in range(dims.num_layers):
        prefix = f"model.layers.{layer}"
        hf[f"{prefix}.input_layernorm.weight"] = rand(hidden)
        hf[f"{prefix}.self_attn.q_proj.weight"] = rand(dims.num_heads * head, hidden)
        hf[f"{prefix}.self_attn.k_proj.weight"] = rand(dims.num_kv_heads * head, hidden)
        hf[f"{prefix}.self_attn.v_proj.weight"] = rand(dims.num_kv_heads * head, hidden)
        hf[f"{prefix}.self_attn.o_proj.weight"] = rand(hidden, dims.num_heads * head)
        hf[f"{prefix}.self_attn.q_norm.weight"] = rand(head)
        hf[f"{prefix}.self_attn.k_norm.weight"] = rand(head)
        hf[f"{prefix}.post_attention_layernorm.weight"] = rand(hidden)
        hf[f"{prefix}.mlp.gate_proj.weight"] = rand(ffn, hidden)
        hf[f"{prefix}.mlp.up_proj.weight"] = rand(ffn, hidden)
        hf[f"{prefix}.mlp.down_proj.weight"] = rand(hidden, ffn)
    if not dims.tie_embeddings:
        hf["lm_head.weight"] = rand(dims.vocab_size, hidden)
    return hf


def _convert(
    hf: dict[str, torch.Tensor],
    dims: Qwen3Dims,
    tp: int,
    mask_token_id: int | None = None,
    layout: str = "te",
) -> list[dict[str, torch.Tensor]]:
    """Run hf_to_megatron_state with the padded and real vocabulary sizes of
    these tests.

    Args:
        hf: Hugging Face state dict.
        dims: Qwen3 sizes.
        tp: Tensor-parallel size.
        mask_token_id: Mask token id, or None.
        layout: Layer spec, ``"te"`` or ``"local"``.

    Returns:
        One Megatron state dict per rank.
    """
    return hf_to_megatron_state(
        hf,
        dims,
        tp=tp,
        padded_vocab=PADDED_VOCAB,
        real_vocab=REAL_VOCAB,
        mask_token_id=mask_token_id,
        layout=layout,
    )


@pytest.mark.parametrize(("base", "tp"), _SHAPES)
@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("layout", ["te", "local"])
def test_round_trip_is_exact_without_mask(
    base: Qwen3Dims, tp: int, tied: bool, layout: str
) -> None:
    """I7: HF -> Megatron -> HF returns every key bit-for-bit when there is no mask token."""
    dims = replace(base, tie_embeddings=tied)
    hf = _random_hf(dims)
    shards = _convert(hf, dims, tp, layout=layout)
    assert len(shards) == tp
    back = megatron_to_hf_state(shards, dims, layout=layout)
    assert set(back) == set(hf)
    for key in hf:
        assert torch.equal(back[key], hf[key]), key


@pytest.mark.parametrize(("base", "tp"), _SHAPES)
@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("layout", ["te", "local"])
def test_round_trip_changes_only_the_mask_row(
    base: Qwen3Dims, tp: int, tied: bool, layout: str
) -> None:
    """I7 with a mask token: only row 37 of each vocabulary matrix changes, to the mean of 0..36."""
    dims = replace(base, tie_embeddings=tied)
    hf = _random_hf(dims, seed=1)
    shards = _convert(hf, dims, tp, mask_token_id=MASK_ID, layout=layout)
    back = megatron_to_hf_state(shards, dims, layout=layout)
    assert set(back) == set(hf)
    for key in hf:
        expected = hf[key].clone()
        if key in _VOCAB_KEYS:
            expected[MASK_ID] = hf[key][:REAL_VOCAB].mean(dim=0)
        assert torch.equal(back[key], expected), key


@pytest.mark.parametrize(("tp", "vocab", "padded"), _TRUNCATED_SHAPES)
@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("layout", ["te", "local"])
def test_truncated_vocab_round_trip_is_exact_below_the_padded_size(
    tp: int, vocab: int, padded: int, tied: bool, layout: str
) -> None:
    """With padded < V_hf, rows below padded round trip exactly, and the rows
    above come back as 0.

    The mask row lies below padded, so it becomes the mean of rows 0..36 as in the other tests.
    """
    dims = replace(_DIMS, vocab_size=vocab, tie_embeddings=tied)
    hf = _random_hf(dims, seed=6)
    shards = hf_to_megatron_state(
        hf,
        dims,
        tp=tp,
        padded_vocab=padded,
        real_vocab=REAL_VOCAB,
        mask_token_id=MASK_ID,
        layout=layout,
    )
    back = megatron_to_hf_state(shards, dims, layout=layout)
    assert set(back) == set(hf)
    width = _DIMS.hidden_size
    for key in hf:
        if key not in _VOCAB_KEYS:
            assert torch.equal(back[key], hf[key]), key
            continue
        expected = hf[key][:padded].clone()
        expected[MASK_ID] = hf[key][:REAL_VOCAB].mean(dim=0)
        assert torch.equal(back[key][:padded], expected), key
        assert torch.equal(back[key][padded:], torch.zeros(vocab - padded, width)), key


def test_restore_vocab_tail_replaces_only_the_rows_from_the_padded_size() -> None:
    """Rows from padded_rows on come from the base, and the rows below
    padded_rows are kept."""
    weight = torch.full((5, 2), -1.0)
    base = torch.arange(10.0).reshape(5, 2)
    restored = restore_vocab_tail(weight, base, padded_rows=3)
    assert torch.equal(restored[:3], weight[:3])
    assert torch.equal(restored[3:], base[3:])
    assert torch.equal(restore_vocab_tail(weight, base, padded_rows=5), weight)
    with pytest.raises(ValueError, match="shape"):
        restore_vocab_tail(weight, torch.zeros(4, 2), padded_rows=3)
    with pytest.raises(ValueError, match="padded_rows"):
        restore_vocab_tail(weight, base, padded_rows=-1)


def test_megatron_shard_shapes_for_tp2() -> None:
    """Every Megatron key has the per-rank shape of spec section 13 at TP 2."""
    shards = _convert(_random_hf(_DIMS), _DIMS, tp=2)
    expected = {
        "embedding.word_embeddings.weight": (24, 16),
        "output_layer.weight": (24, 16),
        "decoder.final_layernorm.weight": (16,),
        "decoder.layers.0.self_attention.linear_qkv.layer_norm_weight": (16,),
        "decoder.layers.0.self_attention.linear_qkv.weight": (32, 16),
        "decoder.layers.0.self_attention.q_layernorm.weight": (8,),
        "decoder.layers.0.self_attention.k_layernorm.weight": (8,),
        "decoder.layers.0.self_attention.linear_proj.weight": (16, 16),
        "decoder.layers.0.mlp.linear_fc1.layer_norm_weight": (16,),
        "decoder.layers.0.mlp.linear_fc1.weight": (24, 16),
        "decoder.layers.0.mlp.linear_fc2.weight": (16, 12),
    }
    for key, shape in expected.items():
        assert tuple(shards[0][key].shape) == shape, key
    assert len(shards[0]) == 2 * 8 + 3


def test_tied_model_has_no_output_layer_key() -> None:
    """output_layer.weight is written only for untied models."""
    untied = _convert(_random_hf(_DIMS), _DIMS, tp=2)
    assert "output_layer.weight" in untied[0]
    tied_dims = replace(_DIMS, tie_embeddings=True)
    tied = _convert(_random_hf(tied_dims), tied_dims, tp=2)
    assert "output_layer.weight" not in tied[0]


def test_tied_lm_head_equal_to_embedding_is_accepted() -> None:
    """A tied model may carry lm_head.weight when it equals the embedding, and
    it is then used."""
    dims = replace(_DIMS, tie_embeddings=True)
    hf = _random_hf(dims)
    hf["lm_head.weight"] = hf["model.embed_tokens.weight"].clone()
    shards = _convert(hf, dims, tp=1)
    back = megatron_to_hf_state(shards, dims)
    assert "lm_head.weight" not in back
    assert torch.equal(back["model.embed_tokens.weight"], hf["model.embed_tokens.weight"])


def test_tied_lm_head_different_from_embedding_raises() -> None:
    """A tied model whose lm_head.weight differs from the embedding is
    rejected."""
    dims = replace(_DIMS, tie_embeddings=True)
    hf = _random_hf(dims)
    hf["lm_head.weight"] = hf["model.embed_tokens.weight"] + 1.0
    with pytest.raises(ValueError, match="lm_head.weight differs"):
        _convert(hf, dims, tp=1)


def test_unused_hf_key_raises_and_rotary_inv_freq_is_allowed() -> None:
    """Unused Hugging Face keys raise, except rotary inv_freq buffers."""
    hf = _random_hf(_DIMS)
    hf["model.rotary_emb.inv_freq"] = torch.zeros(4)
    _convert(hf, _DIMS, tp=1)
    hf["model.layers.0.self_attn.q_proj.bias"] = torch.zeros(32)
    with pytest.raises(ValueError, match="unused Hugging Face keys"):
        _convert(hf, _DIMS, tp=1)


def test_missing_hf_key_raises_key_error() -> None:
    """A required Hugging Face key that is absent raises KeyError."""
    hf = _random_hf(_DIMS)
    del hf["model.layers.1.mlp.up_proj.weight"]
    with pytest.raises(KeyError, match="up_proj"):
        _convert(hf, _DIMS, tp=1)


def test_hf_shape_mismatch_raises() -> None:
    """A Hugging Face tensor with the wrong shape raises ValueError."""
    hf = _random_hf(_DIMS)
    hf["model.layers.0.self_attn.q_norm.weight"] = torch.zeros(4)
    with pytest.raises(ValueError, match="q_norm"):
        _convert(hf, _DIMS, tp=1)


def test_group_count_not_divisible_by_tp_raises() -> None:
    """G % tp != 0 raises ValueError, both in the shard function and in the
    full conversion."""
    full = torch.zeros((4 + 2 * 2) * 8, 16)
    with pytest.raises(ValueError, match="num_kv_heads"):
        shard_qkv(full, _DIMS, tp=4)
    with pytest.raises(ValueError, match="num_kv_heads"):
        _convert(_random_hf(_DIMS), _DIMS, tp=4)


@pytest.mark.parametrize("mask_id", [10, PADDED_VOCAB])
def test_mask_token_outside_the_mask_row_range_raises(mask_id: int) -> None:
    """The mask id must lie in [real_vocab, padded_vocab), so it never
    overwrites a real token."""
    with pytest.raises(ValueError, match="mask_token_id"):
        _convert(_random_hf(_DIMS), _DIMS, tp=1, mask_token_id=mask_id)


def test_unknown_layout_raises() -> None:
    """Only the "te" and "local" layouts are supported."""
    with pytest.raises(ValueError, match="layout"):
        layer_key_table(0, "megatron")
    with pytest.raises(ValueError, match="layout"):
        _convert(_random_hf(_DIMS), _DIMS, tp=1, layout="bogus")
    with pytest.raises(ValueError, match="layout"):
        megatron_to_hf_state([{}], _DIMS, layout="bogus")


def test_local_layout_uses_separate_norm_modules() -> None:
    """The local spec has input_layernorm and pre_mlp_layernorm, and no fused
    layer norms."""
    local = layer_key_table(3, "local")
    assert local["decoder.layers.3.input_layernorm.weight"] == (
        "none",
        "model.layers.3.input_layernorm.weight",
    )
    assert local["decoder.layers.3.pre_mlp_layernorm.weight"] == (
        "none",
        "model.layers.3.post_attention_layernorm.weight",
    )
    assert not any(key.endswith("layer_norm_weight") for key in local)


def test_te_layout_fuses_norms_into_linear_qkv_and_fc1() -> None:
    """The TE spec keeps the input and post-attention norms in linear_qkv and
    linear_fc1."""
    te = layer_key_table(3, "te")
    assert te["decoder.layers.3.self_attention.linear_qkv.layer_norm_weight"] == (
        "none",
        "model.layers.3.input_layernorm.weight",
    )
    assert te["decoder.layers.3.mlp.linear_fc1.layer_norm_weight"] == (
        "none",
        "model.layers.3.post_attention_layernorm.weight",
    )
    assert "decoder.layers.3.input_layernorm.weight" not in te
    assert "decoder.layers.3.pre_mlp_layernorm.weight" not in te


@pytest.mark.parametrize("layout", ["te", "local"])
def test_layer_table_has_eight_entries_with_the_spec_kinds(layout: str) -> None:
    """Each layer maps eight Megatron keys, with the kinds of spec section
    13."""
    table = layer_key_table(0, layout)
    assert len(table) == 8
    kinds = sorted(kind for kind, _ in table.values())
    assert kinds == ["cols", "cols", "fc1", "none", "none", "none", "none", "qkv"]
    assert table["decoder.layers.0.self_attention.linear_qkv.weight"] == (
        "qkv",
        "model.layers.0.self_attn",
    )
    assert table["decoder.layers.0.mlp.linear_fc1.weight"] == ("fc1", "model.layers.0.mlp")


def test_megatron_split_matches_hf_projections() -> None:
    """Splitting linear_qkv as attention.py does gives the HF query, key, and
    value projections."""
    dims = _DIMS
    heads, groups, head = dims.num_heads, dims.num_kv_heads, dims.head_dim
    gen = torch.Generator().manual_seed(2)
    q = torch.randn(heads * head, dims.hidden_size, generator=gen)
    k = torch.randn(groups * head, dims.hidden_size, generator=gen)
    v = torch.randn(groups * head, dims.hidden_size, generator=gen)
    qkv = qkv_hf_to_megatron(q, k, v, dims)
    x = torch.randn(5, dims.hidden_size, generator=gen)

    mixed = (x @ qkv.T).view(5, groups, (heads // groups + 2) * head)
    query, key, value = torch.split(mixed, [heads // groups * head, head, head], dim=-1)
    query = query.reshape(5, heads, head)

    assert torch.allclose(query, (x @ q.T).view(5, heads, head), atol=1e-5)
    assert torch.allclose(key, (x @ k.T).view(5, groups, head), atol=1e-5)
    assert torch.allclose(value, (x @ v.T).view(5, groups, head), atol=1e-5)


def test_shard_qkv_tp2_gives_each_rank_one_query_group() -> None:
    """Rank 0 gets query heads 0..1, key 0, and value 0.

    Rank 1 gets heads 2..3, key 1, value 1.
    """
    dims = _DIMS
    gen = torch.Generator().manual_seed(3)
    q = torch.randn(32, 16, generator=gen)
    k = torch.randn(16, 16, generator=gen)
    v = torch.randn(16, 16, generator=gen)
    full = qkv_hf_to_megatron(q, k, v, dims)
    shards = shard_qkv(full, dims, tp=2)
    assert tuple(shards[0].shape) == (32, 16)
    assert torch.equal(shards[0][0:16], q[0:16])
    assert torch.equal(shards[0][16:24], k[0:8])
    assert torch.equal(shards[0][24:32], v[0:8])
    assert torch.equal(shards[1][0:16], q[16:32])
    assert torch.equal(shards[1][16:24], k[8:16])
    assert torch.equal(shards[1][24:32], v[8:16])
    assert torch.equal(unshard_qkv(shards, dims), full)


def test_shard_fc1_tp2_rank0_is_gate_then_up_of_first_half() -> None:
    """Rank 0 holds gate rows 0..11 followed by up rows 0..11, and unsharding
    restores both."""
    gen = torch.Generator().manual_seed(4)
    gate = torch.randn(24, 16, generator=gen)
    up = torch.randn(24, 16, generator=gen)
    shards = shard_fc1(gate, up, tp=2)
    assert torch.equal(shards[0], torch.cat([gate[0:12], up[0:12]], dim=0))
    assert torch.equal(shards[1], torch.cat([gate[12:24], up[12:24]], dim=0))
    got_gate, got_up = unshard_fc1(shards)
    assert torch.equal(got_gate, gate)
    assert torch.equal(got_up, up)


def test_row_and_column_shards_round_trip_and_own_their_storage() -> None:
    """Shards are independent copies, so torch.save does not store the full
    tensor per rank."""
    gen = torch.Generator().manual_seed(5)
    weight = torch.randn(12, 8, generator=gen)
    row_parts = shard_rows(weight, tp=3)
    col_parts = shard_cols(weight, tp=4)
    assert torch.equal(unshard_rows(row_parts), weight)
    assert torch.equal(unshard_cols(col_parts), weight)
    for part in row_parts + col_parts:
        assert part.untyped_storage().data_ptr() != weight.untyped_storage().data_ptr()


def test_non_divisible_splits_raise() -> None:
    """Rows, columns, and FFN width that do not divide by tp raise
    ValueError."""
    with pytest.raises(ValueError, match="not divisible"):
        shard_rows(torch.zeros(5, 3), tp=2)
    with pytest.raises(ValueError, match="not divisible"):
        shard_cols(torch.zeros(3, 5), tp=2)
    with pytest.raises(ValueError, match="not divisible"):
        shard_fc1(torch.zeros(5, 3), torch.zeros(5, 3), tp=2)


def test_vocab_padding_masking_and_cutting() -> None:
    """Vocabulary rows are copied, zero-padded, given a mean mask row, and cut
    back to size."""
    weight = torch.arange(10.0).reshape(5, 2)
    padded = vocab_to_megatron(weight, padded_rows=8, real_vocab=3, mask_token_id=None)
    assert tuple(padded.shape) == (8, 2)
    assert torch.equal(padded[:5], weight)
    assert torch.equal(padded[5:], torch.zeros(3, 2))
    masked = vocab_to_megatron(weight, padded_rows=8, real_vocab=3, mask_token_id=6)
    assert torch.equal(masked[6], torch.tensor([2.0, 3.0]))
    assert torch.equal(vocab_to_hf(padded, vocab_size_hf=4), weight[:4])


def test_megatron_to_hf_ignores_extra_state_and_rejects_unknown_keys() -> None:
    """The exporter skips *_extra_state keys, but it raises on any other key it
    does not map."""
    dims = replace(_DIMS, tie_embeddings=True)
    hf = _random_hf(dims)
    shards = _convert(hf, dims, tp=1)
    shards[0]["decoder.layers.0.mlp.linear_fc1._extra_state"] = torch.zeros(1)
    back = megatron_to_hf_state(shards, dims)
    assert set(back) == set(hf)
    shards[0]["decoder.layers.0.unexpected.weight"] = torch.zeros(1)
    with pytest.raises(ValueError, match="unused Megatron keys"):
        megatron_to_hf_state(shards, dims)


def test_from_hf_config_applies_head_dim_and_tie_defaults() -> None:
    """head_dim defaults to hidden_size // num_attention_heads, and
    tie_word_embeddings to False."""
    cfg = {
        "num_hidden_layers": 2,
        "hidden_size": 16,
        "intermediate_size": 24,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 40,
    }
    assert Qwen3Dims.from_hf_config(cfg) == replace(_DIMS, head_dim=4)
    explicit = Qwen3Dims.from_hf_config({**cfg, "head_dim": 8, "tie_word_embeddings": True})
    assert explicit == replace(_DIMS, tie_embeddings=True)


_SIZE_CFG = {
    "num_hidden_layers": 2,
    "hidden_size": 16,
    "intermediate_size": 24,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "vocab_size": 40,
}


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        pytest.param({"hidden_act": "gelu"}, "hidden_act", id="hidden-act"),
        pytest.param(
            {"rope_scaling": {"rope_type": "linear", "factor": 2.0}},
            "linear",
            id="rope-scaling",
        ),
        pytest.param(
            {"rope_parameters": {"rope_type": "yarn", "rope_theta": 1e6, "factor": 2.0}},
            "yarn",
            id="rope-parameters",
        ),
        pytest.param({"use_sliding_window": True}, "use_sliding_window", id="sliding-window"),
        pytest.param({"attention_bias": True}, "attention_bias", id="attention-bias"),
    ],
)
def test_from_hf_config_rejects_unsupported_features(extra: dict[str, Any], message: str) -> None:
    """Features that the converter does not map raise ValueError instead of
    being ignored."""
    with pytest.raises(ValueError, match=message):
        Qwen3Dims.from_hf_config({**_SIZE_CFG, **extra})


def test_from_hf_config_accepts_default_rope_and_the_supported_features() -> None:
    """The default RoPE, in either config layout, and SiLU without bias or
    sliding window pass."""
    expected = replace(_DIMS, head_dim=4)
    accepted: list[dict[str, Any]] = [
        {},
        {"rope_scaling": None},
        {"rope_scaling": {"rope_type": "default"}},
        {"rope_parameters": {"rope_type": "default", "rope_theta": 1e6}},
        {"hidden_act": "silu", "use_sliding_window": False, "attention_bias": False},
    ]
    for extra in accepted:
        assert Qwen3Dims.from_hf_config({**_SIZE_CFG, **extra}) == expected, extra


def test_qwen3_dims_rejects_heads_that_are_not_whole_groups() -> None:
    """num_heads must be a multiple of num_kv_heads."""
    with pytest.raises(ValueError, match="multiple of"):
        replace(_DIMS, num_kv_heads=3)


def test_unshard_qkv_rejects_wrong_shard_shape() -> None:
    """A shard with the wrong number of rows raises ValueError."""
    shards = [torch.zeros(31, 16), torch.zeros(31, 16)]
    with pytest.raises(ValueError, match="qkv shard"):
        unshard_qkv(shards, _DIMS)


def test_exporter_rejects_missing_key_and_replicated_mismatch() -> None:
    """The exporter rejects a key missing on one rank, and replicated tensors
    that differ."""
    dims = replace(_DIMS, tie_embeddings=True)
    shards = _convert(_random_hf(dims), dims, tp=2)
    missing = [dict(shard) for shard in shards]
    del missing[1]["decoder.layers.0.mlp.linear_fc2.weight"]
    with pytest.raises(KeyError, match="linear_fc2"):
        megatron_to_hf_state(missing, dims)
    norm_key = "decoder.layers.0.self_attention.linear_qkv.layer_norm_weight"
    differs = [dict(shard) for shard in shards]
    differs[1][norm_key] = shards[1][norm_key] + 1.0
    with pytest.raises(ValueError, match="differs between ranks"):
        megatron_to_hf_state(differs, dims)


def test_unshard_fc1_rejects_odd_row_count() -> None:
    """An fc1 shard holds equal gate and up halves, so an odd row count raises
    ValueError."""
    with pytest.raises(ValueError, match="even row count"):
        unshard_fc1([torch.zeros(5, 16), torch.zeros(5, 16)])


def _hf_cfg(tied: bool = False) -> dict[str, Any]:
    """Return a Hugging Face config dict that matches the tiny dims, in the
    transformers 4.x layout.

    Args:
        tied: True to set tie_word_embeddings.

    Returns:
        A new config dict with rope_theta at the top level.
    """
    return {
        "num_hidden_layers": 2,
        "hidden_size": 16,
        "intermediate_size": 24,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": 40,
        "tie_word_embeddings": tied,
        "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
        "attention_bias": False,
        "use_sliding_window": False,
    }


def _matching_args(*, tied: bool = False, **overrides: Any) -> argparse.Namespace:
    """Return MegaDLMs arguments that match _hf_cfg, with some attributes
    replaced.

    Args:
        tied: True for a tied model. It sets untie_embeddings_and_output_weights to False.
        **overrides: Attributes to set after the matching values are filled in.

    Returns:
        A Namespace with every attribute that check_args_against_hf compares.
    """
    values: dict[str, Any] = {
        "num_layers": 2,
        "hidden_size": 16,
        "ffn_hidden_size": 24,
        "num_attention_heads": 4,
        "num_query_groups": 2,
        "group_query_attention": True,
        "kv_channels": 8,
        "rotary_base": 1000000.0,
        "norm_epsilon": 1e-6,
        "rotary_interleaved": False,
        "untie_embeddings_and_output_weights": not tied,
        "swiglu": True,
        "qk_layernorm": True,
        "add_bias_linear": False,
        "normalization": "RMSNorm",
        "position_embedding_type": "rope",
        "apply_layernorm_1p": False,
        "apply_residual_connection_post_layernorm": False,
        "use_rope_scaling": False,
        "rotary_percent": 1.0,
        "tensor_model_parallel_size": 1,
        "mask_token": MASK_ID,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("layout", ["te", "local"])
def test_check_args_accepts_a_matching_namespace(tied: bool, layout: str) -> None:
    """Arguments that match the Hugging Face config pass, for both tie settings
    and both layouts."""
    check_args_against_hf(_matching_args(tied=tied), _hf_cfg(tied=tied), layout)


_SINGLE_MISMATCHES = [
    pytest.param({"num_layers": 3}, "num_layers", id="num-layers"),
    pytest.param({"hidden_size": 32}, "hidden_size", id="hidden-size"),
    pytest.param({"ffn_hidden_size": 20}, "ffn_hidden_size", id="ffn-hidden-size"),
    pytest.param({"num_attention_heads": 8}, "num_attention_heads", id="num-attention-heads"),
    pytest.param({"num_query_groups": 4}, "num_query_groups", id="num-query-groups"),
    pytest.param({"group_query_attention": False}, "group_query_attention", id="gqa-off"),
    pytest.param({"kv_channels": 4}, "kv_channels", id="kv-channels"),
    pytest.param({"rotary_base": 10000}, "rotary_base", id="rotary-base"),
    pytest.param({"norm_epsilon": 1e-5}, "norm_epsilon", id="norm-epsilon"),
    pytest.param({"rotary_interleaved": True}, "rotary_interleaved", id="rotary-interleaved"),
    pytest.param({"swiglu": False}, "swiglu", id="swiglu"),
    pytest.param({"qk_layernorm": False}, "qk_layernorm", id="qk-layernorm"),
    pytest.param({"add_bias_linear": True}, "add_bias_linear", id="add-bias-linear"),
    pytest.param({"normalization": "LayerNorm"}, "normalization", id="normalization"),
    pytest.param(
        {"position_embedding_type": "learned_absolute"},
        "position_embedding_type",
        id="position-embedding-type",
    ),
    pytest.param({"apply_layernorm_1p": True}, "apply_layernorm_1p", id="layernorm-1p"),
    pytest.param(
        {"apply_residual_connection_post_layernorm": True},
        "apply_residual_connection_post_layernorm",
        id="post-layernorm-residual",
    ),
    pytest.param({"use_rope_scaling": True}, "use_rope_scaling", id="rope-scaling"),
    pytest.param({"rotary_percent": 0.5}, "rotary_percent", id="rotary-percent"),
]


@pytest.mark.parametrize(("overrides", "name"), _SINGLE_MISMATCHES)
def test_check_args_rejects_each_single_mismatch(overrides: dict[str, Any], name: str) -> None:
    """Each single wrong value raises ValueError, and the message names the
    attribute."""
    with pytest.raises(ValueError, match=name):
        check_args_against_hf(_matching_args(**overrides), _hf_cfg(), "te")


@pytest.mark.parametrize("tied", [False, True])
def test_check_args_rejects_an_untie_flag_that_contradicts_the_config(tied: bool) -> None:
    """untie_embeddings_and_output_weights must be the opposite of
    tie_word_embeddings."""
    with pytest.raises(ValueError, match="untie_embeddings_and_output_weights"):
        check_args_against_hf(_matching_args(tied=not tied), _hf_cfg(tied=tied), "te")


def test_check_args_lists_every_mismatch_in_one_error() -> None:
    """The error names every attribute that differs, not only the first one."""
    args = _matching_args(rotary_base=10000, norm_epsilon=1e-5, swiglu=False)
    with pytest.raises(ValueError) as info:
        check_args_against_hf(args, _hf_cfg(), "te")
    for name in ("rotary_base", "norm_epsilon", "swiglu"):
        assert name in str(info.value)


def test_check_args_skips_attributes_that_the_args_lack() -> None:
    """An args object without a compared attribute is not checked for that
    attribute."""
    check_args_against_hf(argparse.Namespace(num_layers=2), _hf_cfg(), "te")
    with pytest.raises(ValueError, match="num_layers"):
        check_args_against_hf(argparse.Namespace(num_layers=3), _hf_cfg(), "te")


def test_check_args_requires_group_query_attention_only_when_kv_heads_are_fewer() -> None:
    """With multi-head attention (as many KV heads as query heads), the flag is
    not required."""
    cfg = _hf_cfg()
    cfg["num_key_value_heads"] = 4
    args = _matching_args(group_query_attention=False, num_query_groups=4)
    check_args_against_hf(args, cfg, "te")


def test_check_args_reads_rope_theta_from_rope_parameters() -> None:
    """Transformers 5.x stores rope_theta in rope_parameters, and that value is
    compared."""
    cfg = _hf_cfg()
    del cfg["rope_theta"]
    cfg["rope_parameters"] = {"rope_type": "default", "rope_theta": 1000000.0}
    check_args_against_hf(_matching_args(), cfg, "te")
    with pytest.raises(ValueError, match="rotary_base"):
        check_args_against_hf(_matching_args(rotary_base=10000), cfg, "te")


def test_check_args_requires_a_rope_theta_in_the_config() -> None:
    """A config with no rope_theta, in either place, raises KeyError."""
    cfg = _hf_cfg()
    del cfg["rope_theta"]
    with pytest.raises(KeyError, match="rope_theta"):
        check_args_against_hf(_matching_args(), cfg, "te")


def test_check_args_rejects_an_unknown_layout() -> None:
    """The layout is validated like the other entry points."""
    with pytest.raises(ValueError, match="layout"):
        check_args_against_hf(_matching_args(), _hf_cfg(), "bogus")
