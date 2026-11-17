"""The Engram per-layer hash seeds must not shift when first-class V layers are inserted."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "submodules" / "Megatron-LM"))

from megatron.core.ssm.engram import engram_hash_layer_number, engram_hash_multipliers  # noqa: E402


def _config(v_layers):
    return types.SimpleNamespace(
        kv_share_layers=v_layers,
        engram_hash_per_layer=True,
        engram_ngram_orders=(2, 3, 4),
        engram_num_heads=8,
        engram_aux_ngram_orders=(),
        engram_aux_num_heads=0,
    )


def test_v_insertion_pins_deep_engram_seed():
    original = _config(None)
    inserted = _config([10, 27, 44, 62, 79, 96])

    # The deep Engram moves from 0-based index 49 / layer number 50 to index 52 / 53.
    assert engram_hash_layer_number(inserted, 53) == 50
    assert engram_hash_layer_number(inserted, 3) == 3
    assert torch.equal(
        engram_hash_multipliers(inserted, 50),
        engram_hash_multipliers(original, 50),
    )
    assert not torch.equal(
        engram_hash_multipliers(inserted, 53),
        engram_hash_multipliers(original, 50),
    )
