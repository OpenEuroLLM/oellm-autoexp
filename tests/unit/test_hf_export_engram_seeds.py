"""Engram hash seeds across V-layer insertions (docs/recall_surgery_ambitious.md §22).

The v4a surgery inserts six V layers, which moves the second Engram layer from decoder
index 49 (layer_number 50) to 52 (layer_number 53). With ``engram_hash_per_layer`` the
n-gram hash multipliers are seeded per layer_number, so the export must keep the seeds of
the pre-insertion position; otherwise the stitched model is not function-preserving and
the export would not match its 35900 parent.
"""
import torch

from oellm_autoexp.hf_export.export_ambitious import (
    engram_hash_layer_number,
    engram_hash_multipliers,
)

V_LAYERS = [10, 27, 44, 62, 79, 96]
ENGRAM_INDICES = [2, 52]  # v4a pattern; the base 98-layer pattern has them at 2 and 49


def _m(**kw):
    m = dict(engram_hash_per_layer=True, engram_ngram_orders=[2, 3], engram_num_heads=2)
    m.update(kw)
    return m


def test_v_layers_do_not_renumber_the_engram_seeds():
    base = _m()
    stitched = _m(kv_share_layers=V_LAYERS)
    # the first Engram layer is not moved by the insertions after it
    assert engram_hash_layer_number(stitched, 2) == 2
    # the second one keeps its pre-insertion position (49, layer_number 50)
    assert engram_hash_layer_number(stitched, 52) == 49
    # a V layer after a layer does not renumber it, one before it does
    assert engram_hash_layer_number(stitched, 20) == 19  # 0-based index 20: one V below (10)
    assert engram_hash_layer_number(stitched, 30) == 28  # two V below (10, 27)
    # without insertions the position is the layer's own
    assert engram_hash_layer_number(base, 52) == 52


def test_pinned_seeds_match_the_base_model_and_differ_from_the_shift():
    base = _m()
    stitched = _m(kv_share_layers=V_LAYERS)
    pinned = engram_hash_multipliers(stitched, engram_hash_layer_number(stitched, 52) + 1)
    assert torch.equal(torch.tensor(pinned), torch.tensor(engram_hash_multipliers(base, 50)))
    shifted = engram_hash_multipliers(stitched, 52 + 1)
    assert not torch.equal(torch.tensor(pinned), torch.tensor(shifted))
    # the first Engram layer is unchanged either way
    assert torch.equal(torch.tensor(engram_hash_multipliers(stitched, 3)),
                       torch.tensor(engram_hash_multipliers(base, 3)))


def test_engram_indices_of_the_v4a_pattern():
    """The indices the rule is applied to are the pattern's Engram positions."""
    pattern = "*ENLELELELVELELELE*ELELELELVELELELE*ELELELELVELELELEN*ELELELELVELELELE*ELELELELVELELELE*ELELELELVELELELE"
    assert [i for i, c in enumerate(pattern) if c == "N"] == ENGRAM_INDICES
