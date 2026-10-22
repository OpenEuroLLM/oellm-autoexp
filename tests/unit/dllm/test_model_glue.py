"""Glue tests for ``BlockDiffusionGPTModel.forward`` with fake Megatron
components.

The model is built without ``GPTModel.__init__``, which reads the Megatron global arguments. The
embedding, RoPE table, decoder, output layer, and language-model loss are small fakes that record
their inputs. The decoder is the identity, so the logits at input index ``j`` are the output layer
applied to the embedding of the token at ``j``. The expected values are recomputed with plain
loops, the dense mask references, and the noise samplers. The tests check the wiring of spec
sections 2 to 7 and 9.

The module is skipped when ``submodules/MegaDLMs`` is not checked out.

Run with:
    pytest tests/unit/dllm/test_model_glue.py -v
"""

from __future__ import annotations

import math
import sys
import types
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[3]
MEGADLMS_DIR = REPO_ROOT / "submodules" / "MegaDLMs"
if not (MEGADLMS_DIR / "megatron").is_dir():
    pytest.skip("submodules/MegaDLMs is not checked out", allow_module_level=True)
if str(MEGADLMS_DIR) not in sys.path:
    sys.path.insert(0, str(MEGADLMS_DIR))
try:
    from torch.nn import functional as F
    from torch.nn.attention.flex_attention import BlockMask

    import oellm_autoexp.dllm.model as model_module
    from oellm_autoexp.dllm import context as bd_context
    from oellm_autoexp.dllm.masks import (
        dense_ar_reference,
        dense_bd3_reference,
        materialize_mask,
    )
    from oellm_autoexp.dllm.model import BlockDiffusionGPTModel
    from oellm_autoexp.dllm.noise import (
        sample_block_noise,
        sample_complementary_noise,
        sample_validation_noise,
    )
    from oellm_autoexp.dllm.schedule import parse_block_schedule
    from oellm_autoexp.dllm.targets import bd3_loss_terms
except ImportError as exc:  # the MegaDLMs Megatron stack is not importable in this environment
    pytest.skip(f"MegaDLMs import failed: {exc}", allow_module_level=True)

_VOCAB = 32  # V
_MASK_ID = 31  # --mask-token, which is V - 1
_EOD_ID = 30  # tokenizer.eod in these tests; x0 ids are drawn from [0, 29]
_HIDDEN = 8  # h
_L = 16  # effective length L = seq_length - 1
_BATCH = 2  # b
_BLOCK = 4  # B, with L % B == 0
_SEED = 7  # args.seed; the training generator is seeded seed + 1000 * dp_rank = 7 at rank 0
_VAL_SEED = 1234  # args.bd_val_seed
_LN2 = math.log(2.0)

# Document ids of the two rows in the shift-1 test: EOD at positions (5, 11) and (0, 9).
_ROW0_DOCS = [0] * 6 + [1] * 6 + [2] * 4
_ROW1_DOCS = [0] + [1] * 9 + [2] * 6

_Outputs = tuple[torch.Tensor, torch.Tensor, dict[str, tuple[torch.Tensor, torch.Tensor]]]


@pytest.fixture(autouse=True)
def _rank_zero_and_clean_context(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Use data-parallel rank 0 and start and end every test with an empty
    BlockMask context."""
    monkeypatch.setattr(model_module.parallel_state, "get_data_parallel_rank", lambda: 0)
    bd_context.clear_block_mask()
    yield
    bd_context.clear_block_mask()


def _rope_table(length: int) -> torch.Tensor:
    """Return a deterministic rotary table with one row per position, shape [n,
    1, 1, 4]."""
    return torch.arange(length * 4, dtype=torch.float32).reshape(length, 1, 1, 4) + 1.0


class _FakeEmbedding(torch.nn.Module):
    """Token embedding that records ``(input_ids, position_ids)`` and returns
    [s, b, h]."""

    def __init__(self) -> None:
        super().__init__()
        self.table = torch.nn.Embedding(_VOCAB, _HIDDEN)
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Embed ``input_ids`` [b, s] and return the Megatron layout [s, b,
        h]."""
        self.calls.append((input_ids, position_ids))
        return self.table(input_ids).transpose(0, 1)


class _FakeRotary(torch.nn.Module):
    """RoPE table fake that records the requested sequence lengths."""

    def __init__(self) -> None:
        super().__init__()
        self.lengths: list[int] = []

    def forward(self, max_seq_len: int) -> torch.Tensor:
        """Return the rotary table for ``max_seq_len`` positions."""
        self.lengths.append(max_seq_len)
        return _rope_table(max_seq_len)


@dataclass
class _DecoderCall:
    """Arguments of one decoder call, with the BlockMask that core attention
    reads."""

    hidden_states: torch.Tensor
    attention_mask: Any
    rotary_pos_emb: torch.Tensor
    block_mask: Any


class _FakeDecoder(torch.nn.Module):
    """Identity decoder that records each call and the BlockMask stored in the
    context."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[_DecoderCall] = []

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Any, rotary_pos_emb: torch.Tensor
    ) -> torch.Tensor:
        """Record the arguments and the stored BlockMask, then return
        ``hidden_states``."""
        block_mask = bd_context.get_block_mask()
        self.calls.append(_DecoderCall(hidden_states, attention_mask, rotary_pos_emb, block_mask))
        return hidden_states


class _FakeOutputLayer(torch.nn.Module):
    """Linear head [s, b, h] -> [s, b, V] that returns ``(logits, None)`` like
    Megatron."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(_HIDDEN, _VOCAB, bias=False)

    def forward(
        self, hidden: torch.Tensor, weight: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, None]:
        """Project ``hidden`` to the vocabulary.

        The weight argument is unused.
        """
        return self.proj(hidden), None


def _fake_lm_loss(labels: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Return the per-token cross entropy of labels [b, s] and logits [s, b,
    V], as [b, s]."""
    return F.cross_entropy(logits.permute(1, 2, 0), labels, reduction="none")


def _make_model(
    *,
    objective: str = "bd3",
    block_size: int = _BLOCK,
    token_shift: int = 0,
    doc_mask: int = 0,
    complementary: int = 0,
    loss_weighting: str = "elbo",
    schedule: str = "",
    curr_iteration: int = 0,
    training: bool = True,
    nld_ar_weight: float = 1.0,
    nld_diffusion_weight: float = 0.3,
    nld_ar_only_fraction: float = 0.2,
) -> BlockDiffusionGPTModel:
    """Build a model whose Megatron parts are fakes and whose attributes match
    ``__init__``.

    Args:
        objective: ``"bd3"``, ``"ar"`` or ``"nld"``.
        block_size: The constant ``--bd-block-size``.
        token_shift: ``--bd-token-shift``.
        doc_mask: ``--bd-doc-mask``.
        complementary: ``--bd-complementary-mask``.
        loss_weighting: ``--bd-loss-weighting``.
        schedule: ``--bd-block-schedule``.
        curr_iteration: The current training iteration.
        training: Whether the model starts in training mode.
        nld_ar_weight: ``--bd-nld-ar-weight``.
        nld_diffusion_weight: ``--bd-nld-diffusion-weight``.
        nld_ar_only_fraction: ``--bd-nld-ar-only-fraction``.

    Returns:
        The model, in the requested mode.
    """
    model = BlockDiffusionGPTModel.__new__(BlockDiffusionGPTModel)
    torch.nn.Module.__init__(model)
    model.args = types.SimpleNamespace(
        bd_objective=objective,
        bd_block_size=block_size,
        bd_block_schedule=schedule,
        bd_token_shift=token_shift,
        bd_complementary_mask=complementary,
        bd_loss_weighting=loss_weighting,
        bd_t_min=0.0,
        bd_t_max=1.0,
        bd_doc_mask=doc_mask,
        bd_val_seed=_VAL_SEED,
        mask_token=_MASK_ID,
        seq_length=_L + 1,
        train_iters=100,
        curr_iteration=curr_iteration,
        seed=_SEED,
        bd_nld_ar_weight=nld_ar_weight,
        bd_nld_diffusion_weight=nld_diffusion_weight,
        bd_nld_ar_only_fraction=nld_ar_only_fraction,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model.embedding = _FakeEmbedding()
        model.rotary_pos_emb = _FakeRotary()
        model.decoder = _FakeDecoder()
        model.output_layer = _FakeOutputLayer()
    # Attributes that BlockDiffusionGPTModel.__init__ sets after super().__init__().
    model.pre_process = True
    model.post_process = True
    model.share_embeddings_and_output_weights = False
    model.eod_id = _EOD_ID
    model.mask_id = _MASK_ID
    model.objective = objective
    model.seq_len_effective = _L
    model.block_schedule = parse_block_schedule(schedule)
    model._train_generator = None
    model._val_generator = None
    model._val_iteration = None
    model._val_reset_pending = True
    model._mask_cache = {}
    model._source_cache = {}
    model.compute_language_model_loss = _fake_lm_loss
    model.train(training)
    return model


def _random_labels(seed: int) -> torch.Tensor:
    """Return labels [b, L + 1] with ids in [0, 29], so no token is EOD or the
    mask id."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, _EOD_ID, (_BATCH, _L + 1), generator=gen)


def _run(
    model: BlockDiffusionGPTModel,
    labels: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
) -> _Outputs:
    """Call the model the way Megatron does, with tokens[:, 1:] == labels[:,
    :-1]."""
    tokens = torch.cat([torch.zeros_like(labels[:, :1]), labels[:, :-1]], dim=1)
    if loss_mask is None:
        loss_mask = torch.ones(labels.shape, dtype=torch.float32)
    return model(input_ids=tokens, labels=labels, loss_mask=loss_mask)


def _train_noise(
    x0: torch.Tensor, block_size: int, complementary: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recompute the training noise of the first forward from a generator
    seeded like the model."""
    gen = torch.Generator().manual_seed(_SEED)
    if complementary:
        return sample_complementary_noise(x0, block_size, _MASK_ID, 0.0, 1.0, gen)
    return sample_block_noise(x0, block_size, _MASK_ID, 0.0, 1.0, gen)


def _validation_nelbo_draws(
    model: BlockDiffusionGPTModel, x0: torch.Tensor, count: int
) -> list[float]:
    """Return the NELBO of each of the first ``count`` validation draws (spec
    section 9)."""
    gen = torch.Generator().manual_seed(_VAL_SEED)
    values: list[float] = []
    for _ in range(count):
        xt, masked, _ = sample_validation_noise(x0, _BLOCK, _MASK_ID, gen)
        ce = _identity_ce(model, torch.cat([xt, x0], dim=1), x0)
        values.append(_manual_nelbo(ce, masked, _scorable_from_position_one(), _BLOCK))
    return values


def _identity_ce(
    model: BlockDiffusionGPTModel, input_ids: torch.Tensor, x0: torch.Tensor
) -> torch.Tensor:
    """Return the cross entropy of x0 for the identity decoder and token shift
    0.

    Output ``i`` reads input index ``i``, so the logits come from the first L embedded inputs.
    """
    with torch.no_grad():
        hidden = model.embedding.table(input_ids).transpose(0, 1)
        logits = model.output_layer(hidden[: x0.shape[1]])[0]
        return F.cross_entropy(logits.permute(1, 2, 0), x0, reduction="none")


def _scorable_from_position_one() -> torch.Tensor:
    """Return the scored positions when every position has a source and
    loss_mask is all ones."""
    scorable = torch.ones(_BATCH, _L, dtype=torch.bool)
    scorable[:, 0] = False
    return scorable


def _manual_bd3_terms(
    ce: torch.Tensor,
    masked: torch.Tensor,
    scorable: torch.Tensor,
    p_block: torch.Tensor,
    block_size: int,
) -> dict[str, float]:
    """Sum the bd3 terms of spec section 7 with plain Python loops."""
    totals = {"elbo_sum": 0.0, "masked_ce_sum": 0.0, "num_targets": 0.0, "scored": 0.0}
    for row in range(ce.shape[0]):
        for pos in range(ce.shape[1]):
            if not bool(scorable[row, pos]):
                continue
            totals["scored"] += 1.0
            if bool(masked[row, pos]):
                value = float(ce[row, pos])
                totals["num_targets"] += 1.0
                totals["masked_ce_sum"] += value
                totals["elbo_sum"] += value / float(p_block[row, pos // block_size])
    return totals


def _manual_nelbo(
    ce: torch.Tensor, masked: torch.Tensor, scorable: torch.Tensor, block_size: int
) -> float:
    """Return the validation NELBO summed over rows, with weight B / l_k per
    block (spec 9)."""
    total = 0.0
    for row in range(ce.shape[0]):
        for start in range(0, ce.shape[1], block_size):
            block = range(start, start + block_size)
            num_masked = sum(bool(masked[row, pos]) for pos in block)
            weight = block_size / num_masked
            for pos in block:
                if bool(masked[row, pos]) and bool(scorable[row, pos]):
                    total += weight * float(ce[row, pos])
    return total


def _manual_shift1_sources(
    length: int, block_size: int, docs: list[int] | None
) -> list[int | None]:
    """Return the source index of each position under token shift 1 (spec
    section 6)."""
    sources: list[int | None] = []
    for pos in range(length):
        if pos == 0 or (docs is not None and docs[pos] != docs[pos - 1]):
            sources.append(None)
        elif pos % block_size != 0:
            sources.append(pos - 1)
        else:
            sources.append(length + pos - 1)
    return sources


def _manual_doc_ids(row: list[int]) -> list[int]:
    """Return doc_ids[i], the number of EOD tokens before position i, with a
    plain loop."""
    ids: list[int] = []
    count = 0
    for token in row:
        ids.append(count)
        if token == _EOD_ID:
            count += 1
    return ids


def _assert_pair(
    pair: tuple[torch.Tensor, torch.Tensor], numerator: float, denominator: float
) -> None:
    """Assert one logged (sum, count) pair against plain Python values."""
    assert float(pair[0].detach()) == pytest.approx(numerator, rel=1e-5, abs=1e-6)
    assert float(pair[1]) == pytest.approx(denominator)


def _assert_same_outputs(first: _Outputs, second: _Outputs) -> None:
    """Assert that two forward results hold equal tensors under equal keys."""
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert first[2].keys() == second[2].keys()
    for key, (numerator, denominator) in first[2].items():
        assert torch.equal(numerator, second[2][key][0]), key
        assert torch.equal(denominator, second[2][key][1]), key


def test_bd3_training_wiring_without_shift_or_documents() -> None:
    """Shift 0 and no document mask: inputs, RoPE, BlockMask, and loss follow spec 3 to 7."""
    model = _make_model()
    labels = _random_labels(seed=101)
    x0 = labels[:, :-1]
    loss_sum, num_tokens, metrics = _run(model, labels)

    xt, masked, p_block = _train_noise(x0, _BLOCK)
    input_ids = torch.cat([xt, x0], dim=1)
    embedding_ids, embedding_pos = model.embedding.calls[-1]
    assert torch.equal(embedding_ids, input_ids)
    assert torch.equal(embedding_pos, (torch.arange(2 * _L) % _L).expand(_BATCH, -1))

    call = model.decoder.calls[-1]
    assert call.attention_mask is None
    assert torch.equal(call.hidden_states, model.embedding.table(input_ids).transpose(0, 1))
    assert model.rotary_pos_emb.lengths == [_L]
    rope2 = torch.cat([_rope_table(_L), _rope_table(_L)], dim=0)
    assert torch.equal(call.rotary_pos_emb, rope2)
    assert isinstance(call.block_mask, BlockMask)
    mask = materialize_mask(call.block_mask.mask_mod, 1, 2 * _L, 2 * _L)
    assert torch.equal(mask[0], dense_bd3_reference(_L, _BLOCK, None))

    # Identity decoder, src[i] = i, and every position except 0 is scorable.
    ce = _identity_ce(model, input_ids, x0)
    scorable = _scorable_from_position_one()
    with torch.no_grad():
        terms = bd3_loss_terms(ce, masked, scorable, p_block, _BLOCK, "elbo")
    manual = _manual_bd3_terms(ce, masked, scorable, p_block, _BLOCK)
    assert manual["scored"] == _BATCH * (_L - 1)
    assert float(num_tokens) == pytest.approx(_BATCH * (_L - 1))
    assert float(terms["num_tokens"]) == pytest.approx(_BATCH * (_L - 1))
    assert float(loss_sum.detach()) == pytest.approx(manual["elbo_sum"], rel=1e-5)
    assert float(terms["loss_sum"].detach()) == pytest.approx(manual["elbo_sum"], rel=1e-5)

    scored = _BATCH * (_L - 1)
    _assert_pair(metrics["lm loss"], manual["elbo_sum"], scored)
    _assert_pair(metrics["masked ce"], manual["masked_ce_sum"], manual["num_targets"])
    _assert_pair(metrics["mask frac"], float(p_block.sum()), p_block.numel())
    _assert_pair(metrics["block size"], _BLOCK, 1)
    _assert_pair(metrics["nelbo bits/token"], manual["elbo_sum"] / _LN2, scored)


def test_bd3_training_wiring_with_shift1_and_documents() -> None:
    """Shift 1 with document ids: sources follow spec 6 and the mask follows the documents."""
    model = _make_model(token_shift=1, doc_mask=1)
    labels = _random_labels(seed=202)
    labels[0, 5] = _EOD_ID
    labels[0, 11] = _EOD_ID
    labels[1, 0] = _EOD_ID
    labels[1, 9] = _EOD_ID
    x0 = labels[:, :-1]
    assert [_manual_doc_ids(row) for row in x0.tolist()] == [_ROW0_DOCS, _ROW1_DOCS]
    loss_sum, num_tokens, metrics = _run(model, labels)

    xt, masked, p_block = _train_noise(x0, _BLOCK)
    call = model.decoder.calls[-1]
    assert torch.equal(model.embedding.calls[-1][0], torch.cat([xt, x0], dim=1))
    assert isinstance(call.block_mask, BlockMask)
    mask = materialize_mask(call.block_mask.mask_mod, _BATCH, 2 * _L, 2 * _L)
    for row, docs in enumerate((_ROW0_DOCS, _ROW1_DOCS)):
        assert torch.equal(mask[row], dense_bd3_reference(_L, _BLOCK, docs))

    # Recompute each scored position from its source with plain loops.
    with torch.no_grad():
        logits = model.output_layer(call.hidden_states)[0]  # [2L, b, V]
    ce = torch.zeros(_BATCH, _L)
    scorable = torch.zeros(_BATCH, _L, dtype=torch.bool)
    for row, docs in enumerate((_ROW0_DOCS, _ROW1_DOCS)):
        for pos, src in enumerate(_manual_shift1_sources(_L, _BLOCK, docs)):
            if src is None:
                continue
            scorable[row, pos] = True
            with torch.no_grad():
                ce[row, pos] = F.cross_entropy(logits[src, row][None], x0[row, pos][None])
    # No source at position 0 and none at the first position of each document after an EOD.
    assert not bool(scorable[0, 0]) and not bool(scorable[1, 0])
    assert not bool(scorable[0, 6]) and not bool(scorable[0, 12])
    assert not bool(scorable[1, 1]) and not bool(scorable[1, 10])
    assert int(scorable.sum()) == 26

    manual = _manual_bd3_terms(ce, masked, scorable, p_block, _BLOCK)
    assert manual["scored"] == 26
    assert float(num_tokens) == pytest.approx(26)
    assert float(loss_sum.detach()) == pytest.approx(manual["elbo_sum"], rel=1e-5)
    _assert_pair(metrics["lm loss"], manual["elbo_sum"], 26)
    _assert_pair(metrics["nelbo bits/token"], manual["elbo_sum"] / _LN2, 26)


def test_eval_forward_returns_validation_nelbo_and_bits_per_token() -> None:
    """Eval mode returns (nelbo_sum, scored, metrics) from the validation
    estimator (spec 9)."""
    model = _make_model(training=False)
    labels = _random_labels(seed=303)
    x0 = labels[:, :-1]
    nelbo_sum, scored, metrics = _run(model, labels)

    (expected,) = _validation_nelbo_draws(model, x0, 1)
    assert float(nelbo_sum.detach()) == pytest.approx(expected, rel=1e-5)
    assert float(scored) == pytest.approx(_BATCH * (_L - 1))
    # Spec 9: the logged value is nelbo_sum / ln 2 over the scored positions.
    _assert_pair(metrics["val nelbo bits/token"], float(nelbo_sum.detach()) / _LN2, float(scored))
    _assert_pair(metrics["val nelbo bits/token"], expected / _LN2, _BATCH * (_L - 1))
    _assert_pair(metrics["block size"], _BLOCK, 1)


def test_eval_call_after_eval_restarts_the_validation_round() -> None:
    """Each eval() call starts a new round, so two forwards after eval() agree
    exactly."""
    model = _make_model()
    labels = _random_labels(seed=304)
    model.eval()
    first = _run(model, labels)
    model.eval()
    second = _run(model, labels)
    _assert_same_outputs(first, second)


def test_eval_generator_continues_within_one_round() -> None:
    """Without an eval() call in between, the second forward draws the next
    validation noise."""
    model = _make_model()
    labels = _random_labels(seed=305)
    x0 = labels[:, :-1]
    model.eval()
    first = _run(model, labels)
    second = _run(model, labels)
    assert not torch.equal(first[0], second[0])

    draw_1, draw_2 = _validation_nelbo_draws(model, x0, 2)
    assert float(first[0].detach()) == pytest.approx(draw_1, rel=1e-5)
    assert float(second[0].detach()) == pytest.approx(draw_2, rel=1e-5)


def test_complementary_masking_doubles_the_batch() -> None:
    """Complementary masking stacks two disjoint copies and scores all 2b rows
    (spec 3)."""
    model = _make_model(complementary=1)
    labels = _random_labels(seed=404)
    x0 = labels[:, :-1]
    _, num_tokens, metrics = _run(model, labels)

    xt, masked, p_block = _train_noise(x0, _BLOCK, complementary=True)
    x0_stacked = torch.cat([x0, x0], dim=0)
    input_ids = torch.cat([xt, x0_stacked], dim=1)
    call = model.decoder.calls[-1]
    assert call.hidden_states.shape == (2 * _L, 2 * _BATCH, _HIDDEN)
    assert torch.equal(model.embedding.calls[-1][0], input_ids)

    # Copy 2 masks exactly the positions that copy 1 left unmasked, with 1 to B - 1 per block.
    copy1, copy2 = masked[:_BATCH], masked[_BATCH:]
    assert bool((copy1 ^ copy2).all())
    per_block = copy1.reshape(_BATCH, _L // _BLOCK, _BLOCK).sum(dim=-1)
    assert bool(((per_block >= 1) & (per_block <= _BLOCK - 1)).all())

    ce = _identity_ce(model, input_ids, x0_stacked)
    scorable = torch.ones(2 * _BATCH, _L, dtype=torch.bool)
    scorable[:, 0] = False
    manual = _manual_bd3_terms(ce, masked, scorable, p_block, _BLOCK)
    assert manual["scored"] == 2 * _BATCH * (_L - 1)
    assert float(num_tokens) == pytest.approx(2 * _BATCH * (_L - 1))
    _assert_pair(metrics["lm loss"], manual["elbo_sum"], 2 * _BATCH * (_L - 1))
    _assert_pair(metrics["mask frac"], float(p_block.sum()), 2 * _BATCH * (_L // _BLOCK))

    mask = materialize_mask(call.block_mask.mask_mod, 1, 2 * _L, 2 * _L)
    assert torch.equal(mask[0], dense_bd3_reference(_L, _BLOCK, None))


def test_ar_objective_runs_causal_model_on_clean_tokens() -> None:
    """Objective ar: x0 alone at positions 0 to L - 1, causal mask, and next-token loss."""
    model = _make_model(objective="ar")
    labels = _random_labels(seed=505)
    loss_mask = torch.ones(labels.shape, dtype=torch.float32)
    loss_mask[0, 5] = 0.0
    loss_mask[1, 9] = 0.0
    x0 = labels[:, :-1]
    loss_sum, num_tokens, metrics = _run(model, labels, loss_mask)

    call = model.decoder.calls[-1]
    embedding_ids, embedding_pos = model.embedding.calls[-1]
    assert torch.equal(embedding_ids, x0)
    assert torch.equal(embedding_pos, torch.arange(_L).expand(_BATCH, -1))
    assert torch.equal(call.hidden_states, model.embedding.table(x0).transpose(0, 1))
    assert model.rotary_pos_emb.lengths == [_L]
    assert torch.equal(call.rotary_pos_emb, _rope_table(_L))
    assert isinstance(call.block_mask, BlockMask)
    mask = materialize_mask(call.block_mask.mask_mod, 1, _L, _L)
    assert torch.equal(mask[0], dense_ar_reference(_L, None))

    # Output i - 1 predicts x0[i] for i >= 1. target_ok comes from loss_mask[:, :-1].
    expected = 0.0
    count = 0
    with torch.no_grad():
        logits = model.output_layer(call.hidden_states)[0]  # [L, b, V]
        for row in range(_BATCH):
            for pos in range(1, _L):
                if float(loss_mask[row, pos]) > 0:
                    target = x0[row, pos][None]
                    expected += float(F.cross_entropy(logits[pos - 1, row][None], target))
                    count += 1
    assert count == 28
    assert float(num_tokens) == pytest.approx(28)
    assert float(loss_sum.detach()) == pytest.approx(expected, rel=1e-5)
    _assert_pair(metrics["lm loss"], expected, 28)
    _assert_pair(metrics["block size"], 1, 1)
    _assert_pair(metrics["nelbo bits/token"], expected / _LN2, 28)


def test_document_mask_is_built_once_and_refreshed_per_layout() -> None:
    """Two training forwards with different document layouts share one mask_mod
    (spec 5)."""
    model = _make_model(doc_mask=1)
    layout_a = _random_labels(seed=606)
    layout_a[0, 3] = _EOD_ID
    layout_a[1, 7] = _EOD_ID
    layout_b = _random_labels(seed=607)
    layout_b[0, 9] = _EOD_ID
    layout_b[1, 2] = _EOD_ID
    layout_b[1, 13] = _EOD_ID
    docs_a = [_manual_doc_ids(row) for row in layout_a[:, :-1].tolist()]
    docs_b = [_manual_doc_ids(row) for row in layout_b[:, :-1].tolist()]

    _run(model, layout_a)
    block_mask_a = model.decoder.calls[-1].block_mask
    # The mask_mod reads one shared document buffer, so layout A is materialized before layout B.
    mask_a = materialize_mask(block_mask_a.mask_mod, _BATCH, 2 * _L, 2 * _L)
    assert len(model._mask_cache) == 1
    (entry,) = model._mask_cache.values()
    mask_mod_cached, buffer_cached = entry[0], entry[1]
    assert block_mask_a.mask_mod is mask_mod_cached

    _run(model, layout_b)
    block_mask_b = model.decoder.calls[-1].block_mask
    mask_b = materialize_mask(block_mask_b.mask_mod, _BATCH, 2 * _L, 2 * _L)
    assert len(model._mask_cache) == 1
    (entry,) = model._mask_cache.values()
    assert entry[0] is mask_mod_cached
    assert entry[1] is buffer_cached
    assert block_mask_b.mask_mod is mask_mod_cached
    assert torch.equal(buffer_cached, torch.tensor(docs_b))

    for row in range(_BATCH):
        assert torch.equal(mask_a[row], dense_bd3_reference(_L, _BLOCK, docs_a[row]))
        assert torch.equal(mask_b[row], dense_bd3_reference(_L, _BLOCK, docs_b[row]))
    assert not torch.equal(mask_a, mask_b)


def test_block_schedule_selects_block_size_by_iteration() -> None:
    """The schedule "0:4,0.5:8" with train_iters 100 picks B = 4 before 50 and
    B = 8 from 50."""
    model = _make_model(schedule="0:4,0.5:8", block_size=8, curr_iteration=10)
    labels = _random_labels(seed=707)
    for iteration, expected_block in ((10, 4), (49, 4), (50, 8), (60, 8)):
        model.args.curr_iteration = iteration
        _, _, metrics = _run(model, labels)
        _assert_pair(metrics["block size"], expected_block, 1)
        call = model.decoder.calls[-1]
        mask = materialize_mask(call.block_mask.mask_mod, 1, 2 * _L, 2 * _L)
        assert torch.equal(mask[0], dense_bd3_reference(_L, expected_block, None))


def test_block_size_before_the_first_step_uses_the_loaded_iteration() -> None:
    """Before the training loop sets curr_iteration (validation of a resumed
    run), the block size follows args.iteration of the loaded checkpoint."""
    model = _make_model(schedule="0:4,0.5:8", block_size=8)
    del model.args.curr_iteration
    for loaded, expected_block in ((0, 4), (49, 4), (60, 8)):
        model.args.iteration = loaded
        assert model.current_block_size() == expected_block


# ---------------------------------------------------------------------- objective nld (section 15)


def _metric(metrics: dict, key: str) -> float:
    numerator, denominator = metrics[key]
    return float(numerator) / float(denominator)


def test_nld_with_ar_weight_zero_gives_the_bd3_loss() -> None:
    """I12: w_ar = 0, w_bd = 1 reproduces the bd3 loss of the same noise (B = 4 here)."""
    labels = _random_labels(seed=901)
    bd3 = _make_model(objective="bd3", curr_iteration=50)
    nld = _make_model(
        objective="nld", curr_iteration=50, nld_ar_weight=0.0, nld_diffusion_weight=1.0
    )
    bd3_loss, bd3_tokens, bd3_metrics = _run(bd3, labels)
    nld_loss, nld_tokens, nld_metrics = _run(nld, labels)
    assert torch.equal(nld_loss, bd3_loss)
    assert float(nld_tokens) == float(bd3_tokens)
    for key in ("masked ce", "mask frac", "nelbo bits/token"):
        assert _metric(nld_metrics, key) == pytest.approx(_metric(bd3_metrics, key))


def test_nld_ar_only_stage_is_the_ar_objective() -> None:
    """I12: before ar_only_fraction * train_iters (20 of 100) a step is objective ar (R10)."""
    labels = _random_labels(seed=902)
    for iteration in (0, 19):
        ar = _make_model(objective="ar", curr_iteration=iteration)
        nld = _make_model(objective="nld", curr_iteration=iteration)
        ar_loss, ar_tokens, ar_metrics = _run(ar, labels)
        nld_loss, nld_tokens, nld_metrics = _run(nld, labels)
        assert torch.equal(nld_loss, ar_loss)
        assert float(nld_tokens) == float(ar_tokens)
        assert set(nld_metrics) == set(ar_metrics)
        assert nld.decoder.calls[-1].hidden_states.shape[0] == _L  # one stream, no [xt, x0]
    joint = _make_model(objective="nld", curr_iteration=20)
    _run(joint, labels)
    assert joint.decoder.calls[-1].hidden_states.shape[0] == 2 * _L


def test_nld_loss_is_weighted_ar_plus_diffusion_over_the_ar_count() -> None:
    """Section 15: loss_sum = w_ar * ar_sum + w_bd * bd_sum, num_tokens = AR count."""
    labels = _random_labels(seed=903)
    x0 = labels[:, :-1]
    model = _make_model(objective="nld", curr_iteration=60)
    loss_sum, num_tokens, metrics = _run(model, labels)
    # The fake decoder is the identity, so the clean output at L + i - 1 is the embedding of
    # x0[i - 1]; the AR term is the cross entropy of its logits against x0[i].
    hidden = model.embedding.table(x0).transpose(0, 1)
    logits, _ = model.output_layer(hidden[:-1])
    ar_ce = _fake_lm_loss(x0[:, 1:], logits)
    ar_sum = ar_ce.sum()
    assert float(num_tokens) == _BATCH * (_L - 1)
    _assert_pair(metrics["ar loss"], float(ar_sum), _BATCH * (_L - 1))
    diffusion_sum = float(metrics["diffusion loss"][0])
    assert float(loss_sum) == pytest.approx(1.0 * float(ar_sum) + 0.3 * diffusion_sum, rel=1e-6)
    _assert_pair(metrics["lm loss"], float(loss_sum), _BATCH * (_L - 1))


def test_nld_block_mask_has_a_token_causal_clean_stream() -> None:
    labels = _random_labels(seed=904)
    model = _make_model(objective="nld", curr_iteration=60)
    _run(model, labels)
    mask = materialize_mask(model.decoder.calls[-1].block_mask.mask_mod, 1, 2 * _L, 2 * _L)
    assert torch.equal(mask[0], dense_bd3_reference(_L, _BLOCK, None, clean_causal=True))
    assert not torch.equal(mask[0], dense_bd3_reference(_L, _BLOCK, None))


def test_nld_validation_reports_nelbo_and_ar_bits() -> None:
    labels = _random_labels(seed=905)
    model = _make_model(objective="nld", curr_iteration=0, training=False)
    _, num_tokens, metrics = _run(model, labels)
    assert {"lm loss", "val nelbo bits/token", "val ar bits/token", "block size"} == set(metrics)
    assert float(num_tokens) == _BATCH * (_L - 1)
