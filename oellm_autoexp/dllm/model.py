"""Block-diffusion GPT model on top of the MegaDLMs difflm GPTModel.

The model implements the BD3/SDAR training formulation of the dllm-bd3-spec: the decoder input is
``[xt, x0]`` (noisy copy, clean copy) of length 2L, RoPE uses position ``i mod L`` for both copies,
attention follows the block-diffusion FlexAttention mask, and only the noisy copy (or, with token
shift, the clean copy at block starts) predicts ``x0``. The ``ar`` objective runs the same weights
as a causal model on ``x0`` alone. The ``nld`` objective (spec section 15) makes the clean copy
token-causal and adds its AR loss to the weighted block-diffusion loss, after an AR-only stage.

``forward`` returns ``(loss_sum, num_tokens, metrics)`` per micro-batch, where ``metrics`` maps a
log key to a ``(numerator, denominator)`` pair. ``pretrain_block_difflm.loss_func`` reduces them.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from megatron.core import parallel_state, tensor_parallel
from megatron.core.models.difflm import GPTModel
from torch import Tensor

from oellm_autoexp.dllm import context as bd_context
from oellm_autoexp.dllm.masks import (
    build_block_mask,
    compute_doc_ids,
    make_ar_mask_mod,
    make_bd3_mask_mod,
)
from oellm_autoexp.dllm.noise import (
    sample_block_noise,
    sample_complementary_noise,
    sample_validation_noise,
    validation_nelbo,
)
from oellm_autoexp.dllm.schedule import block_size_at, parse_block_schedule
from oellm_autoexp.dllm.targets import (
    apply_doc_source_rule,
    ar_loss_terms,
    ar_sources,
    bd3_loss_terms,
    prediction_sources,
    scorable_positions,
)

LN2 = math.log(2.0)

Metrics = dict[str, tuple[Tensor, Tensor]]


class BlockDiffusionGPTModel(GPTModel):
    """MegaDLMs GPTModel with the block-diffusion (bd3) and AR-control (ar)
    objectives.

    The constructor takes the MegaDLMs GPTModel arguments plus ``eod_id``. Everything else comes
    from the Megatron arguments (``--bd-*`` flags, ``--mask-token``, ``--seq-length``, ``--seed``).
    """

    def __init__(self, *args: Any, eod_id: int, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if not (self.pre_process and self.post_process):
            raise ValueError("BlockDiffusionGPTModel requires pipeline_model_parallel_size == 1")
        if self.position_embedding_type != "rope":
            raise ValueError("BlockDiffusionGPTModel requires --position-embedding-type rope")
        self.eod_id = int(eod_id)
        self.mask_id = int(self.args.mask_token)
        self.objective = self.args.bd_objective
        if self.objective not in ("bd3", "ar", "nld"):
            raise ValueError(f"bd_objective must be bd3, ar, or nld, got {self.objective!r}")
        self.seq_len_effective = int(self.args.seq_length) - 1
        self.block_schedule = parse_block_schedule(self.args.bd_block_schedule)
        self._train_generator: torch.Generator | None = None
        self._val_generator: torch.Generator | None = None
        self._val_iteration: int | None = None
        self._val_reset_pending = True
        self._mask_cache: dict[tuple, Any] = {}
        self._source_cache: dict[tuple, tuple[Tensor, Tensor]] = {}

    # ------------------------------------------------------------------ state helpers

    def train(self, mode: bool = True) -> BlockDiffusionGPTModel:
        """Switch train/eval mode; entering eval mode starts a new validation
        round.

        Megatron calls ``eval()`` once at the start of every ``evaluate`` call. Resetting the
        validation generator there (in addition to a change of ``args.curr_iteration``) keeps the
        valid and test rounds at the end of training deterministic as well.
        """
        if not mode:
            self._val_reset_pending = True
        return super().train(mode)

    def current_block_size(self) -> int:
        """Return the block size B for the current iteration (spec section 8).

        Megatron sets ``args.curr_iteration`` inside the training loop. Before the first step,
        for example in a validation round of a resumed run, ``args.iteration`` holds the
        iteration of the loaded checkpoint.
        """
        return block_size_at(
            self._iteration(),
            int(self.args.train_iters),
            self.block_schedule,
            int(self.args.bd_block_size),
        )

    def _iteration(self) -> int:
        """Return ``args.curr_iteration``, or ``args.iteration`` before the
        training loop."""
        iteration = getattr(self.args, "curr_iteration", None)
        if iteration is None:
            iteration = getattr(self.args, "iteration", 0) or 0
        return int(iteration)

    def nld_ar_stage(self) -> bool:
        """Whether an nld training step is in its AR-only stage (spec section
        15)."""
        fraction = float(self.args.bd_nld_ar_only_fraction)
        return self._iteration() < fraction * int(self.args.train_iters)

    def _noise_generator(self, device: torch.device) -> torch.Generator:
        """Return the training-noise generator (seed + 1000 * dp_rank, spec
        section 3)."""
        if self._train_generator is None or self._train_generator.device != device:
            seed = int(self.args.seed) + 1000 * parallel_state.get_data_parallel_rank()
            self._train_generator = torch.Generator(device=device).manual_seed(seed)
        return self._train_generator

    def _validation_generator(self, device: torch.device) -> torch.Generator:
        """Return the validation generator, reset at the first micro-batch of
        each round."""
        iteration = getattr(self.args, "curr_iteration", None)
        if (
            self._val_generator is None
            or self._val_generator.device != device
            or self._val_reset_pending
            or iteration != self._val_iteration
        ):
            self._val_generator = torch.Generator(device=device).manual_seed(
                int(self.args.bd_val_seed)
            )
            self._val_iteration = iteration
            self._val_reset_pending = False
        return self._val_generator

    def _block_mask(
        self, objective: str, seq_len: int, block_size: int, doc_ids: Tensor | None, device
    ) -> Any:
        """Build the FlexAttention BlockMask; reuse it when there are no
        document ids.

        With document ids, the mask_mod closure and its document-id
        buffer are created once per (objective, L, B, batch) and the
        buffer is refreshed in place, so the compiled flex_attention
        always sees the same mask_mod object and does not recompile per
        micro-batch. Refreshing in place is safe because PP == 1 runs
        the forward and backward of one micro-batch before the next
        forward starts (stream order on one device).
        """
        q_len = seq_len if objective == "ar" else 2 * seq_len
        batch = None if doc_ids is None else doc_ids.shape[0]
        key = (objective, seq_len, block_size, batch, str(device))
        if key not in self._mask_cache:
            doc_buffer = None if doc_ids is None else doc_ids.clone()
            if objective == "ar":
                mask_mod = make_ar_mask_mod(seq_len, doc_buffer)
            else:
                mask_mod = make_bd3_mask_mod(
                    seq_len, block_size, doc_buffer, clean_causal=objective == "nld"
                )
            block_mask = None
            if doc_ids is None:
                block_mask = build_block_mask(mask_mod, None, q_len, q_len, device)
            self._mask_cache[key] = (mask_mod, doc_buffer, block_mask)
        mask_mod, doc_buffer, block_mask = self._mask_cache[key]
        if doc_buffer is None:
            return block_mask
        doc_buffer.copy_(doc_ids)
        return build_block_mask(mask_mod, batch, q_len, q_len, device)

    def _sources(
        self, seq_len: int, block_size: int, token_shift: int, device
    ) -> tuple[Tensor, Tensor]:
        key = (seq_len, block_size, token_shift, str(device))
        if key not in self._source_cache:
            self._source_cache[key] = prediction_sources(
                seq_len, block_size, token_shift, device=device
            )
        return self._source_cache[key]

    # ------------------------------------------------------------------ model pieces

    def _decode(self, input_ids: Tensor, rope_len: int, block_mask: Any) -> Tensor:
        """Run embedding and decoder. Returns final hidden states [s, b, h].

        Megatron applies RoPE by sequence index, so a [xt, x0] input of
        length 2L gets the rotary table of L positions concatenated with
        itself (position i mod L, spec section 4).
        """
        seq = input_ids.shape[1]
        if seq % rope_len != 0:
            raise ValueError(f"input length {seq} is not a multiple of the rope length {rope_len}")
        positions = torch.arange(rope_len, device=input_ids.device).repeat(seq // rope_len)
        position_ids = positions.unsqueeze(0).expand(input_ids.shape[0], -1)
        decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
        rope = self.rotary_pos_emb(rope_len)
        rotary_pos_emb = torch.cat([rope] * (seq // rope_len), dim=0)
        bd_context.set_block_mask(block_mask)
        return self.decoder(
            hidden_states=decoder_input, attention_mask=None, rotary_pos_emb=rotary_pos_emb
        )

    def _logits_at(self, hidden: Tensor, src: Tensor) -> Tensor:
        """Gather hidden states [s, b, h] at ``src`` and apply the output layer
        -> [L, b, V/tp]."""
        hidden_src = hidden.index_select(0, src)
        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()
        logits, _ = self.output_layer(hidden_src, weight=output_weight)
        return logits

    def _bd3_doc_ids(self, x0: Tensor) -> Tensor | None:
        return compute_doc_ids(x0, self.eod_id) if int(self.args.bd_doc_mask) else None

    # ------------------------------------------------------------------ forward

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        decoder_input: Tensor | None = None,
        labels: Tensor | None = None,
        inference_params: Any = None,
        packed_seq_params: Any = None,
        extra_block_kwargs: dict | None = None,
        runtime_gather_output: bool | None = None,
        loss_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Metrics]:
        """Compute the per-micro-batch loss terms.

        Args:
            input_ids: Megatron ``tokens`` [b, S]. Unused except for its shape; x0 comes from
                ``labels`` (spec section 2).
            position_ids: Ignored (positions are i mod L).
            attention_mask: Must be None; the mask is built from the spec.
            decoder_input: Unsupported.
            labels: Megatron ``labels`` [b, S]; x0 = labels[:, :-1].
            inference_params: Unsupported.
            packed_seq_params: Unsupported.
            extra_block_kwargs: Unsupported.
            runtime_gather_output: Unsupported.
            loss_mask: Megatron ``loss_mask`` [b, S]; target_ok = loss_mask[:, :-1].

        Returns:
            ``(loss_sum, num_tokens, metrics)`` as described in the module docstring.
        """
        if labels is None or loss_mask is None:
            raise ValueError("BlockDiffusionGPTModel.forward needs labels and loss_mask")
        if attention_mask is not None:
            raise ValueError("attention_mask must be None (set --no-create-attention-mask...)")
        if any(
            v is not None
            for v in (
                decoder_input,
                inference_params,
                packed_seq_params,
                extra_block_kwargs,
                runtime_gather_output,
            )
        ):
            raise NotImplementedError("decoder_input, inference and packed sequences unsupported")
        x0 = labels[:, :-1].contiguous()
        target_ok = loss_mask[:, :-1] > 0
        if x0.shape[1] != self.seq_len_effective:
            raise ValueError(
                f"x0 length {x0.shape[1]} != seq_length - 1 = {self.seq_len_effective}"
            )
        if self.objective == "ar":
            return self._forward_ar(x0, target_ok)
        if self.objective == "nld":
            return self._forward_nld(x0, target_ok)
        return self._forward_bd3(x0, target_ok)

    def _noise(
        self, x0: Tensor, target_ok: Tensor, doc_ids: Tensor | None, block_size: int
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor, Tensor, Tensor]:
        """Draw the noisy copy for training (spec section 3) or validation
        (section 9).

        Returns:
            ``(x0, target_ok, doc_ids, xt, masked, weight)``. With complementary masking the
            first three are repeated on the batch axis. ``weight`` is ``p_block`` in training
            and the validation block weight in eval mode.
        """
        args = self.args
        if not self.training:
            generator = self._validation_generator(x0.device)
            xt, masked, block_weight = sample_validation_noise(
                x0, block_size, self.mask_id, generator
            )
            return x0, target_ok, doc_ids, xt, masked, block_weight
        generator = self._noise_generator(x0.device)
        if int(args.bd_complementary_mask):
            xt, masked, p_block = sample_complementary_noise(
                x0, block_size, self.mask_id, args.bd_t_min, args.bd_t_max, generator
            )
            x0 = torch.cat([x0, x0], dim=0)
            target_ok = torch.cat([target_ok, target_ok], dim=0)
            doc_ids = torch.cat([doc_ids, doc_ids], dim=0) if doc_ids is not None else None
            return x0, target_ok, doc_ids, xt, masked, p_block
        xt, masked, p_block = sample_block_noise(
            x0, block_size, self.mask_id, args.bd_t_min, args.bd_t_max, generator
        )
        return x0, target_ok, doc_ids, xt, masked, p_block

    def _forward_bd3(self, x0: Tensor, target_ok: Tensor) -> tuple[Tensor, Tensor, Metrics]:
        args = self.args
        seq_len = x0.shape[1]
        block_size = self.current_block_size()
        token_shift = int(args.bd_token_shift)
        doc_ids = self._bd3_doc_ids(x0)
        x0, target_ok, doc_ids, xt, masked, weight = self._noise(x0, target_ok, doc_ids, block_size)
        p_block = block_weight = weight

        batch = x0.shape[0]
        block_mask = self._block_mask("bd3", seq_len, block_size, doc_ids, x0.device)
        hidden = self._decode(torch.cat([xt, x0], dim=1), seq_len, block_mask)
        src, base_has_src = self._sources(seq_len, block_size, token_shift, x0.device)
        has_src = apply_doc_source_rule(base_has_src, doc_ids, token_shift, batch)
        ce = self.compute_language_model_loss(x0, self._logits_at(hidden, src)).float()
        scorable = scorable_positions(has_src, target_ok)

        one = torch.ones((), device=x0.device)
        if not self.training:
            nelbo_sum, scored = validation_nelbo(ce, masked, block_weight, scorable)
            metrics: Metrics = {
                "lm loss": (nelbo_sum, scored),
                "val nelbo bits/token": (nelbo_sum / LN2, scored),
                "block size": (one * block_size, one),
            }
            return nelbo_sum, scored, metrics

        terms = bd3_loss_terms(ce, masked, scorable, p_block, block_size, args.bd_loss_weighting)
        metrics = {
            "lm loss": (terms["loss_sum"].detach(), terms["num_tokens"]),
            "masked ce": (terms["masked_ce_sum"].detach(), terms["num_targets"]),
            "mask frac": (p_block.sum().float(), one * p_block.numel()),
            "block size": (one * block_size, one),
            "nelbo bits/token": (terms["elbo_sum"].detach() / LN2, terms["scored"]),
        }
        return terms["loss_sum"], terms["num_tokens"], metrics

    def _forward_ar(self, x0: Tensor, target_ok: Tensor) -> tuple[Tensor, Tensor, Metrics]:
        seq_len = x0.shape[1]
        doc_ids = self._bd3_doc_ids(x0)
        block_mask = self._block_mask("ar", seq_len, 1, doc_ids, x0.device)
        hidden = self._decode(x0, seq_len, block_mask)
        src, has_src = ar_sources(seq_len, device=x0.device)
        ce = self.compute_language_model_loss(x0, self._logits_at(hidden, src)).float()
        scorable = scorable_positions(has_src.unsqueeze(0).expand(x0.shape[0], -1), target_ok)
        loss_sum, num_tokens = ar_loss_terms(ce, scorable)
        one = torch.ones((), device=x0.device)
        metrics: Metrics = {
            "lm loss": (loss_sum.detach(), num_tokens),
            "block size": (one, one),
        }
        key = "nelbo bits/token" if self.training else "val nelbo bits/token"
        metrics[key] = (loss_sum.detach() / LN2, num_tokens)
        return loss_sum, num_tokens, metrics

    def _forward_nld(self, x0: Tensor, target_ok: Tensor) -> tuple[Tensor, Tensor, Metrics]:
        """Hybrid step (spec section 15): AR loss of the token-causal clean
        stream plus the weighted bd3 loss of the noisy stream, over the AR
        token count; an AR-only stage first."""
        if self.training and self.nld_ar_stage():
            return self._forward_ar(x0, target_ok)
        args = self.args
        seq_len = x0.shape[1]
        block_size = self.current_block_size()
        token_shift = int(args.bd_token_shift)
        doc_ids = self._bd3_doc_ids(x0)
        x0, target_ok, doc_ids, xt, masked, weight = self._noise(x0, target_ok, doc_ids, block_size)

        batch = x0.shape[0]
        block_mask = self._block_mask("nld", seq_len, block_size, doc_ids, x0.device)
        hidden = self._decode(torch.cat([xt, x0], dim=1), seq_len, block_mask)
        src, base_has_src = self._sources(seq_len, block_size, token_shift, x0.device)
        has_src = apply_doc_source_rule(base_has_src, doc_ids, token_shift, batch)
        ce = self.compute_language_model_loss(x0, self._logits_at(hidden, src)).float()
        scorable = scorable_positions(has_src, target_ok)
        # AR term: the clean output at index L + i - 1 predicts x0[i].
        ar_src, ar_has_src = ar_sources(seq_len, device=x0.device)
        ar_ce = self.compute_language_model_loss(
            x0, self._logits_at(hidden, ar_src + seq_len)
        ).float()
        ar_scorable = scorable_positions(ar_has_src.unsqueeze(0).expand(batch, -1), target_ok)
        ar_sum, ar_count = ar_loss_terms(ar_ce, ar_scorable)
        ar_weight = float(args.bd_nld_ar_weight)
        diffusion_weight = float(args.bd_nld_diffusion_weight)

        one = torch.ones((), device=x0.device)
        if not self.training:
            nelbo_sum, scored = validation_nelbo(ce, masked, weight, scorable)
            loss_sum = ar_weight * ar_sum + diffusion_weight * nelbo_sum
            metrics: Metrics = {
                "lm loss": (loss_sum, ar_count),
                "val nelbo bits/token": (nelbo_sum / LN2, scored),
                "val ar bits/token": (ar_sum / LN2, ar_count),
                "block size": (one * block_size, one),
            }
            return loss_sum, ar_count, metrics

        terms = bd3_loss_terms(ce, masked, scorable, weight, block_size, args.bd_loss_weighting)
        loss_sum = ar_weight * ar_sum + diffusion_weight * terms["loss_sum"]
        metrics = {
            "lm loss": (loss_sum.detach(), ar_count),
            "ar loss": (ar_sum.detach(), ar_count),
            "diffusion loss": (terms["loss_sum"].detach(), terms["num_tokens"]),
            "masked ce": (terms["masked_ce_sum"].detach(), terms["num_targets"]),
            "mask frac": (weight.sum().float(), one * weight.numel()),
            "block size": (one * block_size, one),
            "nelbo bits/token": (terms["elbo_sum"].detach() / LN2, terms["scored"]),
        }
        return loss_sum, ar_count, metrics

    # ------------------------------------------------------------------ diagnostics

    def ar_logits(self, x0: Tensor, doc_mask: bool = False) -> Tensor:
        """Causal logits for every position of ``x0`` [b, L] -> [b, L, V]
        (gathered over TP).

        Output ``i`` predicts ``x0[i + 1]``. Used by the parity check.
        """
        seq_len = x0.shape[1]
        doc_ids = compute_doc_ids(x0, self.eod_id) if doc_mask else None
        block_mask = self._block_mask("ar", seq_len, 1, doc_ids, x0.device)
        hidden = self._decode(x0, seq_len, block_mask)
        src = torch.arange(seq_len, device=x0.device)
        logits = self._logits_at(hidden, src)
        logits = tensor_parallel.gather_from_tensor_model_parallel_region(logits)
        return logits.transpose(0, 1).contiguous()

    def bd3_logits(
        self, x0: Tensor, xt: Tensor, block_size: int, token_shift: int, doc_mask: bool = False
    ) -> Tensor:
        """Block-diffusion logits at the prediction sources [b, L, V] (gathered
        over TP).

        Output ``i`` predicts ``x0[i]`` (spec section 6). Used by the parity check (I3).
        """
        seq_len = x0.shape[1]
        doc_ids = compute_doc_ids(x0, self.eod_id) if doc_mask else None
        block_mask = self._block_mask("bd3", seq_len, block_size, doc_ids, x0.device)
        hidden = self._decode(torch.cat([xt, x0], dim=1), seq_len, block_mask)
        src, _ = self._sources(seq_len, block_size, token_shift, x0.device)
        logits = self._logits_at(hidden, src)
        logits = tensor_parallel.gather_from_tensor_model_parallel_region(logits)
        return logits.transpose(0, 1).contiguous()

    def nld_clean_logits(
        self, x0: Tensor, xt: Tensor, block_size: int, doc_mask: bool = False
    ) -> Tensor:
        """Clean-stream logits of the nld input ``[xt, x0]`` [b, L, V]
        (gathered over TP).

        Output ``i`` predicts ``x0[i + 1]``. With the token-causal clean stream these equal
        ``ar_logits(x0)`` whatever ``xt`` holds (invariant I11). Used by the parity check.
        """
        seq_len = x0.shape[1]
        doc_ids = compute_doc_ids(x0, self.eod_id) if doc_mask else None
        block_mask = self._block_mask("nld", seq_len, block_size, doc_ids, x0.device)
        hidden = self._decode(torch.cat([xt, x0], dim=1), seq_len, block_mask)
        src = torch.arange(seq_len, 2 * seq_len, device=x0.device)
        logits = self._logits_at(hidden, src)
        logits = tensor_parallel.gather_from_tensor_model_parallel_region(logits)
        return logits.transpose(0, 1).contiguous()
