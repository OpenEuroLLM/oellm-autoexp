"""Qwen3 with Gemma-2 style final-logit softcapping applied at inference.

Megatron-Bridge carries `final_logit_softcapping` only for Gemma 2 and Gemma-VL, and the
Qwen3 HF config has no such field, so a model trained with
`--final-logit-softcapping` exports to an HF checkpoint that runs UNCAPPED. For
likelihood-scored evaluation that is the wrong model: the cap changes the softmax
normalizer, so per-token log-probabilities differ and multiple-choice options can reorder.
(Argmax metrics are unaffected -- ``c * tanh(x / c)`` is strictly monotone.)

This subclass restores the cap. Drop it beside the exported weights and point `auto_map` at
it, with `final_logit_softcapping` added to config.json::

    "architectures": ["Qwen3SoftcapForCausalLM"],
    "auto_map": {"AutoModelForCausalLM": "modeling_qwen3_softcap.Qwen3SoftcapForCausalLM"},
    "final_logit_softcapping": 30.0

Loading it needs `trust_remote_code=True`, which the oellm-eval backend sets by default.

The cap is applied to the returned logits AND, when `labels` are supplied, to the logits the
loss is computed from -- the base class would otherwise compute its loss from the uncapped
tensor before this wrapper ever sees it. lm-eval's loglikelihood path does not pass labels
(it log-softmaxes the returned logits itself), so that branch mainly guards other callers.
"""

import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM


class Qwen3SoftcapForCausalLM(Qwen3ForCausalLM):
    """Qwen3 that applies ``c * tanh(logits / c)`` to the final logits."""

    def forward(self, *args, labels=None, **kwargs):
        """Run the base model uncapped, then cap the logits (and any loss)."""
        # labels are withheld from the base call on purpose: its internal loss would be
        # computed from uncapped logits, which is exactly what this class exists to avoid.
        outputs = super().forward(*args, labels=None, **kwargs)

        cap = getattr(self.config, "final_logit_softcapping", None)
        if cap:
            cap = float(cap)
            logits = outputs.logits
            # fp32 for the tanh: the export is bf16, and doing this in bf16 would add
            # roughly 0.2% relative rounding to every logit for no reason at eval time.
            outputs.logits = (cap * torch.tanh(logits.float() / cap)).to(logits.dtype)

        if labels is not None:
            outputs.loss = self._causal_lm_loss(outputs.logits, labels)

        return outputs

    @staticmethod
    def _causal_lm_loss(logits, labels):
        """Standard shifted causal-LM cross entropy, from the CAPPED logits."""
        shift_logits = logits[..., :-1, :].contiguous().float()
        shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
        return torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
