"""Register a gpt_plus TrainSpec that enforces YAML-provided model sizes."""

from __future__ import annotations

from titan_oellm.models.gpt_plus import (
    Transformer,
    parallelize_gpt_plus,
)
from titan_oellm.models.gpt_plus.model.args import TransformerModelArgs
from titan_oellm.components.lr_scheduler_universal import build_lr_schedulers_auto
from titan_oellm.components.metrics_with_parameter_logging import (
    build_metrics_processor_with_parameter_logging,
)
from titan_oellm.components.validator import build_sci_validator
from titan_oellm.datasets.sci_dataloader import build_sci_dataloader
from titan_oellm.datasets.sci_tokenizers.sci_tokenizer import build_sci_hf_tokenizer
from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.optimizer import build_optimizers
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.protocols.train_spec import register_train_spec, TrainSpec


class CustomTransformerModelArgs(TransformerModelArgs):
    def update_from_config(self, job_config, **kwargs) -> None:  # type: ignore[override]
        # Apply YAML-provided sizing first
        model_cfg = job_config.model
        if getattr(model_cfg, "hidden_dim", None):
            self.dim = model_cfg.hidden_dim
        if getattr(model_cfg, "num_layers", None):
            self.n_layers = model_cfg.num_layers
        if getattr(model_cfg, "num_attention_heads", None):
            self.n_heads = model_cfg.num_attention_heads
        if getattr(model_cfg, "n_kv_heads", None) is not None:
            self.n_kv_heads = model_cfg.n_kv_heads
        if getattr(model_cfg, "multiple_of", None):
            self.multiple_of = model_cfg.multiple_of

        if getattr(model_cfg, "ffn_dim_multiplier", None) is not None:
            self.ffn_dim_multiplier = model_cfg.ffn_dim_multiplier
        elif getattr(model_cfg, "ffn_hidden_size", None) and self.dim:
            self.ffn_dim_multiplier = model_cfg.ffn_hidden_size / float(self.dim)

        # Apply standard updates (rope, qk_norm, etc.)
        super().update_from_config(job_config, **kwargs)


def get_train_spec() -> TrainSpec:
    return TrainSpec(
        model_cls=Transformer,
        model_args={"custom": CustomTransformerModelArgs()},
        parallelize_fn=parallelize_gpt_plus,
        pipelining_fn=pipeline_llm,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers_auto,
        build_dataloader_fn=build_sci_dataloader,
        build_tokenizer_fn=build_sci_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_sci_validator,
        build_metrics_processor_fn=build_metrics_processor_with_parameter_logging,
    )


register_train_spec("gpt_plus_custom", get_train_spec())
