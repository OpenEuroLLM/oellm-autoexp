"""NeMo Automodel configuration schema (auto-generated)."""

from dataclasses import dataclass, field
from typing import Any, Literal

from compoconf import ConfigInterface


@dataclass
class AutomodelBenchmarkConfig:
    """Auto-generated config for benchmark section."""

    # Number of warmup steps before timing.
    warmup_steps: int = 10
    # Peak hardware TFLOPS for MFU calculation (e.g. 989 for H100).
    peak_tflops: int = 989
    # Step to start nsys profiling (-1 to disable).
    nsys_start: int = -1
    # Step to end nsys profiling (-1 to disable).
    nsys_end: int = -1
    # Ranks to profile with nsys.
    nsys_ranks: Any = field(default_factory=lambda: [])
    # Number of nodes for TFLOPS calculation.
    num_nodes: int = 1


@dataclass
class AutomodelStepSchedulerConfig:
    """Auto-generated config for step_scheduler section."""

    # Total number of samples processed per optimizer step across all GPUs. This is the
    # effective batch size for the entire training step.
    global_batch_size: int | None = None
    # Number of samples per micro-batch per GPU. This is the batch size for a single
    # forward/backward pass on one GPU.
    local_batch_size: int | None = None
    # Frequency of checkpoint steps.
    ckpt_every_steps: int | None = None
    # Number of training steps between validation.
    val_every_steps: int | None = None
    # Frequency of remote logging (e.g., WandB, MLflow). Default: 1 (every step).
    log_remote_every_steps: int = 1
    # Frequency of manual garbage collection steps.
    gc_every_steps: int | None = None
    # Initial global step. Used when resuming from checkpoint. Default: 0.
    start_step: int = 0
    # Initial epoch. Used when resuming from checkpoint. Default: 0.
    start_epoch: int = 0
    # Total number of epochs. Default: None or calculated from max_steps if num_epochs is None
    # or 10 if max_steps and num_epochs are both None.
    num_epochs: int | None = None
    # Maximum number of steps to run. If None, calculated from num_epochs.
    max_steps: int | None = None


@dataclass
class AutomodelPipelineConfig:
    """Auto-generated config for distributed.pipeline section."""

    # Pipeline schedule type. Supported values: "1f1b" (one-forward-one-backward), "gpipe",
    # "interleaved_1f1b", "looped_bfs", "dfs", "v_schedule", "zero_bubble". Defaults to
    # "1f1b".
    pp_schedule: str | None = "1f1b"
    # Path to a CSV file defining a custom pipeline schedule. If provided, overrides
    # pp_schedule.
    pp_schedule_csv: str | None = None
    # Size of each microbatch for pipeline execution. pp_batch_size must be divisible by
    # pp_microbatch_size.
    pp_microbatch_size: int = 1
    # Total batch size per pipeline stage. Must be divisible by pp_microbatch_size.
    pp_batch_size: int = 1
    # Number of transformer layers per pipeline stage. If None, layers are split evenly across
    # stages.
    layers_per_stage: int | None = None
    # When using virtual stages (interleaved schedules), round the number of virtual stages to
    # a multiple of pp_size. "up" rounds up, "down" rounds down. If None, no rounding is
    # applied.
    round_virtual_stages_to_pp_multiple: Literal["up", "down"] | None = None
    # Explicit specification of which module FQNs belong to each model part/stage. If
    # provided, overrides automatic layer splitting.
    module_fqns_per_model_part: list[list[str]] | None = None
    # Apply pipeline patches to the inner model (e.g., the base transformer in a CausalLM
    # wrapper). Defaults to True.
    patch_inner_model: bool = True
    # Apply pipeline patches to the CausalLM wrapper model. Defaults to True.
    patch_causal_lm_model: bool = True
    # Patch stage backward to use no_sync context for gradient accumulation efficiency. Useful
    # when combining PP with FSDP.
    patch_stage_backward_maybe_with_nosync: bool = False
    # Data type for pipeline computation. If None, uses the model's default dtype.
    dtype: Any | None = None
    # Scale gradients within the pipeline schedule (by 1/n_microbatches). If False, gradients
    # must be scaled externally. Defaults to False.
    scale_grads_in_schedule: bool = False
    # Sequence length hint for pipeline parallel shape inference. If None, it will be inferred
    # from the dataset config.
    pp_seq_len: int | None = None


@dataclass
class AutomodelDistEnvConfig:
    """Auto-generated config for dist_env section."""

    # Distributed backend.
    backend: str = "nccl"
    # Distributed initialization timeout in minutes.
    timeout_minutes: int = 10


@dataclass
class AutomodelTEFp8Config:
    """Auto-generated config for model.backend.te_fp8 section."""

    recipe: Literal["current", "block"] | Any = "current"


@dataclass
class AutomodelBackendConfig:
    """Auto-generated config for model.backend section."""

    # Attention backend ("te", "sdpa", or "flex").
    attn: Literal["te", "sdpa", "flex"] = "te"
    # Linear layer backend ("torch" or "te").
    linear: Literal["torch", "te"] = "te"
    # RMSNorm backend ("torch", "torch_fp32", or "te").
    rms_norm: Literal["torch", "torch_fp32", "te"] = "torch_fp32"
    # Whether to use fused RoPE (requires TE).
    rope_fusion: bool = True
    # MoE expert GEMM backend. "torch" uses per-expert loop, "te" uses TE GroupedLinear, "gmm"
    # uses grouped_gemm.ops.gmm, "torch_mm" uses torch._grouped_mm.
    experts: Literal["torch", "te", "gmm", "torch_mm"] = "torch_mm"
    # MoE token dispatcher. "torch" uses DTensor all-gather/reduce-scatter, "deepep" uses
    # DeepEP for token dispatch.
    dispatcher: Literal["torch", "deepep", "hybridep"] = "torch"
    dispatcher_num_sms: int = 20
    # If True, replace the learned Gate with FakeBalancedGate that assigns tokens to experts
    # without learned routing weights.
    fake_balanced_gate: bool = False
    # Noise level [0, 1] for FakeBalancedGate. When > 0, uses biased topk selection seeded
    # from the input content so routing varies dynamically across training steps (like real
    # Gate) while remaining deterministic for activation checkpointing recompute (same input =
    # same routing). Only used when fake_balanced_gate=True.
    fake_gate_noise: float = 0.0
    # Whether to enable HuggingFace state dict adapter.
    enable_hf_state_dict_adapter: bool = True
    # Whether to enable FSDP2 optimizations.
    enable_fsdp_optimizations: bool = False
    te_fp8: AutomodelTEFp8Config = field(default_factory=AutomodelTEFp8Config)


@dataclass
class AutomodelModelConfig:
    """Auto-generated config for model section."""

    # Hydra target for model factory.
    _target_: str = "nemo_automodel.NeMoAutoModelForCausalLM.from_config"
    # Trust remote code when loading HF models.
    trust_remote_code: bool = True
    backend: AutomodelBackendConfig = field(default_factory=AutomodelBackendConfig)


@dataclass
class AutomodelDistributedConfig:
    """Auto-generated config for distributed section."""

    # Distribution strategy (fsdp2, megatron_fsdp, ddp).
    strategy: str = "fsdp2"
    # Tensor parallel size.
    tp_size: int = 1
    # Context parallel size.
    cp_size: int = 1
    # Pipeline parallel size.
    pp_size: int = 1
    # Data parallel replicate size (FSDP2 only).
    dp_replicate_size: int = 1
    # Expert parallel size.
    ep_size: int = 1
    # Enable sequence parallelism in TP plan.
    sequence_parallel: bool = False
    # Enable activation checkpointing.
    activation_checkpointing: bool = False
    # Defer FSDP gradient sync to final micro-batch.
    defer_fsdp_grad_sync: bool = True
    pipeline: AutomodelPipelineConfig = field(default_factory=AutomodelPipelineConfig)


@dataclass
class AutomodelCheckpointConfig:
    """Auto-generated config for checkpoint section."""

    # Enable checkpointing.
    enabled: bool = False
    # Directory for saving checkpoints.
    checkpoint_dir: str = "checkpoints/"
    # Checkpoint format: torch_save or safetensors.
    model_save_format: str = "torch_save"
    # Save model in consolidated safetensors format.
    save_consolidated: bool = False
    # Path to restore checkpoint from.
    restore_from: str | None = None


@dataclass
class AutomodelLossFnConfig:
    """Auto-generated config for loss_fn section."""

    # Hydra target for loss class.
    _target_: str = "nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy"
    # if True it will cast logits to float32 before computing cross entropy. Default: True.
    fp32_upcast: bool = True
    # label to ignore in CE calculation. Defaults to -100.
    ignore_index: int = -100
    # type of reduction. Defaults to "sum".
    reduction: str = "sum"


@dataclass
class AutomodelDatasetConfig:
    """Auto-generated config for dataset section."""

    # Hydra target for dataset class.
    _target_: str = (
        "nemo_automodel.components.datasets.llm.mock_iterable_dataset.MockIterableDataset"
    )
    # Sequence length for training.
    seq_len: int = 4096
    # Number of samples in the dataset.
    num_samples: int = 100000
    # Batch size (usually interpolated from step_scheduler).
    batch_size: str = "${..step_scheduler.local_batch_size}"


@dataclass
class AutomodelDataloaderConfig:
    """Auto-generated config for dataloader section."""

    # DataLoader batch size (null when dataset yields batches).
    batch_size: str | None = None
    # Hydra target for DataLoader class.
    _target_: str = "torch.utils.data.DataLoader"
    # Number of DataLoader workers.
    num_workers: int = 0


@dataclass
class AutomodelOptimizerConfig:
    """Auto-generated config for optimizer section."""

    # Hydra target for optimizer class.
    _target_: str = "torch.optim.Adam"
    # Adam beta coefficients.
    betas: Any = field(default_factory=lambda: [0.9, 0.999])
    # Adam epsilon for numerical stability.
    eps: float = 1e-08
    # Learning rate.
    lr: float = 0.0001
    # Weight decay.
    weight_decay: int = 0
    # Use foreach implementation (set false for TE GroupedLinear).
    foreach: bool = False


@dataclass
class AutomodelLrSchedulerConfig:
    """Auto-generated config for lr_scheduler section."""

    # LR decay schedule: cosine, linear, constant.
    lr_decay_style: str = "cosine"
    # Number of warmup steps.
    lr_warmup_steps: int = 500
    # Minimum learning rate after decay.
    min_lr: float = 0.0


@dataclass
class AutomodelValidationDatasetConfig:
    """Auto-generated config for validation_dataset section."""

    # Hydra target for validation dataset class.
    _target_: str = (
        "nemo_automodel.components.datasets.llm.mock_iterable_dataset.MockIterableDataset"
    )
    # Sequence length for validation.
    seq_len: int = 4096
    # Number of validation samples.
    num_samples: int = 1000
    # Validation batch size.
    batch_size: str = "${..step_scheduler.local_batch_size}"


@dataclass
class AutomodelValidationDataloaderConfig:
    """Auto-generated config for validation_dataloader section."""

    # Validation DataLoader batch size (null when dataset yields batches).
    batch_size: str | None = None
    # Hydra target for validation DataLoader class.
    _target_: str = "torch.utils.data.DataLoader"
    # Number of validation DataLoader workers.
    num_workers: int = 0


@dataclass
class AutomodelClipGradNormConfig:
    """Auto-generated config for clip_grad_norm section."""

    # Maximum gradient norm for clipping.
    max_norm: float = 1.0


@dataclass
class AutomodelConfig(ConfigInterface):
    """Typed projection of NeMo Automodel configuration."""

    # Generated by scripts/generate_automodel_config.py

    # Recipe class name.
    recipe: str = "BenchmarkingRecipeForNextTokenPrediction"
    # Random seed.
    seed: int = 1234
    benchmark: AutomodelBenchmarkConfig = field(default_factory=AutomodelBenchmarkConfig)
    step_scheduler: AutomodelStepSchedulerConfig = field(
        default_factory=AutomodelStepSchedulerConfig
    )
    distributed: AutomodelDistributedConfig = field(default_factory=AutomodelDistributedConfig)
    dist_env: AutomodelDistEnvConfig = field(default_factory=AutomodelDistEnvConfig)
    model: AutomodelModelConfig = field(default_factory=AutomodelModelConfig)
    checkpoint: AutomodelCheckpointConfig = field(default_factory=AutomodelCheckpointConfig)
    loss_fn: AutomodelLossFnConfig = field(default_factory=AutomodelLossFnConfig)
    dataset: AutomodelDatasetConfig = field(default_factory=AutomodelDatasetConfig)
    dataloader: AutomodelDataloaderConfig = field(default_factory=AutomodelDataloaderConfig)
    optimizer: AutomodelOptimizerConfig = field(default_factory=AutomodelOptimizerConfig)
    lr_scheduler: AutomodelLrSchedulerConfig = field(default_factory=AutomodelLrSchedulerConfig)
    validation_dataset: AutomodelValidationDatasetConfig = field(
        default_factory=AutomodelValidationDatasetConfig
    )
    validation_dataloader: AutomodelValidationDataloaderConfig = field(
        default_factory=AutomodelValidationDataloaderConfig
    )
    clip_grad_norm: AutomodelClipGradNormConfig = field(default_factory=AutomodelClipGradNormConfig)
    aux: dict[str, Any] = field(default_factory=dict)


__all__ = ["AutomodelConfig"]
