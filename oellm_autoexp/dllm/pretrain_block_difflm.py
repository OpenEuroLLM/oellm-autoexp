"""Block-diffusion (BD3/SDAR-style) training launcher for MegaDLMs.

Run with ``PYTHONPATH=.:submodules/MegaDLMs torchrun ... oellm_autoexp/dllm/pretrain_block_difflm.py``.
It keeps MegaDLMs' data pipeline, argument parser (chained with the ``--bd-*`` flags), and training
loop, and replaces the model and loss with ``BlockDiffusionGPTModel`` (see dllm-bd3-spec).
"""

from __future__ import annotations

import os
import sys

if __package__ in (None, ""):
    # Started as a script: make the repository root and MegaDLMs importable.
    _REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    for _path in (os.path.join(_REPO_ROOT, "submodules", "MegaDLMs"), _REPO_ROOT):
        if _path not in sys.path:
            sys.path.insert(0, _path)

import torch
import torch._dynamo
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.difflm.gpt_layer_specs import (
    get_difflm_layer_with_transformer_engine_spec,
)
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from megatron.core.transformer.attention import (
    DiffLMSelfAttention,
    SelfAttentionSubmodules,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.qwen2_norm import Qwen2RMSNorm
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.transformer.transformer_layer import (
    DiffLMTransformerLayer,
    TransformerLayerSubmodules,
)
from megatron.training import (
    get_args,
    get_timers,
    get_tokenizer,
    pretrain,
    print_rank_0,
)
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.global_vars import get_wandb_writer
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    print_rank_last,
)
from pretrain_difflm import train_valid_test_datasets_provider

from oellm_autoexp.dllm.args import extra_args_provider, validate_bd_args
from oellm_autoexp.dllm.flex_core_attention import BlockDiffusionFlexAttention
from oellm_autoexp.dllm.model import BlockDiffusionGPTModel


def local_block_spec(args, qk_layernorm: bool) -> TransformerBlockSubmodules:
    """Transformer-Engine-free block spec (``--transformer-impl local``).

    Tensor-parallel mcore linears, the HF-exact ``Qwen2RMSNorm`` for every norm (including the
    final one), and the block-diffusion flex core attention. Checkpoint keys follow the "local"
    layout of ``oellm_autoexp.dllm.convert.qwen3_layout`` (``input_layernorm``,
    ``pre_mlp_layernorm``).

    Args:
        args: Megatron arguments (``num_layers`` is read).
        qk_layernorm: Apply RMSNorm to queries and keys (Qwen3).

    Returns:
        The block submodules for ``DiffLMTransformerBlock``.
    """
    norm = Qwen2RMSNorm
    layer = ModuleSpec(
        module=DiffLMTransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=norm,
            self_attention=ModuleSpec(
                module=DiffLMSelfAttention,
                params={"attn_mask_type": AttnMaskType.no_mask},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear,
                    core_attention=BlockDiffusionFlexAttention,
                    linear_proj=RowParallelLinear,
                    q_layernorm=norm if qk_layernorm else IdentityOp,
                    k_layernorm=norm if qk_layernorm else IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=norm,
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear
                ),
            ),
            mlp_bda=get_bias_dropout_add,
        ),
    )
    return TransformerBlockSubmodules(layer_specs=[layer] * args.num_layers, layer_norm=norm)


def model_provider(pre_process: bool = True, post_process: bool = True) -> BlockDiffusionGPTModel:
    """Build the block-diffusion model with flex core attention.

    ``--transformer-impl transformer_engine`` (default) uses the MegaDLMs TE layer spec;
    ``--transformer-impl local`` uses ``local_block_spec`` (no Transformer Engine).

    Args:
        pre_process: Build the embedding (must be True; PP = 1).
        post_process: Build the output layer (must be True; PP = 1).

    Returns:
        The model.
    """
    args = get_args()
    validate_bd_args(args)
    real_vocab = get_tokenizer().vocab_size  # len(tokenizer) for the HF tokenizer
    if args.mask_token != real_vocab:
        raise ValueError(
            f"--mask-token {args.mask_token} must equal len(tokenizer) = {real_vocab} "
            "(dllm-bd3-spec section 12)"
        )
    if args.use_legacy_models:
        raise ValueError("legacy models are not supported")
    if args.spec is not None or args.num_experts:
        raise ValueError("--spec and MoE are not supported by the block-diffusion launcher")
    if args.multi_latent_attention:
        raise ValueError("multi-latent attention is not supported")

    print_rank_0("building block-diffusion GPT model ...")
    config = core_transformer_config_from_args(args)
    if args.transformer_impl == "transformer_engine":
        layer_spec = get_difflm_layer_with_transformer_engine_spec(
            num_experts=None,
            moe_grouped_gemm=False,
            qk_layernorm=args.qk_layernorm,
            multi_latent_attention=False,
            moe_use_legacy_grouped_gemm=False,
            attn_mask_type="no_mask",
            core_attn_implementation="FlexAttention",
        )
        layer_spec.submodules.self_attention.submodules.core_attention = BlockDiffusionFlexAttention
    elif args.transformer_impl == "local":
        if args.gradient_accumulation_fusion:
            raise ValueError("--transformer-impl local needs --no-gradient-accumulation-fusion")
        layer_spec = local_block_spec(args, args.qk_layernorm)
    else:
        raise ValueError(f"unsupported --transformer-impl {args.transformer_impl}")
    model = BlockDiffusionGPTModel(
        config=config,
        transformer_layer_spec=layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        rope_scaling=args.use_rope_scaling,
        eod_id=get_tokenizer().eod,
    )
    print_rank_0(str(model))
    return model


def loss_func(output_tensor: tuple) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Reduce the model outputs for Megatron.

    Args:
        output_tensor: ``(loss_sum, num_tokens, metrics)`` from ``BlockDiffusionGPTModel.forward``.

    Returns:
        ``(loss_sum, num_tokens, report)``. Megatron divides ``loss_sum`` by ``num_tokens``
        (or by the global token count with ``--calculate-per-token-loss``). ``report`` holds
        ``(numerator, denominator)`` pairs summed over the data-parallel group.
    """
    args = get_args()
    loss_sum, num_tokens, metrics = output_tensor
    if args.check_for_nan_in_loss_and_grad:
        get_rerun_state_machine().validate_result(
            result=loss_sum,
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,
            fatal=True,
        )
    keys = list(metrics)
    pairs = torch.stack(
        [
            torch.stack([metrics[k][0].detach().float(), metrics[k][1].detach().float()])
            for k in keys
        ]
    )
    torch.distributed.all_reduce(pairs, group=mpu.get_data_parallel_group())
    report = {k: (pairs[i, 0], pairs[i, 1]) for i, k in enumerate(keys)}
    local_num_tokens = num_tokens.detach().clamp(min=1).to(torch.int)
    return loss_sum, local_num_tokens, report


def forward_step(data_iterator, model: BlockDiffusionGPTModel):
    """Run one micro-batch.

    Args:
        data_iterator: Megatron data iterator.
        model: The (wrapped) model.

    Returns:
        The model output and the loss function.
    """
    timers = get_timers()
    timers("batch-generator", log_level=2).start()
    batch = get_batch_on_this_cp_rank(get_batch_on_this_tp_rank(data_iterator))
    timers("batch-generator").stop()
    output = model(
        batch["tokens"],
        batch["position_ids"],
        None,
        labels=batch["labels"],
        loss_mask=batch["loss_mask"],
    )
    return output, loss_func


def install_throughput_log() -> None:
    """Print data tokens/s/GPU (L tokens per sample) at every log interval
    (metric M4).

    Wraps ``megatron.training.training.training_log``. The value uses Megatron's own
    ``interval-time`` timer, read before the original function resets it.
    """
    import megatron.training.training as megatron_training

    original = megatron_training.training_log

    def training_log(
        loss_dict,
        total_loss_dict,
        learning_rate,
        decoupled_learning_rate,
        iteration,
        loss_scale,
        report_memory_flag,
        skipped_iter,
        grad_norm,
        params_norm,
        num_zeros_in_grad,
    ):
        args = get_args()
        if iteration % args.log_interval == 0:
            n_iters = (
                total_loss_dict.get("advanced iterations", 0)
                + total_loss_dict.get("skipped iterations", 0)
                + 1
            )
            elapsed = get_timers()("interval-time").elapsed(reset=False, barrier=False)
            if elapsed > 0:
                samples = get_num_microbatches() * args.micro_batch_size * args.data_parallel_size
                tokens_per_s_gpu = (
                    n_iters * samples * (args.seq_length - 1) / elapsed / args.world_size
                )
                print_rank_last(
                    f" bd throughput | iteration {iteration:8d} | "
                    f"tokens/s/GPU: {tokens_per_s_gpu:.1f} |"
                )
                writer = get_wandb_writer()
                if writer is not None:
                    writer.log({"bd tokens/s/GPU": tokens_per_s_gpu}, iteration)
        return original(
            loss_dict,
            total_loss_dict,
            learning_rate,
            decoupled_learning_rate,
            iteration,
            loss_scale,
            report_memory_flag,
            skipped_iter,
            grad_norm,
            params_norm,
            num_zeros_in_grad,
        )

    megatron_training.training_log = training_log


def main() -> None:
    """Entry point: Megatron ``pretrain`` with the chained argument provider."""
    train_valid_test_datasets_provider.is_distributed = True
    # Compiled flex_attention compiles once per mask shape (batch size, block size); keep the
    # Dynamo cache large enough that a block schedule or eval batches never fall back to eager.
    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
    install_throughput_log()
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={"tokenizer_type": "HuggingFaceTokenizer"},
        extra_args_provider=extra_args_provider,
    )


if __name__ == "__main__":
    main()
