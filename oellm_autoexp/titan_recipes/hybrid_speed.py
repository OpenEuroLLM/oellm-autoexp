"""TorchTitan (upstream main) twin of the hybrid speed suite's dense 130M reference.

The Megatron reference (config/experiments/korbi/hybrid_speed/speed_base_mn5.yaml, arm
``dense``): hidden 768, 12 blocks of 12 heads of 64, SwiGLU 2048, q/k RMSNorm, RoPE, tied
embeddings, vocabulary 50304, sequence 4096, 4 GPUs x micro batch 8. Here the same model as
a Qwen3 config, run through upstream torchtitan with its own speed features (CUDA graphs,
locally compiled regions, FSDP2), reading the same pre-tokenized Megatron data.

    torchrun --nproc-per-node 4 -m torchtitan.train \\
        --module oellm_autoexp.titan_recipes.hybrid_speed --config dense130m

Recipes (functions returning a ``Trainer.Config``): ``dense130m`` and its variants below;
``dense130m_local`` is a 1-GPU smoke test on any GPU. Environment: TITAN_DATA_PREFIX
(Megatron ``.bin`` / ``.idx`` prefix), TITAN_TOKENIZER (HF tokenizer directory), TITAN_STEPS.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from typing import Any

import numpy as np

from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.data.dataset import SampleProcessor, SingleDatasetConfig, TextSequence
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import AdamW, LRSchedulersContainer, Optim, OptimizersContainer
from torchtitan.config import Configurable, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common import CosSinRoPE, Embedding, Linear
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.common.decoder_sharding import (
    lm_head_sharding_config,
    pre_lm_head_norm_sharding_config,
    token_embedding_sharding_config,
)
from torchtitan.models.qwen3.flavors import (
    _build_qwen3_layers,
    _EMBEDDING_SKIP_INIT,
    _LINEAR_INIT,
    _qwen3_norm,
)
from torchtitan.models.qwen3.model import Qwen3Model
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.trainer import Trainer

FINEWEB = "/gpfs/projects/ehpc533/datasets/FineWeb-gpt-neox/fineweb-10BT_text_document"
NEOX = "/gpfs/projects/ehpc533/datasets/gpt-neox-20b-tokenizer"
SEQ = 4096
# attention backend of the variants: varlen needs flash_attn_interface (FA3) on Hopper; flex,
# flex_flash (FlexAttention's flash backend, Hopper) need nothing extra
ATTN = os.environ.get("TITAN_ATTN", "varlen")

# Megatron's indexed-dataset dtype codes (megatron/core/datasets/indexed_dataset.py)
_DTYPES = {1: np.uint8, 2: np.int8, 3: np.int16, 4: np.int32, 5: np.int64, 6: np.float64,
           7: np.float32, 8: np.uint16}


def _megatron_tokens(prefix: str) -> np.ndarray:
    """The whole token stream of a Megatron ``.bin`` (documents back to back), memory-mapped."""
    with open(prefix + ".idx", "rb") as f:
        assert f.read(9) == b"MMIDIDX\x00\x00", f"{prefix}.idx: not a Megatron index"
        struct.unpack("<Q", f.read(8))  # version
        (code,) = struct.unpack("<B", f.read(1))
    return np.memmap(prefix + ".bin", dtype=_DTYPES[code], mode="r")


class MegatronWindowSource(Configurable):
    """Random access to contiguous ``seq_len + 1`` token windows of a pre-tokenized Megatron
    dataset (one window = one training sequence: plain causal attention, as the Megatron
    reference without a document mask)."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        prefix: str = FINEWEB
        seq_len: int = SEQ

    def __init__(self, config: Config) -> None:
        self._prefix, self._n = config.prefix, config.seq_len + 1
        self._tokens = None

    def _data(self) -> np.ndarray:
        if self._tokens is None:  # opened lazily (per dataloader worker)
            self._tokens = _megatron_tokens(self._prefix)
        return self._tokens

    def __len__(self) -> int:
        return len(self._data()) // self._n

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {"tokens": np.asarray(self._data()[index * self._n:(index + 1) * self._n],
                                     dtype=np.int64)}

    def __getstate__(self):
        return {"_prefix": self._prefix, "_n": self._n, "_tokens": None}


class WindowProcessor(SampleProcessor):
    """A token window -> next-token input / label pair (no tokenizer involved)."""

    @dataclass(kw_only=True, slots=True)
    class Config(SampleProcessor.Config):
        pass

    def __init__(self, config: Config, **kwargs) -> None:
        pass

    def __call__(self, sample: dict[str, Any], rng: np.random.Generator) -> TextSequence:
        del rng
        t = sample["tokens"]
        return TextSequence(input_ids=t[:-1], labels=t[1:])


def _qwen3_130m(attn_backend: str, parallelism: ParallelismConfig,
                vocab_size: int = 50304) -> Qwen3Model.Config:
    """The Megatron reference model as a Qwen3 config (no biases, qk-norm, tied embeddings)."""
    dim, head_dim, n_layers = 768, 64, 12
    sp = parallelism.enable_sequence_parallel
    return Qwen3Model.Config(
        max_context_length=SEQ,
        vocab_size=vocab_size,
        dim=dim,
        norm=_qwen3_norm(dim, sharding_config=pre_lm_head_norm_sharding_config(enable_sp=sp)),
        enable_weight_tying=True,
        tok_embeddings=Embedding.Config(num_embeddings=vocab_size, embedding_dim=dim,
                                        param_init=_EMBEDDING_SKIP_INIT,
                                        sharding_config=token_embedding_sharding_config(enable_sp=sp)),
        lm_head=Linear.Config(in_features=dim, out_features=vocab_size, param_init=_LINEAR_INIT,
                              sharding_config=lm_head_sharding_config()),
        layers=_build_qwen3_layers(
            n_layers=n_layers, dim=dim, n_heads=12, n_kv_heads=12, head_dim=head_dim,
            hidden_dim=2048, attn_backend=attn_backend,
            rope=CosSinRoPE.Config(dim=head_dim, max_context_length=SEQ, theta=10000.0),
            enable_sp=sp,
        ),
        local_compile_regions=["cos_sin_rope", "fused_binary_activation", "loss"],
    )


def dense130m(attn_backend: str | None = None, micro_batch: int = 8) -> Trainer.Config:
    """FSDP over the node, CUDA graphs, compiled RoPE / SwiGLU / loss regions."""
    parallelism = ParallelismConfig(data_parallel_shard_degree=-1)
    model = _qwen3_130m(attn_backend or ATTN, parallelism)
    tokens = micro_batch * SEQ
    dataset = SingleDatasetConfig(
        source=MegatronWindowSource.Config(prefix=os.environ.get("TITAN_DATA_PREFIX", FINEWEB)),
        processor=WindowProcessor.Config(),
    )
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(global_vocab_size=decoder_vocab_size(model)),
        ),
        hf_assets_path=os.environ.get("TITAN_TOKENIZER", NEOX),
        metrics=MetricsProcessor.Config(log_freq=5),
        model=model,
        dataloader=GrainDataLoader.Config(
            max_num_documents=micro_batch,
            dataset=ConcatThenSplitPackingConfig(dataset=dataset),
        ),
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(optimizers=[AdamW.Config(pattern=r".*", lr=1e-3)]),
            lr_scheduler=LRSchedulersContainer.Config(warmup_steps=5),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=tokens,
            max_context_length=SEQ,
            steps=int(os.environ.get("TITAN_STEPS", "100")),
        ),
        parallelism=parallelism,
        checkpointer=None,
        activation_remat=None,
    )


def dense130m_nocg() -> Trainer.Config:
    config = dense130m(attn_backend=ATTN)
    config.training.disable_cuda_graphs = True
    return config


def dense130m_nocompile() -> Trainer.Config:
    config = dense130m(attn_backend=ATTN)
    config.model.local_compile_regions = []
    return config


def dense130m_ddp() -> Trainer.Config:
    config = dense130m(attn_backend=ATTN)
    config.parallelism.data_parallel_shard_degree = 1
    # replicate degrees must be explicit: the world size
    config.parallelism.data_parallel_replicate_degree = int(os.environ.get("WORLD_SIZE", "4"))
    return config


def dense130m_noreshard() -> Trainer.Config:
    config = dense130m(attn_backend=ATTN)
    config.parallelism.fsdp_reshard_after_forward = "never"
    return config


def dense130m_mbs16() -> Trainer.Config:
    return dense130m(attn_backend=ATTN, micro_batch=16)


def dense130m_flex() -> Trainer.Config:
    return dense130m(attn_backend="flex")


def dense130m_flex_flash() -> Trainer.Config:
    return dense130m(attn_backend="flex_flash")


def dense130m_local() -> Trainer.Config:
    """One-GPU smoke test (any GPU: micro batch 2, 10 steps)."""
    config = dense130m(micro_batch=2)
    config.training.steps = int(os.environ.get("TITAN_STEPS", "10"))
    config.metrics.log_freq = 1
    return config
