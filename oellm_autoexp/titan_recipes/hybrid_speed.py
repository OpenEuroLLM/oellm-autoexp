"""TorchTitan (0.3.0, upstream) twin of the hybrid speed suite's dense 130M reference.

The Megatron reference (config/experiments/korbi/hybrid_speed/speed_base_mn5.yaml, arm
``dense``): hidden 768, 12 blocks of 12 heads of 64, SwiGLU 2048, q/k RMSNorm, RoPE, tied
embeddings, vocabulary 50304, sequence 4096, 4 GPUs x micro batch 8. Here the same model as
a Qwen3 config, run through upstream torchtitan with its own speed features (per-block
torch.compile, CUDA graphs, FSDP2), reading the same pre-tokenized Megatron data.

    torchrun --nproc-per-node 4 -m torchtitan.train \\
        --module oellm_autoexp.titan_recipes.hybrid_speed --config dense130m

Recipes (functions returning a ``Trainer.Config``): ``dense130m`` (FSDP, compile, CUDA
graphs, flex_flash attention) and its variants below; ``dense130m_local`` is a 1-GPU smoke
test on any GPU. Environment: TITAN_DATA_PREFIX (Megatron ``.bin`` / ``.idx`` prefix),
TITAN_TOKENIZER (HF tokenizer directory), TITAN_STEPS.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import IterableDataset

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw, LRSchedulersContainer
from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.models.common import CosSinRoPE, Embedding, Linear
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.qwen3 import (
    _build_qwen3_layers,
    _EMBEDDING_SKIP_INIT,
    _LINEAR_INIT,
    _qwen3_norm,
    parallelize_qwen3,
)
from torchtitan.models.qwen3.model import Qwen3Model
from torchtitan.models.qwen3.state_dict_adapter import Qwen3StateDictAdapter
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.trainer import Trainer

FINEWEB = "/gpfs/projects/ehpc533/datasets/FineWeb-gpt-neox/fineweb-10BT_text_document"
NEOX = "/gpfs/projects/ehpc533/datasets/gpt-neox-20b-tokenizer"

# Megatron's indexed-dataset dtype codes (megatron/core/datasets/indexed_dataset.py)
_DTYPES = {1: np.uint8, 2: np.int8, 3: np.int16, 4: np.int32, 5: np.int64, 6: np.float64,
           7: np.float32, 8: np.uint16}


def _megatron_tokens(prefix: str) -> np.ndarray:
    """The whole token stream of a Megatron ``.bin`` (documents back to back), memory-mapped."""
    with open(prefix + ".idx", "rb") as f:
        header = f.read(9)
        assert header == b"MMIDIDX\x00\x00", f"{prefix}.idx: not a Megatron index"
        (version,) = struct.unpack("<Q", f.read(8))
        (code,) = struct.unpack("<B", f.read(1))
    return np.memmap(prefix + ".bin", dtype=_DTYPES[code], mode="r")


class MegatronTokenStream(IterableDataset):
    """Contiguous ``seq_len + 1`` windows of a pre-tokenized Megatron dataset, one disjoint
    slice of the stream per data-parallel rank. Positions run 0..seq_len-1 (one "document"
    per sample: plain causal attention, as the Megatron reference without a document mask)."""

    def __init__(self, prefix: str, seq_len: int, dp_rank: int, dp_world_size: int):
        self.prefix, self.seq_len = prefix, seq_len
        self.dp_rank, self.dp_world_size = dp_rank, dp_world_size
        self._sample = 0

    def __iter__(self):
        tokens = _megatron_tokens(self.prefix)
        n = self.seq_len + 1
        per_rank = len(tokens) // self.dp_world_size // n
        base = self.dp_rank * per_rank * n
        positions = torch.arange(self.seq_len)
        while True:
            i = self._sample % per_rank
            window = torch.from_numpy(tokens[base + i * n: base + (i + 1) * n].astype(np.int64))
            self._sample += 1
            yield {"input": window[:-1], "positions": positions}, window[1:]

    def state_dict(self):
        return {"sample": self._sample}

    def load_state_dict(self, state_dict):
        self._sample = state_dict["sample"]


class MegatronTokenDataLoader(ParallelAwareDataloader):
    """Dataloader over :class:`MegatronTokenStream`."""

    @dataclass(kw_only=True, slots=True)
    class Config(ParallelAwareDataloader.Config):
        data_prefix: str = FINEWEB

    def __init__(self, config: Config, *, dp_world_size: int, dp_rank: int, seq_len: int,
                 local_batch_size: int, snapshot_every_n_steps: int | None = 1, **kwargs):
        super().__init__(
            MegatronTokenStream(config.data_prefix, seq_len, dp_rank, dp_world_size),
            dp_rank=dp_rank, dp_world_size=dp_world_size, num_workers=config.num_workers,
            persistent_workers=config.persistent_workers, pin_memory=config.pin_memory,
            prefetch_factor=config.prefetch_factor, snapshot_every_n_steps=snapshot_every_n_steps,
            batch_size=local_batch_size,
        )


def _qwen3_130m(attn_backend: str, vocab_size: int = 50304) -> Qwen3Model.Config:
    """The Megatron reference model as a Qwen3 config (no biases, qk-norm, tied embeddings)."""
    dim, head_dim, n_layers = 768, 64, 12
    return Qwen3Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        norm=_qwen3_norm(dim),
        enable_weight_tying=True,
        tok_embeddings=Embedding.Config(num_embeddings=vocab_size, embedding_dim=dim,
                                        param_init=_EMBEDDING_SKIP_INIT),
        lm_head=Linear.Config(in_features=dim, out_features=vocab_size, param_init=_LINEAR_INIT),
        layers=_build_qwen3_layers(
            fuse_qkv=True, n_layers=n_layers, dim=dim, n_heads=12, n_kv_heads=12,
            head_dim=head_dim, hidden_dim=2048, attn_backend=attn_backend,
            rope=CosSinRoPE.Config(dim=head_dim, max_seq_len=4096, theta=10000.0),
        ),
    )


def _model_spec(attn_backend: str) -> ModelSpec:
    return ModelSpec(name="qwen3", flavor="130M", model=_qwen3_130m(attn_backend),
                     parallelize_fn=parallelize_qwen3, pipelining_fn=None,
                     post_optimizer_build_fn=None, state_dict_adapter=Qwen3StateDictAdapter)


def dense130m(attn_backend: str = "flex_flash", local_batch_size: int = 8) -> Trainer.Config:
    """FSDP over the node, per-block compile, CUDA graphs (titan's defaults for speed)."""
    model_spec = _model_spec(attn_backend)
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(global_vocab_size=decoder_vocab_size(model_spec)),
        ),
        hf_assets_path=os.environ.get("TITAN_TOKENIZER", NEOX),
        metrics=MetricsProcessor.Config(log_freq=5),
        model_spec=model_spec,
        dataloader=MegatronTokenDataLoader.Config(
            data_prefix=os.environ.get("TITAN_DATA_PREFIX", FINEWEB), num_workers=2,
            pin_memory=True,
        ),
        optimizer=default_adamw(lr=1e-3),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=5),
        training=TrainingConfig(
            local_batch_size=local_batch_size, seq_len=4096,
            steps=int(os.environ.get("TITAN_STEPS", "100")), max_norm=1.0,
        ),
        parallelism=ParallelismConfig(data_parallel_shard_degree=-1),
        compile=CompileConfig(enable=True, components=["model", "loss"]),
        checkpoint=CheckpointManager.Config(enable=False),
    )


def dense130m_nocg() -> Trainer.Config:
    config = dense130m()
    config.training.disable_cuda_graphs = True
    return config


def dense130m_eager() -> Trainer.Config:
    config = dense130m()
    config.compile.enable = False
    return config


def dense130m_ddp() -> Trainer.Config:
    config = dense130m()
    config.parallelism.data_parallel_shard_degree = 1
    config.parallelism.data_parallel_replicate_degree = -1
    return config


def dense130m_noreshard() -> Trainer.Config:
    config = dense130m()
    config.parallelism.fsdp_reshard_after_forward = "never"
    return config


def dense130m_lbs16() -> Trainer.Config:
    return dense130m(local_batch_size=16)


def dense130m_flex() -> Trainer.Config:
    return dense130m(attn_backend="flex")


def dense130m_varlen() -> Trainer.Config:
    return dense130m(attn_backend="varlen")


def dense130m_local() -> Trainer.Config:
    """One-GPU smoke test (any GPU: flex attention, 10 steps, batch 2)."""
    config = dense130m(attn_backend="flex", local_batch_size=2)
    config.training.steps = int(os.environ.get("TITAN_STEPS", "10"))
    config.metrics.log_freq = 1
    return config
