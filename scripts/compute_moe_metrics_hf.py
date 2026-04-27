"""MoE router metrics for HuggingFace models (Mixtral, DBRX, OLMoE, etc.)

Adapts the Megatron-based metrics script for HuggingFace transformer MoE models.
Generates expert co-activation heatmaps and JSON metrics.

Single-dataset usage:
    python scripts/compute_moe_metrics_hf.py \
  --model-name mistralai/Mixtral-8x7B-v0.1 \
  --dataset-path /path/to/c4/en \
  --dataset-split validation \
  --dataset-percentage 1.0 \
  --seed 1234 \
  --num-batches 200 \
  --batch-size 1 \
  --max-length 2048 \
  --output-jsonl results/c4_saturation.jsonl \
  --coactivation-output results/c4_coactivation.jsonl \
  --expert-token-distribution-output results/c4_expert_token_dist.jsonl

Multi-process data-parallel (full model replica per rank; ``torchrun`` sets ``WORLD_SIZE``>1):
    torchrun --standalone --nproc_per_node=4 python scripts/compute_moe_metrics_hf.py \\
      --no-device-map --model-name Qwen/Qwen3-30B-A3B ... \\
      # Requires enough VRAM per GPU for the entire checkpoint; batches are strided across ranks.

Multi-dataset usage (one model load, N datasets, combined outputs):
    python scripts/compute_moe_metrics_hf.py \
  --model-name Qwen/Qwen3-30B-A3B \
  --batch-size 1 --max-length 2048 \
  --dataset-entry '{"label":"c4_en","dataset":"allenai/c4","dataset_config":"en","dataset_split":"validation","dataset_percentage":1.0,"num_batches":200}' \
  --dataset-entry '{"label":"wikitext_103","dataset":"wikitext","dataset_config":"wikitext-103-raw-v1","dataset_split":"test","dataset_percentage":100.0,"num_batches":200}' \
  --output-jsonl results/saturation.jsonl \
  --coactivation-output results/coactivation.jsonl \
  --expert-token-distribution-output results/expert_token_dist.jsonl

Outputs (jsonl, one record per dataset label):
    - saturation output: --output-jsonl
    - coactivation output: --coactivation-output
    - expert token distribution output: --expert-token-distribution-output
    - coactivation_<label>_<layer>.png heatmaps (when --plot is set)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from datetime import timedelta
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from tqdm.auto import tqdm

try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False


class RouterMetricsCollector:
    """Collects expert utilization metrics with all GPU→CPU transfers deferred to finalize().

    Calling .item() or .cpu() inside a forward hook synchronizes the CUDA stream at every
    MoE layer, stalling the GPU pipeline for every batch.  This class accumulates all
    statistics as GPU tensors during update() and only moves data to CPU in finalize(),
    which is called once after all batches have been processed.
    """

    def __init__(self) -> None:
        # Lazily initialized on first update() for each layer name.
        # All accumulator tensors remain on GPU until finalize().
        self._batch_counts: Dict[str, int] = {}
        self._saturation_sums: Dict[str, torch.Tensor] = {}   # 0-d float32 GPU tensor
        self._coact_sums: Dict[str, torch.Tensor] = {}        # [E, E] float32 GPU tensor
        self._single_sums: Dict[str, torch.Tensor] = {}       # [E]    float32 GPU tensor
        # CPU tensors populated by finalize() — accessed by downstream plot/JSON code.
        self.coactivation_sums: Dict[str, torch.Tensor] = {}
        self.single_counts: Dict[str, torch.Tensor] = {}

    def update(
        self,
        name: str,
        routing_indices: torch.Tensor,  # Shape: [batch*seq, topk]
        num_experts: int,
        topk: int,
    ) -> None:
        """Accumulate routing statistics entirely on GPU — no CPU/scalar sync."""
        device = routing_indices.device
        routing_indices = routing_indices.long()

        # bincount is cheaper than building a [tokens, E] bool map just to sum.
        single_counts = torch.bincount(routing_indices.reshape(-1), minlength=num_experts).float()
        # 0-d GPU tensor: no .item() call here.
        saturation_val = (single_counts > 0).float().mean()

        if topk > 1:
            first = routing_indices[:, :-1].reshape(-1)
            second = routing_indices[:, 1:].reshape(-1)
            pair_counts = torch.zeros((num_experts, num_experts), dtype=torch.float32, device=device)
            ones = torch.ones(first.shape[0], dtype=torch.float32, device=device)
            pair_counts.index_put_((first, second), ones, accumulate=True)
            pair_counts.index_put_((second, first), ones, accumulate=True)
        else:
            pair_counts = torch.zeros((num_experts, num_experts), dtype=torch.float32, device=device)

        if name not in self._batch_counts:
            self._batch_counts[name] = 0
            self._saturation_sums[name] = torch.zeros((), dtype=torch.float32, device=device)
            self._coact_sums[name] = torch.zeros((num_experts, num_experts), dtype=torch.float32, device=device)
            self._single_sums[name] = torch.zeros(num_experts, dtype=torch.float32, device=device)

        self._batch_counts[name] += 1
        self._saturation_sums[name].add_(saturation_val)
        self._coact_sums[name].add_(pair_counts)
        self._single_sums[name].add_(single_counts)

    def init_layer(self, name: str, num_experts: int, device: torch.device) -> None:
        """Pre-initialize per-layer accumulators so all ranks share identical merge keys."""
        if name in self._batch_counts:
            return
        self._batch_counts[name] = 0
        self._saturation_sums[name] = torch.zeros((), dtype=torch.float32, device=device)
        self._coact_sums[name] = torch.zeros((num_experts, num_experts), dtype=torch.float32, device=device)
        self._single_sums[name] = torch.zeros(num_experts, dtype=torch.float32, device=device)

    def finalize(self) -> dict:
        """Transfer GPU accumulators to CPU and compute per-layer averages.

        GPU→CPU transfers and scalar syncs are intentionally deferred to here so they
        happen once after the loop rather than inside each forward-hook call.
        """
        results = {}
        for name in self._batch_counts:
            batches = max(self._batch_counts[name], 1)

            # One .item() sync per layer — acceptable outside the hot loop.
            saturation = self._saturation_sums[name].item() / batches

            coact = self._coact_sums[name]
            sc = self._single_sums[name]
            normalized = coact / sc.clamp(min=1.0).unsqueeze(1)
            off_diagonal = ~torch.eye(coact.shape[0], dtype=torch.bool, device=coact.device)
            coactivation_rate = normalized[off_diagonal].mean().item()

            # Populate CPU tensors consumed by downstream plot/JSON output.
            self.coactivation_sums[name] = coact.cpu()
            self.single_counts[name] = sc.cpu()

            results[name] = {
                "saturation": saturation,
                "expert_coactivation": coactivation_rate,
            }
        return results

    def pack_for_distributed(self) -> Dict[str, Any]:
        """CPU snapshot of GPU accumulators for ``torch.distributed.all_gather_object``."""
        packed: Dict[str, Any] = {}
        for name in self._batch_counts:
            packed[name] = {
                "batches": int(self._batch_counts[name]),
                "sat_sum": self._saturation_sums[name].detach().float().cpu(),
                "coact": self._coact_sums[name].detach().float().cpu(),
                "single": self._single_sums[name].detach().float().cpu(),
            }
        return packed

    def import_merged_state(self, merged: Dict[str, Any]) -> None:
        """Replace accumulators from a global merge (CPU float tensors); then call ``finalize()``."""
        self._batch_counts.clear()
        self._saturation_sums.clear()
        self._coact_sums.clear()
        self._single_sums.clear()
        self.coactivation_sums.clear()
        self.single_counts.clear()
        for name, v in merged.items():
            self._batch_counts[name] = int(v["batches"])
            self._saturation_sums[name] = v["sat_sum"].clone()
            self._coact_sums[name] = v["coact"].clone()
            self._single_sums[name] = v["single"].clone()


def merge_router_collector_packed_states(
    packed_per_rank: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Sum per-rank MoE statistics into one packed dict (CPU tensors, integer batch counts)."""
    merged: Dict[str, Any] = {}
    for rank_payload in packed_per_rank:
        if not rank_payload:
            continue
        for name, v in rank_payload.items():
            if name not in merged:
                merged[name] = {
                    "batches": 0,
                    "sat_sum": v["sat_sum"].clone(),
                    "coact": v["coact"].clone(),
                    "single": v["single"].clone(),
                }
            else:
                m = merged[name]
                m["batches"] += int(v["batches"])
                m["sat_sum"] = m["sat_sum"] + v["sat_sum"]
                m["coact"] = m["coact"] + v["coact"]
                m["single"] = m["single"] + v["single"]
    return merged


def _maybe_init_distributed() -> tuple[bool, int, int, int]:
    """Return ``(use_ddp, rank, world_size, local_rank)`` for ``torchrun`` / SLURM multi-proc.

    Initializes ``torch.distributed`` when ``WORLD_SIZE > 1``. Uses ``nccl`` when CUDA is
    available, otherwise ``gloo`` (CPU-only smoke tests). Rank and world size are taken
    from the process environment (``torchrun`` sets ``MASTER_ADDR`` / ``MASTER_PORT`` / ``RANK``).
    """
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    if ws <= 1:
        return False, 0, 1, 0

    import torch.distributed as dist

    if dist.is_initialized():
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        return True, rank, dist.get_world_size(), local_rank

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend = "nccl"
    else:
        backend = "gloo"

    # Multi-node dataset I/O can skew rank progress; use a longer timeout than
    # PyTorch's default (10 min) to avoid false watchdog failures.
    dist.init_process_group(backend=backend, timeout=timedelta(minutes=60))
    return True, dist.get_rank(), dist.get_world_size(), local_rank


def _maybe_create_status_sync_group():
    """Create a CPU/Gloo group for tiny control collectives under NCCL runs."""
    import torch.distributed as dist

    if not dist.is_initialized():
        return None
    if dist.get_backend() != "nccl":
        return None
    try:
        return dist.new_group(backend="gloo", timeout=timedelta(minutes=60))
    except Exception as exc:
        if dist.get_rank() == 0:
            print(
                f"Warning: failed to create Gloo status-sync group ({type(exc).__name__}: {exc}); "
                "falling back to default NCCL group for status checks."
            )
        return None


def _distributed_merge_collectors(collector: RouterMetricsCollector) -> None:
    """In-place all-reduce merge of collector accumulators across ranks.

    This avoids ``all_gather_object`` of large Python payloads, which can allocate
    substantial temporary buffers and is fragile under tight post-inference memory.
    """
    import torch.distributed as dist

    if not dist.is_initialized():
        return

    layer_names = sorted(collector._coact_sums.keys())
    if not layer_names:
        return

    if collector._saturation_sums:
        device = next(iter(collector._saturation_sums.values())).device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Free cached blocks before collectives to reduce late-run OOM risk.
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()

    for name in layer_names:
        batch_count = torch.tensor(
            float(collector._batch_counts[name]), dtype=torch.float32, device=device
        )
        dist.all_reduce(batch_count, op=dist.ReduceOp.SUM)
        collector._batch_counts[name] = int(round(batch_count.item()))

        dist.all_reduce(collector._saturation_sums[name], op=dist.ReduceOp.SUM)
        dist.all_reduce(collector._coact_sums[name], op=dist.ReduceOp.SUM)
        dist.all_reduce(collector._single_sums[name], op=dist.ReduceOp.SUM)


def _build_hf_distributed_config(enable_expert_parallel: bool):
    """Return HF DistributedConfig (or None) for EP-capable inference loads."""
    if not enable_expert_parallel:
        return None
    try:
        from transformers.distributed.configuration_utils import DistributedConfig
    except Exception as exc:  # pragma: no cover - depends on transformers version
        raise RuntimeError(
            "--enable-expert-parallel requested, but this transformers build does not "
            "expose transformers.distributed.configuration_utils.DistributedConfig. "
            "Upgrade transformers to a version with HF expert parallelism support."
        ) from exc
    return DistributedConfig(enable_expert_parallel=True)


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    """Append one JSON object line and force-flush to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, sort_keys=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def _build_coactivation_layers(
    collector: RouterMetricsCollector,
    top_n: int,
) -> Dict[str, dict]:
    """Build per-layer coactivation JSON structure for one dataset."""
    layers_out: Dict[str, dict] = {}
    for layer_name, matrix in collector.coactivation_sums.items():
        matrix_normalized = _normalize_coactivation_matrix(
            matrix,
            collector.single_counts.get(layer_name),
        )
        top_indices = _select_top_experts_pairwise(matrix_normalized, top_n)
        submatrix = matrix_normalized[top_indices][:, top_indices]
        layers_out[layer_name] = {
            "experts": top_indices if isinstance(top_indices, list) else top_indices.tolist(),
            "matrix": (submatrix * 100.0).tolist(),
        }
    return layers_out


def _load_jsonl_dataset_labels(path: Path) -> Set[str]:
    """Return dataset labels present in a JSONL checkpoint file."""
    labels: Set[str] = set()
    if not path.exists():
        return labels
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                print(f"Warning: skipping invalid JSONL line {lineno} in {path}")
                continue
            if not isinstance(rec, dict):
                continue
            label = rec.get("label")
            if isinstance(label, str) and label:
                labels.add(label)
    return labels


# Sparse-MoE class names that follow the Mixtral-style interface:
#   attributes: gate (Linear), experts (ModuleList), top_k (int)
#   forward output: (hidden_states, router_logits)  where router_logits is [tokens, num_experts]
_SPARSE_MOE_CLASS_NAMES = {
    "MixtralSparseMoeBlock",  # Mixtral-8x7B / 8x22B
    "Qwen3MoeSparseMoeBlock", # Qwen3-MoE (e.g. 30B-A3B)
}


def register_sparse_moe_hooks(model, collector: RouterMetricsCollector):
    """Register forward hooks for any Mixtral-style sparse MoE block.

    Covers Mixtral, Qwen3-MoE, and any other architecture whose MoE block
    exposes ``gate`` (Linear) + ``experts`` (ModuleList) + ``top_k`` (int)
    and returns ``(hidden_states, router_logits)`` from its forward pass.
    """
    hooks = []
    
    for name, module in model.named_modules():
        # Hook only the actual sparse MoE block, not its child modules (for example .gate Linear).
        is_sparse_moe_block = module.__class__.__name__ in _SPARSE_MOE_CLASS_NAMES
        has_sparse_block_api = hasattr(module, "gate") and hasattr(module, "experts")
        if is_sparse_moe_block or has_sparse_block_api:
            layer_name = name
            try:
                num_experts = len(module.experts)
            except Exception:
                num_experts = None
            if num_experts is not None:
                gate_weight = getattr(getattr(module, "gate", None), "weight", None)
                if torch.is_tensor(gate_weight):
                    collector.init_layer(layer_name, int(num_experts), gate_weight.device)
            
            def hook_fn(module, inputs, outputs, layer_name=layer_name):
                # Defensive guard: skip modules that do not expose sparse-MoE block attributes.
                if not hasattr(module, "gate"):
                    return

                topk = module.top_k if hasattr(module, 'top_k') else 2

                # Prefer router logits returned by the block to avoid recomputing gate(hidden_states).
                router_logits = None
                if isinstance(outputs, tuple) and len(outputs) > 1 and torch.is_tensor(outputs[1]):
                    router_logits = outputs[1]

                hidden_states = inputs[0]
                if router_logits is None:
                    gate_output = module.gate(hidden_states)
                    if isinstance(gate_output, tuple):
                        router_logits = gate_output[0]
                    else:
                        router_logits = gate_output

                num_experts = router_logits.shape[-1]
                effective_topk = min(topk, num_experts)
                
                # Get top-k experts
                selected_experts = torch.topk(router_logits, effective_topk, dim=-1).indices
                selected_experts_flat = selected_experts.view(-1, effective_topk)
                
                collector.update(
                    layer_name,
                    selected_experts_flat,
                    num_experts,
                    effective_topk
                )
            
            hooks.append(module.register_forward_hook(hook_fn))
    
    return hooks


def register_dbrx_hooks(model, collector: RouterMetricsCollector):
    """Register forward hooks for DBRX MoE blocks."""
    hooks = []
    
    for name, module in model.named_modules():
        if "ffn" in name.lower() and hasattr(module, 'router'):
            layer_name = name
            num_experts = int(getattr(module, "moe_num_experts", 0))
            router_weight = getattr(getattr(module, "router", None), "weight", None)
            if num_experts > 0 and torch.is_tensor(router_weight):
                collector.init_layer(layer_name, num_experts, router_weight.device)
            
            def hook_fn(module, inputs, outputs, layer_name=layer_name):
                num_experts = module.moe_num_experts
                topk = module.moe_top_k
                
                hidden_states = inputs[0]
                router_logits = module.router(hidden_states)
                
                routing_weights, selected_experts = torch.topk(
                    router_logits, topk, dim=-1
                )
                
                batch_size, seq_len, _ = hidden_states.shape
                selected_experts_flat = selected_experts.view(-1, topk)
                
                collector.update(
                    layer_name,
                    selected_experts_flat,
                    num_experts,
                    topk
                )
            
            hooks.append(module.register_forward_hook(hook_fn))
    
    return hooks


def get_model_type(model) -> str:
    """Detect MoE model type from config."""
    model_type = model.config.model_type.lower()
    
    if "mixtral" in model_type:
        return "mixtral"
    elif "olmoe" in model_type:
        return "olmoe"
    elif "dbrx" in model_type:
        return "dbrx"
    elif "qwen3_moe" in model_type or "qwen3-moe" in model_type:
        return "qwen3_moe"
    else:
        return "unknown"


def _resolve_model_source(model_name_or_path: str, local_files_only: bool) -> str:
    """Resolve a model repo id to a local filesystem path when offline.

    Passing a repo id string to some tokenizer loaders can trigger metadata calls
    even when local-only intent is set. Returning a local snapshot path avoids
    those code paths entirely.
    """
    model_path = Path(model_name_or_path).expanduser()
    if model_path.exists():
        return str(model_path)

    if not local_files_only:
        return model_name_or_path

    try:
        from huggingface_hub import snapshot_download

        local_snapshot = snapshot_download(
            repo_id=model_name_or_path,
            local_files_only=True,
        )
        return local_snapshot
    except Exception as exc:
        raise RuntimeError(
            "Offline mode is enabled but no local cached snapshot was found for "
            f"'{model_name_or_path}'. Pre-download it on a networked node or pass a local path. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc


def load_dataset_texts(
    dataset_name: Optional[str],
    dataset_config: Optional[str],
    dataset_path: Optional[str],
    split: str,
    text_field: str,
    percentage: float,
    cache_dir: Optional[str],
    num_samples: int,
    seed: int,
    local_files_only: bool,
) -> List[str]:
    """Load texts from a HuggingFace dataset name or a local dataset path.

    Local paths support both datasets saved with `save_to_disk` and shard-style
    layouts containing `*.json` or `*.json.gz` files.
    """
    if percentage <= 0 or percentage > 100:
        raise ValueError("--dataset-percentage must be in the range (0, 100].")

    if dataset_path:
        dataset_path_obj = Path(dataset_path).expanduser()
        print(
            f"Loading local dataset from path={dataset_path_obj} split={split} percentage={percentage}%"
        )
        if not dataset_path_obj.exists():
            raise FileNotFoundError(f"Local dataset path not found: {dataset_path_obj}")

        try:
            dataset_obj = load_from_disk(str(dataset_path_obj))
        except Exception as exc:
            print(
                f"load_from_disk failed ({type(exc).__name__}); trying local shard files"
            )

            shard_dir = dataset_path_obj
            if dataset_config:
                config_subdir = shard_dir / dataset_config
                if config_subdir.is_dir():
                    shard_dir = config_subdir
                else:
                    print(
                        f"Configured subdir not found under dataset path: {config_subdir}; using {shard_dir}"
                    )

            split_aliases = [split]
            if split == "validation":
                split_aliases.append("valid")
            elif split == "valid":
                split_aliases.append("validation")

            shard_files: List[Path] = []
            for alias in split_aliases:
                shard_files.extend(sorted(shard_dir.glob(f"*{alias}*.json")))
                shard_files.extend(sorted(shard_dir.glob(f"*{alias}*.json.gz")))
                shard_files.extend(sorted(shard_dir.glob(f"*{alias}*.jsonl")))
                shard_files.extend(sorted(shard_dir.glob(f"*{alias}*.jsonl.gz")))

            if not shard_files:
                shard_files.extend(sorted(shard_dir.glob("*.json")))
                shard_files.extend(sorted(shard_dir.glob("*.json.gz")))
                shard_files.extend(sorted(shard_dir.glob("*.jsonl")))
                shard_files.extend(sorted(shard_dir.glob("*.jsonl.gz")))

            deduped_files: List[str] = []
            seen: Set[str] = set()
            for file_path in shard_files:
                key = str(file_path)
                if key not in seen:
                    seen.add(key)
                    deduped_files.append(key)

            if not deduped_files:
                raise FileNotFoundError(
                    f"No local json/json.gz shards found in {shard_dir}"
                )

            print(f"Loading {len(deduped_files)} local shard file(s) from {shard_dir}")
            dataset = load_dataset(
                "json",
                data_files=deduped_files,
                split="train",
                cache_dir=cache_dir,
            )
            print(f"Loaded local shard dataset from {shard_dir}")
        else:
            if isinstance(dataset_obj, DatasetDict):
                if split not in dataset_obj:
                    raise ValueError(
                        f"Split '{split}' not found in local dataset. Available: {list(dataset_obj.keys())}"
                    )
                dataset = dataset_obj[split]
            elif isinstance(dataset_obj, Dataset):
                dataset = dataset_obj
            else:
                raise TypeError(
                    f"Unsupported local dataset type: {type(dataset_obj).__name__}"
                )

            print(f"Loaded local dataset from {dataset_path_obj}")
    else:
        if not dataset_name:
            raise ValueError("Provide either --dataset or --dataset-path.")

        print(
            f"Loading dataset={dataset_name} config={dataset_config or 'default'} split={split} percentage={percentage}%"
        )
        load_kwargs = {
            "path": dataset_name,
            "split": split,
            "cache_dir": cache_dir,
        }
        if dataset_config:
            load_kwargs["name"] = dataset_config
        if local_files_only:
            # Reuse local cache only; do not attempt downloads/checks.
            load_kwargs["download_mode"] = "reuse_dataset_if_exists"
        dataset = load_dataset(**load_kwargs)
        print(f"Loaded {dataset_name} successfully")

    sample_fraction = percentage / 100.0
    sample_size = min(max(int(len(dataset) * sample_fraction), num_samples), len(dataset))
    print(f"Sampling {sample_size} entries ({percentage}%) from dataset with {len(dataset)} total entries")

    indices = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(seed)
    ).tolist()[:sample_size]
    sampled = dataset.select(indices)

    texts: List[str] = []
    for item in sampled:
        value = item.get(text_field)
        if not value:
            continue
        if isinstance(value, str):
            texts.append(value)
        elif isinstance(value, list):
            texts.extend(entry for entry in value if isinstance(entry, str))

    if not texts:
        raise ValueError(
            f"No text entries found using field '{text_field}' in the sampled dataset."
        )

    print(f"Loaded {len(texts)} text entries after sampling")
    return texts


def collect_router_logits_metrics(
    collector: RouterMetricsCollector,
    router_logits,
    topk: int,
    layer_prefix: str = "layers",
) -> None:
    """Collect metrics from per-layer router logits tensors.

    Expected per-layer logits shape is either [tokens, num_experts]
    or [batch, seq, num_experts].
    """
    if router_logits is None:
        return

    if isinstance(router_logits, torch.Tensor):
        per_layer_logits = [router_logits]
    else:
        per_layer_logits = list(router_logits)

    for layer_idx, logits in enumerate(per_layer_logits):
        if logits is None:
            continue

        logits = logits.float()
        if logits.ndim not in (2, 3):
            continue

        num_experts = logits.shape[-1]
        effective_topk = min(topk, num_experts)
        selected_experts = torch.topk(logits, effective_topk, dim=-1).indices
        selected_experts_flat = selected_experts.view(-1, effective_topk)

        collector.update(
            f"{layer_prefix}.{layer_idx}",
            selected_experts_flat,
            num_experts,
            effective_topk,
        )


def _normalize_coactivation_matrix(
    matrix: torch.Tensor,
    single_counts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Normalize co-activation counts to conditional rates."""
    matrix = matrix.clone().float()
    if single_counts is None:
        raise ValueError("single_counts must be provided for pairwise-adjacent co-activation")
    matrix_normalized = matrix / single_counts.clamp(min=1.0).unsqueeze(1)
    matrix_normalized = matrix_normalized - torch.diag(torch.diagonal(matrix_normalized))
    return matrix_normalized


def _select_top_experts_pairwise(matrix_normalized: torch.Tensor, top_n: int) -> List[int]:
    """Select top experts using the original OLMoE pair-priority strategy."""
    num_experts = matrix_normalized.shape[0]
    if num_experts <= top_n:
        return list(range(num_experts))

    selected = []
    flat = matrix_normalized.clone()
    flat.fill_diagonal_(0.0)
    values, indices = torch.sort(flat.reshape(-1), descending=True)

    for value, flat_idx in zip(values.tolist(), indices.tolist()):
        if value <= 0:
            break
        a = flat_idx // num_experts
        b = flat_idx % num_experts
        if a not in selected:
            selected.append(a)
        if b not in selected:
            selected.append(b)
        if len(selected) >= top_n:
            return selected[:top_n]

    max_per_expert = matrix_normalized.max(dim=1).values
    fill_order = torch.argsort(max_per_expert, descending=True).tolist()
    for expert in fill_order:
        if expert not in selected:
            selected.append(expert)
        if len(selected) >= top_n:
            break
    return selected[:top_n]


def build_expert_token_distribution_payload(
    single_counts: Dict[str, torch.Tensor],
    *,
    model_name: str,
    model_type: str,
    dataset: str,
    dataset_path: Optional[str],
    dataset_split: str,
    dataset_text_field: str,
    dataset_percentage: float,
    seed: int,
    num_batches: int,
) -> dict:
    """Build a JSON-serializable structure of per-layer expert selection distributions.

    ``single_counts`` are accumulated in ``RouterMetricsCollector`` as a bincount over
    all selected expert indices (shape flattened from ``[tokens, top_k]``), so each
    layer's ``counts[e]`` is how often expert ``e`` was chosen across the run, and
    ``probs`` is that vector normalized to sum to 1. This is comparable across datasets
    when using the same ``num_batches`` / sampling setup.
    """
    layers_out: Dict[str, dict] = {}
    for layer_name, counts in sorted(single_counts.items()):
        counts_f = counts.float()
        total = float(counts_f.sum().item())
        if total <= 0:
            probs_list: List[float] = [0.0] * int(counts_f.numel())
        else:
            probs_list = (counts_f / total).tolist()
        layers_out[layer_name] = {
            "num_experts": int(counts_f.numel()),
            "total_routed_selections": int(round(total)),
            "counts_per_expert": counts_f.long().tolist(),
            "prob_mass_per_expert": probs_list,
        }

    return {
        "schema": "expert_token_distribution_v1",
        "description": (
            "Per-layer histogram of routed expert indices. Counts are over all "
            "top-k selections (each token contributes up to top_k counts). "
            "prob_mass_per_expert is counts normalized within the layer."
        ),
        "model": model_name,
        "model_type": model_type,
        "dataset": dataset,
        "dataset_path": dataset_path,
        "dataset_split": dataset_split,
        "dataset_text_field": dataset_text_field,
        "dataset_percentage": dataset_percentage,
        "seed": seed,
        "num_batches": num_batches,
        "layers": layers_out,
    }


def plot_coactivation_heatmaps(
    coactivation_sums: dict,
    plot_dir: Path,
    top_n: int = 32,
    single_counts: Optional[dict] = None,
    filename_prefix: str = "",
) -> None:
    """Plot co-activation matrices as heatmaps for each layer.

    ``filename_prefix`` is prepended to the output PNG filename (use e.g. ``"c4_en_"`` in
    multi-dataset runs so plots for different datasets land in the same directory
    without colliding).
    """
    if not HAS_PLOTTING:
        print("Warning: matplotlib not available, skipping co-activation plots.")
        return
    
    for layer_name, matrix in coactivation_sums.items():
        layer_single_counts = None if single_counts is None else single_counts.get(layer_name)
        matrix_normalized = _normalize_coactivation_matrix(
            matrix,
            layer_single_counts,
        )
        
        top_indices = _select_top_experts_pairwise(matrix_normalized, top_n)
        submatrix = matrix_normalized[top_indices][:, top_indices]
        heatmap_values = submatrix.numpy() * 100.0
        
        plt.figure(figsize=(5, 4))
        plt.imshow(heatmap_values, cmap="RdPu", aspect="auto", vmin=0, vmax=60)
        plt.colorbar(ticks=[0, 15, 30, 45, 60])
        plt.title(f"Co-activation: {layer_name}")
        plt.xlabel("Expert")
        plt.ylabel("Expert")
        tick_positions = list(range(len(top_indices)))
        tick_labels = [str(idx) for idx in top_indices]
        plt.xticks(tick_positions, tick_labels, rotation=90)
        plt.yticks(tick_positions, tick_labels)
        plt.tight_layout()
        
        safe_name = layer_name.replace(".", "_").replace("/", "_")
        plot_path = plot_dir / f"coactivation_{filename_prefix}{safe_name}.png"
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        print(f"Saved co-activation heatmap to: {plot_path}")
        plt.close()


_LABEL_RE = re.compile(r"^[A-Za-z0-9._-]+$")

_DATASET_ENTRY_FIELDS = {
    "label",
    "dataset",
    "dataset_config",
    "dataset_path",
    "dataset_split",
    "dataset_text_field",
    "dataset_percentage",
    "num_batches",
    "seed",
}


@dataclass
class DatasetSpec:
    """One dataset to score for router metrics."""

    label: str
    dataset: Optional[str] = None
    dataset_config: Optional[str] = None
    dataset_path: Optional[str] = None
    dataset_split: str = "test"
    dataset_text_field: str = "text"
    dataset_percentage: float = 100.0
    num_batches: Optional[int] = None
    seed: int = 1234


def _build_dataset_spec(entry: dict, defaults: dict) -> DatasetSpec:
    merged = {**defaults, **entry}
    unknown = set(merged.keys()) - _DATASET_ENTRY_FIELDS
    if unknown:
        raise ValueError(
            f"Unknown --dataset-entry fields: {sorted(unknown)}. "
            f"Allowed: {sorted(_DATASET_ENTRY_FIELDS)}"
        )
    label = merged.get("label")
    if not isinstance(label, str) or not label:
        raise ValueError(f"--dataset-entry must include a non-empty 'label': {entry!r}")
    if not _LABEL_RE.match(label):
        raise ValueError(
            f"--dataset-entry label '{label}' must match [A-Za-z0-9._-]+ (filename-safe)"
        )
    if not merged.get("dataset") and not merged.get("dataset_path"):
        raise ValueError(
            f"--dataset-entry '{label}' must provide 'dataset' or 'dataset_path'"
        )
    fields = {k: merged[k] for k in _DATASET_ENTRY_FIELDS if k in merged}
    return DatasetSpec(**fields)


def _parse_dataset_specs(args: argparse.Namespace) -> List[DatasetSpec]:
    """Build the list of DatasetSpec from CLI args.

    Precedence:
        1. If one or more ``--dataset-entry`` flags are given: each is parsed as JSON
           and CLI-level ``--dataset-*`` flags become fallbacks for fields absent on
           the entry.
        2. Otherwise, fall back to the single-dataset CLI flags (legacy mode).
    """
    if args.dataset_entry:
        defaults: Dict[str, Any] = {
            "dataset_split": args.dataset_split,
            "dataset_text_field": args.dataset_text_field,
            "dataset_percentage": args.dataset_percentage,
            "seed": args.seed,
        }
        if args.num_batches is not None:
            defaults["num_batches"] = args.num_batches
        if args.dataset is not None:
            defaults["dataset"] = args.dataset
        if args.dataset_config is not None:
            defaults["dataset_config"] = args.dataset_config
        if args.dataset_path is not None:
            defaults["dataset_path"] = args.dataset_path

        specs: List[DatasetSpec] = []
        for raw in args.dataset_entry:
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"--dataset-entry is not valid JSON: {raw!r}") from exc
            if not isinstance(entry, dict):
                raise ValueError(
                    f"--dataset-entry must be a JSON object, got {type(entry).__name__}: {raw!r}"
                )
            specs.append(_build_dataset_spec(entry, defaults))

        labels = [s.label for s in specs]
        if len(set(labels)) != len(labels):
            dupes = sorted({l for l in labels if labels.count(l) > 1})
            raise ValueError(f"--dataset-entry labels must be unique; duplicates: {dupes}")
        return specs

    legacy: Dict[str, Any] = {
        "label": "default",
        "dataset": args.dataset or "wikitext",
        "dataset_config": args.dataset_config,
        "dataset_path": args.dataset_path,
        "dataset_split": args.dataset_split,
        "dataset_text_field": args.dataset_text_field,
        "dataset_percentage": args.dataset_percentage,
        "seed": args.seed,
    }
    if args.num_batches is not None:
        legacy["num_batches"] = args.num_batches
    return [_build_dataset_spec(legacy, defaults={})]


def _build_expert_distribution_layers(
    single_counts: Dict[str, torch.Tensor],
) -> Dict[str, dict]:
    """Return the per-layer counts / probabilities dict used inside expert-token-dist JSON."""
    layers_out: Dict[str, dict] = {}
    for layer_name, counts in sorted(single_counts.items()):
        counts_f = counts.float()
        total = float(counts_f.sum().item())
        if total <= 0:
            probs_list: List[float] = [0.0] * int(counts_f.numel())
        else:
            probs_list = (counts_f / total).tolist()
        layers_out[layer_name] = {
            "num_experts": int(counts_f.numel()),
            "total_routed_selections": int(round(total)),
            "counts_per_expert": counts_f.long().tolist(),
            "prob_mass_per_expert": probs_list,
        }
    return layers_out


def _infer_input_device(model) -> torch.device:
    """Device for token ids / attention_mask when using HF accelerate ``device_map``."""
    device = getattr(model, "device", None)
    if device is not None:
        return device if isinstance(device, torch.device) else torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_one_dataset(
    *,
    model,
    tokenizer,
    model_type: str,
    spec: DatasetSpec,
    batch_size: int,
    max_length: int,
    local_files_only: bool,
    dataset_cache_dir: Optional[str],
    pad_to_multiple_of: Optional[int],
    ddp_rank: int = 0,
    ddp_world_size: int = 1,
    shard_batches_across_ranks: bool = True,
    show_progress: bool = True,
) -> tuple[RouterMetricsCollector, dict]:
    """Run inference for one dataset spec on the already-loaded model.

    Returns the ``RouterMetricsCollector`` (not yet finalized; keep it so callers can
    still access ``coactivation_sums`` / ``single_counts`` after ``finalize``) and a
    metadata dict describing the dataset configuration actually used.
    """
    if spec.num_batches is not None and spec.num_batches > 0:
        planned_samples = spec.num_batches * batch_size
    else:
        planned_samples = 1

    texts = load_dataset_texts(
        dataset_name=spec.dataset,
        dataset_config=spec.dataset_config,
        dataset_path=spec.dataset_path,
        split=spec.dataset_split,
        text_field=spec.dataset_text_field,
        percentage=spec.dataset_percentage,
        cache_dir=dataset_cache_dir,
        num_samples=planned_samples,
        seed=spec.seed,
        local_files_only=local_files_only,
    )

    if spec.num_batches is not None and spec.num_batches > 0:
        effective_num_batches = spec.num_batches
    else:
        inferred = len(texts) // batch_size
        if inferred <= 0:
            raise ValueError(
                f"[{spec.label}] Not enough sampled texts ({len(texts)}) to form one batch "
                f"at batch_size={batch_size}. Increase dataset_percentage or reduce batch_size."
            )
        effective_num_batches = inferred
        print(
            f"[{spec.label}] Inferred num_batches={effective_num_batches} "
            f"from sampled texts={len(texts)}"
        )

    total_items = min(len(texts), effective_num_batches * batch_size)
    dropped_texts = len(texts) - total_items
    if ddp_rank == 0:
        print(
            f"[{spec.label}] Coverage: consumed_texts={total_items}/{len(texts)} "
            f"(batch_size={batch_size}, num_batches={effective_num_batches}), "
            f"dropped_texts={dropped_texts}"
        )

    collector = RouterMetricsCollector()
    if model_type in ("mixtral", "qwen3_moe"):
        hooks = register_sparse_moe_hooks(model, collector)
    elif model_type == "olmoe":
        hooks = []
    elif model_type == "dbrx":
        hooks = register_dbrx_hooks(model, collector)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    input_device = _infer_input_device(model)
    tok_kw: Dict[str, Any] = {
        "return_tensors": "pt",
        "padding": True,
        "truncation": True,
        "max_length": max_length,
    }
    if pad_to_multiple_of is not None and pad_to_multiple_of > 0:
        tok_kw["pad_to_multiple_of"] = pad_to_multiple_of

    try:
        batch_starts = list(range(0, total_items, batch_size))
        if shard_batches_across_ranks:
            local_batch_starts = [
                i
                for batch_idx, i in enumerate(batch_starts)
                if ddp_world_size <= 1 or (batch_idx % ddp_world_size == ddp_rank)
            ]
        else:
            # EP/TP collectives expect all ranks to execute identical forward schedules.
            local_batch_starts = batch_starts
        # Full-sequence forwards only: disable KV cache to cut memory traffic and work
        # tied to returning/storing past_key_values when not generating.
        with torch.inference_mode():
            for batch_idx, i in enumerate(
                tqdm(
                    local_batch_starts,
                    total=len(local_batch_starts),
                    desc=f"[{spec.label}]",
                    leave=False,
                    disable=not show_progress,
                )
            ):
                batch_texts = texts[i : i + batch_size]
                inputs = tokenizer(batch_texts, **tok_kw).to(
                    input_device, non_blocking=True
                )

                if inputs["input_ids"].shape[1] == 0:
                    continue

                if model_type == "olmoe":
                    olmoe_inputs = dict(inputs)
                    olmoe_inputs.pop("attention_mask", None)
                    outputs = model(
                        **olmoe_inputs,
                        output_router_logits=True,
                        use_cache=False,
                    )
                    router_logits = getattr(outputs, "router_logits", None)
                    if router_logits is None and isinstance(outputs, dict):
                        router_logits = outputs.get("router_logits")
                    topk = int(getattr(model.config, "num_experts_per_tok", 8))
                    collect_router_logits_metrics(
                        collector=collector,
                        router_logits=router_logits,
                        topk=topk,
                    )
                else:
                    _ = model(**inputs, use_cache=False)

                if show_progress and (batch_idx + 1) % 10 == 0:
                    print(f"[{spec.label}]   Processed {batch_idx + 1} local batches (rank {ddp_rank})")
    finally:
        for hook in hooks:
            hook.remove()

    metadata = {
        "dataset": spec.dataset,
        "dataset_config": spec.dataset_config,
        "dataset_path": spec.dataset_path,
        "dataset_split": spec.dataset_split,
        "dataset_text_field": spec.dataset_text_field,
        "dataset_percentage": spec.dataset_percentage,
        "seed": spec.seed,
        "num_batches": effective_num_batches,
        "num_texts_loaded": len(texts),
        "num_texts_consumed": total_items,
        "num_texts_dropped": dropped_texts,
    }
    return collector, metadata


def main():
    parser = argparse.ArgumentParser(description="Compute MoE metrics for HuggingFace models")
    parser.add_argument("--model-name", type=str, required=True, help="HuggingFace model name or path")

    # Legacy single-dataset flags. Also used as fallback defaults for --dataset-entry fields
    # that are not explicitly set on an entry.
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name (legacy single-dataset / per-entry default)")
    parser.add_argument("--dataset-path", type=str, default=None, help="Local dataset path (legacy single-dataset / per-entry default)")
    parser.add_argument("--dataset-config", type=str, default=None, help="Dataset config (legacy / per-entry default)")
    parser.add_argument("--dataset-split", type=str, default="test", help="Dataset split (per-entry default)")
    parser.add_argument("--dataset-text-field", type=str, default="text", help="Text field name (per-entry default)")
    parser.add_argument("--dataset-percentage", type=float, default=100.0, help="Percentage of dataset to sample in (0, 100] (per-entry default)")
    parser.add_argument("--dataset-cache-dir", type=str, default=None, help="HuggingFace dataset cache directory")
    parser.add_argument("--num-batches", type=int, default=None, help="Number of batches (per-entry default). If omitted, inferred per dataset from sampled text count.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for dataset sampling (per-entry default)")

    # Multi-dataset input (repeatable).
    parser.add_argument(
        "--dataset-entry",
        action="append",
        default=[],
        metavar="JSON",
        help=(
            "Repeatable JSON object describing one dataset. Required field: 'label' "
            "(unique, filename-safe). Optional: dataset, dataset_config, dataset_path, "
            "dataset_split, dataset_text_field, dataset_percentage, num_batches, seed. "
            "When one or more --dataset-entry flags are given, the legacy --dataset / "
            "--dataset-path flags only serve as per-entry defaults."
        ),
    )

    # Shared run-wide options.
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (shared across datasets)")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length (shared across datasets)")

    # Outputs (combined across datasets).
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default="saturation_results.jsonl",
        help="Per-dataset checkpoint output (jsonl). One JSON object per dataset.",
    )
    parser.add_argument(
        "--resume-from-jsonl",
        type=str,
        default=None,
        help=(
            "Path to a previous per-dataset jsonl checkpoint. Existing labels in that file "
            "are treated as already processed and skipped in this run."
        ),
    )
    parser.add_argument(
        "--coactivation-output",
        type=str,
        default="coactivation_results.jsonl",
        help=(
            "Per-dataset co-activation checkpoint (jsonl). "
            "One JSON object per dataset. Default: coactivation_results.jsonl"
        ),
    )
    parser.add_argument(
        "--expert-token-distribution-output",
        type=str,
        default="expert_token_distribution.jsonl",
        help=(
            "Per-dataset expert-token-distribution checkpoint (jsonl). "
            "One JSON object per dataset. Default: expert_token_distribution.jsonl"
        ),
    )
    parser.add_argument("--coactivation-top-n", type=int, default=16, help="Top N experts for coactivation")
    parser.add_argument("--plot", action="store_true", help="Generate per-dataset co-activation heatmaps")
    parser.add_argument("--plot-dir", type=str, default=None, help="Directory to save plots (defaults to output-jsonl directory)")

    # HF / device.
    parser.add_argument("--no-local-files-only", action="store_true", help="Allow downloading from HuggingFace (default: offline)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (ignored if device-map=auto)")
    parser.add_argument("--no-device-map", action="store_true", help="Disable accelerate device_map='auto'")
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="sdpa",
        metavar="NAME",
        help=(
            "Transformers attention backend (passed to from_pretrained). "
            "'sdpa' uses PyTorch scaled-dot-product attention (usually faster than eager on PT 2.x). "
            "Use 'eager' if load or forward fails on your stack. "
            "'flash_attention_2' requires flash-attn and compatible hardware."
        ),
    )
    parser.add_argument(
        "--pad-to-multiple-of",
        type=int,
        default=None,
        metavar="N",
        help=(
            "If set, pad sequences to a multiple of N (tokenizer pad_to_multiple_of). "
            "Can improve matmul efficiency but adds pad tokens that still participate in MoE routing, "
            "so router metrics may shift slightly vs unpadded runs."
        ),
    )
    parser.add_argument(
        "--ddp-disable",
        action="store_true",
        help=(
            "Ignore multi-process launch (WORLD_SIZE>1) and run as a single process. "
            "Use if WORLD_SIZE is set in the environment for unrelated reasons."
        ),
    )
    parser.add_argument(
        "--enable-expert-parallel",
        action="store_true",
        help=(
            "Enable Hugging Face expert parallelism via DistributedConfig. "
            "Requires torchrun (WORLD_SIZE>1)."
        ),
    )

    args = parser.parse_args()

    use_ddp, ddp_rank, ddp_world_size, ddp_local_rank = (False, 0, 1, 0)
    if not args.ddp_disable:
        use_ddp, ddp_rank, ddp_world_size, ddp_local_rank = _maybe_init_distributed()
    status_sync_group = _maybe_create_status_sync_group() if use_ddp else None
    if args.enable_expert_parallel and not use_ddp:
        raise ValueError(
            "--enable-expert-parallel requires multi-process launch (e.g. torchrun --nproc_per_node=N)."
        )

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.num_batches is not None and args.num_batches <= 0:
        raise ValueError("--num-batches must be > 0 when provided")

    dataset_specs = _parse_dataset_specs(args)

    output_jsonl_path = Path(args.output_jsonl)
    coactivation_jsonl_path = Path(args.coactivation_output)
    expert_dist_jsonl_path = Path(args.expert_token_distribution_output)
    resume_jsonl_path = Path(args.resume_from_jsonl) if args.resume_from_jsonl else None
    processed_labels: Set[str] = set()
    if resume_jsonl_path is not None:
        processed_labels = _load_jsonl_dataset_labels(resume_jsonl_path)
        if ddp_rank == 0:
            print(
                f"Resume checkpoint: {resume_jsonl_path} "
                f"(processed labels found: {len(processed_labels)})"
            )

    if processed_labels:
        dataset_specs_all = dataset_specs
        dataset_specs = [s for s in dataset_specs_all if s.label not in processed_labels]
        if ddp_rank == 0:
            skipped = [s.label for s in dataset_specs_all if s.label in processed_labels]
            print(
                f"Skipping {len(skipped)} already-processed dataset(s): {', '.join(skipped)}"
            )
            print(f"Remaining dataset(s) to process: {len(dataset_specs)}")
    if ddp_rank == 0:
        print(f"Configured {len(dataset_specs)} dataset(s):")
        for spec in dataset_specs:
            print(
                f"  - {spec.label}: dataset={spec.dataset} dataset_path={spec.dataset_path} "
                f"config={spec.dataset_config} split={spec.dataset_split} "
                f"text_field={spec.dataset_text_field} percentage={spec.dataset_percentage} "
                f"num_batches={spec.num_batches} seed={spec.seed}"
            )
        if use_ddp and args.enable_expert_parallel:
            print(
                f"Distributed expert-parallel inference: world_size={ddp_world_size}. "
                "All ranks execute the same batches; model handles distributed routing/collectives."
            )
        elif use_ddp:
            print(
                f"Distributed data-parallel inference: world_size={ddp_world_size} "
                f"(strided batch split; full replica per rank). Requires --no-device-map and enough "
                f"VRAM per GPU for the whole model."
            )

    if ddp_rank == 0:
        output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        coactivation_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        expert_dist_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        if resume_jsonl_path is not None:
            if not resume_jsonl_path.exists():
                raise FileNotFoundError(
                    f"--resume-from-jsonl path does not exist: {resume_jsonl_path}"
                )
            if resume_jsonl_path.resolve() != output_jsonl_path.resolve():
                shutil.copyfile(resume_jsonl_path, output_jsonl_path)
            else:
                # In-place resume: keep existing content and append missing datasets.
                pass
            resume_coact = resume_jsonl_path.with_name(coactivation_jsonl_path.name)
            resume_dist = resume_jsonl_path.with_name(expert_dist_jsonl_path.name)
            if resume_coact.exists():
                if resume_coact.resolve() != coactivation_jsonl_path.resolve():
                    shutil.copyfile(resume_coact, coactivation_jsonl_path)
            else:
                coactivation_jsonl_path.write_text("", encoding="utf-8")
            if resume_dist.exists():
                if resume_dist.resolve() != expert_dist_jsonl_path.resolve():
                    shutil.copyfile(resume_dist, expert_dist_jsonl_path)
            else:
                expert_dist_jsonl_path.write_text("", encoding="utf-8")
        else:
            output_jsonl_path.write_text("", encoding="utf-8")
            coactivation_jsonl_path.write_text("", encoding="utf-8")
            expert_dist_jsonl_path.write_text("", encoding="utf-8")
        print(f"Per-dataset incremental checkpoint: {output_jsonl_path}")
        print(f"Per-dataset coactivation checkpoint: {coactivation_jsonl_path}")
        print(f"Per-dataset expert-distribution checkpoint: {expert_dist_jsonl_path}")

    local_files_only = not args.no_local_files_only
    if local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        if ddp_rank == 0:
            print("Offline mode enabled: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1")

    model_source = _resolve_model_source(args.model_name, local_files_only)
    if model_source != args.model_name and ddp_rank == 0:
        print(f"Resolved cached local model path: {model_source}")

    if ddp_rank == 0:
        print(f"Loading model: {args.model_name}")

    # Prefer fast SDP kernels where the backend exposes them (CUDA/HIP builds often do).
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "sdp"):
        sdp = torch.backends.cuda.sdp
        for fn_name in ("enable_flash_sdp", "enable_mem_efficient_sdp", "enable_math_sdp"):
            fn = getattr(sdp, fn_name, None)
            if callable(fn):
                fn(True)

    def _model_load_kw(include_attn: bool) -> dict:
        kw: Dict[str, Any] = {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
            "local_files_only": local_files_only,
        }
        distributed_config = _build_hf_distributed_config(args.enable_expert_parallel)
        if distributed_config is not None:
            kw["distributed_config"] = distributed_config
        if include_attn and args.attn_implementation:
            kw["attn_implementation"] = args.attn_implementation
        elif not include_attn:
            # Second attempt: force eager so we do not silently re-use config.json default
            # (often still "sdpa") after a failed sdpa load.
            kw["attn_implementation"] = "eager"
        return kw

    use_device_map = not args.no_device_map
    if use_ddp:
        if use_device_map and ddp_rank == 0:
            print(
                "Warning: DDP mode forces single-GPU replica per rank; ignoring device_map='auto' "
                "(use --no-device-map explicitly to silence)."
            )
        use_device_map = False

    infer_device = args.device
    if use_ddp and torch.cuda.is_available():
        infer_device = f"cuda:{ddp_local_rank}"

    model: torch.nn.Module
    for include_attn in (True, False):
        try:
            if use_device_map:
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_source,
                        device_map="auto",
                        low_cpu_mem_usage=True,
                        **_model_load_kw(include_attn),
                    )
                    if ddp_rank == 0:
                        print("Model loaded with automatic device placement across available GPUs")
                    break
                except (ImportError, ValueError) as e:
                    if ddp_rank == 0:
                        print(
                            f"Warning: device_map='auto' failed ({e}). "
                            "Install accelerate or use --no-device-map."
                        )
                        print("Falling back to single device (may OOM for large models)...")
                    use_device_map = False

            model = AutoModelForCausalLM.from_pretrained(
                model_source,
                **_model_load_kw(include_attn),
            )
            model = model.to(infer_device)
            break
        except Exception as e:
            if include_attn:
                if ddp_rank == 0:
                    print(
                        f"Warning: model load failed with attn_implementation="
                        f"{args.attn_implementation!r} ({type(e).__name__}: {e}). Retrying without that kwarg."
                    )
            else:
                raise

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_type = get_model_type(model)
    if ddp_rank == 0:
        print(f"Detected model type: {model_type}")

    plot_dir: Optional[Path] = None
    if args.plot:
        if not HAS_PLOTTING:
            raise RuntimeError(
                "matplotlib and seaborn are required for --plot but are not available. "
                "Please install: pip install matplotlib seaborn"
            )
        plot_dir = Path(args.plot_dir) if args.plot_dir else Path(args.output_jsonl).parent
        plot_dir.mkdir(parents=True, exist_ok=True)

    # Process each dataset; accumulate finalized collectors for combined co-activation /
    # expert-distribution outputs. A failure on one dataset is recorded as status and does
    # not abort the others — preserving partial results is worth it after a large model load.
    per_dataset: Dict[str, dict] = {}
    collectors: Dict[str, RouterMetricsCollector] = {}
    for spec in dataset_specs:
        if ddp_rank == 0:
            print(f"\n=== Dataset '{spec.label}' ===")
        collector: Optional[RouterMetricsCollector] = None
        metadata: Dict[str, Any] = {}
        local_status = "ok"
        try:
            collector, metadata = run_one_dataset(
                model=model,
                tokenizer=tokenizer,
                model_type=model_type,
                spec=spec,
                batch_size=args.batch_size,
                max_length=args.max_length,
                local_files_only=local_files_only,
                dataset_cache_dir=args.dataset_cache_dir,
                pad_to_multiple_of=args.pad_to_multiple_of,
                ddp_rank=ddp_rank,
                ddp_world_size=ddp_world_size,
                shard_batches_across_ranks=(use_ddp and not args.enable_expert_parallel),
                show_progress=(ddp_rank == 0),
            )
        except Exception as exc:
            status = f"error: {type(exc).__name__}: {exc}"
            local_status = status
            if ddp_rank == 0:
                print(f"[{spec.label}] {status}")
            metadata = {
                "dataset": spec.dataset,
                "dataset_config": spec.dataset_config,
                "dataset_path": spec.dataset_path,
                "dataset_split": spec.dataset_split,
                "dataset_text_field": spec.dataset_text_field,
                "dataset_percentage": spec.dataset_percentage,
                "seed": spec.seed,
                "num_batches": spec.num_batches,
                "num_texts_loaded": 0,
            }

        if use_ddp:
            import torch.distributed as dist

            if status_sync_group is not None:
                fail_tensor = torch.tensor(
                    0 if local_status == "ok" else 1, dtype=torch.int32, device="cpu"
                )
                dist.all_reduce(fail_tensor, op=dist.ReduceOp.MAX, group=status_sync_group)
            else:
                status_device = (
                    torch.device(f"cuda:{ddp_local_rank}")
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                )
                if status_device.type == "cuda":
                    torch.cuda.synchronize(status_device)
                    torch.cuda.empty_cache()
                fail_tensor = torch.tensor(
                    0 if local_status == "ok" else 1, dtype=torch.int32, device=status_device
                )
                dist.all_reduce(fail_tensor, op=dist.ReduceOp.MAX)
            any_rank_failed = int(fail_tensor.item()) != 0
            if any_rank_failed:
                if ddp_rank == 0 and local_status == "ok":
                    print(
                        f"[{spec.label}] error: at least one rank failed before merge; "
                        "marking dataset as failed on all ranks to keep collectives aligned."
                    )
                final_status = local_status if local_status != "ok" else "error: peer rank failed"
                per_dataset[spec.label] = {
                    "status": final_status,
                    **metadata,
                    "layers": {},
                }
                if ddp_rank == 0:
                    _append_jsonl(
                        output_jsonl_path,
                        {
                            "schema": "moe_metrics_dataset_result_v1",
                            "model": args.model_name,
                            "model_type": model_type,
                            "label": spec.label,
                            "status": final_status,
                            **metadata,
                            "layers": {},
                        },
                    )
                    _append_jsonl(
                        coactivation_jsonl_path,
                        {
                            "schema": "moe_coactivation_dataset_result_v1",
                            "model": args.model_name,
                            "model_type": model_type,
                            "label": spec.label,
                            "status": final_status,
                            "coactivation_top_n": args.coactivation_top_n,
                            **metadata,
                            "layers": {},
                        },
                    )
                    _append_jsonl(
                        expert_dist_jsonl_path,
                        {
                            "schema": "expert_token_distribution_dataset_result_v1",
                            "model": args.model_name,
                            "model_type": model_type,
                            "label": spec.label,
                            "status": final_status,
                            **metadata,
                            "layers": {},
                        },
                    )
                continue

        if collector is None:
            per_dataset[spec.label] = {
                "status": local_status,
                **metadata,
                "layers": {},
            }
            if ddp_rank == 0:
                _append_jsonl(
                    output_jsonl_path,
                    {
                        "schema": "moe_metrics_dataset_result_v1",
                        "model": args.model_name,
                        "model_type": model_type,
                        "label": spec.label,
                        "status": local_status,
                        **metadata,
                        "layers": {},
                    },
                )
                _append_jsonl(
                    coactivation_jsonl_path,
                    {
                        "schema": "moe_coactivation_dataset_result_v1",
                        "model": args.model_name,
                        "model_type": model_type,
                        "label": spec.label,
                        "status": local_status,
                        "coactivation_top_n": args.coactivation_top_n,
                        **metadata,
                        "layers": {},
                    },
                )
                _append_jsonl(
                    expert_dist_jsonl_path,
                    {
                        "schema": "expert_token_distribution_dataset_result_v1",
                        "model": args.model_name,
                        "model_type": model_type,
                        "label": spec.label,
                        "status": local_status,
                        **metadata,
                        "layers": {},
                    },
                )
            continue

        if use_ddp and not args.enable_expert_parallel:
            _distributed_merge_collectors(collector)
            metadata = {
                **metadata,
                "ddp_world_size": ddp_world_size,
                "ddp_batch_striding": f"batch_idx % {ddp_world_size} == rank",
            }
        elif use_ddp and args.enable_expert_parallel:
            metadata = {
                **metadata,
                "distributed_world_size": ddp_world_size,
                "distributed_mode": "expert_parallel",
            }

        results = collector.finalize()
        per_dataset[spec.label] = {
            "status": "ok",
            **metadata,
            "layers": results,
        }
        if ddp_rank == 0:
            _append_jsonl(
                output_jsonl_path,
                {
                    "schema": "moe_metrics_dataset_result_v1",
                    "model": args.model_name,
                    "model_type": model_type,
                    "label": spec.label,
                    "status": "ok",
                    **metadata,
                    "layers": results,
                },
            )
            coact_layers = _build_coactivation_layers(collector, args.coactivation_top_n)
            _append_jsonl(
                coactivation_jsonl_path,
                {
                    "schema": "moe_coactivation_dataset_result_v1",
                    "model": args.model_name,
                    "model_type": model_type,
                    "label": spec.label,
                    "status": "ok",
                    "coactivation_top_n": args.coactivation_top_n,
                    **metadata,
                    "layers": coact_layers,
                },
            )
            _append_jsonl(
                expert_dist_jsonl_path,
                {
                    "schema": "expert_token_distribution_dataset_result_v1",
                    "model": args.model_name,
                    "model_type": model_type,
                    "label": spec.label,
                    "status": "ok",
                    **metadata,
                    "layers": _build_expert_distribution_layers(collector.single_counts),
                },
            )
        collectors[spec.label] = collector

        if plot_dir is not None and collector.coactivation_sums and ddp_rank == 0:
            print(f"[{spec.label}] Generating co-activation plots in {plot_dir}...")
            plot_coactivation_heatmaps(
                collector.coactivation_sums,
                plot_dir,
                args.coactivation_top_n,
                collector.single_counts,
                filename_prefix=f"{spec.label}_",
            )

    if ddp_rank == 0:
        ok_count = sum(1 for e in per_dataset.values() if e.get("status") == "ok")
        print(
            f"\nCompleted datasets: ok={ok_count}/{len(per_dataset)} | "
            f"saturation={output_jsonl_path} | "
            f"coactivation={coactivation_jsonl_path} | "
            f"expert_dist={expert_dist_jsonl_path}"
        )

    if use_ddp:
        import torch.distributed as dist

        if status_sync_group is not None:
            try:
                dist.destroy_process_group(status_sync_group)
            except Exception:
                pass

        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
