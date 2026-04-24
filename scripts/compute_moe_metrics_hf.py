"""MoE router metrics for HuggingFace models (Mixtral, DBRX, OLMoE, etc.)

Adapts the Megatron-based metrics script for HuggingFace transformer MoE models.
Generates expert co-activation heatmaps and JSON metrics.

Usage:
    python scripts/compute_moe_metrics_hf.py \
  --model-name mistralai/Mixtral-8x7B-v0.1 \
  --dataset-path /path/to/c4/en \
  --dataset-split validation \
  --dataset-percentage 1.0 \
  --seed 1234 \
  --num-batches 200 \
  --batch-size 1 \
  --max-length 2048 \
  --output-json results/c4_saturation.json \
  --coactivation-output results/c4_coactivation.json \
  --expert-token-distribution-output results/c4_expert_token_dist.json

Outputs:
    - saturation_results.json: Router saturation metrics
    - coactivation_results.json: Co-activation matrices for top-N experts
    - coactivation_*.png: Co-activation heatmaps for each layer
    - expert_token_distribution.json (optional): Per-layer counts and probabilities
      over all routed slots (flattened top-k selections per token), for cross-dataset
      comparison of router load.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

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

            if not shard_files:
                shard_files.extend(sorted(shard_dir.glob("*.json")))
                shard_files.extend(sorted(shard_dir.glob("*.json.gz")))

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
) -> None:
    """Plot co-activation matrices as heatmaps for each layer."""
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
        # Show the actual selected expert IDs (not 0..N-1 matrix indices).
        tick_positions = list(range(len(top_indices)))
        tick_labels = [str(idx) for idx in top_indices]
        plt.xticks(tick_positions, tick_labels, rotation=90)
        plt.yticks(tick_positions, tick_labels)
        plt.tight_layout()
        
        # Sanitize layer name for filename
        safe_name = layer_name.replace(".", "_").replace("/", "_")
        plot_path = plot_dir / f"coactivation_{safe_name}.png"
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        print(f"Saved co-activation heatmap to: {plot_path}")
        plt.close()


def main():
    raw_argv = sys.argv[1:]
    parser = argparse.ArgumentParser(description="Compute MoE metrics for HuggingFace models")
    parser.add_argument("--model-name", type=str, required=True, help="HuggingFace model name or path")
    parser.add_argument("--dataset", type=str, default="wikitext", help="Dataset name")
    parser.add_argument("--dataset-path", type=str, default=None, help="Local dataset path. Supports save_to_disk datasets and raw json/json.gz shard directories")
    parser.add_argument("--dataset-config", type=str, default=None, help="Dataset config")
    parser.add_argument("--dataset-split", type=str, default="test", help="Dataset split")
    parser.add_argument("--dataset-text-field", type=str, default="text", help="Text field name inside the dataset")
    parser.add_argument("--dataset-percentage", type=float, default=100.0, help="Random percentage of the dataset to sample, in the range (0, 100]")
    parser.add_argument("--dataset-cache-dir", type=str, default=None, help="Optional HuggingFace dataset cache directory")
    parser.add_argument("--num-batches", type=int, default=None, help="Number of batches to process. If omitted, inferred from sampled dataset size and batch-size")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed used for dataset sampling")
    parser.add_argument("--output-json", type=str, default="saturation_results.json", help="Output JSON path")
    parser.add_argument("--coactivation-output", type=str, default=None, help="Co-activation output JSON")
    parser.add_argument(
        "--expert-token-distribution-output",
        type=str,
        default=None,
        help=(
            "Optional JSON path for per-layer expert selection counts and normalized "
            "probabilities (routed slots / flattened top-k), useful for comparing router "
            "load across datasets."
        ),
    )
    parser.add_argument("--coactivation-top-n", type=int, default=16, help="Top N experts for coactivation")
    parser.add_argument("--plot", action="store_true", help="Generate saturation and co-activation plots")
    parser.add_argument("--plot-dir", type=str, default=None, help="Directory to save plots (defaults to output-json directory)")
    parser.add_argument("--no-local-files-only", action="store_true", help="Allow downloading from HuggingFace (default: use only cached files)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (ignored if device-map is auto)")
    parser.add_argument("--no-device-map", action="store_true", help="Disable automatic device placement (requires model fit on single device)")
    
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    num_batches_provided = any(
        token == "--num-batches" or token.startswith("--num-batches=")
        for token in raw_argv
    )

    if not num_batches_provided and args.num_batches is None:
        planned_samples = 1
    elif args.num_batches is not None and args.num_batches > 0:
        planned_samples = args.num_batches * args.batch_size
    else:
        raise ValueError("--num-batches must be > 0 when provided")
    
    # Default to local files only to avoid network issues; override with --no-local-files-only if needed
    local_files_only = not args.no_local_files_only

    if local_files_only:
        # Enforce strict offline mode across HuggingFace stack.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        print("Offline mode enabled: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1")

    model_source = _resolve_model_source(args.model_name, local_files_only)
    if model_source != args.model_name:
        print(f"Resolved cached local model path: {model_source}")
    
    print(f"Loading model: {args.model_name}")
    
    # Try device_map="auto" for multi-GPU placement (requires accelerate)
    use_device_map = not args.no_device_map
    
    if use_device_map:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_source,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                local_files_only=local_files_only,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            print(f"Model loaded with automatic device placement across available GPUs")
        except (ImportError, ValueError) as e:
            print(f"Warning: device_map='auto' failed ({e}). Install accelerate or use --no-device-map.")
            print("Falling back to single device (may OOM for large models)...")
            model = AutoModelForCausalLM.from_pretrained(
                model_source,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                local_files_only=local_files_only
            )
            model = model.to(args.device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            local_files_only=local_files_only
        )
        model = model.to(args.device)
    
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        trust_remote_code=True,
        local_files_only=local_files_only
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    texts = load_dataset_texts(
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        dataset_path=args.dataset_path,
        split=args.dataset_split,
        text_field=args.dataset_text_field,
        percentage=args.dataset_percentage,
        cache_dir=args.dataset_cache_dir,
        num_samples=planned_samples,
        seed=args.seed,
        local_files_only=local_files_only,
    )

    if num_batches_provided:
        effective_num_batches = args.num_batches
    else:
        inferred = len(texts) // args.batch_size
        if inferred <= 0:
            raise ValueError(
                "Not enough sampled texts to form one batch. Increase --dataset-percentage or reduce --batch-size."
            )
        effective_num_batches = inferred
        print(
            f"Inferred num_batches={effective_num_batches} from sampled texts={len(texts)} and batch_size={args.batch_size}"
        )
    
    # Register hooks based on model type
    model_type = get_model_type(model)
    print(f"Detected model type: {model_type}")

    collector = RouterMetricsCollector()
    
    if model_type == "mixtral":
        hooks = register_sparse_moe_hooks(model, collector)
    elif model_type == "qwen3_moe":
        hooks = register_sparse_moe_hooks(model, collector)
    elif model_type == "olmoe":
        hooks = []
    elif model_type == "dbrx":
        hooks = register_dbrx_hooks(model, collector)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    print(f"Processing {effective_num_batches} batches...")
    
    # Run inference with progress bar over validation batches.
    total_items = min(len(texts), effective_num_batches * args.batch_size)
    batch_starts = range(0, total_items, args.batch_size)
    with torch.no_grad():
        for i in tqdm(batch_starts, total=effective_num_batches, desc="Validation", leave=False):
            batch_texts = texts[i:i + args.batch_size]
            
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length
            ).to(model.device)
            
            # Some datasets contain empty text rows that can tokenize to length 0 for specific tokenizers.
            # Skip those batches to avoid undefined routing statistics.
            if inputs["input_ids"].shape[1] == 0:
                continue

            if model_type == "olmoe":
                # Work around a Transformers OLMoE bug in load_balancing_loss_func where
                # attention_mask can trigger a divide-by-zero for certain routed token layouts.
                olmoe_inputs = dict(inputs)
                olmoe_inputs.pop("attention_mask", None)
                outputs = model(**olmoe_inputs, output_router_logits=True)
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
                _ = model(**inputs)
            
            if (i // args.batch_size + 1) % 10 == 0:
                print(f"  Processed {i // args.batch_size + 1} batches")
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Get results
    results = collector.finalize()
    
    # Determine plot directory
    if args.plot:
        if args.plot_dir:
            plot_dir = Path(args.plot_dir)
        else:
            plot_dir = Path(args.output_json).parent
        plot_dir.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "model": args.model_name,
        "model_type": model_type,
        "coactivation_mode": "pairwise-adjacent",
        "dataset": args.dataset,
        "dataset_path": args.dataset_path,
        "dataset_split": args.dataset_split,
        "dataset_text_field": args.dataset_text_field,
        "dataset_percentage": args.dataset_percentage,
        "seed": args.seed,
        "num_batches": effective_num_batches,
        "num_batches_provided": num_batches_provided,
        "layers": results
    }
    
    print("\nResults:")
    print(json.dumps(output_data, indent=2))
    
    # Save saturation results
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_data, indent=2))
    print(f"\nSaved saturation results to: {output_path}")
    
    # Save co-activation results
    if args.coactivation_output:
        top_n = args.coactivation_top_n
        coactivation_summary = {
            "model": args.model_name,
            "layers": {}
        }
        
        for layer_name, matrix in collector.coactivation_sums.items():
            matrix_normalized = _normalize_coactivation_matrix(
                matrix,
                collector.single_counts.get(layer_name),
            )

            top_indices = _select_top_experts_pairwise(matrix_normalized, top_n)
            submatrix = matrix_normalized[top_indices][:, top_indices]
            matrix_to_save = (submatrix * 100.0).tolist()
            
            coactivation_summary["layers"][layer_name] = {
                "experts": top_indices if isinstance(top_indices, list) else top_indices.tolist(),
                "matrix": matrix_to_save,
            }
        
        coact_path = Path(args.coactivation_output)
        coact_path.parent.mkdir(parents=True, exist_ok=True)
        coact_path.write_text(json.dumps(coactivation_summary, indent=2))
        print(f"Saved co-activation results to: {coact_path}")

    if args.expert_token_distribution_output:
        dist_payload = build_expert_token_distribution_payload(
            collector.single_counts,
            model_name=args.model_name,
            model_type=model_type,
            dataset=args.dataset,
            dataset_path=args.dataset_path,
            dataset_split=args.dataset_split,
            dataset_text_field=args.dataset_text_field,
            dataset_percentage=args.dataset_percentage,
            seed=args.seed,
            num_batches=effective_num_batches,
        )
        dist_path = Path(args.expert_token_distribution_output)
        dist_path.parent.mkdir(parents=True, exist_ok=True)
        dist_path.write_text(json.dumps(dist_payload, indent=2))
        print(f"Saved expert token distribution results to: {dist_path}")
    
    # Generate co-activation plots if requested
    if args.plot:
        if not HAS_PLOTTING:
            raise RuntimeError(
                "matplotlib and seaborn are required for --plot but are not available. "
                "Please install: pip install matplotlib seaborn"
            )
        if collector.coactivation_sums:
            print(f"\nGenerating co-activation plots in {plot_dir}...")
            plot_coactivation_heatmaps(
                collector.coactivation_sums,
                plot_dir,
                args.coactivation_top_n,
                collector.single_counts,
            )
        else:
            print("\nNo co-activation data to plot.")


if __name__ == "__main__":
    main()
