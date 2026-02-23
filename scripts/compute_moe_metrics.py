#!/usr/bin/env python3
"""Post-hoc MoE router metrics from a Megatron checkpoint.

Computes expert utilization, co-activation rates (per OLMoE paper), and saturation metrics
across checkpoint iterations. Supports comparison against multiple earlier checkpoints.

Usage:
    python compute_moe_metrics.py \\
      --tensor-model-parallel-size 1 \\
      --pipeline-model-parallel-size 1 \\
      --load <checkpoint_directory> \\
      --use-checkpoint-args \\
      --ckpt-format torch_dist \\
      --vocab-file <path_to_vocab_file> \\
      --merge-file <path_to_merges_file> \\
      --legacy-tokenizer \\
      --no-bias-dropout-fusion \\
      --moe-data-type hf_dataset \\
      --moe-dataset-name wikitext \\ 
      --moe-dataset-config wikitext-2-raw-v1 \\
      --moe-dataset-split test \\
      --moe-dataset-percentage 100.0 \\
      --moe-dataset-cache-dir <path_to_cache> \\
      --moe-compare-ckpts 500,2000,3500,4500 \\
      --moe-output-json saturation_results.json \\
      --moe-coactivation-output coactivation_results.json \\
      --moe-plot \\
      --moe-plot-dir <output_plot_directory>

Data type options:
    - random: Generate random token inputs (no tokenizer needed, fastest)
    - hf_dataset: Load from HuggingFace datasets (wikitext, openwebtext, etc.)
    - prompts: Load text prompts from file (--moe-prompts-file <path>)

Output:
    - saturation_results.json: Router saturation metrics per checkpoint comparison
    - coactivation_results.json: Co-activation matrices for top-N experts per layer
    - router_saturation_vs_final.png: Saturation evolution plot
    - coactivation_*.png: Co-activation heatmaps for each layer
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm

REPO_ROOT = Path(__file__).resolve().parents[1]
MEGATRON_ROOT = REPO_ROOT / "submodules" / "Megatron-LM"
sys.path.append(str(MEGATRON_ROOT))

from functools import partial  # noqa: E402
from gpt_builders import gpt_builder  # noqa: E402
from model_provider import model_provider  # noqa: E402
from megatron.core.transformer.moe.router import TopKRouter  # noqa: E402
from megatron.training import get_args, get_model, print_rank_0  # noqa: E402
from megatron.training.tokenizer import build_tokenizer  # noqa: E402
from megatron.training.checkpointing import (
    checkpoint_exists,
    get_checkpoint_tracker_filename,
    load_checkpoint,
    read_metadata,
)  # noqa: E402
from megatron.training.initialize import initialize_megatron  # noqa: E402
from megatron.training.utils import get_ltor_masks_and_position_ids  # noqa: E402


class RouterMetricsCollector:
    def __init__(self) -> None:
        self.totals = {}

    def update(
        self,
        name: str,
        routing_map: torch.Tensor,
        topk: int,
        probs: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        reference_routing_map: Optional[torch.Tensor] = None,
    ) -> None:
        routing_map = routing_map.detach()
        if padding_mask is not None:
            padding_mask = padding_mask.reshape(-1)
            routing_map = routing_map[~padding_mask]
            if probs is not None:
                probs = probs.detach()
                probs = probs[~padding_mask]

        routing_map = routing_map.float()
        tokens_per_expert = routing_map.sum(dim=0)
        saturation = (tokens_per_expert > 0).float().mean().item()

        if topk > 1:
            num_experts = routing_map.shape[1]
            if probs is None:
                co_occurrence = routing_map.t().matmul(routing_map)
                expert_counts = torch.diagonal(co_occurrence)
                normalized = co_occurrence.float() / expert_counts.clamp(min=1.0).unsqueeze(
                    1
                )
                off_diagonal = ~torch.eye(
                    num_experts, dtype=torch.bool, device=normalized.device
                )
                coactivation_rate = normalized[off_diagonal].mean().item()
            else:
                topk_indices = torch.topk(probs, k=topk, dim=-1).indices
                single_counts = torch.zeros(
                    num_experts, device=topk_indices.device, dtype=torch.float32
                )
                pair_counts = torch.zeros(
                    (num_experts, num_experts),
                    device=topk_indices.device,
                    dtype=torch.float32,
                )

                single_counts.scatter_add_(
                    0,
                    topk_indices.reshape(-1),
                    torch.ones_like(topk_indices.reshape(-1), dtype=torch.float32),
                )

                a = topk_indices[:, :-1].reshape(-1)
                b = topk_indices[:, 1:].reshape(-1)
                pair_counts.index_put_(
                    (a, b),
                    torch.ones_like(a, dtype=torch.float32),
                    accumulate=True,
                )
                pair_counts.index_put_(
                    (b, a),
                    torch.ones_like(a, dtype=torch.float32),
                    accumulate=True,
                )

                normalized = pair_counts / single_counts.clamp(min=1.0).unsqueeze(1)
                off_diagonal = ~torch.eye(
                    num_experts, dtype=torch.bool, device=normalized.device
                )
                coactivation_rate = normalized[off_diagonal].mean().item()
        else:
            coactivation_rate = 0.0

        if name not in self.totals:
            self.totals[name] = {
                "batches": 0,
                "saturation_sum": 0.0,
                "coactivation_sum": 0.0,
                "compare_saturation_sum": 0.0,
                "compare_tokens": 0,
            }

        self.totals[name]["batches"] += 1
        self.totals[name]["saturation_sum"] += saturation
        self.totals[name]["coactivation_sum"] += coactivation_rate

        if reference_routing_map is not None:
            if reference_routing_map.shape != routing_map.shape:
                raise ValueError(
                    f"Routing map shape mismatch for {name}: "
                    f"{routing_map.shape} vs {reference_routing_map.shape}."
                )
            routing_bool = routing_map.bool()
            intersection = (routing_bool & reference_routing_map).sum(dim=1).float()
            denom = max(topk, 1)
            overlap = intersection / denom
            self.totals[name]["compare_saturation_sum"] += overlap.sum().item()
            self.totals[name]["compare_tokens"] += overlap.numel()

    def finalize(self) -> dict:
        results = {}
        for name, vals in self.totals.items():
            batches = max(vals["batches"], 1)
            compare_tokens = max(vals["compare_tokens"], 1)
            results[name] = {
                "expert_coactivation": vals["coactivation_sum"] / batches,
                "compare_saturation": vals["compare_saturation_sum"] / compare_tokens,
            }
        return results


def add_metrics_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group(title="MoE metrics")
    group.add_argument(
        "--moe-data-type",
        type=str,
        choices=["random", "prompts", "hf_dataset"],
        default="random",
        help="Data source: 'random' (default), 'prompts' (from file), or 'hf_dataset' (any HuggingFace dataset).",
    )
    group.add_argument("--moe-prompts-file", type=str, default=None)
    group.add_argument(
        "--moe-dataset-name",
        type=str,
        default=None,
        help="HuggingFace dataset name (e.g., 'allenai/c4', 'wikitext', 'openwebtext').",
    )
    group.add_argument(
        "--moe-dataset-config",
        type=str,
        default=None,
        help="Dataset configuration name (e.g., 'en' for C4, 'wikitext-2-raw-v1' for wikitext).",
    )
    group.add_argument(
        "--moe-dataset-split",
        type=str,
        default="validation",
        help="Dataset split to use (e.g., 'train', 'validation', 'test').",
    )
    group.add_argument(
        "--moe-dataset-text-field",
        type=str,
        default="text",
        help="Name of the text field in the dataset (default: 'text').",
    )
    group.add_argument(
        "--moe-dataset-percentage",
        type=float,
        default=0.5,
        help="Percentage of dataset to use (0-100, default: 0.5).",
    )
    group.add_argument(
        "--moe-dataset-cache-dir",
        type=str,
        default=None,
        help="Cache directory for HuggingFace datasets.",
    )
    group.add_argument("--moe-batch-size", type=int, default=1)
    group.add_argument("--moe-num-batches", type=int, default=10)
    group.add_argument("--moe-seq-length", type=int, default=None)
    group.add_argument("--moe-output-json", type=str, default=None)
    group.add_argument(
        "--moe-compare-ckpts",
        type=str,
        default=None,
        help="Comma-separated checkpoint steps to compare against the final checkpoint.",
    )
    group.add_argument(
        "--moe-coactivation-output",
        type=str,
        default=None,
        help="Write layerwise coactivation summary for the final checkpoint.",
    )
    group.add_argument(
        "--moe-coactivation-top-n",
        type=int,
        default=32,
        help="Number of experts to keep for the coactivation summary.",
    )
    group.add_argument(
        "--moe-plot",
        action="store_true",
        default=False,
        help="Save saturation and coactivation plots.",
    )
    group.add_argument(
        "--moe-plot-dir",
        type=str,
        default=None,
        help="Directory to write plots (defaults next to output JSON).",
    )
    group.add_argument("--moe-seed", type=int, default=1234)
    return parser


def _load_prompts(path: Optional[str]) -> List[str]:
    if not path:
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _load_hf_dataset_texts(
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    text_field: str,
    percentage: float,
    cache_dir: Optional[str],
    num_samples: int,
    seed: int,
) -> List[str]:
    """Load texts from any HuggingFace dataset.

    Tries internet first, falls back to cache, then returns empty list if both fail.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets library required. Install with: pip install datasets"
        )

    print_rank_0(
        f"Loading {dataset_name} ({dataset_config or 'default'}) {split} ({percentage}%)..."
    )

    load_kwargs = {"path": dataset_name, "split": split, "cache_dir": cache_dir}
    if dataset_config:
        load_kwargs["name"] = dataset_config

    # Try loading from internet first
    try:
        dataset = load_dataset(**load_kwargs)
        print_rank_0(f"Loaded {dataset_name} successfully.")
    except Exception as e:
        # Fall back to cache-only mode
        print_rank_0(f"Internet load failed ({type(e).__name__}). Trying cache-only...")
        try:
            load_kwargs_offline = load_kwargs.copy()
            load_kwargs_offline["download_mode"] = (
                "force_redownload" if cache_dir else "reuse_cache_if_exists"
            )
            dataset = load_dataset(**load_kwargs_offline)
            print_rank_0(f"Loaded {dataset_name} from cache.")
        except Exception as e2:
            print_rank_0(
                f"Cache load failed ({type(e2).__name__}). Falling back to random inputs."
            )
            return []

    sample_fraction = percentage / 100.0
    sample_size = min(
        max(int(len(dataset) * sample_fraction), num_samples), len(dataset)
    )

    indices = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(seed)
    ).tolist()[:sample_size]
    sampled = dataset.select(indices)

    texts = []
    for item in sampled:
        if text_field in item and item[text_field]:
            text = item[text_field]
            if isinstance(text, str):
                texts.append(text)
            elif isinstance(text, list):
                texts.extend([t for t in text if isinstance(t, str)])

    print_rank_0(f"Loaded {len(texts)} texts")
    return texts


def _iter_token_batches(
    prompts: List[str],
    tokenizer,
    batch_size: int,
    seq_length: int,
    pad_id: int,
    vocab_size: int,
    num_batches: int,
    random_inputs: bool,
    device: torch.device,
    generator: torch.Generator,
) -> Iterable[torch.Tensor]:
    prompt_idx = 0
    for _ in range(num_batches):
        batch = []
        for _ in range(batch_size):
            if random_inputs or not prompts:
                ids = torch.randint(
                    0, vocab_size, (seq_length,), device=device, generator=generator
                ).tolist()
            else:
                text = prompts[prompt_idx % len(prompts)]
                prompt_idx += 1
                ids = tokenizer.tokenize(text)[:seq_length]
                ids = ids + [pad_id] * (seq_length - len(ids))
            batch.append(ids)
        yield torch.tensor(batch, device=device, dtype=torch.long)


def _parse_ckpt_steps(value: Optional[str]) -> List[int]:
    if not value:
        return []
    return [int(s.strip()) for s in value.split(",") if s.strip()]


def _get_final_checkpoint_step(load_dir: str) -> Tuple[Optional[int], bool]:
    tracker = get_checkpoint_tracker_filename(load_dir)
    if not Path(tracker).is_file():
        return None, False
    iteration, release = read_metadata(tracker)
    return iteration, release


def _build_token_batches(
    prompts: List[str],
    tokenizer,
    batch_size: int,
    seq_length: int,
    pad_id: int,
    vocab_size: int,
    num_batches: int,
    random_inputs: bool,
    device: torch.device,
    seed: int,
) -> List[torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return list(
        _iter_token_batches(
            prompts,
            tokenizer,
            batch_size,
            seq_length,
            pad_id,
            vocab_size,
            num_batches,
            random_inputs,
            device,
            generator,
        )
    )


def _collect_final_routing_and_coactivation(
    model,
    token_batches: List[torch.Tensor],
    eod_token: int,
    pad_token: int,
    reset_position_ids: bool,
    reset_attention_mask: bool,
    eod_mask_loss: bool,
    pad_mask_loss: bool,
) -> Tuple[Dict[str, List[torch.Tensor]], Dict[str, torch.Tensor], Dict[str, int]]:
    routing_maps: Dict[str, List[torch.Tensor]] = {}
    coactivation_sums: Dict[str, torch.Tensor] = {}
    topk_by_layer: Dict[str, int] = {}

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, TopKRouter):
            routing_maps[name] = []

            def _hook(mod, _inputs, outputs, layer_name=name):
                _probs, routing_map = outputs
                padding_mask = None
                if len(_inputs) > 1:
                    padding_mask = _inputs[1]
                routing_map = routing_map.detach()
                if padding_mask is not None:
                    padding_mask = padding_mask.reshape(-1)
                    routing_map = routing_map[~padding_mask]
                topk_by_layer[layer_name] = mod.topk
                routing_maps[layer_name].append(
                    routing_map.to(dtype=torch.bool, device="cpu")
                )

                routing_float = routing_map.float()
                coactivation = routing_float.t().matmul(routing_float)
                if layer_name not in coactivation_sums:
                    coactivation_sums[layer_name] = coactivation.detach().cpu()
                else:
                    coactivation_sums[layer_name] += coactivation.detach().cpu()

            hooks.append(module.register_forward_hook(_hook))

    with torch.no_grad():
        for tokens in token_batches:
            attention_mask, _loss_mask, position_ids = get_ltor_masks_and_position_ids(
                tokens,
                eod_token,
                pad_token,
                reset_position_ids,
                reset_attention_mask,
                eod_mask_loss,
                pad_mask_loss,
            )
            padding_mask = tokens.eq(pad_token)
            _ = model(
                tokens,
                position_ids,
                attention_mask,
                padding_mask=padding_mask,
            )

    for handle in hooks:
        handle.remove()

    return routing_maps, coactivation_sums, topk_by_layer


def _compare_saturation_against_final(
    model,
    token_batches: List[torch.Tensor],
    eod_token: int,
    pad_token: int,
    reset_position_ids: bool,
    reset_attention_mask: bool,
    eod_mask_loss: bool,
    pad_mask_loss: bool,
    final_routing_maps: Dict[str, List[torch.Tensor]],
    topk_by_layer: Dict[str, int],
) -> dict:
    collector = RouterMetricsCollector()
    batch_index = {"value": 0}

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, TopKRouter):

            def _hook(mod, _inputs, outputs, layer_name=name):
                _probs, routing_map = outputs
                padding_mask = None
                if len(_inputs) > 1:
                    padding_mask = _inputs[1]
                idx = batch_index["value"]
                reference = final_routing_maps[layer_name][idx].to(routing_map.device)
                collector.update(
                    layer_name,
                    routing_map,
                    topk_by_layer[layer_name],
                    probs=_probs,
                    padding_mask=padding_mask,
                    reference_routing_map=reference,
                )

            hooks.append(module.register_forward_hook(_hook))

    with torch.no_grad():
        for tokens in token_batches:
            attention_mask, _loss_mask, position_ids = get_ltor_masks_and_position_ids(
                tokens,
                eod_token,
                pad_token,
                reset_position_ids,
                reset_attention_mask,
                eod_mask_loss,
                pad_mask_loss,
            )
            padding_mask = tokens.eq(pad_token)
            _ = model(
                tokens,
                position_ids,
                attention_mask,
                padding_mask=padding_mask,
            )
            batch_index["value"] += 1

    for handle in hooks:
        handle.remove()

    return collector.finalize()


def main() -> None:
    initialize_megatron(
        extra_args_provider=add_metrics_args,
        args_defaults={
            # Change these defaults if necessary
            "no_load_rng": True,
            "no_load_optim": True,
            "micro_batch_size": 1,
            "exit_on_missing_checkpoint": True,
            "enable_msc": False,
            "use_cpu_initialization": True,
            "standalone_embedding_stage": False,
        },
    )
    args = get_args()

    if args.pipeline_model_parallel_size > 1:
        raise RuntimeError("Pipeline parallelism >1 is not supported in this script.")

    load_dir = getattr(args, "load", None)
    if not load_dir or not checkpoint_exists(load_dir):
        raise RuntimeError(
            "No checkpoint found. Provide --load pointing to a valid checkpoint directory."
        )

    ddp_model = get_model(partial(model_provider, gpt_builder), wrap_with_ddp=False)
    model = ddp_model[0]
    model.eval()

    # Determine if we need a tokenizer
    use_random = args.moe_data_type == "random"

    if use_random:
        tokenizer = None
        pad_id = 0
    else:
        tokenizer = build_tokenizer(args)
        # Legacy tokenizer doesn't have 'pad', use eod as padding
        pad_id = tokenizer.eod

    seq_length = (
        args.moe_seq_length
        if args.moe_seq_length is not None
        else args.max_position_embeddings
    )
    if seq_length is None:
        raise ValueError("Specify --seq-length or --max-position-embeddings.")

    # Determine data source and load prompts
    prompts = []

    if args.moe_data_type == "prompts" or (args.moe_prompts_file and not use_random):
        prompts = _load_prompts(args.moe_prompts_file)
        if not prompts:
            print_rank_0("No prompts loaded. Falling back to random inputs.")
            use_random = True

    if args.moe_data_type == "hf_dataset" and not use_random:
        if not args.moe_dataset_name:
            print_rank_0(
                "ERROR: --moe-dataset-name is required when using --moe-data-type hf_dataset"
            )
            use_random = True
        else:
            prompts = _load_hf_dataset_texts(
                dataset_name=args.moe_dataset_name,
                dataset_config=args.moe_dataset_config,
                split=args.moe_dataset_split,
                text_field=args.moe_dataset_text_field,
                percentage=args.moe_dataset_percentage,
                cache_dir=args.moe_dataset_cache_dir,
                num_samples=args.moe_num_batches * args.moe_batch_size * 2,
                seed=args.moe_seed,
            )
            if not prompts:
                use_random = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if use_random:
        eod_token = 0
        vocab_size = args.padded_vocab_size
        print_rank_0(
            "Using random inputs (set --moe-data-type c4 or --moe-prompts-file for real data)"
        )
    else:
        eod_token = tokenizer.eod
        vocab_size = args.padded_vocab_size or tokenizer.vocab_size

    pad_token = pad_id
    reset_position_ids = getattr(args, "reset_position_ids", False)
    reset_attention_mask = getattr(args, "reset_attention_mask", False)
    eod_mask_loss = getattr(args, "eod_mask_loss", False)
    pad_mask_loss = getattr(args, "pad_mask_loss", False)

    compare_steps = _parse_ckpt_steps(args.moe_compare_ckpts)
    final_step, is_release = _get_final_checkpoint_step(load_dir)
    if final_step is None and not is_release:
        raise RuntimeError(
            "Could not determine final checkpoint from latest_checkpointed_iteration.txt"
        )

    token_batches = _build_token_batches(
        prompts,
        tokenizer,
        args.moe_batch_size,
        seq_length,
        pad_id,
        vocab_size,
        args.moe_num_batches,
        use_random,
        device,
        args.moe_seed,
    )

    args.ckpt_step = None if is_release else final_step
    load_checkpoint(ddp_model, None, None, strict=False)
    final_routing_maps, coactivation_sums, topk_by_layer = (
        _collect_final_routing_and_coactivation(
            model,
            token_batches,
            eod_token,
            pad_token,
            reset_position_ids,
            reset_attention_mask,
            eod_mask_loss,
            pad_mask_loss,
        )
    )

    saturation_results = {
        "final_checkpoint": "release" if is_release else final_step,
        "compare_steps": compare_steps,
        "layers": {},
    }
    for step in compare_steps:
        args.ckpt_step = step
        load_checkpoint(ddp_model, None, None, strict=False)
        saturation_results["layers"][str(step)] = _compare_saturation_against_final(
            model,
            token_batches,
            eod_token,
            pad_token,
            reset_position_ids,
            reset_attention_mask,
            eod_mask_loss,
            pad_mask_loss,
            final_routing_maps,
            topk_by_layer,
        )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() != 0:
            return

    print_rank_0(json.dumps(saturation_results, indent=2, sort_keys=True))
    if args.moe_output_json:
        output_path = Path(args.moe_output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(saturation_results, indent=2, sort_keys=True))
        default_plot_dir = output_path.parent
    else:
        default_plot_dir = Path.cwd()

    if args.moe_coactivation_output:
        top_n = max(args.moe_coactivation_top_n, 1)
        coactivation_summary = {
            "final_checkpoint": "release" if is_release else final_step,
            "layers": {},
        }
        for layer_name, matrix in coactivation_sums.items():
            matrix = matrix.clone().float()
            diag = torch.diagonal(matrix)  # N_Ei: total activations per expert
            # Normalize by expert activation counts to match paper formula:
            # Expert co-activation(E_i, E_j) = N_{E_i, E_j} / N_{E_i}
            diag_safe = diag.clamp(min=1.0)  # Avoid division by zero
            matrix_normalized = matrix / diag_safe.unsqueeze(1)
            # Zero out diagonal (self-activation rates are not meaningful)
            matrix_normalized = matrix_normalized - torch.diag(
                torch.diagonal(matrix_normalized)
            )
            # Select top-N experts with highest maximum co-activation rates
            max_per_expert = matrix_normalized.max(dim=1).values
            top_indices = torch.topk(
                max_per_expert, k=min(top_n, matrix_normalized.shape[0])
            ).indices
            top_indices = top_indices.sort().values
            submatrix = matrix_normalized[top_indices][:, top_indices]
            coactivation_summary["layers"][layer_name] = {
                "experts": top_indices.tolist(),
                "matrix": submatrix.tolist(),
            }

        output_path = Path(args.moe_coactivation_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(coactivation_summary, indent=2, sort_keys=True)
        )

    if args.moe_plot:
        if plt is None:
            raise RuntimeError(
                "matplotlib is required for --moe-plot but is not available."
            )

        plot_dir = Path(args.moe_plot_dir) if args.moe_plot_dir else default_plot_dir
        plot_dir.mkdir(parents=True, exist_ok=True)

        if compare_steps:
            steps_sorted = sorted(compare_steps)
            layer_names = set()
            for step in steps_sorted:
                layer_names.update(saturation_results["layers"][str(step)].keys())
            layer_names = sorted(layer_names)

            plt.figure(figsize=(14, 7))

            cmap = cm.get_cmap("tab20" if len(layer_names) <= 20 else "hsv")
            colors = [
                cmap(i / max(len(layer_names) - 1, 1)) for i in range(len(layer_names))
            ]

            for idx, layer_name in enumerate(layer_names):
                y_vals = []
                for step in steps_sorted:
                    layer_data = saturation_results["layers"][str(step)].get(layer_name)
                    if layer_data is None:
                        y_vals.append(float("nan"))
                    else:
                        y_vals.append(layer_data.get("compare_saturation", 0.0) * 100.0)
                plt.plot(
                    steps_sorted,
                    y_vals,
                    marker="o",
                    label=layer_name,
                    color=colors[idx],
                    linewidth=2.5,
                    markersize=7,
                )

            plt.xlabel("Checkpoint step", fontsize=12, fontweight="bold")
            plt.ylabel("Router saturation (%)", fontsize=12, fontweight="bold")
            plt.title(
                "Router saturation vs final checkpoint", fontsize=14, fontweight="bold"
            )
            plt.grid(True, alpha=0.3, linestyle="--")
            plt.ylim(0, 105)
            # Place legend outside plot area, to the right
            plt.legend(
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
                fontsize=9,
                framealpha=0.95,
                edgecolor="black",
            )
            plt.tight_layout()
            plt.savefig(plot_dir / "router_saturation_vs_final.png", dpi=200)
            plt.close()

        if args.moe_coactivation_output:
            for layer_name, layer_info in coactivation_summary["layers"].items():
                matrix = torch.tensor(layer_info["matrix"], dtype=torch.float32)
                plt.figure(figsize=(5, 4))
                plt.imshow(matrix, cmap="RdPu", aspect="auto")
                plt.colorbar()
                plt.title(f"Co-activation: {layer_name}")
                plt.xlabel("Expert")
                plt.ylabel("Expert")
                plt.tight_layout()
                safe_name = layer_name.replace("/", "_")
                plt.savefig(plot_dir / f"coactivation_{safe_name}.png", dpi=200)
                plt.close()


if __name__ == "__main__":
    main()
