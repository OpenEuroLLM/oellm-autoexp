"""Post-hoc MoE router metrics from a Megatron checkpoint.

Computes expert utilization, co-activation rates (per OLMoE paper), and saturation
metrics across checkpoint iterations. Supports comparison against multiple earlier
checkpoints to track how routing behaviour evolves during training.

===============================================================================
USAGE EXAMPLE (with experiment YAML config)
===============================================================================
    We are loading the dataset from a local folder here:
    python scripts/moe/compute_moe_metrics.py \
        --load chkpt_folder \
        --use-checkpoint-args \
        --moe-experiment-config moe_config.yaml \
        --moe-data-type hf_dataset \
        --moe-dataset-path c4 \
        --moe-dataset-config en \
        --moe-dataset-split validation \
        --moe-dataset-percentage 0.5 \
        --moe-seq-length 2048 \
        --moe-batch-size 1 \
        --moe-compare-ckpts 500,2000,3500,4500 \
        --moe-output-json saturation_results.json \
        --moe-coactivation-output coactivation_results.json \
        --moe-plot \
        --moe-plot-dir plots/
    
    For running with a HF dataset (wikitext here):
    python scripts/moe/compute_moe_metrics.py \
        --load chkpt_folder \
        --use-checkpoint-args \
        --moe-experiment-config moe_config.yaml \
        --moe-data-type hf_dataset \
        --moe-dataset-name wikitext \
        --moe-dataset-config wikitext-2-raw-v1\
        --moe-dataset-split test \
        --moe-num-batches 2000 \
        --moe-seq-length 2048 \
        --moe-batch-size 1 \
        --moe-compare-ckpts 500,2000,3500,4500 \
        --moe-output-json saturation_results.json \
        --moe-coactivation-output coactivation_results.json \
        --moe-plot \
        --moe-plot-dir plots/

With color output preserved in a log file:

    python scripts/moe/compute_moe_metrics.py ... --moe-force-color 2>&1 | tee run.log
    less -R run.log

Re-enable verbose Megatron output for debugging:

    ... --no-moe-quiet-init --no-moe-suppress-warnings --no-moe-silence-megatron-logging

===============================================================================
ARGUMENTS REFERENCE
===============================================================================

--- Experiment config (YAML) ------------------------------------------------

  --moe-experiment-config PATH
        Path to the experiment YAML file (e.g. moe_config.yaml).
        Values under the `backend.megatron` key are used as defaults for
        Megatron model arguments and evaluated-before initialization.
        Explicit CLI flags always take priority over YAML values.
        Hydra-style `${.key}` self-references within that section are resolved.

--- Data source --------------------------------------------------------------

  --moe-data-type {random,prompts,hf_dataset}   [default: random]
        How to produce token batches for the forward passes.
          random      – generate random token IDs (no tokenizer needed, fastest)
          prompts     – read plain-text lines from --moe-prompts-file
          hf_dataset  – fetch from a HuggingFace dataset

  --moe-prompts-file PATH
        Path to a plain-text file; one prompt per line.
        Required when --moe-data-type prompts.

  --moe-dataset-name NAME                       [default: None]
        HuggingFace dataset name (e.g. "wikitext", "openwebtext",
      "allenai/c4").  Required when --moe-data-type hf_dataset unless
      --moe-dataset-path is provided.

  --moe-dataset-path PATH                       [default: None]
      Path to a HuggingFace dataset saved on local disk
      (created with datasets.Dataset.save_to_disk).
      When provided, this is loaded with load_from_disk and uses the
      same split/text-field/percentage settings as remote datasets.

  --moe-dataset-config NAME                     [default: None]
        Dataset configuration (e.g. "wikitext-2-raw-v1", "en").

  --moe-dataset-split NAME                      [default: validation]
        Dataset split to use: "train", "validation", "test", etc.

  --moe-dataset-text-field FIELD                [default: text]
        Name of the string field that contains the document text.

  --moe-dataset-percentage FLOAT                [default: 0.5]
        Percentage of the dataset to sample (0–100).

  --moe-dataset-cache-dir PATH                  [default: None]
        Local directory for HuggingFace dataset / model cache.

--- Evaluation config --------------------------------------------------------

  --moe-batch-size INT                          [default: 1]
        Number of sequences per forward-pass batch.

  --moe-num-batches INT                         [default: 10]
        Total number of batches to run; determines evaluation coverage.

  --moe-seq-length INT                          [default: None → from checkpoint]
        Sequence length used for tokenization and padding.
        Defaults to the value stored in the checkpoint args.

  --moe-seed INT                                [default: 1234]
        Random seed for batch generation and dataset sampling.

--- Checkpoint comparison ----------------------------------------------------

  --moe-compare-ckpts STEPS                     [default: None]
        Comma-separated list of earlier checkpoint iteration numbers
        (e.g. "500,2000,3500") to compare routing saturation against the
        final checkpoint.  The final checkpoint is always evaluated first.

--- Output files -------------------------------------------------------------

  --moe-output-json PATH                        [default: None]
        File path for saturation metrics JSON output.
        Contains per-layer, per-step saturation and co-activation values.

  --moe-coactivation-output PATH                [default: None]
        File path for co-activation summary JSON for the final checkpoint.
        Stores the top-N expert pairwise co-activation rates per layer.

  --moe-coactivation-top-n INT                  [default: 32]
        Number of experts to include in the top-N co-activation summary.

  --moe-plot                                    [default: off]
        Save saturation-evolution and co-activation heatmap plots.

  --moe-plot-dir PATH                           [default: same as output JSON]
        Directory where plots are written.
        Generated files:
          router_saturation_vs_final.png     – saturation over training steps
          coactivation_layer_<N>.png         – co-activation heatmap per layer

--- Noise suppression (all default ON) --------------------------------------

  --moe-quiet-init / --no-moe-quiet-init        [default: True]
        Suppress stdout/stderr during Megatron initialization and checkpoint
        loading.  Use --no-moe-quiet-init to see full Megatron startup logs.

  --moe-suppress-warnings / --no-moe-suppress-warnings   [default: True]
        Suppress Python `warnings` module output during execution.

  --moe-silence-megatron-logging / --no-moe-silence-megatron-logging
                                                [default: True]
        Set all loggers under the `megatron.*` namespace to CRITICAL level,
        eliminating RNG, tensor-parallel, and other routine log lines.

--- Display / console output -------------------------------------------------

  --moe-color-output / --no-moe-color-output    [default: True]
        Use ANSI colour codes for stage banners and log lines.
        Automatically disabled when stdout is not a TTY (e.g. log files),
        unless --moe-force-color is also set.

  --moe-force-color / --no-moe-force-color      [default: False]
        Force ANSI colour codes even when stdout is redirected.
        Useful when piping with `tee` and viewing with `less -R`.

  --moe-print-json / --no-moe-print-json        [default: False]
        Print the full saturation JSON to stdout after evaluation.
        Can be very verbose for large models with many compare steps.

===============================================================================
OUTPUTS
===============================================================================

  saturation_results.json
      Per-step, per-layer routing saturation and co-activation averages.
      Also contains an "args" block summarising the run configuration.

  coactivation_results.json
      Top-N expert co-activation rates for the final checkpoint,
      one entry per layer with expert indices and conditional rates.

  router_saturation_vs_final.png
      Line plot of mean routing saturation across layers vs. training step.

  coactivation_layer_<N>.png
      Heatmap of top-N expert co-activation for each MoE layer.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from tqdm.auto import tqdm
REPO_ROOT = Path(__file__).resolve().parents[2]
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


# -----------------------------------------------------------------------------
# Console / log presentation helpers
# -----------------------------------------------------------------------------
_MOE_COLOR_OUTPUT = True
_MOE_FORCE_COLOR = False


def _colorize(text: str, color: str) -> str:
    # Keep log formatting deterministic when color is disabled or output is redirected.
    if not _MOE_COLOR_OUTPUT:
        return text
    if os.getenv("NO_COLOR") and not _MOE_FORCE_COLOR:
        return text
    if not _MOE_FORCE_COLOR and not sys.stdout.isatty():
        return text
    colors = {
        "cyan": "\033[36m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "reset": "\033[0m",
    }
    return f"{colors.get(color, '')}{text}{colors['reset']}"


def _log_stage(title: str) -> None:
    banner = f"===== {title} ====="
    print_rank_0(_colorize(banner, "blue"))


def _log_info(message: str) -> None:
    print_rank_0(_colorize(f"[INFO] {message}", "cyan"))


def _log_warn(message: str) -> None:
    print_rank_0(_colorize(f"[WARN] {message}", "yellow"))


def _log_success(message: str) -> None:
    print_rank_0(_colorize(f"[OK] {message}", "green"))


# -----------------------------------------------------------------------------
# Metrics collection
# -----------------------------------------------------------------------------
def _off_diagonal_mean(matrix: torch.Tensor) -> float:
    num_experts = matrix.shape[0]
    mask = ~torch.eye(num_experts, dtype=torch.bool, device=matrix.device)
    return matrix[mask].mean().item()


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
        # Remove padding positions before any metric computation so all rates are token-valid.
        routing_map = routing_map.detach()
        if padding_mask is not None:
            padding_mask = padding_mask.reshape(-1)
            routing_map = routing_map[~padding_mask]
            if probs is not None:
                probs = probs.detach()
                probs = probs[~padding_mask]

        routing_map = routing_map.float()
        # Saturation is the fraction of experts that received at least one token in this batch.
        tokens_per_expert = routing_map.sum(dim=0)
        saturation = (tokens_per_expert > 0).float().mean().item()

        if topk > 1:
            num_experts = routing_map.shape[1]
            if probs is None:
                # Fallback path: infer co-activation from routing map overlap only.
                co_occurrence = routing_map.t().matmul(routing_map)
                expert_counts = torch.diagonal(co_occurrence)
                normalized = co_occurrence.float() / expert_counts.clamp(min=1.0).unsqueeze(
                    1
                )
                coactivation_rate = _off_diagonal_mean(normalized)
            else:
                # Main path: use ordered top-k experts to build OLMoE-style adjacent pair counts.
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
                coactivation_rate = _off_diagonal_mean(normalized)
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
            # Compare saturation against final checkpoint by per-token top-k overlap.
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


# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------
def _progress_iter(items, description: str):
    # Single wrapper so progress behavior can be adjusted in one place.
    return tqdm(items, desc=description, leave=False)


def add_metrics_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group(title="MoE metrics")
    group.add_argument(
        "--moe-experiment-config",
        type=str,
        default=None,
        help=(
            "Path to experiment YAML (for example moe_config.yaml). "
            "Values under backend.megatron are used as defaults for this script. "
            "Explicit CLI flags still take precedence."
        ),
    )
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
        help="HuggingFace dataset name (e.g., 'allenai/c4', 'wikitext', 'openwebtext'). Optional if --moe-dataset-path is used.",
    )
    group.add_argument(
        "--moe-dataset-path",
        type=str,
        default=None,
        help="Local path to a HuggingFace dataset saved with save_to_disk.",
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
    group.add_argument(
        "--moe-num-batches",
        type=int,
        default=None,
        help=(
            "Total number of batches to run. If omitted for --moe-data-type hf_dataset, "
            "the script infers it from sampled texts and --moe-batch-size."
        ),
    )
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
    group.add_argument(
        "--moe-quiet-init",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Suppress stdout/stderr noise during Megatron initialization and checkpoint loading.",
    )
    group.add_argument(
        "--moe-suppress-warnings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Suppress Python warnings during script execution.",
    )
    group.add_argument(
        "--moe-silence-megatron-logging",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Silence loggers under the megatron.* namespace.",
    )
    group.add_argument(
        "--moe-color-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use ANSI colors for script stage logs.",
    )
    group.add_argument(
        "--moe-print-json",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print full saturation JSON to stdout (can be very verbose).",
    )
    group.add_argument(
        "--moe-force-color",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force ANSI colors even when stdout is redirected (useful with tee/log files).",
    )
    return parser


@contextlib.contextmanager
def _suppress_output(enabled: bool):
    if not enabled:
        yield
        return

    with open(Path("/dev/null"), "w", encoding="utf-8") as sink:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            yield


def _configure_logging_suppression(enabled: bool) -> None:
    if not enabled:
        return

    # Keep non-megatron loggers untouched and silence only megatron namespaces.
    megatron_logger = logging.getLogger("megatron")
    megatron_logger.handlers = [logging.NullHandler()]
    megatron_logger.setLevel(logging.CRITICAL)
    megatron_logger.propagate = False

    # Clamp existing megatron child loggers that may already be instantiated.
    for name, obj in logging.Logger.manager.loggerDict.items():
        if not isinstance(obj, logging.Logger):
            continue
        if name == "megatron" or name.startswith("megatron."):
            obj.handlers = [logging.NullHandler()]
            obj.setLevel(logging.CRITICAL)
            obj.propagate = False


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
def _load_prompts(path: Optional[str]) -> List[str]:
    if not path:
        return []
    prompts_path = Path(path).expanduser()
    with prompts_path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _load_hf_dataset_texts(
    dataset_path: Optional[str],
    dataset_name: Optional[str],
    dataset_config: Optional[str],
    split: str,
    text_field: str,
    percentage: float,
    cache_dir: Optional[str],
    num_samples: int,
    seed: int,
) -> List[str]:
    """Load texts from any HuggingFace dataset.

    For remote datasets, performs a single load_dataset call and returns empty
    list on failure.
    """
    try:
        from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
    except ImportError:
        raise ImportError(
            "datasets library required. Install with: pip install datasets"
        )

    if dataset_path:
        dataset_path_obj = Path(dataset_path).expanduser()
        _log_info(
            f"Loading local dataset from path={dataset_path_obj} split={split} percentage={percentage}%"
        )
        if not dataset_path_obj.exists():
            _log_warn(
                f"Local dataset path not found: {dataset_path_obj} (cwd={Path.cwd()}); falling back to random inputs"
            )
            return []
        try:
            dataset_obj = load_from_disk(str(dataset_path_obj))
        except Exception as exc:
            # Non-save_to_disk layouts (for example C4 json.gz shards) are common on HPC storage.
            _log_warn(
                f"load_from_disk failed ({type(exc).__name__}); trying local shard files"
            )

            shard_dir = dataset_path_obj
            if dataset_config:
                config_subdir = shard_dir / dataset_config
                if config_subdir.is_dir():
                    shard_dir = config_subdir
                else:
                    _log_warn(
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

            # If split-specific discovery fails, allow loading all json shards in the chosen directory.
            if not shard_files:
                shard_files.extend(sorted(shard_dir.glob("*.json")))
                shard_files.extend(sorted(shard_dir.glob("*.json.gz")))

            # Preserve order while removing duplicates.
            deduped_files: List[str] = []
            seen: Set[str] = set()
            for file_path in shard_files:
                key = str(file_path)
                if key not in seen:
                    seen.add(key)
                    deduped_files.append(key)

            if not deduped_files:
                _log_warn(
                    f"No local json/json.gz shards found in {shard_dir}; falling back to random inputs"
                )
                return []

            _log_info(
                f"Loading {len(deduped_files)} local shard file(s) from {shard_dir}"
            )
            try:
                dataset = load_dataset(
                    "json",
                    data_files=deduped_files,
                    split="train",
                    cache_dir=cache_dir,
                )
                _log_success(f"Loaded local shard dataset from {shard_dir}")
            except Exception as shard_exc:
                _log_warn(
                    f"Local shard load failed ({type(shard_exc).__name__}); falling back to random inputs"
                )
                return []

        else:
            if isinstance(dataset_obj, DatasetDict):
                if split not in dataset_obj:
                    _log_warn(
                        f"Split '{split}' not found in local dataset. Available: {list(dataset_obj.keys())}. Falling back to random inputs"
                    )
                    return []
                dataset = dataset_obj[split]
            elif isinstance(dataset_obj, Dataset):
                dataset = dataset_obj
            else:
                _log_warn(
                    f"Unsupported local dataset type: {type(dataset_obj).__name__}; falling back to random inputs"
                )
                return []

            _log_success(f"Loaded local dataset from {dataset_path_obj}")
    else:
        _log_info(
            f"Loading dataset={dataset_name} config={dataset_config or 'default'} split={split} percentage={percentage}%"
        )

        load_kwargs = {"path": dataset_name, "split": split, "cache_dir": cache_dir}
        if dataset_config:
            load_kwargs["name"] = dataset_config

        # Single remote-load attempt. Offline/caching policy should be controlled externally.
        try:
            dataset = load_dataset(**load_kwargs)
            _log_success(f"Loaded {dataset_name} successfully")
        except Exception as e:
            _log_warn(
                f"Dataset load failed ({type(e).__name__}); falling back to random inputs"
            )
            return []

    sample_fraction = percentage / 100.0
    # Respect percentage but guarantee enough samples to fill planned evaluation batches.
    sample_size = min(
        max(int(len(dataset) * sample_fraction), num_samples), len(dataset)
    )
    _log_info(f"Sampling {sample_size} entries ({percentage}%) from dataset with {len(dataset)} total entries")



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

    _log_info(f"Loaded {len(texts)} text entries")
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


# -----------------------------------------------------------------------------
# Experiment-config parsing and defaults
# -----------------------------------------------------------------------------
def _load_yaml_file(path: str) -> Dict[str, Any]:
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise ImportError("Could not load YAML config. Install pyyaml.") from exc

    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"YAML config must be a mapping at top-level: {path}")
    return loaded


def _deep_get(data: Dict[str, Any], path: List[str]) -> Any:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _collect_cli_flags(argv: List[str]) -> Set[str]:
    flags: Set[str] = set()
    for token in argv:
        if token.startswith("--"):
            flags.add(token.split("=", 1)[0])
    return flags


def _preparse_bool_flag(
    argv: List[str],
    positive_flag: str,
    negative_flag: str,
    default: bool,
) -> bool:
    if negative_flag in argv:
        return False
    if positive_flag in argv:
        return True
    return default


def _extract_flag_value(argv: List[str], flag: str) -> Optional[str]:
    for idx, token in enumerate(argv):
        if token == flag and idx + 1 < len(argv):
            return argv[idx + 1]
        if token.startswith(flag + "="):
            return token.split("=", 1)[1]
    return None


_LOCAL_REF_RE = re.compile(r"^\$\{\.([A-Za-z0-9_]+)\}$")


def _resolve_backend_value(value: Any, backend: Dict[str, Any], max_depth: int = 16) -> Any:
    """Resolve simple local Hydra-style references like ${.num_attention_heads}."""
    current = value
    for _ in range(max_depth):
        if not isinstance(current, str):
            return current
        match = _LOCAL_REF_RE.match(current.strip())
        if not match:
            return current
        ref_key = match.group(1)
        if ref_key not in backend:
            return current
        current = backend[ref_key]
    return current


def _infer_tokenizer_type(backend: Dict[str, Any]) -> Optional[str]:
    """Infer tokenizer_type from YAML fields when it is not explicitly set."""
    explicit = backend.get("tokenizer_type")
    if explicit:
        return str(explicit)
    # GPT-2 BPE tokenizer is uniquely identified by the presence of both vocab and merge files.
    if backend.get("vocab_file") and backend.get("merge_file"):
        return "GPT2BPETokenizer"
    return None


def _build_preinit_defaults_from_experiment_config() -> Dict[str, Any]:
    """Load the minimal set of YAML fields needed before initialize_megatron.

    Architecture args come from the checkpoint via use_checkpoint_args.
    Only the fields below are pulled from the experiment config:
      ckpt_format, vocab_file, merge_file, legacy_tokenizer, bias_dropout_fusion.
    """
    config_path = _extract_flag_value(sys.argv[1:], "--moe-experiment-config")
    if not config_path:
        return {}
    cli_flags = _collect_cli_flags(sys.argv[1:])

    config = _load_yaml_file(config_path)
    backend = _deep_get(config, ["backend", "megatron"])
    if not isinstance(backend, dict):
        raise ValueError(
            "Expected backend.megatron mapping in --moe-experiment-config YAML"
        )

    defaults: Dict[str, Any] = {}

    def add_default(arg_name: str, key: str, transform=None) -> None:
        value = backend.get(key)
        if value is None:
            return
        value = _resolve_backend_value(value, backend)
        defaults[arg_name] = transform(value) if transform else value

    for key in ["vocab_file", "merge_file", "ckpt_format"]:
        add_default(key, key, str)
    for key in ["legacy_tokenizer", "bias_dropout_fusion"]:
        add_default(key, key, bool)

    # Infer tokenizer_type from vocab/merge files when not explicitly set.
    if "--tokenizer-type" not in cli_flags and backend.get("tokenizer_type") is None:
        inferred = _infer_tokenizer_type(defaults)
        if inferred:
            defaults["tokenizer_type"] = inferred

    return defaults


def _apply_experiment_config_defaults(args) -> None:
    """Apply post-init YAML defaults.

    Only the same minimal fields set during pre-init are reapplied here (in
    case use_checkpoint_args overwrote them), plus enabling use_checkpoint_args
    so all architecture fields are sourced from the checkpoint.
    """
    config_path = getattr(args, "moe_experiment_config", None)
    if not config_path:
        return

    config = _load_yaml_file(config_path)
    backend = _deep_get(config, ["backend", "megatron"])
    if not isinstance(backend, dict):
        raise ValueError(
            "Expected backend.megatron mapping in --moe-experiment-config YAML"
        )

    cli_flags = _collect_cli_flags(sys.argv[1:])

    def maybe_set(flag: str, attr: str, value: Any, transform=None) -> None:
        if value is None or flag in cli_flags:
            return
        value = _resolve_backend_value(value, backend)
        if transform is not None:
            value = transform(value)
        setattr(args, attr, value)

    maybe_set("--ckpt-format", "ckpt_format", backend.get("ckpt_format"), str)
    maybe_set("--vocab-file", "vocab_file", backend.get("vocab_file"), str)
    maybe_set("--merge-file", "merge_file", backend.get("merge_file"), str)

    if "--legacy-tokenizer" not in cli_flags and backend.get("legacy_tokenizer") is not None:
        args.legacy_tokenizer = bool(_resolve_backend_value(backend.get("legacy_tokenizer"), backend))
    if "--no-bias-dropout-fusion" not in cli_flags and backend.get("bias_dropout_fusion") is not None:
        args.bias_dropout_fusion = bool(_resolve_backend_value(backend.get("bias_dropout_fusion"), backend))

    # Architecture comes from checkpoint; enable unless caller already set this.
    if "--use-checkpoint-args" not in cli_flags:
        args.use_checkpoint_args = True


# -----------------------------------------------------------------------------
# Batch prep and co-activation post-processing
# -----------------------------------------------------------------------------
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
    # Prebuild token batches once so all checkpoints are evaluated on identical inputs.
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


def _normalize_coactivation_matrix(
    matrix: torch.Tensor,
    single_counts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Normalize co-activation counts to conditional rates."""
    matrix = matrix.clone().float()
    if single_counts is None:
        raise ValueError(
            "single_counts must be provided for pairwise-adjacent co-activation"
        )
    matrix_normalized = matrix / single_counts.clamp(min=1.0).unsqueeze(1)
    # Drop diagonal to focus on cross-expert interactions only.
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
        # Greedy pair-first fill: add both endpoints of the strongest pair.
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
    # If pairs are exhausted, backfill by strongest remaining single-expert association.
    fill_order = torch.argsort(max_per_expert, descending=True).tolist()
    for expert in fill_order:
        if expert not in selected:
            selected.append(expert)
        if len(selected) >= top_n:
            break
    return selected[:top_n]


def _collect_final_routing_and_coactivation(
    model,
    token_batches: List[torch.Tensor],
    eod_token: int,
    pad_token: int,
    reset_position_ids: bool,
    reset_attention_mask: bool,
    eod_mask_loss: bool,
    pad_mask_loss: bool,
) -> Tuple[
    Dict[str, List[torch.Tensor]],
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
    Dict[str, int],
]:
    routing_maps: Dict[str, List[torch.Tensor]] = {}
    coactivation_sums: Dict[str, torch.Tensor] = {}
    single_counts_sums: Dict[str, torch.Tensor] = {}
    topk_by_layer: Dict[str, int] = {}

    # Hook every TopK router once and accumulate final-checkpoint routing/co-activation stats.
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
                _probs = _probs.detach()
                if padding_mask is not None:
                    padding_mask = padding_mask.reshape(-1)
                    routing_map = routing_map[~padding_mask]
                    _probs = _probs[~padding_mask]
                topk_by_layer[layer_name] = mod.topk
                routing_maps[layer_name].append(
                    routing_map.to(dtype=torch.bool, device="cpu")
                )

                num_experts = _probs.shape[-1]
                topk = min(mod.topk, num_experts)
                if topk > 1:
                    # OLMoE-style adjacent pair counting from ordered top-k indices.
                    topk_indices = torch.topk(_probs, k=topk, dim=-1).indices
                    single_counts = torch.bincount(
                        topk_indices.reshape(-1), minlength=num_experts
                    ).float()
                    a = topk_indices[:, :-1].reshape(-1)
                    b = topk_indices[:, 1:].reshape(-1)
                    coactivation = torch.zeros(
                        (num_experts, num_experts),
                        device=topk_indices.device,
                        dtype=torch.float32,
                    )
                    ones = torch.ones_like(a, dtype=torch.float32)
                    coactivation.index_put_((a, b), ones, accumulate=True)
                    coactivation.index_put_((b, a), ones, accumulate=True)
                else:
                    single_counts = routing_map.float().sum(dim=0)
                    coactivation = torch.zeros(
                        (num_experts, num_experts),
                        device=routing_map.device,
                        dtype=torch.float32,
                    )

                if layer_name not in coactivation_sums:
                    # First batch initializes layer accumulators.
                    coactivation_sums[layer_name] = coactivation.detach().cpu()
                    single_counts_sums[layer_name] = single_counts.detach().cpu()
                else:
                    # Subsequent batches are additive.
                    coactivation_sums[layer_name] += coactivation.detach().cpu()
                    single_counts_sums[layer_name] += single_counts.detach().cpu()

            hooks.append(module.register_forward_hook(_hook))

    with torch.no_grad():
        for tokens in _progress_iter(token_batches, "Evaluating final checkpoint"):
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

    return routing_maps, coactivation_sums, single_counts_sums, topk_by_layer


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
    # Mutable index shared with hooks to align current batch against final-routing reference.
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
        for tokens in _progress_iter(token_batches, "Comparing checkpoint"):
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


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------
def main() -> None:
    global _MOE_COLOR_OUTPUT, _MOE_FORCE_COLOR
    raw_argv = sys.argv[1:]
    quiet_init = _preparse_bool_flag(
        raw_argv,
        "--moe-quiet-init",
        "--no-moe-quiet-init",
        default=True,
    )
    suppress_warnings = _preparse_bool_flag(
        raw_argv,
        "--moe-suppress-warnings",
        "--no-moe-suppress-warnings",
        default=True,
    )
    silence_megatron_logging = _preparse_bool_flag(
        raw_argv,
        "--moe-silence-megatron-logging",
        "--no-moe-silence-megatron-logging",
        default=True,
    )
    if suppress_warnings:
        warnings.filterwarnings("ignore")
    _configure_logging_suppression(silence_megatron_logging)

    # Safe defaults for a metrics-only run (no optimizer/rng restoration required).
    init_defaults = {
        # Change these defaults if necessary
        "no_load_rng": True,
        "no_load_optim": True,
        "micro_batch_size": 1,
        "exit_on_missing_checkpoint": True,
        "enable_msc": False,
        "use_cpu_initialization": True,
        "standalone_embedding_stage": False,
    }
    init_defaults.update(_build_preinit_defaults_from_experiment_config())

    with _suppress_output(quiet_init):
        initialize_megatron(
            extra_args_provider=add_metrics_args,
            args_defaults=init_defaults,
        )
    args = get_args()
    _MOE_COLOR_OUTPUT = args.moe_color_output
    _MOE_FORCE_COLOR = args.moe_force_color
    _apply_experiment_config_defaults(args)

    _log_stage("Configuration")
    cli_flags = _collect_cli_flags(raw_argv)
    num_batches_provided = "--moe-num-batches" in cli_flags

    if args.moe_batch_size <= 0:
        raise ValueError("--moe-batch-size must be > 0")
    if num_batches_provided and (args.moe_num_batches is None or args.moe_num_batches <= 0):
        raise ValueError("--moe-num-batches must be > 0 when provided")

    _log_info(
        f"data_type={args.moe_data_type} batches={args.moe_num_batches} batch_size={args.moe_batch_size}"
    )

    if args.pipeline_model_parallel_size > 1:
        raise RuntimeError("Pipeline parallelism >1 is not supported in this script.")

    load_dir = getattr(args, "load", None)
    if not load_dir or not checkpoint_exists(load_dir):
        raise RuntimeError(
            "No checkpoint found. Provide --load pointing to a valid checkpoint directory."
        )

    # Build model in eval mode once; checkpoints are swapped into this model in-place.
    ddp_model = get_model(partial(model_provider, gpt_builder), wrap_with_ddp=False)
    model = ddp_model[0]
    model.eval()
    _log_success("Model initialized")

    # Tokenizer is only needed for prompt/HF dataset modes.
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

    # Resolve text source according to mode and fallback rules.
    prompts = []

    if args.moe_data_type == "prompts" or (args.moe_prompts_file and not use_random):
        prompts = _load_prompts(args.moe_prompts_file)
        if not prompts:
            _log_warn("No prompts loaded; falling back to random inputs")
            use_random = True

    if args.moe_data_type == "hf_dataset" and not use_random:
        if not args.moe_dataset_name and not args.moe_dataset_path:
            _log_warn(
                "Either --moe-dataset-name or --moe-dataset-path is required for --moe-data-type hf_dataset"
            )
            use_random = True
        else:
            planned_samples = (
                args.moe_num_batches * args.moe_batch_size
                if num_batches_provided
                else 1
            )
            prompts = _load_hf_dataset_texts(
                dataset_path=args.moe_dataset_path,
                dataset_name=args.moe_dataset_name,
                dataset_config=args.moe_dataset_config,
                split=args.moe_dataset_split,
                text_field=args.moe_dataset_text_field,
                percentage=args.moe_dataset_percentage,
                cache_dir=args.moe_dataset_cache_dir,
                num_samples=planned_samples,
                seed=args.moe_seed,
            )
            if not prompts:
                use_random = True

    if use_random:
        if args.moe_num_batches is None:
            raise ValueError(
                "Provide --moe-num-batches when using random/prompt fallback inputs. "
                "Automatic inference only applies to successfully loaded hf_dataset texts."
            )
        effective_num_batches = args.moe_num_batches
    elif args.moe_data_type == "hf_dataset" and not num_batches_provided:
        inferred = len(prompts) // args.moe_batch_size
        if inferred <= 0:
            raise ValueError(
                "Not enough sampled texts to form one batch. Increase --moe-dataset-percentage or reduce --moe-batch-size."
            )
        effective_num_batches = inferred
        _log_info(
            f"Inferred moe_num_batches={effective_num_batches} from sampled texts={len(prompts)} and batch_size={args.moe_batch_size}"
        )
    else:
        effective_num_batches = args.moe_num_batches

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if use_random:
        eod_token = 0
        vocab_size = args.padded_vocab_size
        _log_info("Using random inputs")
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

    # Build once so all compared checkpoints see exactly the same token inputs.
    token_batches = _build_token_batches(
        prompts,
        tokenizer,
        args.moe_batch_size,
        seq_length,
        pad_id,
        vocab_size,
        effective_num_batches,
        use_random,
        device,
        args.moe_seed,
    )
    _log_info(f"Prepared {len(token_batches)} token batches (seq_length={seq_length})")

    args.ckpt_step = None if is_release else final_step
    _log_stage("Final Checkpoint Evaluation")
    with _suppress_output(args.moe_quiet_init):
        load_checkpoint(ddp_model, None, None, strict=False)
    final_routing_maps, coactivation_sums, single_counts_sums, topk_by_layer = (
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

    # Compare each requested checkpoint against routing captured from the final checkpoint.
    saturation_results = {
        "final_checkpoint": "release" if is_release else final_step,
        "compare_steps": compare_steps,
        "layers": {},
    }
    _log_stage("Checkpoint Comparisons")
    for step in _progress_iter(compare_steps, "Checkpoint comparisons"):
        args.ckpt_step = step
        with _suppress_output(args.moe_quiet_init):
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

    _log_stage("Results")
    if args.moe_print_json:
        print_rank_0(json.dumps(saturation_results, indent=2, sort_keys=True))
    else:
        _log_info(
            f"Computed saturation comparisons for {len(saturation_results['layers'])} checkpoint(s)"
        )
    # Persist saturation summary.
    if args.moe_output_json:
        output_path = Path(args.moe_output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(saturation_results, indent=2, sort_keys=True))
        _log_success(f"Wrote saturation JSON: {output_path}")
        default_plot_dir = output_path.parent
    else:
        default_plot_dir = Path.cwd()

    # Persist co-activation matrices for selected experts.
    if args.moe_coactivation_output:
        top_n = max(args.moe_coactivation_top_n, 1)
        coactivation_summary = {
            "final_checkpoint": "release" if is_release else final_step,
            "coactivation_mode": "pairwise-adjacent",
            "layers": {},
        }
        for layer_name, matrix in coactivation_sums.items():
            matrix_normalized = _normalize_coactivation_matrix(
                matrix,
                single_counts_sums.get(layer_name),
            )

            top_indices = _select_top_experts_pairwise(matrix_normalized, top_n)
            submatrix = matrix_normalized[top_indices][:, top_indices]
            coactivation_summary["layers"][layer_name] = {
                "experts": top_indices,
                "matrix": (submatrix * 100.0).tolist(),
            }

        output_path = Path(args.moe_coactivation_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(coactivation_summary, indent=2, sort_keys=True)
        )
        _log_success(f"Wrote coactivation JSON: {output_path}")

    # Optional plotting for both checkpoint-comparison curves and co-activation heatmaps.
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
            _log_success(f"Wrote plot: {plot_dir / 'router_saturation_vs_final.png'}")

        if args.moe_coactivation_output:
            for layer_name, layer_info in coactivation_summary["layers"].items():
                matrix = torch.tensor(layer_info["matrix"], dtype=torch.float32)
                experts = layer_info.get("experts", list(range(matrix.shape[0])))
                plt.figure(figsize=(5, 4))
                plt.imshow(matrix, cmap="RdPu", aspect="auto", vmin=0, vmax=60)
                plt.colorbar(ticks=[0, 15, 30, 45, 60])
                plt.title(f"Co-activation: {layer_name}")
                plt.xlabel("Expert")
                plt.ylabel("Expert")
                tick_positions = list(range(len(experts)))
                tick_labels = [str(idx) for idx in experts]
                plt.xticks(tick_positions, tick_labels, rotation=90)
                plt.yticks(tick_positions, tick_labels)
                plt.tight_layout()
                safe_name = layer_name.replace("/", "_")
                plt.savefig(plot_dir / f"coactivation_{safe_name}.png", dpi=200)
                plt.close()
            _log_success(f"Wrote coactivation plots in: {plot_dir}")


if __name__ == "__main__":
    main()
