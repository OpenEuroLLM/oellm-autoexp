#!/usr/bin/env python3
"""Scan Megatron-LM ``torch_dist`` checkpoints for parameter/optimizer anomalies.

The scanner is CPU-only and reads one distributed-checkpoint tensor at a time.
It is intended for comparing checkpoints around a training instability without
constructing the model or optimizer.

Outputs
-------
``tensor_stats.csv``
    Per tensor/layer scalar statistics: finite/zero fractions, mean, standard
    deviation, RMS, extrema, sampled absolute quantiles, aggregate change from
    the first checkpoint, and deterministic sampled elementwise drift from the
    first checkpoint (delta RMS, relative delta, cosine, and sign flips).
``channel_stats.csv``
    Distribution of row/input-column RMS values (or absolute values for 1-D
    gains), including fractions below a configurable within-tensor threshold
    and ratios to the first checkpoint.
``channel_outliers.csv``
    The lowest/highest channels in every tensor plus the channels that shrank
    or grew most relative to the first checkpoint.
``metadata.csv``
    Selected checkpoint keys, shapes, dtypes, and estimated load sizes.
``skipped.csv``
    Unsupported, oversized, or failed tensors and the reason.

The first selected checkpoint is the comparison baseline.  A "shut down"
matrix channel is operationally defined as a row or column whose RMS is much
smaller than the median channel RMS for the same tensor and layer.  The default
threshold is 1%; change it with ``--dead-ratio``.

Optimizer formats
-----------------
Fully reshardable Megatron checkpoints expose named keys such as
``optimizer.state.exp_avg.<parameter>``.  Those receive the same per-layer and
per-channel analysis as weights.

The dense-32B production configuration currently uses ``dp_reshardable``.
That format stores optimizer states as flat gradient-buffer buckets, without a
recoverable parameter/channel name in DCP metadata.  For those checkpoints this
script reports exact per-bucket scalar health (non-finite values, negative
``exp_avg_sq``, zeros, RMS, extrema, and checkpoint-to-checkpoint drift), but it
does not pretend that bucket offsets are named model channels.  Named optimizer
channel analysis requires a fully reshardable checkpoint or model-aware loading.

Examples
--------
Scan selected checkpoints under a checkpoint root::

    apptainer exec <training.sif> python3 scripts/scan_checkpoint_stats.py \\
      /e/scratch/.../checkpoints \\
      --iterations 60000,64000,68000,72000,75126 \\
      --output-dir checkpoint_stats

Start with MLP/norm model tensors::

    apptainer exec <training.sif> python3 scripts/scan_checkpoint_stats.py \\
      /e/scratch/.../checkpoints \\
      --iterations 64000,68000,72000 \\
      --include 'mlp|layernorm' \\
      --optimizer off \\
      --output-dir checkpoint_stats_mlp

Inspect keys and estimated I/O without loading tensors::

    apptainer exec <training.sif> python3 scripts/scan_checkpoint_stats.py \
      /e/scratch/.../checkpoints/iter_0064000 --metadata-only

Some older checkpoints pickle Megatron/Transformer Engine metadata.  Run this
inside the matching training container.  On a login node without libcuda, use
the same stub-libcuda workaround documented in ``scripts/scan_weight_stats.py``.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence


ITER_RE = re.compile(r"iter_(\d+)")
EXPLICIT_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
STACKED_LAYER_RE = re.compile(r"(?:^|\.)layers\.(?!\d+\.)")
PARAMETER_NAME_RE = re.compile(r"(?:weight|bias)$")

KNOWN_OPTIMIZER_STATES = (
    "exp_avg_sq",
    "momentum_buffer",
    "master_param",
    "fp32_param",
    "exp_avg",
    "param",
)

TENSOR_FIELDS = [
    "run",
    "checkpoint",
    "iteration",
    "kind",
    "state",
    "tensor",
    "storage_key",
    "layer",
    "shape",
    "dtype",
    "numel",
    "sample_numel",
    "comparison_sample_numel",
    "finite_frac",
    "zero_frac",
    "negative_frac",
    "min",
    "max",
    "mean",
    "std",
    "rms",
    "abs_mean",
    "abs_max",
    "abs_q001",
    "abs_q01",
    "abs_q05",
    "abs_q50",
    "abs_q95",
    "abs_q99",
    "abs_q999",
    "rms_vs_baseline",
    "abs_max_vs_baseline",
    "sample_delta_rms",
    "sample_relative_delta_rms",
    "sample_cosine",
    "sample_sign_flip_frac",
]

CHANNEL_FIELDS = [
    "run",
    "checkpoint",
    "iteration",
    "kind",
    "state",
    "tensor",
    "storage_key",
    "layer",
    "axis",
    "metric",
    "channels",
    "values_per_channel",
    "finite_channel_frac",
    "zero_channel_frac",
    "min",
    "q001",
    "q01",
    "q05",
    "median",
    "q95",
    "q99",
    "q999",
    "max",
    "min_to_median",
    "frac_below_dead_ratio",
    "dead_ratio",
    "median_vs_baseline",
    "ratio_q001",
    "ratio_q01",
    "ratio_median",
    "ratio_q99",
    "ratio_q999",
    "frac_below_0_1x_baseline",
    "frac_below_0_5x_baseline",
    "frac_above_2x_baseline",
    "frac_above_10x_baseline",
]

OUTLIER_FIELDS = [
    "run",
    "checkpoint",
    "iteration",
    "kind",
    "state",
    "tensor",
    "storage_key",
    "layer",
    "axis",
    "metric",
    "direction",
    "rank",
    "channel",
    "magnitude",
    "median",
    "ratio_to_median",
    "baseline_magnitude",
    "ratio_to_baseline",
]

METADATA_FIELDS = [
    "checkpoint",
    "iteration",
    "kind",
    "state",
    "tensor",
    "storage_key",
    "shape",
    "dtype",
    "estimated_gib",
    "optimizer_layout",
]

SKIPPED_FIELDS = ["checkpoint", "iteration", "storage_key", "reason", "detail"]


@dataclass(frozen=True)
class TensorDescriptor:
    storage_key: str
    kind: str
    state: str
    tensor: str
    shape: tuple[int, ...]
    dtype: Any
    estimated_bytes: int
    optimizer_layout: str = ""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "checkpoint",
        nargs="+",
        type=Path,
        help="iter_* directory, or one or more roots containing iter_* directories",
    )
    parser.add_argument("--run", default="", help="optional run label written to CSV")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoint_stats"),
        help="output directory (default: ./checkpoint_stats)",
    )
    parser.add_argument(
        "--iterations",
        help="comma-separated exact iterations to scan, e.g. 60000,64000,68000",
    )
    parser.add_argument("--first", type=int, default=None)
    parser.add_argument("--last", type=int, default=None)
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        metavar="REGEX",
        help="include keys matching any supplied regex (repeatable)",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="REGEX",
        help="exclude keys matching any supplied regex (repeatable)",
    )
    parser.add_argument(
        "--optimizer",
        choices=("auto", "on", "off"),
        default="auto",
        help="scan optimizer moments when present (default: auto)",
    )
    parser.add_argument(
        "--optimizer-states",
        default="exp_avg,exp_avg_sq",
        help="comma-separated optimizer tensor states to load",
    )
    parser.add_argument(
        "--all-model-tensors",
        action="store_true",
        help="include non-parameter model tensors as well as weights/biases",
    )
    parser.add_argument(
        "--no-channels",
        action="store_true",
        help="skip row/column channel analysis",
    )
    parser.add_argument(
        "--dead-ratio",
        type=float,
        default=0.01,
        help="flag channels below this fraction of their peer median (default: 0.01)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="number of low/high channel outliers to retain per axis (default: 8)",
    )
    parser.add_argument(
        "--sample-elements",
        type=int,
        default=250_000,
        help="deterministic sample size for whole-tensor quantiles (default: 250000)",
    )
    parser.add_argument(
        "--comparison-sample-elements",
        type=int,
        default=16_384,
        help=(
            "deterministic elements retained per logical tensor for elementwise "
            "comparison with the first checkpoint; 0 disables (default: 16384)"
        ),
    )
    parser.add_argument(
        "--max-tensor-gib",
        type=float,
        default=0.0,
        help="skip individual tensors larger than this; 0 disables the guard",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="list selected keys and estimated sizes without loading tensor data",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="stop on the first unreadable tensor instead of recording it in skipped.csv",
    )
    parser.add_argument(
        "--fail-on-nonfinite",
        action="store_true",
        help="exit nonzero after writing reports if any loaded tensor is non-finite",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing report CSVs in --output-dir",
    )
    args = parser.parse_args(argv)
    if args.dead_ratio <= 0:
        parser.error("--dead-ratio must be positive")
    if args.top_k < 0:
        parser.error("--top-k must be non-negative")
    if args.sample_elements < 1:
        parser.error("--sample-elements must be at least 1")
    if args.comparison_sample_elements < 0:
        parser.error("--comparison-sample-elements must be non-negative")
    return args


def checkpoint_iteration(path: Path) -> int:
    match = ITER_RE.search(path.name)
    if not match:
        raise ValueError(f"checkpoint directory is not named iter_<N>: {path}")
    return int(match.group(1))


def discover_checkpoints(
    paths: Sequence[Path], selected: set[int] | None, first: int | None, last: int | None
) -> list[Path]:
    found: dict[Path, int] = {}
    for raw in paths:
        path = raw.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        candidates = [path] if ITER_RE.search(path.name) else list(path.glob("iter_*"))
        for candidate in candidates:
            if not candidate.is_dir() or not ITER_RE.search(candidate.name):
                continue
            iteration = checkpoint_iteration(candidate)
            if selected is not None and iteration not in selected:
                continue
            if first is not None and iteration < first:
                continue
            if last is not None and iteration > last:
                continue
            found[candidate] = iteration
    checkpoints = sorted(found, key=lambda p: (found[p], str(p)))
    if not checkpoints:
        raise ValueError("no matching iter_* checkpoint directories found")
    return checkpoints


def parse_optimizer_key(key: str) -> tuple[str, str, str] | None:
    """Return ``(state, parameter-or-bucket-name, layout)`` for optimizer tensors."""
    if "optimizer" not in key:
        return None
    for state in KNOWN_OPTIMIZER_STATES:
        marker = f".state.{state}."
        if marker in key:
            return state, key.split(marker, 1)[1], "named_parameter"
        prefix = f"optimizer.state.{state}."
        if key.startswith(prefix):
            return state, key[len(prefix) :], "named_parameter"
    for state in KNOWN_OPTIMIZER_STATES:
        suffix = f".{state}"
        if key.endswith(suffix):
            return state, key[: -len(suffix)], "bucket"
    return None


def is_parameter_key(key: str) -> bool:
    if "_extra_state" in key:
        return False
    return PARAMETER_NAME_RE.search(key) is not None


def dtype_nbytes(torch_module: Any, dtype: Any) -> int:
    return torch_module.empty((), dtype=dtype).element_size()


def describe_tensor(
    key: str, metadata: Any, torch_module: Any, all_model_tensors: bool
) -> TensorDescriptor | None:
    if not hasattr(metadata, "size") or not hasattr(metadata, "properties"):
        return None
    shape = tuple(int(v) for v in metadata.size)
    dtype = metadata.properties.dtype
    estimated_bytes = math.prod(shape) * dtype_nbytes(torch_module, dtype)
    optimizer = parse_optimizer_key(key)
    if optimizer is not None:
        state, tensor, layout = optimizer
        return TensorDescriptor(
            key, "optimizer", state, tensor, shape, dtype, estimated_bytes, layout
        )
    if is_parameter_key(key):
        return TensorDescriptor(key, "model", "weight", key, shape, dtype, estimated_bytes)
    if all_model_tensors and "_extra_state" not in key:
        return TensorDescriptor(key, "model_aux", "tensor", key, shape, dtype, estimated_bytes)
    return None


def compile_patterns(patterns: Sequence[str]) -> list[re.Pattern[str]]:
    return [re.compile(pattern) for pattern in patterns]


def descriptor_selected(
    descriptor: TensorDescriptor,
    include: Sequence[re.Pattern[str]],
    exclude: Sequence[re.Pattern[str]],
    optimizer_mode: str,
    optimizer_states: set[str],
) -> bool:
    searchable = f"{descriptor.storage_key}\n{descriptor.tensor}"
    if include and not any(pattern.search(searchable) for pattern in include):
        return False
    if any(pattern.search(searchable) for pattern in exclude):
        return False
    if descriptor.kind == "optimizer":
        return optimizer_mode != "off" and descriptor.state in optimizer_states
    return True


def logical_slices(key: str, tensor: Any) -> Iterator[tuple[int, Any]]:
    """Yield logical per-layer tensors from Megatron's homogeneous layer stacks."""
    explicit = EXPLICIT_LAYER_RE.search(key)
    if explicit:
        yield int(explicit.group(1)), tensor
    elif STACKED_LAYER_RE.search(key) and tensor.ndim >= 2:
        for layer in range(tensor.shape[0]):
            yield layer, tensor[layer]
    else:
        yield -1, tensor


def deterministic_sample(flat: Any, limit: int) -> Any:
    if flat.numel() <= limit:
        return flat
    stride = math.ceil(flat.numel() / limit)
    return flat[::stride][:limit]


def quantiles(values: Any, probabilities: Sequence[float], torch_module: Any) -> list[float]:
    if values.numel() == 0:
        return [float("nan")] * len(probabilities)
    q = torch_module.tensor(probabilities, dtype=torch_module.float32)
    return [float(value) for value in torch_module.quantile(values.float(), q).tolist()]


def scalar_stats(tensor: Any, sample_elements: int, torch_module: Any) -> dict[str, Any]:
    values = tensor.detach().float().reshape(-1)
    numel = values.numel()
    finite_mask = torch_module.isfinite(values)
    finite_count = int(finite_mask.sum().item())
    zero_count = int((values == 0).sum().item())
    negative_count = int((values < 0).sum().item())
    finite = values if finite_count == numel else values[finite_mask]

    if finite_count:
        mean = float((finite.sum(dtype=torch_module.float64) / finite_count).item())
        norm = float(
            torch_module.linalg.vector_norm(finite, ord=2, dtype=torch_module.float64).item()
        )
        rms = norm / math.sqrt(finite_count)
        std = math.sqrt(max(rms * rms - mean * mean, 0.0))
        abs_values = finite.abs()
        abs_mean = float(
            (abs_values.sum(dtype=torch_module.float64) / finite_count).item()
        )
        abs_max = float(abs_values.max().item())
        minimum = float(finite.min().item())
        maximum = float(finite.max().item())
        sample = deterministic_sample(abs_values, sample_elements)
        q001, q01, q05, q50, q95, q99, q999 = quantiles(
            sample, (0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999), torch_module
        )
        sample_numel = sample.numel()
    else:
        mean = std = rms = abs_mean = abs_max = minimum = maximum = float("nan")
        q001 = q01 = q05 = q50 = q95 = q99 = q999 = float("nan")
        sample_numel = 0

    return {
        "numel": numel,
        "sample_numel": sample_numel,
        "finite_frac": finite_count / numel if numel else float("nan"),
        "zero_frac": zero_count / numel if numel else float("nan"),
        "negative_frac": negative_count / numel if numel else float("nan"),
        "min": minimum,
        "max": maximum,
        "mean": mean,
        "std": std,
        "rms": rms,
        "abs_mean": abs_mean,
        "abs_max": abs_max,
        "abs_q001": q001,
        "abs_q01": q01,
        "abs_q05": q05,
        "abs_q50": q50,
        "abs_q95": q95,
        "abs_q99": q99,
        "abs_q999": q999,
    }


def sampled_elementwise_change(
    tensor: Any,
    baseline: Any | None,
    sample_elements: int,
    torch_module: Any,
) -> tuple[dict[str, Any], Any | None]:
    """Compare the same deterministic element sample with the first checkpoint.

    Full tensors can be tens of GiB, so retaining a baseline copy for every
    parameter is impractical.  A fixed-stride sample preserves element identity
    across equal-shaped checkpoints and catches directional changes that scalar
    RMS/quantiles cannot, while keeping baseline memory bounded.
    """
    empty = {
        "comparison_sample_numel": 0,
        "sample_delta_rms": float("nan"),
        "sample_relative_delta_rms": float("nan"),
        "sample_cosine": float("nan"),
        "sample_sign_flip_frac": float("nan"),
    }
    if sample_elements == 0:
        return empty, None

    current = deterministic_sample(
        tensor.detach().float().reshape(-1), sample_elements
    ).cpu().clone()
    result = {**empty, "comparison_sample_numel": current.numel()}
    if baseline is None:
        result.update(
            {
                "sample_delta_rms": 0.0,
                "sample_relative_delta_rms": 0.0,
                "sample_cosine": 1.0,
                "sample_sign_flip_frac": 0.0,
            }
        )
        return result, current
    if tuple(baseline.shape) != tuple(current.shape):
        return result, current

    finite = torch_module.isfinite(current) & torch_module.isfinite(baseline)
    current_finite = current[finite].double()
    baseline_finite = baseline[finite].double()
    if current_finite.numel() == 0:
        return result, current

    delta = current_finite - baseline_finite
    delta_rms = float(delta.square().mean().sqrt().item())
    baseline_rms = float(baseline_finite.square().mean().sqrt().item())
    current_norm = float(torch_module.linalg.vector_norm(current_finite).item())
    baseline_norm = float(torch_module.linalg.vector_norm(baseline_finite).item())
    cosine = (
        float(torch_module.dot(current_finite, baseline_finite).item())
        / (current_norm * baseline_norm)
        if current_norm and baseline_norm
        else float("nan")
    )
    nonzero = (current_finite != 0) & (baseline_finite != 0)
    sign_flip = (
        float((torch_module.sign(current_finite[nonzero]) != torch_module.sign(
            baseline_finite[nonzero]
        )).float().mean().item())
        if bool(nonzero.any().item())
        else 0.0
    )
    result.update(
        {
            "sample_delta_rms": delta_rms,
            "sample_relative_delta_rms": safe_ratio(delta_rms, baseline_rms),
            "sample_cosine": cosine,
            "sample_sign_flip_frac": sign_flip,
        }
    )
    return result, current


def channel_magnitudes(
    tensor: Any, state: str, torch_module: Any
) -> Iterator[tuple[str, str, int, Any]]:
    """Yield ``(axis, metric name, values/channel, magnitude vector)``."""
    values = tensor.detach().float()
    if values.ndim == 0:
        return

    def reduce_for_axis(axis: int) -> tuple[Any, int]:
        reduce_dims = tuple(dim for dim in range(values.ndim) if dim != axis)
        values_per_channel = math.prod(values.shape[dim] for dim in reduce_dims)
        finite = torch_module.isfinite(values)
        safe = values if bool(finite.all().item()) else torch_module.where(
            finite, values, torch_module.zeros_like(values)
        )
        if state == "exp_avg_sq":
            magnitude = safe.clamp_min(0).sum(dim=reduce_dims) / values_per_channel
            magnitude = magnitude.sqrt()
        else:
            magnitude = (safe.square().sum(dim=reduce_dims) / values_per_channel).sqrt()
        has_nonfinite = (~finite).any(dim=reduce_dims)
        magnitude[has_nonfinite] = float("inf")
        return magnitude.reshape(-1), values_per_channel

    if values.ndim == 1:
        finite = torch_module.isfinite(values)
        if state == "exp_avg_sq":
            magnitude = values.clamp_min(0).sqrt()
            metric = "sqrt_value"
        else:
            magnitude = values.abs()
            metric = "abs_value"
        magnitude = magnitude.clone()
        magnitude[~finite] = float("inf")
        yield "element", metric, 1, magnitude
        return

    output, output_width = reduce_for_axis(values.ndim - 2)
    metric = "sqrt_mean_exp_avg_sq" if state == "exp_avg_sq" else "rms"
    yield "output_row", metric, output_width, output
    inputs, input_width = reduce_for_axis(values.ndim - 1)
    yield "input_column", metric, input_width, inputs


def safe_ratio(value: float, baseline: float) -> float:
    if not math.isfinite(value) or not math.isfinite(baseline):
        return float("nan")
    if baseline == 0:
        return 1.0 if value == 0 else float("inf")
    return value / baseline


def ratio_vector(current: Any, baseline: Any, torch_module: Any) -> Any:
    current = current.float()
    baseline = baseline.float()
    ratio = torch_module.empty_like(current)
    baseline_zero = baseline == 0
    ratio[~baseline_zero] = current[~baseline_zero] / baseline[~baseline_zero]
    ratio[baseline_zero & (current == 0)] = 1.0
    ratio[baseline_zero & (current != 0)] = float("inf")
    invalid = ~torch_module.isfinite(current) | ~torch_module.isfinite(baseline)
    ratio[invalid] = float("nan")
    return ratio


def channel_summary(
    magnitudes: Any,
    baseline: Any | None,
    dead_ratio: float,
    torch_module: Any,
) -> dict[str, Any]:
    finite_mask = torch_module.isfinite(magnitudes)
    finite = magnitudes[finite_mask]
    channels = magnitudes.numel()
    if finite.numel():
        q001, q01, q05, median, q95, q99, q999 = quantiles(
            finite, (0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999), torch_module
        )
        minimum = float(finite.min().item())
        maximum = float(finite.max().item())
        zero_channel_frac = float((finite == 0).sum().item()) / channels
        min_to_median = safe_ratio(minimum, median)
        frac_dead = float((finite < median * dead_ratio).sum().item()) / channels
    else:
        q001 = q01 = q05 = median = q95 = q99 = q999 = float("nan")
        minimum = maximum = zero_channel_frac = min_to_median = frac_dead = float("nan")

    result = {
        "channels": channels,
        "finite_channel_frac": float(finite_mask.sum().item()) / channels if channels else 0.0,
        "zero_channel_frac": zero_channel_frac,
        "min": minimum,
        "q001": q001,
        "q01": q01,
        "q05": q05,
        "median": median,
        "q95": q95,
        "q99": q99,
        "q999": q999,
        "max": maximum,
        "min_to_median": min_to_median,
        "frac_below_dead_ratio": frac_dead,
        "dead_ratio": dead_ratio,
        "median_vs_baseline": float("nan"),
        "ratio_q001": float("nan"),
        "ratio_q01": float("nan"),
        "ratio_median": float("nan"),
        "ratio_q99": float("nan"),
        "ratio_q999": float("nan"),
        "frac_below_0_1x_baseline": float("nan"),
        "frac_below_0_5x_baseline": float("nan"),
        "frac_above_2x_baseline": float("nan"),
        "frac_above_10x_baseline": float("nan"),
    }
    if baseline is None or tuple(baseline.shape) != tuple(magnitudes.shape):
        return result

    ratios = ratio_vector(magnitudes, baseline, torch_module)
    finite_ratios = ratios[torch_module.isfinite(ratios)]
    if finite_ratios.numel():
        rq001, rq01, rq50, rq99, rq999 = quantiles(
            finite_ratios, (0.001, 0.01, 0.5, 0.99, 0.999), torch_module
        )
        finite_baseline = baseline[torch_module.isfinite(baseline)]
        baseline_median = (
            quantiles(finite_baseline, (0.5,), torch_module)[0]
            if finite_baseline.numel()
            else float("nan")
        )
        result.update(
            {
                "median_vs_baseline": safe_ratio(median, baseline_median),
                "ratio_q001": rq001,
                "ratio_q01": rq01,
                "ratio_median": rq50,
                "ratio_q99": rq99,
                "ratio_q999": rq999,
                "frac_below_0_1x_baseline": float((finite_ratios < 0.1).sum().item())
                / channels,
                "frac_below_0_5x_baseline": float((finite_ratios < 0.5).sum().item())
                / channels,
                "frac_above_2x_baseline": float((finite_ratios > 2.0).sum().item())
                / channels,
                "frac_above_10x_baseline": float((finite_ratios > 10.0).sum().item())
                / channels,
            }
        )
    return result


def extreme_indices(values: Any, k: int, largest: bool, torch_module: Any) -> list[int]:
    if k == 0:
        return []
    finite_indices = torch_module.nonzero(torch_module.isfinite(values), as_tuple=False).reshape(-1)
    if finite_indices.numel() == 0:
        return []
    finite_values = values[finite_indices]
    count = min(k, finite_values.numel())
    selected = torch_module.topk(finite_values, count, largest=largest).indices
    return [int(index) for index in finite_indices[selected].tolist()]


def outlier_rows(
    base: dict[str, Any],
    magnitudes: Any,
    baseline: Any | None,
    median: float,
    top_k: int,
    torch_module: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    directions = [
        ("low_absolute", magnitudes, False),
        ("high_absolute", magnitudes, True),
    ]
    compatible_baseline = (
        baseline
        if baseline is not None and tuple(baseline.shape) == tuple(magnitudes.shape)
        else None
    )
    ratios = None
    if compatible_baseline is not None:
        ratios = ratio_vector(magnitudes, compatible_baseline, torch_module)
        directions.extend(
            [
                ("drop_vs_baseline", ratios, False),
                ("growth_vs_baseline", ratios, True),
            ]
        )

    for direction, ranking_values, largest in directions:
        for rank, index in enumerate(
            extreme_indices(ranking_values, top_k, largest, torch_module), start=1
        ):
            magnitude = float(magnitudes[index].item())
            baseline_magnitude = (
                float(compatible_baseline[index].item())
                if compatible_baseline is not None
                else float("nan")
            )
            rows.append(
                {
                    **base,
                    "direction": direction,
                    "rank": rank,
                    "channel": index,
                    "magnitude": magnitude,
                    "median": median,
                    "ratio_to_median": safe_ratio(magnitude, median),
                    "baseline_magnitude": baseline_magnitude,
                    "ratio_to_baseline": (
                        float(ratios[index].item()) if ratios is not None else float("nan")
                    ),
                }
            )
    return rows


def csv_value(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.9g}"
    return value


class CsvSink:
    def __init__(self, path: Path, fields: Sequence[str]):
        self.path = path
        self.fields = list(fields)
        self.handle = path.open("w", newline="")
        self.writer = csv.DictWriter(self.handle, fieldnames=self.fields)
        self.writer.writeheader()

    def write(self, row: dict[str, Any]) -> None:
        self.writer.writerow({field: csv_value(row.get(field, "")) for field in self.fields})
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def prepare_outputs(output_dir: Path, overwrite: bool) -> dict[str, CsvSink]:
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = {
        "tensor": ("tensor_stats.csv", TENSOR_FIELDS),
        "channel": ("channel_stats.csv", CHANNEL_FIELDS),
        "outlier": ("channel_outliers.csv", OUTLIER_FIELDS),
        "metadata": ("metadata.csv", METADATA_FIELDS),
        "skipped": ("skipped.csv", SKIPPED_FIELDS),
    }
    existing = [
        output_dir / filename
        for filename, _ in specs.values()
        if (output_dir / filename).exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            "report files already exist; choose another --output-dir or pass --overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    return {
        name: CsvSink(output_dir / filename, fields) for name, (filename, fields) in specs.items()
    }


def load_tensor(
    reader: Any, descriptor: TensorDescriptor, torch_module: Any, dcp_module: Any
) -> Any:
    state = {
        descriptor.storage_key: torch_module.empty(descriptor.shape, dtype=descriptor.dtype)
    }
    dcp_module.load(state, storage_reader=reader)
    return state.pop(descriptor.storage_key)


def common_fields(
    args: argparse.Namespace,
    checkpoint: Path,
    iteration: int,
    descriptor: TensorDescriptor,
    layer: int,
) -> dict[str, Any]:
    return {
        "run": args.run,
        "checkpoint": str(checkpoint),
        "iteration": iteration,
        "kind": descriptor.kind,
        "state": descriptor.state,
        "tensor": descriptor.tensor,
        "storage_key": descriptor.storage_key,
        "layer": layer,
    }


def scan_checkpoint(
    checkpoint: Path,
    args: argparse.Namespace,
    sinks: dict[str, CsvSink] | None,
    baseline_scalars: dict[tuple[str, str, str, int], dict[str, Any]],
    baseline_channels: dict[tuple[str, str, str, int, str], Any],
    baseline_samples: dict[tuple[str, str, str, int], Any],
    torch_module: Any,
    dcp_module: Any,
) -> dict[str, Any]:
    iteration = checkpoint_iteration(checkpoint)
    reader = dcp_module.FileSystemReader(checkpoint)
    raw_metadata = reader.read_metadata().state_dict_metadata
    include = compile_patterns(args.include)
    exclude = compile_patterns(args.exclude)
    optimizer_states = {
        state.strip() for state in args.optimizer_states.split(",") if state.strip()
    }

    descriptors = []
    for key, metadata in raw_metadata.items():
        descriptor = describe_tensor(str(key), metadata, torch_module, args.all_model_tensors)
        if descriptor and descriptor_selected(
            descriptor, include, exclude, args.optimizer, optimizer_states
        ):
            descriptors.append(descriptor)
    descriptors.sort(key=lambda d: (d.kind, d.tensor, d.state, d.storage_key))

    optimizer_layouts = sorted(
        {d.optimizer_layout for d in descriptors if d.kind == "optimizer"}
    )
    total_gib = sum(d.estimated_bytes for d in descriptors) / 2**30
    print(
        f"{checkpoint.name}: {len(descriptors)} tensors, {total_gib:.1f} GiB selected"
        + (f", optimizer={','.join(optimizer_layouts)}" if optimizer_layouts else "")
    )

    for descriptor in descriptors:
        metadata_row = {
            "checkpoint": str(checkpoint),
            "iteration": iteration,
            "kind": descriptor.kind,
            "state": descriptor.state,
            "tensor": descriptor.tensor,
            "storage_key": descriptor.storage_key,
            "shape": "x".join(str(v) for v in descriptor.shape),
            "dtype": str(descriptor.dtype),
            "estimated_gib": descriptor.estimated_bytes / 2**30,
            "optimizer_layout": descriptor.optimizer_layout,
        }
        if sinks is not None:
            sinks["metadata"].write(metadata_row)
        if args.metadata_only:
            print(
                f"  {descriptor.estimated_bytes / 2**30:8.3f} GiB  "
                f"{str(descriptor.dtype):>14}  {descriptor.shape!s:>22}  "
                f"{descriptor.kind}:{descriptor.state}  {descriptor.storage_key}"
            )

    if args.metadata_only:
        return {
            "iteration": iteration,
            "selected_tensors": len(descriptors),
            "selected_gib": total_gib,
            "loaded_tensors": 0,
            "nonfinite_tensors": 0,
            "optimizer_layouts": optimizer_layouts,
        }

    loaded = 0
    nonfinite_tensors = 0
    skipped = 0
    channel_groups = 0
    started = time.monotonic()

    for index, descriptor in enumerate(descriptors, start=1):
        gib = descriptor.estimated_bytes / 2**30
        if args.max_tensor_gib and gib > args.max_tensor_gib:
            skipped += 1
            sinks["skipped"].write(
                {
                    "checkpoint": str(checkpoint),
                    "iteration": iteration,
                    "storage_key": descriptor.storage_key,
                    "reason": "max_tensor_gib",
                    "detail": f"{gib:.3f} GiB > {args.max_tensor_gib:.3f} GiB",
                }
            )
            continue
        try:
            tensor = load_tensor(reader, descriptor, torch_module, dcp_module)
        except Exception as error:  # checkpoint readers expose backend-specific exceptions
            if args.strict:
                raise
            skipped += 1
            sinks["skipped"].write(
                {
                    "checkpoint": str(checkpoint),
                    "iteration": iteration,
                    "storage_key": descriptor.storage_key,
                    "reason": type(error).__name__,
                    "detail": str(error),
                }
            )
            print(f"  skipped {descriptor.storage_key}: {type(error).__name__}: {error}")
            continue

        loaded += 1
        print(f"  [{index}/{len(descriptors)}] {gib:.3f} GiB {descriptor.storage_key}")
        for layer, logical_tensor in logical_slices(descriptor.tensor, tensor):
            base = common_fields(args, checkpoint, iteration, descriptor, layer)
            stats = scalar_stats(logical_tensor, args.sample_elements, torch_module)
            scalar_id = (descriptor.kind, descriptor.state, descriptor.tensor, layer)
            baseline_stats = baseline_scalars.get(scalar_id)
            if baseline_stats is None:
                baseline_scalars[scalar_id] = {
                    "rms": stats["rms"],
                    "abs_max": stats["abs_max"],
                }
            sample_change, sampled_values = sampled_elementwise_change(
                logical_tensor,
                baseline_samples.get(scalar_id),
                args.comparison_sample_elements,
                torch_module,
            )
            if scalar_id not in baseline_samples and sampled_values is not None:
                baseline_samples[scalar_id] = sampled_values
            row = {
                **base,
                "shape": "x".join(str(v) for v in logical_tensor.shape),
                "dtype": str(logical_tensor.dtype),
                **stats,
                **sample_change,
                "rms_vs_baseline": (
                    1.0
                    if baseline_stats is None
                    else safe_ratio(stats["rms"], baseline_stats["rms"])
                ),
                "abs_max_vs_baseline": (
                    1.0
                    if baseline_stats is None
                    else safe_ratio(stats["abs_max"], baseline_stats["abs_max"])
                ),
            }
            sinks["tensor"].write(row)
            if stats["finite_frac"] < 1.0:
                nonfinite_tensors += 1

            analyze_channels = not args.no_channels and not (
                descriptor.kind == "optimizer" and descriptor.optimizer_layout == "bucket"
            )
            if not analyze_channels:
                continue
            for axis, metric, values_per_channel, magnitudes in channel_magnitudes(
                logical_tensor, descriptor.state, torch_module
            ):
                channel_groups += 1
                channel_id = (*scalar_id, axis)
                baseline = baseline_channels.get(channel_id)
                summary = channel_summary(
                    magnitudes, baseline, args.dead_ratio, torch_module
                )
                channel_base = {
                    **base,
                    "axis": axis,
                    "metric": metric,
                }
                sinks["channel"].write(
                    {**channel_base, "values_per_channel": values_per_channel, **summary}
                )
                for outlier in outlier_rows(
                    channel_base,
                    magnitudes,
                    baseline,
                    summary["median"],
                    args.top_k,
                    torch_module,
                ):
                    sinks["outlier"].write(outlier)
                if baseline is None:
                    baseline_channels[channel_id] = magnitudes.cpu().clone()

        del tensor
        gc.collect()

    elapsed = time.monotonic() - started
    return {
        "iteration": iteration,
        "selected_tensors": len(descriptors),
        "selected_gib": total_gib,
        "loaded_tensors": loaded,
        "skipped_tensors": skipped,
        "channel_groups": channel_groups,
        "nonfinite_tensors": nonfinite_tensors,
        "optimizer_layouts": optimizer_layouts,
        "elapsed_seconds": elapsed,
    }


def write_manifest(
    output_dir: Path,
    args: argparse.Namespace,
    checkpoints: Sequence[Path],
    reports: Sequence[dict[str, Any]],
) -> None:
    payload = {
        "command": sys.argv,
        "run": args.run,
        "baseline_checkpoint": str(checkpoints[0]),
        "checkpoints": [str(path) for path in checkpoints],
        "settings": {
            "include": args.include,
            "exclude": args.exclude,
            "optimizer": args.optimizer,
            "optimizer_states": args.optimizer_states,
            "all_model_tensors": args.all_model_tensors,
            "channels": not args.no_channels,
            "dead_ratio": args.dead_ratio,
            "top_k": args.top_k,
            "sample_elements": args.sample_elements,
            "comparison_sample_elements": args.comparison_sample_elements,
            "max_tensor_gib": args.max_tensor_gib,
        },
        "reports": list(reports),
    }
    (output_dir / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    selected = (
        {int(value.strip()) for value in args.iterations.split(",") if value.strip()}
        if args.iterations
        else None
    )
    try:
        checkpoints = discover_checkpoints(args.checkpoint, selected, args.first, args.last)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error

    try:
        import torch
        import torch.distributed.checkpoint as dcp
    except ImportError as error:
        raise SystemExit(
            "PyTorch with torch.distributed.checkpoint is required; "
            "run inside the training container"
        ) from error

    sinks = None if args.metadata_only else prepare_outputs(args.output_dir, args.overwrite)
    baseline_scalars: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    baseline_channels: dict[tuple[str, str, str, int, str], Any] = {}
    baseline_samples: dict[tuple[str, str, str, int], Any] = {}
    reports = []
    try:
        for checkpoint in checkpoints:
            try:
                report = scan_checkpoint(
                    checkpoint,
                    args,
                    sinks,
                    baseline_scalars,
                    baseline_channels,
                    baseline_samples,
                    torch,
                    dcp,
                )
            except Exception as error:  # metadata can be missing from incomplete saves
                if args.strict:
                    raise
                print(f"{checkpoint.name}: skipped: {type(error).__name__}: {error}")
                if sinks is not None:
                    sinks["skipped"].write(
                        {
                            "checkpoint": str(checkpoint),
                            "iteration": checkpoint_iteration(checkpoint),
                            "storage_key": "",
                            "reason": type(error).__name__,
                            "detail": str(error),
                        }
                    )
                report = {
                    "iteration": checkpoint_iteration(checkpoint),
                    "selected_tensors": 0,
                    "selected_gib": 0.0,
                    "loaded_tensors": 0,
                    "skipped_tensors": 1,
                    "channel_groups": 0,
                    "nonfinite_tensors": 0,
                    "optimizer_layouts": [],
                    "error": f"{type(error).__name__}: {error}",
                }
            reports.append(report)
    finally:
        if sinks is not None:
            for sink in sinks.values():
                sink.close()

    if not args.metadata_only:
        write_manifest(args.output_dir, args, checkpoints, reports)
        print(f"reports written to {args.output_dir}")

    optimizer_found = any(report["optimizer_layouts"] for report in reports)
    if args.optimizer == "on" and not optimizer_found:
        print("ERROR: --optimizer on was requested, but no supported optimizer tensors were found")
        return 2
    if not args.metadata_only and not any(report["loaded_tensors"] for report in reports):
        print("ERROR: no checkpoint tensors were loaded")
        return 2
    nonfinite = sum(report["nonfinite_tensors"] for report in reports)
    if args.fail_on_nonfinite and nonfinite:
        print(f"ERROR: found non-finite values in {nonfinite} logical tensors")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
