#!/usr/bin/env python3
"""Scan Hugging Face safetensors checkpoints using the 32B weight-statistics logic.

This is the Hugging Face counterpart to ``scan_checkpoint_stats.py``.  It
accepts downloaded ``iter_<N>`` directories containing either a sharded
``model.safetensors.index.json`` checkpoint or standalone ``*.safetensors``
files.  Its CSV output has the same schema, so it can be passed directly to
``compare_checkpoint_stats.py``.

Only model weights are scanned; Hugging Face optimizer files are not read.
Tensor names are preserved as stored in the checkpoint.  Consequently,
within-HF trajectories compare directly, while paired HF-versus-Megatron
reports require a separate canonical name mapping.

Example::

    python3 scripts/scan_hf_checkpoint_stats.py \
      /e/home/jusers/luukkonen1/jupiter/e-sta-workdir/oellm-autoexp/hf_checkpoints/prelude \
      --run prelude8b \
      --output-dir checkpoint_stats/prelude8b \
      --strict --fail-on-nonfinite

Then build the usual trajectory report::

    python3 scripts/compare_checkpoint_stats.py \
      --scan prelude8b=checkpoint_stats/prelude8b \
      --output-dir checkpoint_comparison/prelude8b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

from scan_checkpoint_stats import (
    checkpoint_iteration,
    discover_checkpoints,
    prepare_outputs,
    scan_checkpoint,
    write_manifest,
)


SAFETENSORS_INDEX = "model.safetensors.index.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "checkpoint",
        nargs="+",
        type=Path,
        help="iter_* directory, or roots containing iter_* directories",
    )
    parser.add_argument("--run", default="", help="optional run label written to CSV")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("checkpoint_stats_hf")
    )
    parser.add_argument(
        "--iterations", help="comma-separated exact iterations to scan"
    )
    parser.add_argument("--first", type=int)
    parser.add_argument("--last", type=int)
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        metavar="REGEX",
        help="include tensor names matching any supplied regex (repeatable)",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="REGEX",
        help="exclude tensor names matching any supplied regex (repeatable)",
    )
    parser.add_argument(
        "--all-model-tensors",
        action="store_true",
        help="include tensors whose names do not end in weight or bias",
    )
    parser.add_argument("--no-channels", action="store_true")
    parser.add_argument("--dead-ratio", type=float, default=0.01)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--sample-elements", type=int, default=250_000)
    parser.add_argument("--comparison-sample-elements", type=int, default=16_384)
    parser.add_argument(
        "--max-tensor-gib",
        type=float,
        default=0.0,
        help="skip individual tensors larger than this; 0 disables the guard",
    )
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--fail-on-nonfinite", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if args.dead_ratio <= 0:
        parser.error("--dead-ratio must be positive")
    if args.top_k < 0:
        parser.error("--top-k must be non-negative")
    if args.sample_elements < 1:
        parser.error("--sample-elements must be at least 1")
    if args.comparison_sample_elements < 0:
        parser.error("--comparison-sample-elements must be non-negative")
    if args.max_tensor_gib < 0:
        parser.error("--max-tensor-gib must be non-negative")

    # Fields consumed by the shared scanner. HF snapshots contain model weights
    # only, so optimizer loading is deliberately disabled.
    args.optimizer = "off"
    args.optimizer_states = ""
    return args


class SafetensorsReader:
    """Minimal distributed-checkpoint reader interface over HF safetensors."""

    def __init__(self, checkpoint: Path, safe_open: Any, torch_module: Any):
        self.checkpoint = Path(checkpoint)
        self.safe_open = safe_open
        self.torch = torch_module
        self.weight_map = self._discover_weight_map()
        self._metadata = self._read_tensor_metadata()

    def _discover_weight_map(self) -> dict[str, Path]:
        index_path = self.checkpoint / SAFETENSORS_INDEX
        if index_path.is_file():
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            raw_weight_map = payload.get("weight_map")
            if not isinstance(raw_weight_map, dict) or not raw_weight_map:
                raise ValueError(f"missing or empty weight_map in {index_path}")
            weight_map = {
                str(key): self.checkpoint / str(filename)
                for key, filename in raw_weight_map.items()
            }
            missing = sorted({path for path in weight_map.values() if not path.is_file()})
            if missing:
                raise FileNotFoundError(
                    "safetensors shards referenced by the index are missing: "
                    + ", ".join(str(path) for path in missing)
                )
            return weight_map

        files = sorted(self.checkpoint.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(
                f"no {SAFETENSORS_INDEX} or *.safetensors files in {self.checkpoint}"
            )
        weight_map: dict[str, Path] = {}
        for path in files:
            with self.safe_open(path, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    if key in weight_map:
                        raise ValueError(
                            f"duplicate tensor {key!r} in {weight_map[key]} and {path}"
                        )
                    weight_map[key] = path
        return weight_map

    def _read_tensor_metadata(self) -> dict[str, Any]:
        by_file: dict[Path, list[str]] = {}
        for key, path in self.weight_map.items():
            by_file.setdefault(path, []).append(key)

        metadata: dict[str, Any] = {}
        for path, keys in sorted(by_file.items(), key=lambda item: str(item[0])):
            with self.safe_open(path, framework="pt", device="cpu") as handle:
                available = set(handle.keys())
                for key in keys:
                    if key not in available:
                        raise KeyError(f"{key!r} is indexed but absent from {path}")
                    tensor_slice = handle.get_slice(key)
                    dtype = safetensors_dtype_to_torch(
                        tensor_slice.get_dtype(), self.torch
                    )
                    metadata[key] = SimpleNamespace(
                        size=tuple(int(value) for value in tensor_slice.get_shape()),
                        properties=SimpleNamespace(dtype=dtype),
                    )
        return metadata

    def read_metadata(self) -> Any:
        return SimpleNamespace(state_dict_metadata=self._metadata)

    def load_one(self, key: str) -> Any:
        path = self.weight_map[key]
        with self.safe_open(path, framework="pt", device="cpu") as handle:
            return handle.get_tensor(key)


def safetensors_dtype_to_torch(dtype: str, torch_module: Any) -> Any:
    names = {
        "BOOL": "bool",
        "U8": "uint8",
        "I8": "int8",
        "I16": "int16",
        "I32": "int32",
        "I64": "int64",
        "F16": "float16",
        "BF16": "bfloat16",
        "F32": "float32",
        "F64": "float64",
        "F8_E4M3": "float8_e4m3fn",
        "F8_E5M2": "float8_e5m2",
    }
    name = names.get(str(dtype))
    torch_dtype = getattr(torch_module, name, None) if name else None
    if torch_dtype is None:
        raise TypeError(f"unsupported safetensors dtype: {dtype}")
    return torch_dtype


class SafetensorsAdapter:
    """Object exposing the two DCP calls used by the shared scanner."""

    def __init__(self, safe_open: Any, torch_module: Any):
        self.safe_open = safe_open
        self.torch = torch_module

    def FileSystemReader(self, checkpoint: Path) -> SafetensorsReader:  # noqa: N802
        return SafetensorsReader(checkpoint, self.safe_open, self.torch)

    @staticmethod
    def load(state: dict[str, Any], storage_reader: SafetensorsReader) -> None:
        if len(state) != 1:
            raise ValueError("safetensors adapter expects exactly one requested tensor")
        key = next(iter(state))
        state[key] = storage_reader.load_one(key)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    selected = (
        {int(value.strip()) for value in args.iterations.split(",") if value.strip()}
        if args.iterations
        else None
    )
    try:
        checkpoints = discover_checkpoints(
            args.checkpoint, selected, args.first, args.last
        )
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error

    try:
        import torch
        from safetensors import safe_open
    except ImportError as error:
        raise SystemExit(
            "PyTorch and safetensors are required; install with: pip install torch safetensors"
        ) from error

    adapter = SafetensorsAdapter(safe_open, torch)
    sinks = None if args.metadata_only else prepare_outputs(args.output_dir, args.overwrite)
    baseline_scalars: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    baseline_channels: dict[tuple[str, str, str, int, str], Any] = {}
    baseline_samples: dict[tuple[str, str, str, int], Any] = {}
    reports: list[dict[str, Any]] = []

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
                    adapter,
                )
            except Exception as error:
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
        if not any(report["loaded_tensors"] for report in reports):
            print("ERROR: no checkpoint tensors were loaded")
            return 2

    nonfinite = sum(report["nonfinite_tensors"] for report in reports)
    if args.fail_on_nonfinite and nonfinite:
        print(f"ERROR: found non-finite values in {nonfinite} logical tensors")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
