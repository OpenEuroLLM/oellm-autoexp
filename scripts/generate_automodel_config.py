#!/usr/bin/env python3
"""Generate annotated YAML defaults and typed dataclass schema for NeMo
Automodel.

Introspects Automodel's Python dataclasses and YAML conventions to produce:
  1. config/backend/automodel/base_defaults.yaml  — annotated YAML with all defaults
  2. oellm_autoexp/backends/automodel/config_schema.py — typed dataclass for compoconf

Usage:
    python scripts/generate_automodel_config.py           # both outputs
    python scripts/generate_automodel_config.py --yaml    # YAML only
    python scripts/generate_automodel_config.py --dataclass  # dataclass only
    python scripts/generate_automodel_config.py --stdout  # print to stdout
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import re
import sys
import textwrap
from collections import OrderedDict
from dataclasses import MISSING, fields
from pathlib import Path
from typing import Any, get_args, get_origin
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Mock unavailable GPU libraries before importing Automodel modules
# ---------------------------------------------------------------------------
import importlib.util as _importlib_util

# Patch importlib.util.find_spec so HAVE_TE / HAVE_DEEP_EP / HAVE_GMM resolve
# to False even when mock modules are in sys.modules.
_orig_find_spec = _importlib_util.find_spec

_MOCKED_PACKAGES = {"transformer_engine", "deep_ep", "grouped_gemm"}


def _patched_find_spec(name: str, *args: Any, **kwargs: Any):
    if name in _MOCKED_PACKAGES or any(name.startswith(p + ".") for p in _MOCKED_PACKAGES):
        return None
    return _orig_find_spec(name, *args, **kwargs)


_importlib_util.find_spec = _patched_find_spec  # type: ignore[assignment]

for mod_name in [
    "transformer_engine",
    "transformer_engine.pytorch",
    "transformer_engine.pytorch.router",
    "transformer_engine.pytorch.cpp_extensions",
    "transformer_engine.pytorch.distributed",
    "transformer_engine.pytorch.tensor",
    "transformer_engine.pytorch.float8_tensor",
    "transformer_engine.pytorch.fp8",
    "transformer_engine.pytorch.module",
    "transformer_engine.pytorch.module.rmsnorm",
    "transformer_engine.pytorch.module.linear",
    "transformer_engine.pytorch.module.grouped_linear",
    "transformer_engine.pytorch.quantization",
    "transformer_engine.common",
    "transformer_engine.common.recipe",
    "deep_ep",
    "grouped_gemm",
    "grouped_gemm.ops",
]:
    sys.modules.setdefault(mod_name, MagicMock())

# Add Automodel submodule to path
_REPO_ROOT = Path(__file__).resolve().parent.parent
_AUTOMODEL_ROOT = _REPO_ROOT / "submodules" / "Automodel"
if str(_AUTOMODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(_AUTOMODEL_ROOT))

# Now import Automodel config classes
from nemo_automodel.components.distributed.config import (  # noqa: E402
    FSDP2Config,
)
from nemo_automodel.components.distributed.pipelining.config import PipelineConfig  # noqa: E402
from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy  # noqa: E402
from nemo_automodel.components.models.common.utils import (  # noqa: E402
    BackendConfig,
    TEFp8Config,
)
from nemo_automodel.components.training.step_scheduler import StepScheduler  # noqa: E402

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
_DEFAULT_YAML_OUTPUT = _REPO_ROOT / "config" / "backend" / "automodel" / "base_defaults.yaml"
_DEFAULT_SCHEMA_OUTPUT = (
    _REPO_ROOT / "oellm_autoexp" / "backends" / "automodel" / "config_schema.py"
)


# ---------------------------------------------------------------------------
# Docstring parsing
# ---------------------------------------------------------------------------
def _extract_attr_help(cls: type) -> dict[str, str]:
    """Parse 'Attributes:' section from class docstring."""
    doc = inspect.getdoc(cls) or ""
    result: dict[str, str] = {}
    in_attrs = False
    current_name: str | None = None
    current_lines: list[str] = []

    for line in doc.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("attributes:"):
            in_attrs = True
            continue
        if not in_attrs:
            continue
        # End of Attributes section on blank line or new section header
        if not stripped:
            if current_name:
                result[current_name] = " ".join(current_lines).strip()
                current_name = None
                current_lines = []
            continue
        # New attribute line: "name (type): description" or "name: description"
        m = re.match(r"^(\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)", stripped)
        if m:
            if current_name:
                result[current_name] = " ".join(current_lines).strip()
            current_name = m.group(1)
            current_lines = [m.group(2)] if m.group(2) else []
        elif current_name:
            # Continuation line
            current_lines.append(stripped)

    if current_name:
        result[current_name] = " ".join(current_lines).strip()
    return result


def _extract_init_help(cls: type) -> dict[str, str]:
    """Parse 'Args:' section from __init__ docstring."""
    doc = inspect.getdoc(cls.__init__) or inspect.getdoc(cls) or ""
    result: dict[str, str] = {}
    in_args = False
    current_name: str | None = None
    current_lines: list[str] = []

    for line in doc.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("args:"):
            in_args = True
            continue
        if in_args and stripped.lower().startswith(("returns:", "raises:", "note:", "yields:")):
            in_args = False
            break
        if not in_args:
            continue
        if not stripped:
            if current_name:
                result[current_name] = " ".join(current_lines).strip()
                current_name = None
                current_lines = []
            continue
        m = re.match(r"^(\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)", stripped)
        if m:
            if current_name:
                result[current_name] = " ".join(current_lines).strip()
            current_name = m.group(1)
            current_lines = [m.group(2)] if m.group(2) else []
        elif current_name:
            current_lines.append(stripped)

    if current_name:
        result[current_name] = " ".join(current_lines).strip()
    return result


# ---------------------------------------------------------------------------
# Introspection: dataclass fields
# ---------------------------------------------------------------------------
def _introspect_dataclass(cls: type) -> list[dict[str, Any]]:
    """Extract field info from a dataclass: name, type annotation, default, help."""
    help_map = _extract_attr_help(cls)
    result = []
    for f in fields(cls):
        if f.name.startswith("_"):
            continue
        # Skip InitVar fields (they don't appear in the instance)
        if isinstance(f.type, str) and "InitVar" in f.type:
            continue
        if hasattr(dataclasses, "InitVar") and isinstance(f.default, dataclasses.InitVar):
            continue

        default = f.default if f.default is not MISSING else None
        if f.default_factory is not MISSING:
            try:
                default = f.default_factory()
            except Exception:
                default = None

        result.append(
            {
                "name": f.name,
                "type": f.type,
                "default": default,
                "help": help_map.get(f.name, ""),
            }
        )
    return result


def _introspect_init(cls: type, skip: set[str] | None = None) -> list[dict[str, Any]]:
    """Extract parameter info from a class __init__."""
    help_map = _extract_init_help(cls)
    skip = skip or set()
    sig = inspect.signature(cls.__init__)
    result = []
    for name, param in sig.parameters.items():
        if name in ("self",) or name in skip:
            continue
        default = param.default if param.default is not inspect.Parameter.empty else None
        result.append(
            {
                "name": name,
                "type": param.annotation
                if param.annotation is not inspect.Parameter.empty
                else None,
                "default": default,
                "help": help_map.get(name, ""),
            }
        )
    return result


# ---------------------------------------------------------------------------
# Type representation for schema generation
# ---------------------------------------------------------------------------
_SKIP_TYPES = {"MixedPrecisionPolicy", "CPUOffloadPolicy", "Callable", "torch.dtype"}


def _type_to_str(annotation: Any) -> str | None:
    """Convert a type annotation to a string suitable for a dataclass field."""
    if annotation is None or annotation is inspect.Parameter.empty:
        return "Any"

    # Handle string annotations
    if isinstance(annotation, str):
        # Clean up forward references
        ann = annotation.replace("typing.", "")
        if any(skip in ann for skip in _SKIP_TYPES):
            return None  # skip this field
        return ann

    origin = get_origin(annotation)
    args = get_args(annotation)

    # Literal[...]
    if origin is type(None):
        return "None"

    # Check for typing.Literal
    import typing

    if origin is typing.Literal:
        literals = ", ".join(repr(a) for a in args)
        return f"Literal[{literals}]"

    # Optional[X] = Union[X, None]
    import types

    if origin is types.UnionType or (hasattr(typing, "Union") and origin is typing.Union):
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            inner = _type_to_str(non_none[0])
            if inner is None:
                return None
            return f"{inner} | None"
        parts = []
        for a in args:
            s = _type_to_str(a)
            if s is None:
                return None
            parts.append(s)
        return " | ".join(parts)

    # List[X]
    if origin is list:
        if args:
            inner = _type_to_str(args[0])
            return f"list[{inner}]"
        return "list"

    # Optional (bare)
    if origin is type(None):
        return "None"

    # dict
    if origin is dict:
        return "dict"

    # Basic types
    if annotation is bool:
        return "bool"
    if annotation is int:
        return "int"
    if annotation is float:
        return "float"
    if annotation is str:
        return "str"

    # Check for skipped types by name
    name = getattr(annotation, "__name__", str(annotation))
    if any(skip in name for skip in _SKIP_TYPES):
        return None

    return "Any"


def _format_default(value: Any) -> str:
    """Format a default value for Python source."""
    if isinstance(value, list):
        return f"field(default_factory=lambda: {repr(value)})"
    if isinstance(value, dict):
        return f"field(default_factory=lambda: {repr(value)})"
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    if isinstance(value, str):
        return repr(value)
    return repr(value)


def _yaml_value(value: Any) -> Any:
    """Convert a Python default to a YAML-compatible value."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, str)):
        return value
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return value
    return str(value)


# ---------------------------------------------------------------------------
# Config structure definition
# ---------------------------------------------------------------------------

# YAML-convention sections (no Python class — taken from reference YAML)
_YAML_CONVENTIONS: dict[str, OrderedDict[str, dict[str, Any]]] = {
    "benchmark": OrderedDict(
        [
            ("warmup_steps", {"default": 10, "help": "Number of warmup steps before timing."}),
            (
                "peak_tflops",
                {
                    "default": 989,
                    "help": "Peak hardware TFLOPS for MFU calculation (e.g. 989 for H100).",
                },
            ),
            (
                "nsys_start",
                {"default": -1, "help": "Step to start nsys profiling (-1 to disable)."},
            ),
            ("nsys_end", {"default": -1, "help": "Step to end nsys profiling (-1 to disable)."}),
            ("nsys_ranks", {"default": [], "help": "Ranks to profile with nsys."}),
            ("num_nodes", {"default": 1, "help": "Number of nodes for TFLOPS calculation."}),
        ]
    ),
    "dataset": OrderedDict(
        [
            (
                "_target_",
                {
                    "default": "nemo_automodel.components.datasets.llm.mock_iterable_dataset.MockIterableDataset",
                    "help": "Hydra target for dataset class.",
                },
            ),
            ("seq_len", {"default": 4096, "help": "Sequence length for training."}),
            ("num_samples", {"default": 100000, "help": "Number of samples in the dataset."}),
            (
                "batch_size",
                {
                    "default": "${..step_scheduler.local_batch_size}",
                    "help": "Batch size (usually interpolated from step_scheduler).",
                },
            ),
        ]
    ),
    "dataloader": OrderedDict(
        [
            (
                "batch_size",
                {
                    "default": None,
                    "help": "DataLoader batch size (null when dataset yields batches).",
                },
            ),
            (
                "_target_",
                {
                    "default": "torch.utils.data.DataLoader",
                    "help": "Hydra target for DataLoader class.",
                },
            ),
            ("num_workers", {"default": 0, "help": "Number of DataLoader workers."}),
        ]
    ),
    "optimizer": OrderedDict(
        [
            (
                "_target_",
                {"default": "torch.optim.Adam", "help": "Hydra target for optimizer class."},
            ),
            ("betas", {"default": [0.9, 0.999], "help": "Adam beta coefficients."}),
            ("eps", {"default": 1.0e-8, "help": "Adam epsilon for numerical stability."}),
            ("lr", {"default": 0.0001, "help": "Learning rate."}),
            ("weight_decay", {"default": 0, "help": "Weight decay."}),
            (
                "foreach",
                {
                    "default": False,
                    "help": "Use foreach implementation (set false for TE GroupedLinear).",
                },
            ),
        ]
    ),
    "checkpoint": OrderedDict(
        [
            ("enabled", {"default": False, "help": "Enable checkpointing."}),
            (
                "checkpoint_dir",
                {"default": "checkpoints/", "help": "Directory for saving checkpoints."},
            ),
            (
                "model_save_format",
                {"default": "torch_save", "help": "Checkpoint format: torch_save or safetensors."},
            ),
            (
                "save_consolidated",
                {"default": False, "help": "Save model in consolidated safetensors format."},
            ),
            ("restore_from", {"default": None, "help": "Path to restore checkpoint from."}),
        ]
    ),
    "lr_scheduler": OrderedDict(
        [
            (
                "lr_decay_style",
                {"default": "cosine", "help": "LR decay schedule: cosine, linear, constant."},
            ),
            ("lr_warmup_steps", {"default": 500, "help": "Number of warmup steps."}),
            ("min_lr", {"default": 0.0, "help": "Minimum learning rate after decay."}),
        ]
    ),
    "validation_dataset": OrderedDict(
        [
            (
                "_target_",
                {
                    "default": "nemo_automodel.components.datasets.llm.mock_iterable_dataset.MockIterableDataset",
                    "help": "Hydra target for validation dataset class.",
                },
            ),
            ("seq_len", {"default": 4096, "help": "Sequence length for validation."}),
            ("num_samples", {"default": 1000, "help": "Number of validation samples."}),
            (
                "batch_size",
                {
                    "default": "${..step_scheduler.local_batch_size}",
                    "help": "Validation batch size.",
                },
            ),
        ]
    ),
    "validation_dataloader": OrderedDict(
        [
            (
                "batch_size",
                {
                    "default": None,
                    "help": "Validation DataLoader batch size (null when dataset yields batches).",
                },
            ),
            (
                "_target_",
                {
                    "default": "torch.utils.data.DataLoader",
                    "help": "Hydra target for validation DataLoader class.",
                },
            ),
            ("num_workers", {"default": 0, "help": "Number of validation DataLoader workers."}),
        ]
    ),
    "clip_grad_norm": OrderedDict(
        [
            ("max_norm", {"default": 1.0, "help": "Maximum gradient norm for clipping."}),
        ]
    ),
    "dist_env": OrderedDict(
        [
            ("backend", {"default": "nccl", "help": "Distributed backend."}),
            (
                "timeout_minutes",
                {"default": 10, "help": "Distributed initialization timeout in minutes."},
            ),
        ]
    ),
}


# BackendConfig overrides: show TE-enabled defaults since that's what users want
_BACKEND_OVERRIDES = {
    "attn": "te",
    "linear": "te",
    "rope_fusion": True,
    "experts": "torch_mm",
    "dispatcher": "torch",
}


def _build_all_sections() -> OrderedDict[str, list[dict[str, Any]]]:
    """Build the ordered config structure with all sections."""
    sections: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()

    # Top-level scalars
    sections["_top"] = [
        {
            "name": "recipe",
            "default": "BenchmarkingRecipeForNextTokenPrediction",
            "help": "Recipe class name.",
            "type": str,
        },
        {"name": "seed", "default": 1234, "help": "Random seed.", "type": int},
    ]

    # Benchmark (YAML convention)
    sections["benchmark"] = [
        {
            "name": k,
            "default": v["default"],
            "help": v["help"],
            "type": type(v["default"]) if v["default"] is not None else str,
        }
        for k, v in _YAML_CONVENTIONS["benchmark"].items()
    ]

    # StepScheduler (from __init__)
    ss_fields = _introspect_init(StepScheduler, skip={"dp_size", "dataloader"})
    sections["step_scheduler"] = ss_fields

    # Distributed (YAML convention for size params + FSDP2Config fields)
    dist_fields: list[dict[str, Any]] = [
        {
            "name": "strategy",
            "default": "fsdp2",
            "help": "Distribution strategy (fsdp2, megatron_fsdp, ddp).",
            "type": str,
        },
        {"name": "tp_size", "default": 1, "help": "Tensor parallel size.", "type": int},
        {"name": "cp_size", "default": 1, "help": "Context parallel size.", "type": int},
        {"name": "pp_size", "default": 1, "help": "Pipeline parallel size.", "type": int},
        {
            "name": "dp_replicate_size",
            "default": 1,
            "help": "Data parallel replicate size (FSDP2 only).",
            "type": int,
        },
        {"name": "ep_size", "default": 1, "help": "Expert parallel size.", "type": int},
    ]
    # Add FSDP2Config fields (skip runtime-only ones)
    for f_info in _introspect_dataclass(FSDP2Config):
        if f_info["name"] in ("tp_plan", "mp_policy", "offload_policy", "backend"):
            continue
        dist_fields.append(f_info)
    sections["distributed"] = dist_fields

    # Pipeline config
    pipe_fields = []
    for f_info in _introspect_dataclass(PipelineConfig):
        type_str = _type_to_str(f_info["type"])
        if type_str is None:  # skip Callable, torch.dtype
            continue
        pipe_fields.append(f_info)
    sections["distributed.pipeline"] = pipe_fields

    # dist_env
    sections["dist_env"] = [
        {
            "name": k,
            "default": v["default"],
            "help": v["help"],
            "type": type(v["default"]) if v["default"] is not None else str,
        }
        for k, v in _YAML_CONVENTIONS["dist_env"].items()
    ]

    # Model section (YAML convention + backend)
    model_fields: list[dict[str, Any]] = [
        {
            "name": "_target_",
            "default": "nemo_automodel.NeMoAutoModelForCausalLM.from_config",
            "help": "Hydra target for model factory.",
            "type": str,
        },
        {
            "name": "trust_remote_code",
            "default": True,
            "help": "Trust remote code when loading HF models.",
            "type": bool,
        },
    ]
    sections["model"] = model_fields

    # Model backend (BackendConfig)
    backend_fields = []
    for f_info in _introspect_dataclass(BackendConfig):
        name = f_info["name"]
        # Skip deprecated and runtime-only
        if name in ("enable_deepep", "te_fp8", "gate_precision"):
            continue
        # Override with TE-enabled defaults
        if name in _BACKEND_OVERRIDES:
            f_info = dict(f_info)  # copy
            f_info["default"] = _BACKEND_OVERRIDES[name]
        backend_fields.append(f_info)
    sections["model.backend"] = backend_fields

    # TEFp8Config
    sections["model.backend.te_fp8"] = _introspect_dataclass(TEFp8Config)

    # Checkpoint
    sections["checkpoint"] = [
        {
            "name": k,
            "default": v["default"],
            "help": v["help"],
            "type": type(v["default"]) if v["default"] is not None else str,
        }
        for k, v in _YAML_CONVENTIONS["checkpoint"].items()
    ]

    # Loss function
    sections["loss_fn"] = [
        {
            "name": "_target_",
            "default": "nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy",
            "help": "Hydra target for loss class.",
            "type": str,
        },
    ] + _introspect_init(MaskedCrossEntropy)

    # Dataset
    sections["dataset"] = [
        {
            "name": k,
            "default": v["default"],
            "help": v["help"],
            "type": type(v["default"]) if v["default"] is not None else str,
        }
        for k, v in _YAML_CONVENTIONS["dataset"].items()
    ]

    # Dataloader
    sections["dataloader"] = [
        {
            "name": k,
            "default": v["default"],
            "help": v["help"],
            "type": type(v["default"]) if v["default"] is not None else str,
        }
        for k, v in _YAML_CONVENTIONS["dataloader"].items()
    ]

    # Optimizer
    sections["optimizer"] = [
        {
            "name": k,
            "default": v["default"],
            "help": v["help"],
            "type": type(v["default"]) if v["default"] is not None else str,
        }
        for k, v in _YAML_CONVENTIONS["optimizer"].items()
    ]

    # LR Scheduler (training only)
    sections["lr_scheduler"] = [
        {
            "name": k,
            "default": v["default"],
            "help": v["help"],
            "type": type(v["default"]) if v["default"] is not None else str,
        }
        for k, v in _YAML_CONVENTIONS["lr_scheduler"].items()
    ]

    # Validation dataset (training only)
    sections["validation_dataset"] = [
        {
            "name": k,
            "default": v["default"],
            "help": v["help"],
            "type": type(v["default"]) if v["default"] is not None else str,
        }
        for k, v in _YAML_CONVENTIONS["validation_dataset"].items()
    ]

    # Validation dataloader (training only)
    sections["validation_dataloader"] = [
        {
            "name": k,
            "default": v["default"],
            "help": v["help"],
            "type": type(v["default"]) if v["default"] is not None else str,
        }
        for k, v in _YAML_CONVENTIONS["validation_dataloader"].items()
    ]

    # Gradient clipping (training only)
    sections["clip_grad_norm"] = [
        {
            "name": k,
            "default": v["default"],
            "help": v["help"],
            "type": type(v["default"]) if v["default"] is not None else str,
        }
        for k, v in _YAML_CONVENTIONS["clip_grad_norm"].items()
    ]

    return sections


# ---------------------------------------------------------------------------
# YAML generation
# ---------------------------------------------------------------------------


def _generate_yaml(sections: OrderedDict[str, list[dict[str, Any]]]) -> str:
    """Generate annotated YAML from the sections structure."""
    lines = [
        "# NeMo Automodel configuration defaults (auto-generated).",
        "# Generated by scripts/generate_automodel_config.py",
        "",
    ]

    for section_key, fields_list in sections.items():
        if section_key == "_top":
            for f in fields_list:
                val = _yaml_value(f["default"])
                comment = f.get("help", "")
                line = f"{f['name']}: {_yaml_scalar(val)}"
                if comment:
                    line += f"  # {comment}"
                lines.append(line)
            lines.append("")
            continue

        # Determine nesting depth from dotted key
        parts = section_key.split(".")
        indent = "  " * (len(parts) - 1)
        # Only emit section header for top-level
        if len(parts) == 1:
            lines.append(f"{section_key}:")
        else:
            lines.append(f"{indent}{parts[-1]}:")

        field_indent = "  " * len(parts)
        for f in fields_list:
            val = _yaml_value(f["default"])
            comment = f.get("help", "")
            line = f"{field_indent}{f['name']}: {_yaml_scalar(val)}"
            if comment:
                line += f"  # {comment}"
            lines.append(line)
        lines.append("")

    return "\n".join(lines) + "\n"


def _yaml_scalar(value: Any) -> str:
    """Format a scalar value for inline YAML."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        # Strings that look like interpolations or contain special chars
        if value.startswith("${") or ":" in value or "#" in value:
            return repr(value)
        return value
    if isinstance(value, list):
        if not value:
            return "[]"
        inner = ", ".join(_yaml_scalar(v) for v in value)
        return f"[{inner}]"
    if isinstance(value, float):
        # Scientific notation for very small values
        if 0 < abs(value) < 0.001:
            return f"{value:.1e}"
        return str(value)
    return str(value)


# ---------------------------------------------------------------------------
# Dataclass schema generation
# ---------------------------------------------------------------------------


def _generate_schema(sections: OrderedDict[str, list[dict[str, Any]]]) -> str:
    """Generate a typed dataclass schema from the sections structure."""
    typing_imports: set[str] = {"Any"}
    needs_field = False

    # Build sub-dataclasses
    sub_classes: list[str] = []

    # Map section keys to dataclass names
    class_map = {
        "benchmark": "AutomodelBenchmarkConfig",
        "step_scheduler": "AutomodelStepSchedulerConfig",
        "distributed": "AutomodelDistributedConfig",
        "distributed.pipeline": "AutomodelPipelineConfig",
        "dist_env": "AutomodelDistEnvConfig",
        "model": "AutomodelModelConfig",
        "model.backend": "AutomodelBackendConfig",
        "model.backend.te_fp8": "AutomodelTEFp8Config",
        "checkpoint": "AutomodelCheckpointConfig",
        "loss_fn": "AutomodelLossFnConfig",
        "dataset": "AutomodelDatasetConfig",
        "dataloader": "AutomodelDataloaderConfig",
        "optimizer": "AutomodelOptimizerConfig",
        "lr_scheduler": "AutomodelLrSchedulerConfig",
        "validation_dataset": "AutomodelValidationDatasetConfig",
        "validation_dataloader": "AutomodelValidationDataloaderConfig",
        "clip_grad_norm": "AutomodelClipGradNormConfig",
    }

    # Track which sub-configs are nested inside others
    nested = {
        "distributed.pipeline": ("distributed", "pipeline"),
        "model.backend": ("model", "backend"),
        "model.backend.te_fp8": ("model.backend", "te_fp8"),
    }

    # Emit order: leaves first so forward references are avoided.
    # Deepest-nested sections first, then their parents.
    emit_order = [
        "benchmark",
        "step_scheduler",
        "distributed.pipeline",
        "dist_env",
        "model.backend.te_fp8",
        "model.backend",
        "model",
        "distributed",
        "checkpoint",
        "loss_fn",
        "dataset",
        "dataloader",
        "optimizer",
        "lr_scheduler",
        "validation_dataset",
        "validation_dataloader",
        "clip_grad_norm",
    ]

    # Generate each sub-dataclass
    for section_key in emit_order:
        fields_list = sections.get(section_key, [])
        class_name = class_map.get(section_key)
        if not class_name:
            continue

        field_lines = []
        for f in fields_list:
            type_str = _type_to_str(f.get("type"))
            if type_str is None:
                continue

            default = f["default"]
            default_str = _format_default(default)
            if "field(default_factory" in default_str:
                needs_field = True

            # Add None union if default is None and type doesn't include None
            if default is None and "None" not in type_str:
                type_str = f"{type_str} | None"

            if "Literal" in type_str:
                typing_imports.add("Literal")

            help_text = f.get("help", "")
            if help_text:
                wrapped = textwrap.wrap(help_text, width=88)
                for hl in wrapped:
                    field_lines.append(f"    # {hl}")
            field_lines.append(f"    {f['name']}: {type_str} = {default_str}")

        # Add nested sub-config fields
        for nested_key, (parent, attr_name) in nested.items():
            if parent == section_key and nested_key in class_map:
                nested_cls = class_map[nested_key]
                needs_field = True
                field_lines.append(
                    f"    {attr_name}: {nested_cls} = field(default_factory={nested_cls})"
                )

        cls_block = f"@dataclass\nclass {class_name}:\n"
        cls_block += f'    """Auto-generated config for {section_key} section."""\n\n'
        cls_block += "\n".join(field_lines) + "\n"
        sub_classes.append(cls_block)

    # Build top-level AutomodelConfig
    top_lines = []
    for f in sections.get("_top", []):
        type_str = _type_to_str(f.get("type")) or "str"
        default_str = _format_default(f["default"])
        help_text = f.get("help", "")
        if help_text:
            top_lines.append(f"    # {help_text}")
        top_lines.append(f"    {f['name']}: {type_str} = {default_str}")

    # Add sub-config fields
    needs_field = True  # We'll need field() for sub-configs
    top_section_fields = [
        ("benchmark", "AutomodelBenchmarkConfig"),
        ("step_scheduler", "AutomodelStepSchedulerConfig"),
        ("distributed", "AutomodelDistributedConfig"),
        ("dist_env", "AutomodelDistEnvConfig"),
        ("model", "AutomodelModelConfig"),
        ("checkpoint", "AutomodelCheckpointConfig"),
        ("loss_fn", "AutomodelLossFnConfig"),
        ("dataset", "AutomodelDatasetConfig"),
        ("dataloader", "AutomodelDataloaderConfig"),
        ("optimizer", "AutomodelOptimizerConfig"),
        ("lr_scheduler", "AutomodelLrSchedulerConfig"),
        ("validation_dataset", "AutomodelValidationDatasetConfig"),
        ("validation_dataloader", "AutomodelValidationDataloaderConfig"),
        ("clip_grad_norm", "AutomodelClipGradNormConfig"),
    ]
    for attr_name, cls_name in top_section_fields:
        top_lines.append(f"    {attr_name}: {cls_name} = field(default_factory={cls_name})")

    # Extension field
    top_lines.append("    aux: dict[str, Any] = field(default_factory=dict)")

    # Assemble file
    header = [
        '"""NeMo Automodel configuration schema (auto-generated)."""',
        "",
    ]
    dc_import = (
        "from dataclasses import dataclass, field"
        if needs_field
        else "from dataclasses import dataclass"
    )
    header.append(dc_import)
    header.append(f"from typing import {', '.join(sorted(typing_imports))}")
    header.append("")
    header.append("from compoconf import ConfigInterface")
    header.append("")
    header.append("")

    body = "\n".join(header)
    body += "\n\n".join(sub_classes)
    body += "\n\n"
    body += "@dataclass\n"
    body += "class AutomodelConfig(ConfigInterface):\n"
    body += '    """Typed projection of NeMo Automodel configuration."""\n\n'
    body += "    # Generated by scripts/generate_automodel_config.py\n\n"
    body += "\n".join(top_lines) + "\n"
    body += '\n\n__all__ = ["AutomodelConfig"]\n'

    return body


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml", action="store_true", help="Generate YAML defaults only.")
    parser.add_argument("--dataclass", action="store_true", help="Generate dataclass schema only.")
    parser.add_argument(
        "--stdout", action="store_true", help="Print to stdout instead of writing files."
    )
    parser.add_argument("--yaml-output", type=Path, default=_DEFAULT_YAML_OUTPUT)
    parser.add_argument("--schema-output", type=Path, default=_DEFAULT_SCHEMA_OUTPUT)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_both = not args.yaml and not args.dataclass

    sections = _build_all_sections()

    if args.yaml or generate_both:
        yaml_text = _generate_yaml(sections)
        if args.stdout:
            print("# === base_defaults.yaml ===")
            print(yaml_text)
        else:
            args.yaml_output.parent.mkdir(parents=True, exist_ok=True)
            args.yaml_output.write_text(yaml_text)
            if not args.quiet:
                print(f"Wrote {args.yaml_output}")

    if args.dataclass or generate_both:
        schema_text = _generate_schema(sections)
        if args.stdout:
            print("# === config_schema.py ===")
            print(schema_text)
        else:
            args.schema_output.parent.mkdir(parents=True, exist_ok=True)
            args.schema_output.write_text(schema_text)
            if not args.quiet:
                print(f"Wrote {args.schema_output}")


if __name__ == "__main__":
    main()
