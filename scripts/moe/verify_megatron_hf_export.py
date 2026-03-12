#!/usr/bin/env python3
"""Sanity-check an exported HuggingFace checkpoint against its source Megatron checkpoint.

The script walks every tensor in the Megatron checkpoint, re-exports it through the
bridge weight-mapping to get its HF key, then compares it against the corresponding
tensor in the on-disk HF model.  Two comparison modes are supported:

  allclose  (default)  – tolerance-based comparison via ``torch.allclose``.
  exact     (opt-in)   – bitwise equality via ``torch.equal``; also checks dtypes.

Usage
-----
Single-GPU sanity check with default tolerances::

    python scripts/verify_megatron_hf_export.py \\
        --hf-path      exports/my_qwen3_moe_hf \\
        --megatron-path /path/to/megatron/checkpoint

Strict bitwise check with an interactive debug shell::

    python scripts/verify_megatron_hf_export.py \\
        --hf-path      exports/my_qwen3_moe_hf \\
        --megatron-path /path/to/megatron/checkpoint \\
        --exact-match  \\
        --debug-interact
"""

from __future__ import annotations

import argparse
import code
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml

from megatron.bridge import AutoBridge
from megatron.bridge.training.model_load_save import build_and_load_model, temporary_distributed_context

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Tensors whose names contain any of these substrings are cast to float32
# before comparison (allclose mode only).  These tensors accumulate in fp32
# inside the model regardless of the global dtype, so comparing them in
# bf16/fp16 would raise spurious failures for an otherwise-correct export.
_DEFAULT_FP32_SUBSTRINGS: tuple[str, ...] = (
    "e_score_correction_bias",  # Qwen3-MoE router load-balancing correction
    "A_log",                    # Mamba/SSM log-eigenvalue matrix
    "linear_attn.norm.weight",  # Hybrid-attention layer-norm scale
)

# Candidate YAML filenames written alongside checkpoints by various training
# pipelines to record the exact hyper-parameters used when saving.
_CONFIG_CANDIDATES: tuple[str, ...] = (
    "modelopt_run_config.yaml",
    "run_config.yaml",
    "config.yaml",
)

# Mapping from checkpoint YAML keys to the corresponding provider attribute
# names.  Only integer-valued hyper-parameters are handled here; booleans
# (embedding tying flags) are resolved separately due to naming inconsistencies
# across different training frameworks.
_PROVIDER_INT_FIELDS: dict[str, str] = {
    "num_layers":          "num_layers",
    "hidden_size":         "hidden_size",
    "num_attention_heads": "num_attention_heads",
    "num_query_groups":    "num_query_groups",
    "kv_channels":         "kv_channels",
    "ffn_hidden_size":     "ffn_hidden_size",
    "moe_ffn_hidden_size": "moe_ffn_hidden_size",
    "num_moe_experts":     "num_moe_experts",
    "moe_router_topk":     "moe_router_topk",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CompareSummary:
    """Running tallies of per-tensor comparison outcomes.

    Counters are updated in-place by ``_compare_tensor`` as it processes
    each tensor.  ``_print_report`` reads these at the end of the run.
    """

    checked: int = 0             # total tensors visited
    missing_in_hf: int = 0       # key absent from the HF state dict
    shape_mismatches: int = 0    # shapes differ (usually a TP-degree mismatch)
    dtype_mismatches: int = 0    # dtypes differ (only checked in exact mode)
    value_mismatches: int = 0    # values differ beyond tolerance / not bit-equal
    max_abs_diff: float = 0.0    # worst per-element absolute difference seen
    max_abs_diff_name: str = ""  # name of the tensor producing max_abs_diff


@dataclass
class _LoadedModels:
    """Bundles every live object at the time of the debug breakpoint.

    Providing a single container keeps ``_open_debug_shell`` independent of
    ``main``'s local variables while still exposing all useful names.
    """

    bridge: Any
    provider: Any
    hf_model: Any
    hf_state: dict[str, torch.Tensor]
    megatron_model: list[Any]
    checkpoint_config: dict[str, Any] | None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare exported HF weights against a Megatron checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Paths
    p.add_argument("--hf-path", required=True, metavar="DIR",
                   help="Path to the exported HuggingFace model directory.")
    p.add_argument("--megatron-path", required=True, metavar="DIR",
                   help="Path to the Megatron checkpoint directory.")

    # Loading options
    p.add_argument("--backend", default="gloo", choices=["gloo", "nccl"],
                   help="torch.distributed backend for the temporary process group.")
    p.add_argument("--local-files-only", action="store_true",
                   help="Prevent HuggingFace from downloading any remote artifacts.")
    p.add_argument("--show-progress", action="store_true",
                   help="Display a progress bar while iterating export_hf_weights.")

    # Comparison options
    p.add_argument("--atol", type=float, default=1e-1,
                   help="Absolute tolerance for allclose mode.")
    p.add_argument("--rtol", type=float, default=1e-3,
                   help="Relative tolerance for allclose mode.")
    p.add_argument("--exact-match", action="store_true",
                   help="Require bitwise-exact dtype and value match (torch.equal).")
    p.add_argument("--max-report", type=int, default=20,
                   help="Maximum number of per-tensor mismatch lines to print.")
    p.add_argument(
        "--fp32-compare-substring", action="append", default=None, metavar="SUBSTR",
        help=(
            "Cast tensors whose name contains SUBSTR to float32 before comparing "
            "(ignored in --exact-match mode).  May be repeated."
        ),
    )

    # Debug options
    p.add_argument(
        "--debug-interact", action="store_true",
        help=(
            "Open an interactive Python shell on rank 0 after both models are loaded. "
            "Other ranks wait at a barrier until you exit with Ctrl-D."
        ),
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Checkpoint config helpers
# ---------------------------------------------------------------------------

def _find_config_file(checkpoint_dir: str) -> Path | None:
    """Search *checkpoint_dir* and its parent for a known training config YAML.

    Many training pipelines write a side-car YAML that records the exact
    hyper-parameters used when saving the checkpoint.  Using it prevents
    shape mismatches caused by TP/PP parallelism settings that are embedded
    implicitly in tensor shapes rather than model configs.
    """
    base = Path(checkpoint_dir)
    for directory in (base, base.parent):
        for name in _CONFIG_CANDIDATES:
            candidate = directory / name
            if candidate.exists():
                return candidate
    return None


def _load_checkpoint_config(checkpoint_dir: str) -> dict[str, Any] | None:
    """Return the training config dict for *checkpoint_dir*, or ``None`` if absent."""
    config_path = _find_config_file(checkpoint_dir)
    if config_path is None:
        print(
            "Warning: no checkpoint config YAML found — "
            "provider will use defaults derived from the HF model config."
        )
        return None

    with config_path.open(encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    print(f"Checkpoint config loaded: {config_path}")
    return config


# ---------------------------------------------------------------------------
# Provider alignment helpers
# ---------------------------------------------------------------------------

def _coerce_int(value: Any) -> int | None:
    """Coerce *value* to ``int``, handling string representations like ``"4096"``."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    return int(float(value))  # handles "4096", "4.0e3", scientific notation, etc.


def _coerce_bool(value: Any) -> bool:
    """Coerce *value* to ``bool``, accepting common string literals."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.strip().lower() in {"1", "true", "yes", "y", "on"}:
            return True
        if value.strip().lower() in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def _align_provider_to_config(provider: Any, config: dict[str, Any] | None) -> None:
    """Overwrite provider attributes with values from the checkpoint config.

    This step is critical for avoiding shape/key mismatches when the checkpoint
    was saved with a parallelism degree that differs from what ``AutoBridge``
    assumes (e.g. a TP=2 checkpoint loaded through a TP=1 provider would
    yield tensors that are half the expected width).

    When *config* is ``None`` a safe conservative fallback is applied: tied
    embeddings, which is the common default for most GPT-family checkpoints.
    """
    if config is None:
        _set_provider_tying(provider, untie=False)
        print("No checkpoint config: defaulting to tied embedding/output weights.")
        return

    # Apply integer hyper-parameters (num_layers, hidden_size, …).
    for config_key, attr in _PROVIDER_INT_FIELDS.items():
        if config_key in config and hasattr(provider, attr):
            setattr(provider, attr, _coerce_int(config[config_key]))

    # kv_channels is sometimes omitted or stale in older configs.  Re-derive it
    # from (hidden_size / num_attention_heads) as a safety net.
    _maybe_fix_kv_channels(provider, config)

    # Resolve embedding/output-weight tying across different naming conventions.
    untie = _resolve_untie_embeddings(config)
    _set_provider_tying(provider, untie=untie)
    print(f"Provider tying configured: untie_embeddings_and_output_weights={untie}")


def _maybe_fix_kv_channels(provider: Any, config: dict[str, Any]) -> None:
    """Re-derive ``kv_channels`` from head geometry when the stored value looks stale."""
    if not hasattr(provider, "kv_channels"):
        return
    num_heads = _coerce_int(config.get("num_attention_heads"))
    hidden_size = _coerce_int(config.get("hidden_size"))
    if not (num_heads and hidden_size):
        return

    derived = hidden_size // num_heads
    stored = _coerce_int(config.get("kv_channels"))
    if stored is None or stored != derived:
        provider.kv_channels = derived
        print(f"kv_channels re-derived from hidden_size/num_heads: {hidden_size}/{num_heads} → {derived}")


def _resolve_untie_embeddings(config: dict[str, Any]) -> bool:
    """Return whether the checkpoint has untied embedding/output weights.

    Different frameworks use different flag names for the same concept.  We
    check them in priority order; fall back to ``False`` (tied) if none present.
    """
    if "untie_embeddings_and_output_weights" in config:
        return _coerce_bool(config["untie_embeddings_and_output_weights"])
    if "share_embeddings_and_output_weights" in config:
        return not _coerce_bool(config["share_embeddings_and_output_weights"])
    if "tie_word_embeddings" in config:
        return not _coerce_bool(config["tie_word_embeddings"])
    # Flag absent: conservative default.
    print("Tying flag absent from checkpoint config; defaulting to tied embeddings.")
    return False


def _set_provider_tying(provider: Any, *, untie: bool) -> None:
    """Write both tying-related attributes on *provider* consistently."""
    if hasattr(provider, "untie_embeddings_and_output_weights"):
        provider.untie_embeddings_and_output_weights = untie
    if hasattr(provider, "share_embeddings_and_output_weights"):
        provider.share_embeddings_and_output_weights = not untie


# ---------------------------------------------------------------------------
# Debug shell
# ---------------------------------------------------------------------------

def _open_debug_shell(loaded: _LoadedModels, args: argparse.Namespace) -> None:
    """Open a rank-0 interactive Python shell; all other ranks wait at a barrier.

    The shell namespace merges module globals, the caller's local variables,
    and every attribute of *loaded*, so the following names are directly
    accessible without any prefix:

        bridge, provider, hf_model, hf_state, megatron_model, checkpoint_config

    Exit with Ctrl-D (or ``exit()``) to resume execution.
    """
    rank = torch.distributed.get_rank()

    if rank == 0:
        # Build the shell namespace in layers so that explicit objects from
        # `loaded` take precedence over any shadowed names in globals/locals.
        frame = inspect.currentframe()
        caller_locals: dict[str, Any] = {}
        if frame is not None and frame.f_back is not None:
            caller_locals = dict(frame.f_back.f_locals)

        namespace: dict[str, Any] = dict(globals())
        namespace.update(caller_locals)
        namespace.update(vars(loaded))  # bridge, provider, hf_model, hf_state, …
        namespace["args"] = args
        namespace["rank"] = rank

        banner = (
            "\n=== DEBUG SHELL — both HF and Megatron models are loaded ===\n"
            "Available: bridge, provider, hf_model, hf_state, "
            "megatron_model, checkpoint_config, args\n"
            "Press Ctrl-D to continue.\n"
        )
        code.interact(banner=banner, local=namespace)

    # Park non-zero ranks here until rank 0 exits the interactive shell.
    torch.distributed.barrier()


# ---------------------------------------------------------------------------
# Tensor comparison
# ---------------------------------------------------------------------------

def _maybe_cast_fp32(
    name: str,
    x: torch.Tensor,
    y: torch.Tensor,
    fp32_substrings: Iterable[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Upcast *x* and *y* to float32 when *name* matches a known fp32 substring.

    Certain tensors accumulate in float32 regardless of the global dtype
    (router biases, SSM eigenvalues).  Upcasting before comparison prevents
    spurious allclose failures for otherwise-correct exports.
    """
    if any(sub in name for sub in fp32_substrings):
        return x.float(), y.float()
    return x, y


def _update_max_diff(
    name: str,
    x: torch.Tensor,
    y: torch.Tensor,
    summary: CompareSummary,
) -> float:
    """Compute max per-element absolute diff and update *summary* if it is worst so far."""
    if x.shape != y.shape or x.numel() == 0:
        return 0.0
    max_diff = float((x.float() - y.float()).abs().max().item())
    if max_diff > summary.max_abs_diff:
        summary.max_abs_diff = max_diff
        summary.max_abs_diff_name = name
    return max_diff


def _record_mismatch(samples: list[str], text: str, max_report: int) -> None:
    """Append *text* to *samples* up to the *max_report* limit."""
    if len(samples) < max_report:
        samples.append(text)


def _compare_tensor(
    *,
    name: str,
    megatron_param: torch.Tensor,
    hf_state: dict[str, torch.Tensor],
    fp32_substrings: Iterable[str],
    summary: CompareSummary,
    samples: list[str],
    atol: float,
    rtol: float,
    exact_match: bool,
    max_report: int,
) -> None:
    """Compare one tensor from the Megatron export against its HF counterpart.

    For each tensor, the function checks in order:
      1. Existence  — is the key present in the HF state dict?
      2. Shape      — do the shapes agree?  (mismatch → skip value check)
      3. Value      — exact equality (``torch.equal``) or tolerance-based
                      (``torch.allclose``), depending on *exact_match*.

    *summary* counters and *samples* are updated in-place on failure.
    """
    summary.checked += 1

    # 1. Existence check.
    if name not in hf_state:
        summary.missing_in_hf += 1
        _record_mismatch(samples, f"MISSING  {name}", max_report)
        return

    x = megatron_param.detach().cpu()
    y = hf_state[name].detach().cpu()

    # Track worst absolute diff globally (computed before any casting).
    max_diff = _update_max_diff(name, x, y, summary)

    # 2. Shape check.
    # A shape mismatch almost always means TP-degree is wrong — report it and
    # skip the value check, since the data layout is fundamentally different.
    if x.shape != y.shape:
        summary.shape_mismatches += 1
        _record_mismatch(
            samples,
            f"SHAPE    {name}  megatron={tuple(x.shape)}  hf={tuple(y.shape)}",
            max_report,
        )
        return

    # 3. Value check.
    if exact_match:
        # Exact mode: require identical dtype first, then bitwise value equality.
        if x.dtype != y.dtype:
            summary.dtype_mismatches += 1
            _record_mismatch(
                samples,
                f"DTYPE    {name}  megatron={x.dtype}  hf={y.dtype}",
                max_report,
            )
            return
        if not torch.equal(x, y):
            summary.value_mismatches += 1
            mean_diff = float((x.float() - y.float()).abs().mean().item()) if x.numel() > 0 else 0.0
            _record_mismatch(
                samples,
                f"EXACT    {name}  max_abs={max_diff:.4e}  mean_abs={mean_diff:.4e}  dtype={x.dtype}",
                max_report,
            )
    else:
        # Allclose mode: optionally upcast to fp32, then use torch.allclose.
        x_cmp, y_cmp = _maybe_cast_fp32(name, x, y, fp32_substrings)
        if not torch.allclose(x_cmp, y_cmp, atol=atol, rtol=rtol):
            summary.value_mismatches += 1
            mean_diff = float((x_cmp - y_cmp).abs().mean().item()) if x_cmp.numel() > 0 else 0.0
            _record_mismatch(
                samples,
                (
                    f"VALUE    {name}  max_abs={max_diff:.4e}  mean_abs={mean_diff:.4e}"
                    f"  dtype(meg)={x_cmp.dtype}  dtype(hf)={y_cmp.dtype}"
                ),
                max_report,
            )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_header(args: argparse.Namespace) -> None:
    """Print a human-readable summary of the run configuration."""
    mode = "exact" if args.exact_match else f"allclose (atol={args.atol}, rtol={args.rtol})"
    pad = 20
    print("=" * 46)
    print("  Megatron → HF Checkpoint Verification")
    print("=" * 46)
    print(f"  {'HF path':{pad}}: {args.hf_path}")
    print(f"  {'Megatron path':{pad}}: {args.megatron_path}")
    print(f"  {'Backend':{pad}}: {args.backend}")
    print(f"  {'Compare mode':{pad}}: {mode}")
    print(f"  {'Local files only':{pad}}: {args.local_files_only}")
    print(f"  {'Debug shell':{pad}}: {args.debug_interact}")
    print("=" * 46)
    print()


def _print_report(summary: CompareSummary, samples: list[str]) -> int:
    """Print the final comparison report and return 0 (PASS) or 1 (FAIL)."""
    total_failures = (
        summary.missing_in_hf
        + summary.shape_mismatches
        + summary.dtype_mismatches
        + summary.value_mismatches
    )

    sep = "=" * 46
    print(f"\n{sep}")
    print("  Comparison Report")
    print(sep)
    print(f"  {'Tensors checked':<22}: {summary.checked}")
    print(f"  {'Missing in HF':<22}: {summary.missing_in_hf}")
    print(f"  {'Shape mismatches':<22}: {summary.shape_mismatches}")
    print(f"  {'DType mismatches':<22}: {summary.dtype_mismatches}")
    print(f"  {'Value mismatches':<22}: {summary.value_mismatches}")
    print(f"  {'─' * 42}")
    print(f"  {'Total failures':<22}: {total_failures}")
    print(f"  {'Max abs diff':<22}: {summary.max_abs_diff:.4e}")
    print(f"  {'Worst tensor':<22}: {summary.max_abs_diff_name or 'N/A'}")
    print(sep)

    if samples:
        print(f"\nMismatch details (first {len(samples)}):")
        for line in samples:
            print(f"  {line}")

    if total_failures == 0:
        print("\nPASS — HF export matches the Megatron checkpoint for all compared tensors.")
        return 0

    print("\nFAIL — mismatches detected; see the report above.")
    return 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    args = _parse_args()
    fp32_substrings = tuple(args.fp32_compare_substring or _DEFAULT_FP32_SUBSTRINGS)
    _print_header(args)

    # ------------------------------------------------------------------
    # Phase 1 — load the HF model through AutoBridge.
    #
    # AutoBridge reads the HF model config and weights, and builds the
    # key-mapping tables used later to translate Megatron parameter names
    # into their HF equivalents.
    # ------------------------------------------------------------------
    print("Step 1/3  Loading HF model via AutoBridge …")
    bridge = AutoBridge.from_hf_pretrained(
        args.hf_path,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )

    # ------------------------------------------------------------------
    # Phase 2 — build a Megatron model provider and align it to the
    # checkpoint's hyper-parameters.
    #
    # We do NOT load weights here; weights will be streamed from disk
    # inside the distributed context below.  Aligning the provider before
    # loading prevents shape mismatches from TP/PP parallelism differences.
    # ------------------------------------------------------------------
    print("Step 2/3  Building Megatron provider and aligning to checkpoint config …")
    provider = bridge.to_megatron_provider(load_weights=False)
    checkpoint_config = _load_checkpoint_config(args.megatron_path)
    _align_provider_to_config(provider, checkpoint_config)

    # ------------------------------------------------------------------
    # Phase 3 — load Megatron weights and compare tensors.
    #
    # build_and_load_model requires a torch.distributed process group.
    # We spin up a temporary single-rank group with the "gloo" backend
    # (no GPU required) and tear it down automatically on exit.
    # ------------------------------------------------------------------
    print("Step 3/3  Loading Megatron checkpoint and comparing tensors …")
    summary = CompareSummary()
    samples: list[str] = []

    with temporary_distributed_context(backend=args.backend):
        megatron_model = build_and_load_model(
            checkpoint_path=args.megatron_path,
            model_cfg=provider,
            use_cpu_init=True,
        )
        # build_and_load_model may return a single model or a list of pipeline chunks.
        if not isinstance(megatron_model, list):
            megatron_model = [megatron_model]

        # Bundle all live objects so the debug shell and comparison loop share the same view.
        loaded = _LoadedModels(
            bridge=bridge,
            provider=provider,
            hf_model=bridge.hf_pretrained,
            hf_state=dict(bridge.hf_pretrained.state),
            megatron_model=megatron_model,
            checkpoint_config=checkpoint_config,
        )

        # Optional interactive shell — drop into it while both models are in memory.
        # Rank 0 gets the shell; all other ranks wait at a distributed barrier.
        if args.debug_interact:
            _open_debug_shell(loaded, args)

        # Iterate every tensor the bridge can export, compare against the HF state dict.
        with torch.inference_mode():
            for name, megatron_param in bridge.export_hf_weights(
                megatron_model, show_progress=args.show_progress
            ):
                _compare_tensor(
                    name=name,
                    megatron_param=megatron_param,
                    hf_state=loaded.hf_state,
                    fp32_substrings=fp32_substrings,
                    summary=summary,
                    samples=samples,
                    atol=args.atol,
                    rtol=args.rtol,
                    exact_match=args.exact_match,
                    max_report=args.max_report,
                )

    return _print_report(summary, samples)


if __name__ == "__main__":
    raise SystemExit(main())
