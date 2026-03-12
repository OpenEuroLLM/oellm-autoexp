#!/usr/bin/env python3
"""
Export Megatron checkpoint to HuggingFace Qwen3 MoE format.

This script converts a Megatron-LM checkpoint (with custom config) to 
HuggingFace Qwen3 MoE format that can be loaded with transformers.

The script auto-detects your model configuration from the Megatron checkpoint
and displays it before export. You provide a reference Qwen3 MoE model (or use
the default) for tokenizer and structure, but the actual architecture comes from
your checkpoint.

Usage:
    # Simplest - uses default reference model, auto-detects your config
    python export_megatron_to_qwen3_moe.py \
        --megatron-path ./checkpoints/my_qwen3_moe \
        --hf-output-path ./exports/my_qwen3_moe_hf
    
    # Specify reference model explicitly (for tokenizer/structure)
    python export_megatron_to_qwen3_moe.py \
        --megatron-path ./checkpoints/my_qwen3_moe \
        --hf-output-path ./exports/my_qwen3_moe_hf \
        --reference-model Qwen/Qwen3-30B-A3B
    
    # Or use a predefined provider for a specific model
    python export_megatron_to_qwen3_moe.py \
        --megatron-path ./checkpoints/my_qwen3_moe \
        --hf-output-path ./exports/my_qwen3_moe_hf \
        --model-provider qwen3_moe_30b
"""

import argparse
import json
import os
import sys
import shutil
import atexit
import tempfile
from pathlib import Path
from typing import TextIO

import torch
import yaml
from rich.console import Console

from megatron.bridge import AutoBridge


console = Console()


# -----------------------------------------------------------------------------
# Logging and type-coercion helpers
# -----------------------------------------------------------------------------


class TeeStream:
    """Mirror writes to the terminal stream and a log file stream."""

    def __init__(self, terminal_stream: TextIO, log_stream: TextIO):
        self._terminal_stream = terminal_stream
        self._log_stream = log_stream

    def write(self, data):
        self._terminal_stream.write(data)
        self._log_stream.write(data)

    def flush(self):
        self._terminal_stream.flush()
        self._log_stream.flush()

    def isatty(self):
        return self._terminal_stream.isatty()


def _setup_stdout_stderr_tee(log_file_path: str) -> None:
    """Tee process stdout/stderr to a log file while preserving terminal output."""
    log_path = Path(log_file_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "a", buffering=1, encoding="utf-8")
    atexit.register(log_file.close)

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(original_stdout, log_file)
    sys.stderr = TeeStream(original_stderr, log_file)


def _to_int(value):
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(float(value))
    return int(value)


def _to_float(value):
    if value is None:
        return None
    if isinstance(value, float):
        return value
    if isinstance(value, str):
        return float(value)
    return float(value)


def _to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


# Validate that checkpoint fields needed by Qwen3-MoE export are present and sane
# before any model construction starts.
def validate_checkpoint_config_for_qwen3_moe(cfg: dict) -> tuple[list[str], list[str]]:
    """Validate checkpoint config has required Qwen3-MoE fields."""
    errors: list[str] = []
    warnings: list[str] = []

    required_int_fields = [
        "num_layers",
        "hidden_size",
        "num_attention_heads",
        "num_query_groups",
        "ffn_hidden_size",
        "moe_ffn_hidden_size",
        "num_moe_experts",
        "moe_router_topk",
    ]
    for key in required_int_fields:
        if key not in cfg:
            errors.append(f"missing required field: {key}")
            continue
        try:
            _ = _to_int(cfg[key])
        except Exception:
            errors.append(f"invalid integer field {key}: {cfg[key]}")

    if "num_moe_experts" in cfg and _to_int(cfg.get("num_moe_experts", 0)) <= 0:
        errors.append("num_moe_experts must be > 0 for Qwen3 MoE")
    if "moe_router_topk" in cfg and _to_int(cfg.get("moe_router_topk", 0)) <= 0:
        errors.append("moe_router_topk must be > 0 for Qwen3 MoE")

    if cfg.get("gated_linear_unit") is False:
        warnings.append("gated_linear_unit=False (Qwen3 MoE usually uses gated_linear_unit=True)")
    if cfg.get("qk_layernorm") is False:
        warnings.append("qk_layernorm=False (Qwen3 MoE usually uses qk_layernorm=True)")
    if cfg.get("normalization") not in (None, "RMSNorm"):
        warnings.append(f"normalization={cfg.get('normalization')} (expected RMSNorm)")

    return errors, warnings


# Copy architecture-defining values from checkpoint config into HF config using a
# stable key mapping, including derived fields like head_dim when needed.
def sync_hf_config_from_checkpoint(hf_config, cfg: dict) -> list[str]:
    """Apply checkpoint architecture values onto HF config and return change log."""
    mapping = {
        "num_layers": "num_hidden_layers",
        "hidden_size": "hidden_size",
        "num_attention_heads": "num_attention_heads",
        "num_query_groups": "num_key_value_heads",
        "kv_channels": "head_dim",
        "ffn_hidden_size": "intermediate_size",
        "moe_ffn_hidden_size": "moe_intermediate_size",
        "num_moe_experts": "num_experts",
        "moe_router_topk": "num_experts_per_tok",
        "vocab_size": "vocab_size",
        "max_position_embeddings": "max_position_embeddings",
        "layernorm_epsilon": "rms_norm_eps",
        "rotary_base": "rope_theta",
        "attention_dropout": "attention_dropout",
    }

    changes: list[str] = []
    for ckpt_key, hf_key in mapping.items():
        if ckpt_key not in cfg or not hasattr(hf_config, hf_key):
            continue
        raw_value = cfg[ckpt_key]
        if ckpt_key in {
            "num_layers",
            "hidden_size",
            "num_attention_heads",
            "num_query_groups",
            "ffn_hidden_size",
            "moe_ffn_hidden_size",
            "num_moe_experts",
            "moe_router_topk",
            "vocab_size",
            "max_position_embeddings",
        }:
            new_value = _to_int(raw_value)
        elif ckpt_key == "kv_channels":
            new_value = _to_int(raw_value)
        else:
            new_value = _to_float(raw_value)

        old_value = getattr(hf_config, hf_key)
        if old_value != new_value:
            setattr(hf_config, hf_key, new_value)
            changes.append(f"{hf_key}: {old_value} -> {new_value}")

    # Ensure model type stays correct for output config.
    if hasattr(hf_config, "model_type") and hf_config.model_type != "qwen3_moe":
        changes.append(f"model_type: {hf_config.model_type} -> qwen3_moe")
        hf_config.model_type = "qwen3_moe"

    # Derive head_dim from hidden_size // num_attention_heads if kv_channels not explicit.
    if "kv_channels" not in cfg and hasattr(hf_config, "head_dim"):
        nh = _to_int(cfg.get("num_attention_heads"))
        hs = _to_int(cfg.get("hidden_size"))
        if nh and hs:
            derived = hs // nh
            if getattr(hf_config, "head_dim", None) != derived:
                changes.append(f"head_dim: {getattr(hf_config, 'head_dim', None)} -> {derived} (derived)")
                hf_config.head_dim = derived

    # Align tied/untied embeddings with checkpoint semantics.
    untie = None
    if "untie_embeddings_and_output_weights" in cfg:
        untie = _to_bool(cfg["untie_embeddings_and_output_weights"])
    elif "share_embeddings_and_output_weights" in cfg:
        untie = not _to_bool(cfg["share_embeddings_and_output_weights"])
    elif "tie_word_embeddings" in cfg:
        untie = not _to_bool(cfg["tie_word_embeddings"])

    if untie is not None and hasattr(hf_config, "tie_word_embeddings"):
        target_tie = not untie
        if hf_config.tie_word_embeddings != target_tie:
            changes.append(f"tie_word_embeddings: {hf_config.tie_word_embeddings} -> {target_tie}")
            hf_config.tie_word_embeddings = target_tie

    return changes


# Apply mapped architecture values and preserve raw checkpoint config in the output
# config for reproducibility and debugging.
def apply_full_checkpoint_config_to_hf(hf_config, cfg: dict) -> tuple[list[str], int]:
    """
    Apply checkpoint config comprehensively to HF config.

    1) Sync canonical architecture fields (HF-native keys)
    2) Preserve full raw checkpoint config under `megatron_modelopt_run_config`

    Returns:
        (changes, num_raw_keys_preserved)
    """
    changes = sync_hf_config_from_checkpoint(hf_config, cfg)

    # Preserve the entire checkpoint config for reproducibility/auditing.
    setattr(hf_config, "megatron_modelopt_run_config", dict(cfg))

    # Also expose a concise marker to indicate this came from Megatron checkpoint config.
    setattr(hf_config, "megatron_config_source", "modelopt_run_config.yaml")

    return changes, len(cfg)


# Compare exported HF config with checkpoint config over mapped fields to catch
# accidental drift when exact matching is requested.
def find_hf_checkpoint_config_mismatches(hf_cfg: dict, ckpt_cfg: dict) -> list[str]:
    """Return a list of config mismatches between exported HF config and Megatron checkpoint config."""
    mapping = {
        "num_layers": "num_hidden_layers",
        "hidden_size": "hidden_size",
        "num_attention_heads": "num_attention_heads",
        "num_query_groups": "num_key_value_heads",
        "kv_channels": "head_dim",
        "ffn_hidden_size": "intermediate_size",
        "moe_ffn_hidden_size": "moe_intermediate_size",
        "num_moe_experts": "num_experts",
        "moe_router_topk": "num_experts_per_tok",
        "vocab_size": "vocab_size",
        "max_position_embeddings": "max_position_embeddings",
        "layernorm_epsilon": "rms_norm_eps",
        "rotary_base": "rope_theta",
        "attention_dropout": "attention_dropout",
    }

    mismatches: list[str] = []
    for ckpt_key, hf_key in mapping.items():
        if ckpt_key not in ckpt_cfg or hf_key not in hf_cfg:
            continue

        ckpt_raw = ckpt_cfg[ckpt_key]
        if ckpt_key in {
            "num_layers",
            "hidden_size",
            "num_attention_heads",
            "num_query_groups",
            "kv_channels",
            "ffn_hidden_size",
            "moe_ffn_hidden_size",
            "num_moe_experts",
            "moe_router_topk",
            "vocab_size",
            "max_position_embeddings",
        }:
            ckpt_value = _to_int(ckpt_raw)
        else:
            ckpt_value = _to_float(ckpt_raw)

        hf_value = hf_cfg[hf_key]
        if hf_value != ckpt_value:
            mismatches.append(f"{hf_key}: HF={hf_value} vs checkpoint={ckpt_value}")

    if "untie_embeddings_and_output_weights" in ckpt_cfg and "tie_word_embeddings" in hf_cfg:
        expected_tie = not _to_bool(ckpt_cfg["untie_embeddings_and_output_weights"])
        if hf_cfg["tie_word_embeddings"] != expected_tie:
            mismatches.append(
                f"tie_word_embeddings: HF={hf_cfg['tie_word_embeddings']} vs checkpoint_expected={expected_tie}"
            )

    return mismatches


# Resolve model ID to a concrete local path in offline/local-only mode, preferring
# local cache snapshots to avoid network metadata calls.
def _resolve_reference_model_path(
    reference_model: str,
    hf_cache_dir: str | None,
    local_files_only: bool,
) -> str:
    """Resolve a HF model ID to a local path when offline/local-only mode is used."""
    ref_path = Path(reference_model)
    if ref_path.exists():
        return str(ref_path)

    if not local_files_only:
        return reference_model

    # In local-files-only mode, resolve to an existing local snapshot to avoid hub metadata calls.
    cache_candidates: list[Path] = []
    if hf_cache_dir:
        cache_candidates.append(Path(hf_cache_dir))
    if os.getenv("HF_HUB_CACHE"):
        cache_candidates.append(Path(os.getenv("HF_HUB_CACHE")))
    if os.getenv("HF_HOME"):
        cache_candidates.append(Path(os.getenv("HF_HOME")) / "hub")

    # Fallback default HF cache locations
    cache_candidates.extend(
        [
            Path.home() / ".cache" / "huggingface" / "hub",
            Path.home() / ".cache" / "huggingface",
        ]
    )

    org, name = reference_model.split("/", 1) if "/" in reference_model else ("", reference_model)
    repo_dir_name = f"models--{org}--{name}" if org else f"models--{name}"

    for base in cache_candidates:
        if not base.exists():
            continue

        # Handle both .../hub and HF_HOME-style roots.
        hub_base = base / "hub" if (base / "hub").exists() else base
        repo_dir = hub_base / repo_dir_name
        snapshots_dir = repo_dir / "snapshots"
        if snapshots_dir.exists():
            snapshots = sorted([p for p in snapshots_dir.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime)
            if snapshots:
                resolved = str(snapshots[-1])
                console.print(f"[green]✓[/green] Resolved cached snapshot: {resolved}")
                return resolved

    # Last attempt via huggingface_hub local-only snapshot resolver.
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(
            repo_id=reference_model,
            local_files_only=True,
            cache_dir=hf_cache_dir,
        )
    except Exception as e:
        raise FileNotFoundError(
            f"Could not resolve local cache for reference model '{reference_model}'. "
            f"Pass a local path to --reference-model or set --hf-cache-dir correctly. Last error: {e}"
        ) from e


# Create a temporary local HF template with checkpoint-sized architecture so bridge
# export uses the intended target layout (instead of reference-model full-size layout).
def _build_checkpoint_sized_reference_template(
    reference_model_or_path: str,
    checkpoint_config: dict,
    hf_cache_dir: str | None,
    local_files_only: bool,
) -> str:
    """Create a local HF reference model with checkpoint-matched architecture.

    This prevents export from inheriting the original reference model shard/key map
    (e.g., 48 layers / 128 experts) when the checkpoint has a smaller shape.
    """
    from transformers import AutoConfig, AutoTokenizer, GenerationConfig, Qwen3MoeForCausalLM

    resolved_reference = _resolve_reference_model_path(reference_model_or_path, hf_cache_dir, local_files_only)
    base_config = AutoConfig.from_pretrained(
        resolved_reference,
        trust_remote_code=True,
        local_files_only=local_files_only,
        cache_dir=hf_cache_dir,
    )

    # Keep tokenizer + artifact layout from the reference model, but force architecture
    # fields to the checkpoint dimensions so export targets the intended shape.
    sync_hf_config_from_checkpoint(base_config, checkpoint_config)
    if hasattr(base_config, "model_type"):
        base_config.model_type = "qwen3_moe"
    setattr(base_config, "architectures", ["Qwen3MoeForCausalLM"])

    template_dir = Path(tempfile.mkdtemp(prefix="qwen3_moe_ckpt_ref_"))

    # Build a local skeleton model only to materialize a checkpoint-sized HF template.
    # This is intentionally temporary and removed after export.
    template_model = Qwen3MoeForCausalLM(base_config)
    template_model.save_pretrained(template_dir, safe_serialization=True, max_shard_size="50GB")
    del template_model

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            resolved_reference,
            trust_remote_code=True,
            local_files_only=local_files_only,
            cache_dir=hf_cache_dir,
        )
        tokenizer.save_pretrained(template_dir)
    except Exception as e:
        console.print(f"[yellow]⚠[/yellow] Could not copy tokenizer into template: {e}")

    try:
        generation_config = GenerationConfig.from_pretrained(
            resolved_reference,
            local_files_only=local_files_only,
            cache_dir=hf_cache_dir,
        )
        generation_config.save_pretrained(template_dir)
    except Exception:
        # Optional file; no action needed if absent.
        pass

    console.print(f"[green]✓[/green] Created checkpoint-sized local reference template: {template_dir}")
    return str(template_dir)


# Discover and load checkpoint-side YAML config from top-level or latest iter_* dir.
def load_megatron_config(megatron_path: Path) -> dict:
    """
    Load model configuration from Megatron checkpoint.
    
    Args:
        megatron_path: Path to Megatron checkpoint directory
        
    Returns:
        dict: Parsed configuration
    """
    # Look for config files
    config_files = [
        megatron_path / "modelopt_run_config.yaml",
        megatron_path / "run_config.yaml",
        megatron_path / "config.yaml",
    ]
    
    # Also check in iter_* subdirectories
    iter_dirs = sorted([d for d in megatron_path.iterdir() if d.is_dir() and d.name.startswith("iter_")])
    if iter_dirs:
        latest_iter = iter_dirs[-1]
        config_files.extend([
            latest_iter / "modelopt_run_config.yaml",
            latest_iter / "run_config.yaml",
            latest_iter / "config.yaml",
        ])
    
    for config_file in config_files:
        if config_file.exists():
            console.print(f"[green]✓[/green] Found config: {config_file}")
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            return config
    
    raise FileNotFoundError(
        f"Could not find config file in {megatron_path}. "
        f"Looked for: modelopt_run_config.yaml, run_config.yaml, config.yaml"
    )


def export_megatron_to_qwen3_moe(
    megatron_path: str,
    hf_output_path: str,
    checkpoint_config: dict | None = None,
    reference_model: str = None,
    model_provider: str = None,
    hf_cache_dir: str | None = None,
    local_files_only: bool = False,
    offline: bool = False,
    show_progress: bool = True,
    strict: bool = False,
    exact_config_match: bool = False,
    use_checkpoint_sized_reference_template: bool = True,
) -> None:
    """
    Export Megatron checkpoint to HuggingFace Qwen3 MoE format.
    
    Args:
        megatron_path: Path to Megatron checkpoint directory
        hf_output_path: Output path for HuggingFace model
        checkpoint_config: Parsed Megatron checkpoint config for validation/sync
        reference_model: HuggingFace model ID to use as config reference (optional)
        model_provider: Specific model provider to use (optional)
        hf_cache_dir: Optional path to local HuggingFace cache directory
        local_files_only: Force loading only from local files/cache (no downloads)
        offline: Enable strict offline mode via environment variables
        show_progress: Show progress bar during export
        strict: Strict weight matching during export
        use_checkpoint_sized_reference_template: Build a temporary local HF reference
            with checkpoint-matched architecture to avoid inheriting full-size
            source shard/key expectations.
    """
    console.print("\n[bold cyan]Exporting Megatron → HuggingFace Qwen3 MoE[/bold cyan]")
    console.print(f"Source: {megatron_path}")
    console.print(f"Target: {hf_output_path}")
    
    # Validate megatron path exists
    megatron_path_obj = Path(megatron_path)
    if not megatron_path_obj.exists():
        console.print(f"[red]Error:[/red] Megatron path does not exist: {megatron_path}")
        sys.exit(1)
    
    # Create output directory
    hf_output_path_obj = Path(hf_output_path)
    hf_output_path_obj.mkdir(parents=True, exist_ok=True)

    # Configure explicit HuggingFace cache paths when provided.
    if hf_cache_dir:
        cache_path = Path(hf_cache_dir)
        # Accept both HF_HOME-style and direct hub cache paths.
        if cache_path.name == "hub":
            os.environ["HF_HUB_CACHE"] = str(cache_path)
            os.environ["HF_HOME"] = str(cache_path.parent)
        else:
            os.environ["HF_HOME"] = str(cache_path)
            os.environ["HF_HUB_CACHE"] = str(cache_path / "hub")

    # Enforce strict offline behavior when requested.
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        local_files_only = True
        console.print("[cyan]Offline mode enabled (HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1)[/cyan]")
    
    # ------------------------------------------------------------------
    # 1) Resolve reference artifacts (config/tokenizer) source.
    # ------------------------------------------------------------------
    if not reference_model and not model_provider:
        console.print("[red]Error:[/red] Must specify either --reference-model or --model-provider")
        sys.exit(1)

    if model_provider:
        console.print(f"Using model provider: {model_provider}")
        provider_to_ref = {
            "qwen3_moe": "Qwen/Qwen3-30B-A3B",
            "qwen3_moe_30b": "Qwen/Qwen3-30B-A3B",
            "qwen3_moe_235b": "Qwen/Qwen3-235B-A22B",
        }
        if model_provider.lower() not in provider_to_ref:
            console.print(f"[red]Error:[/red] Unknown provider: {model_provider}")
            console.print(f"Available providers: {list(provider_to_ref.keys())}")
            sys.exit(1)
        reference_model = provider_to_ref[model_provider.lower()]
        console.print(f"Resolved provider to reference model: {reference_model}")
    else:
        console.print(f"Using reference model for config: {reference_model}")

    resolved_reference = _resolve_reference_model_path(reference_model, hf_cache_dir, local_files_only)
    if resolved_reference != reference_model:
        console.print(f"Using local snapshot path: {resolved_reference}")

    # ------------------------------------------------------------------
    # 2) Optionally build a checkpoint-sized local HF template.
    # ------------------------------------------------------------------
    template_reference_path: str | None = None
    bridge_source_path = resolved_reference
    if checkpoint_config is not None and use_checkpoint_sized_reference_template:
        try:
            template_reference_path = _build_checkpoint_sized_reference_template(
                reference_model_or_path=resolved_reference,
                checkpoint_config=checkpoint_config,
                hf_cache_dir=hf_cache_dir,
                local_files_only=local_files_only,
            )
            bridge_source_path = template_reference_path
        except Exception as e:
            console.print(
                "[yellow]⚠[/yellow] Could not build checkpoint-sized reference template; "
                "falling back to direct reference model."
            )
            console.print(f"[yellow]Reason:[/yellow] {e}")
            console.print(
                "[yellow]Tip:[/yellow] Re-run with --no-checkpoint-sized-reference-template "
                "to skip template creation explicitly."
            )

    # Bridge is initialized from either:
    # - checkpoint-sized template (preferred for clean target shard/index layout), or
    # - original reference model (fallback path).
    bridge = AutoBridge.from_hf_pretrained(
        bridge_source_path,
        trust_remote_code=True,
        local_files_only=local_files_only,
        cache_dir=hf_cache_dir,
    )
    
    detected_model_type = getattr(getattr(bridge.hf_pretrained, "config", None), "model_type", "unknown")
    console.print(f"[green]✓[/green] Bridge initialized for model type: {detected_model_type}")

    # ------------------------------------------------------------------
    # 3) Validate checkpoint config and stamp it into HF config metadata.
    # ------------------------------------------------------------------
    if checkpoint_config is not None:
        errors, warnings = validate_checkpoint_config_for_qwen3_moe(checkpoint_config)
        if warnings:
            for warning in warnings:
                console.print(f"[yellow]⚠[/yellow] Checkpoint config warning: {warning}")
        if errors:
            for error in errors:
                console.print(f"[red]✗[/red] Checkpoint config error: {error}")
            raise ValueError("Checkpoint config validation failed. Refusing to export blindly.")

        cfg_changes, raw_key_count = apply_full_checkpoint_config_to_hf(bridge.hf_pretrained.config, checkpoint_config)
        if cfg_changes:
            console.print("[cyan]Applying checkpoint config to HF config:[/cyan]")
            for item in cfg_changes:
                console.print(f"  - {item}")
        else:
            console.print("[green]✓[/green] HF config already matches checkpoint config")
        console.print(f"[green]✓[/green] Preserved full checkpoint config payload ({raw_key_count} keys) in HF config")
    
    # ------------------------------------------------------------------
    # 4) Build Megatron model, load checkpoint, and export to HF.
    # ------------------------------------------------------------------
    console.print("\n[bold]Starting export...[/bold]")
    try:
        from megatron.bridge.training.model_load_save import build_and_load_model, temporary_distributed_context

        # Get the Qwen3MoE-aware model provider from the bridge.
        # This builds the correct Qwen3MoE architecture (not a generic GPTModel),
        # which the bridge knows how to map to HF weight layout.
        # load_weights=False because we supply weights from the Megatron checkpoint.
        console.print("[cyan]Getting Qwen3MoE model provider from bridge...[/cyan]")
        provider = bridge.to_megatron_provider(load_weights=False)
        console.print("[green]✓[/green] Got Qwen3MoE provider")

        # Override provider dimensions from checkpoint config so the model shape matches.
        if checkpoint_config is not None:
            dim_map = {
                "num_layers": "num_layers",
                "hidden_size": "hidden_size",
                "num_attention_heads": "num_attention_heads",
                "num_query_groups": "num_query_groups",
                "kv_channels": "kv_channels",
                "ffn_hidden_size": "ffn_hidden_size",
                "moe_ffn_hidden_size": "moe_ffn_hidden_size",
                "num_moe_experts": "num_moe_experts",
                "moe_router_topk": "moe_router_topk",
            }
            for ckpt_key, prov_attr in dim_map.items():
                if ckpt_key in checkpoint_config and hasattr(provider, prov_attr):
                    setattr(provider, prov_attr, _to_int(checkpoint_config[ckpt_key]))

            # Derive kv_channels from hidden_size // num_attention_heads when not explicit.
            # The reference model (e.g. 30B) has a different head_dim; leaving it would
            # cause a shape mismatch in linear_proj weights.
            if "kv_channels" not in checkpoint_config:
                nh = _to_int(checkpoint_config.get("num_attention_heads"))
                hs = _to_int(checkpoint_config.get("hidden_size"))
                if nh and hs and hasattr(provider, "kv_channels"):
                    derived_kv = hs // nh
                    provider.kv_channels = derived_kv
                    console.print(f"  Derived kv_channels={derived_kv} from hidden_size({hs}) // num_attention_heads({nh})")

            # Align embedding/output tying so checkpoint metadata and expected keys match.
            untie = None
            if "untie_embeddings_and_output_weights" in checkpoint_config:
                untie = _to_bool(checkpoint_config["untie_embeddings_and_output_weights"])
            elif "share_embeddings_and_output_weights" in checkpoint_config:
                untie = not _to_bool(checkpoint_config["share_embeddings_and_output_weights"])
            elif "tie_word_embeddings" in checkpoint_config:
                untie = not _to_bool(checkpoint_config["tie_word_embeddings"])

            # Most GPT checkpoints with missing output_layer.weight are tied.
            if untie is None:
                untie = False

            if hasattr(provider, "untie_embeddings_and_output_weights"):
                provider.untie_embeddings_and_output_weights = untie
            if hasattr(provider, "share_embeddings_and_output_weights"):
                provider.share_embeddings_and_output_weights = not untie

            console.print(
                f"  Embedding/output tying: untie_embeddings_and_output_weights={untie}"
            )

            console.print("[green]✓[/green] Applied checkpoint dimensions to provider")

        # Keep loading + export in one distributed context so parallel_state groups remain valid.
        console.print("[cyan]Loading Megatron checkpoint into Qwen3MoE model...[/cyan]")
        with temporary_distributed_context(backend="gloo"):
            megatron_model = build_and_load_model(
                checkpoint_path=megatron_path,
                model_cfg=provider,
                use_cpu_init=True,
            )

            console.print("[green]✓[/green] Loaded Megatron model from checkpoint")

            if not isinstance(megatron_model, list):
                megatron_model = [megatron_model]

            console.print("[cyan]Exporting to HuggingFace format...[/cyan]")
            bridge.save_hf_pretrained(
                megatron_model,
                hf_output_path,
                show_progress=show_progress,
                strict=strict,
            )

        # Preserve original Megatron config file alongside exported HF artifacts.
        candidate_cfg_files = [
            Path(megatron_path) / "modelopt_run_config.yaml",
            Path(megatron_path) / "run_config.yaml",
            Path(megatron_path) / "config.yaml",
        ]
        for cfg_file in candidate_cfg_files:
            if cfg_file.exists():
                shutil.copy2(cfg_file, Path(hf_output_path) / cfg_file.name)
                console.print(f"[green]✓[/green] Copied source config file: {cfg_file.name}")
                break

        if exact_config_match and checkpoint_config is not None:
            exported_config_file = Path(hf_output_path) / "config.json"
            if not exported_config_file.exists():
                raise FileNotFoundError(f"Expected exported config file was not found: {exported_config_file}")

            with open(exported_config_file, "r", encoding="utf-8") as f:
                exported_cfg = json.load(f)

            mismatches = find_hf_checkpoint_config_mismatches(exported_cfg, checkpoint_config)
            if mismatches:
                console.print("[red]✗[/red] Exact config check failed. Mismatches:")
                for item in mismatches:
                    console.print(f"  - {item}")
                raise ValueError("Exported HF config does not exactly match checkpoint config (mapped fields).")

            console.print("[green]✓[/green] Exact config check passed (mapped fields match checkpoint).")

        console.print("\n[green]✓[/green] Successfully exported to HuggingFace format")
    except Exception as e:
        console.print(f"\n[red]✗[/red] Export failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if 'template_reference_path' in locals() and template_reference_path:
            try:
                shutil.rmtree(template_reference_path, ignore_errors=True)
            except Exception:
                pass
    
    # ------------------------------------------------------------------
    # 5) Summarize outputs and run a lightweight load check.
    # ------------------------------------------------------------------
    if hf_output_path_obj.exists():
        console.print("\n[bold]Output structure:[/bold]")
        files = sorted(hf_output_path_obj.iterdir())
        total_size = 0
        
        for item in files:
            if item.is_dir():
                console.print(f"  📂 {item.name}/")
            else:
                size_mb = item.stat().st_size / (1024 * 1024)
                total_size += size_mb
                console.print(f"  📄 {item.name} ({size_mb:.2f} MB)")
        
        console.print(f"\n[bold]Total size:[/bold] {total_size:.2f} MB")
    
    # Verify the model can be loaded
    console.print("\n[bold]Verifying exported model...[/bold]")
    try:
        from transformers import Qwen3MoeForCausalLM
        model = Qwen3MoeForCausalLM.from_pretrained(
            hf_output_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            local_files_only=local_files_only,
            cache_dir=hf_cache_dir,
        )
        num_params = sum(p.numel() for p in model.parameters())
        console.print(f"[green]✓[/green] Model loads successfully")
        console.print(f"  Parameters: {num_params / 1e9:.2f}B")
        console.print(f"  Layers: {model.config.num_hidden_layers}")
        console.print(f"  Hidden size: {model.config.hidden_size}")
        console.print(f"  Num experts: {model.config.num_experts}")
        console.print(f"  Experts per token: {model.config.num_experts_per_tok}")
    except Exception as e:
        console.print(f"[yellow]⚠[/yellow] Could not verify model loading: {e}")
    
    console.print(f"\n[bold green]Export complete![/bold green]")
    console.print(f"HuggingFace Qwen3 MoE model saved to: {hf_output_path}")


def main():
    # Keep CLI focused: reference/model provider selection + export behavior flags.
    parser = argparse.ArgumentParser(
        description="Export Megatron checkpoint to HuggingFace Qwen3 MoE format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Required arguments
    parser.add_argument(
        "--megatron-path",
        type=str,
        required=True,
        help="Path to Megatron checkpoint directory",
    )
    parser.add_argument(
        "--hf-output-path",
        type=str,
        required=True,
        help="Output path for HuggingFace model",
    )
    
    # Config source (one required)
    config_group = parser.add_argument_group("Config Source (optional - auto-detects from checkpoint if not specified)")
    config_group.add_argument(
        "--reference-model",
        type=str,
        help="HuggingFace model ID to use as config reference (e.g., Qwen/Qwen3-30B-A3B)",
    )
    config_group.add_argument(
        "--model-provider",
        type=str,
        choices=["qwen3_moe", "qwen3_moe_30b", "qwen3_moe_235b"],
        help="Predefined model provider to use",
    )
    
    # Export options
    export_group = parser.add_argument_group("Export Options")
    export_group.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar",
    )
    export_group.add_argument(
        "--strict",
        action="store_true",
        help="Enable strict weight matching",
    )
    export_group.add_argument(
        "--exact-config-match",
        action="store_true",
        help="Fail export if mapped HF config fields do not exactly match checkpoint config",
    )
    export_group.add_argument(
        "--offline",
        action="store_true",
        help="Strict offline mode (no internet calls; sets HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE)",
    )
    export_group.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load HuggingFace artifacts from local cache/files only",
    )
    export_group.add_argument(
        "--hf-cache-dir",
        type=str,
        default=None,
        help="Path to HuggingFace cache directory (e.g., /path/to/HF_CACHE)",
    )
    export_group.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional log file path to tee stdout/stderr while preserving terminal output",
    )
    export_group.add_argument(
        "--no-checkpoint-sized-reference-template",
        action="store_true",
        help=(
            "Disable auto-creation of a checkpoint-sized local HF reference template. "
            "Use this only if you intentionally want source-model shard/key expectations."
        ),
    )
    
    args = parser.parse_args()

    if args.log_file:
        _setup_stdout_stderr_tee(args.log_file)
        global console
        console = Console()
        console.print(f"[cyan]Logging stdout/stderr to:[/cyan] {args.log_file}")
    
    # Auto-detect and display config from checkpoint
    megatron_config = None
    if args.megatron_path:
        try:
            console.print("[cyan]Reading model config from Megatron checkpoint...[/cyan]")
            megatron_config = load_megatron_config(Path(args.megatron_path))
            console.print("[green]✓[/green] Successfully loaded config from checkpoint\n")
        
            # Display detected configuration
            console.print("[bold]Auto-detected Model Configuration:[/bold]")
            console.print(f"  Layers: {megatron_config.get('num_layers', 'N/A')}")
            console.print(f"  Hidden size: {megatron_config.get('hidden_size', 'N/A')}")
            console.print(f"  Attention heads: {megatron_config.get('num_attention_heads', 'N/A')}")
            console.print(f"  Query groups (GQA): {megatron_config.get('num_query_groups', 'N/A')}")
            console.print(f"  FFN hidden size: {megatron_config.get('ffn_hidden_size', 'N/A')}")
            console.print(f"  MoE FFN hidden size: {megatron_config.get('moe_ffn_hidden_size', 'N/A')}")
            console.print(f"  Num experts: {megatron_config.get('num_moe_experts', 'N/A')}")
            console.print(f"  Experts per token: {megatron_config.get('moe_router_topk', 'N/A')}")
            console.print()
        except Exception as e:
            console.print(f"[yellow]⚠[/yellow] Could not auto-detect config: {e}\n")

    # Determine reference model to use
    reference_model_to_use = args.reference_model
    
    # Default to Qwen3-30B-A3B if no source specified
    if not reference_model_to_use and not args.model_provider:
        console.print("[cyan]No --reference-model specified. Using Qwen/Qwen3-30B-A3B for tokenizer/config structure.[/cyan]")
        console.print("[cyan](The actual model dimensions are determined by your checkpoint)[/cyan]\n")
        reference_model_to_use = "Qwen/Qwen3-30B-A3B"
    
    # Use the standard export path with reference model
    export_megatron_to_qwen3_moe(
        megatron_path=args.megatron_path,
        hf_output_path=args.hf_output_path,
        checkpoint_config=megatron_config,
        reference_model=reference_model_to_use,
        model_provider=args.model_provider,
        hf_cache_dir=args.hf_cache_dir,
        local_files_only=args.local_files_only,
        offline=args.offline,
        show_progress=not args.no_progress,
        strict=args.strict,
        exact_config_match=args.exact_config_match,
        use_checkpoint_sized_reference_template=not args.no_checkpoint_sized_reference_template,
    )


if __name__ == "__main__":
    main()
