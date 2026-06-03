"""
Utilities shared between progress_tracker and any sweep-validation tooling.

Provides:
  _resolve_defaults          -- merge Hydra-style defaults into a parsed cfg dict
  render_job_name            -- render a job-name template for a given combo/stage
  substitute_omegaconf_path_vars -- substitute ${key} refs from a context dict
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml


# ── helpers ────────────────────────────────────────────────────────────────────

def _deep_merge(base: dict, override: dict) -> dict:
    """Merge *override* into *base* in-place (nested dicts are merged recursively)."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def _find_config_root(config_path: str) -> Path:
    """Walk up from config_path until we find the ancestor directory named 'config'."""
    p = Path(config_path).resolve()
    for ancestor in p.parents:
        if ancestor.name == "config":
            return ancestor
    # fallback: assume config/experiments/.../file.yaml layout (4 levels up)
    return p.parent.parent.parent.parent


def _read_package(yaml_text: str) -> str | None:
    """Return the value of a '# @package <name>' directive from the first 5 lines."""
    for line in yaml_text.splitlines()[:5]:
        m = re.match(r"#\s*@package\s+(\S+)", line)
        if m:
            return m.group(1)
    return None


def _navigate(cfg: dict, dotted_path: str) -> dict:
    """Return (creating as needed) the nested dict at *dotted_path* inside *cfg*."""
    node = cfg
    for key in dotted_path.split("."):
        if key not in node or not isinstance(node[key], dict):
            node[key] = {}
        node = node[key]
    return node


def _apply_default(cfg: dict, config_root: Path, raw_key: str, value: str) -> None:
    """Load one defaults-list entry and merge it into *cfg*."""
    # /sweep/multilingual_scaling@sweep → group=sweep/multilingual_scaling, pkg_override=sweep
    if "@" in raw_key:
        config_group, pkg_override = raw_key.rsplit("@", 1)
    else:
        config_group, pkg_override = raw_key, None

    config_group = config_group.strip("/")
    yaml_file = config_root / config_group / f"{value}.yaml"
    if not yaml_file.is_file():
        return

    text = yaml_file.read_text(errors="replace")
    sub_cfg = yaml.safe_load(text) or {}

    # Determine target package (in dotted form, e.g. "backend.megatron")
    if pkg_override is not None:
        package = pkg_override
    else:
        file_pkg = _read_package(text)
        if file_pkg and file_pkg != "_global_":
            package = file_pkg
        else:
            # infer from config group path: backend/megatron → backend.megatron
            package = config_group.replace("/", ".")

    target = _navigate(cfg, package) if package else cfg
    _deep_merge(target, sub_cfg)

    # Recurse into the sub-config's own defaults (one level only — enough for base.yaml)
    for entry in sub_cfg.get("defaults", []):
        if entry == "_self_" or isinstance(entry, str):
            continue
        if not isinstance(entry, dict):
            continue
        for sub_key, sub_val in entry.items():
            # Resolve relative to the sub-config's directory
            sub_root = yaml_file.parent
            # Absolute keys start with /; relative keys are under the same group dir
            if str(sub_key).startswith("/"):
                _apply_default(cfg, config_root, sub_key, str(sub_val))
            else:
                rel_key = f"{config_group}/{sub_key}".strip("/")
                _apply_default(cfg, config_root, f"/{rel_key}", str(sub_val))


def _resolve_defaults(cfg: dict, config_path: str) -> None:
    """Merge Hydra-style defaults list entries into *cfg* in-place.

    Handles the subset needed by parse_config:
      - sweep YAML (via the @sweep key)  → cfg["sweep"]
      - model/data YAMLs with @package backend.megatron → cfg["backend"]["megatron"]
    """
    config_root = _find_config_root(config_path)
    for entry in cfg.get("defaults", []):
        if entry == "_self_" or isinstance(entry, str):
            continue
        if not isinstance(entry, dict):
            continue
        for raw_key, value in entry.items():
            _apply_default(cfg, config_root, str(raw_key), str(value))


# ── job-name rendering ─────────────────────────────────────────────────────────

def substitute_omegaconf_path_vars(template: str, ctx: dict[str, Any]) -> str:
    """Replace ${key} references in *template* with values from *ctx*.

    Unresolvable references are left as-is.
    """
    def _replace(m: re.Match) -> str:
        key = m.group(1)
        return str(ctx[key]) if key in ctx else m.group(0)

    return re.sub(r"\$\{([^}]+)\}", _replace, template)


def render_job_name(
    tpl: str,
    num_experts: int,
    lr: float,
    gbsz: int,
    seed: int,
    stage: str,
    tokens: int | None = None,
) -> str:
    """Render a job-name template for a given (lr, gbsz, stage, tokens) combo.

    *tokens* is the stable-phase token budget (sets job_horizon_suffix to e.g.
    "50BT").  For decay stages *tokens* is omitted — the suffix is empty because
    the budget is already embedded in *stage* (e.g. "decay6BT").

    The template uses escaped OmegaConf syntax (``\\${...}``) so that Hydra
    doesn't interpolate it at config-load time; we unescape before substituting.
    """
    # Unescape \${ → ${
    unescaped = tpl.replace("\\${", "${")

    horizon_suffix = f"{tokens // 1_000_000_000}BT" if tokens is not None else ""

    ctx: dict[str, Any] = {
        "backend.megatron.lr":                  str(lr),
        "backend.megatron.global_batch_size":   str(gbsz),
        "backend.megatron.seed":                str(seed),
        "backend.megatron.num_experts":         str(num_experts),
        "stage":                                stage,
        "backend.megatron.aux.job_horizon_suffix": horizon_suffix,
    }
    return substitute_omegaconf_path_vars(unescaped, ctx)
