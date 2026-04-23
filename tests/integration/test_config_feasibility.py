"""Integration tests: verify all canonical experiment configs remain schema-compatible.

Checks that each config in config/experiments/ (excluding personal subdirs) can be:
  1. Composed by Hydra (all defaults: references resolve, no broken interpolations)
  2. Parsed by the compoconf schema (no renamed/missing required fields)

These tests run entirely offline — no cluster, GPU, or real filesystem paths needed.
Dummy values are injected for cluster-specific env vars.

Run with:
    pytest tests/integration/test_config_feasibility.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from oellm_autoexp.config.loader import load_config_reference
from oellm_autoexp.config.schema import ConfigSetup

# ── Config discovery ──────────────────────────────────────────────────────────

_CONFIG_DIR = Path("config")
_EXPERIMENTS_DIR = _CONFIG_DIR / "experiments"

# Top-level files that are base fragments, not standalone runnable experiments.
_EXCLUDED_FILES: frozenset[str] = frozenset({"base.yaml"})


def _discover_experiment_configs() -> list[pytest.param]:
    params = []
    for yaml_file in sorted(_EXPERIMENTS_DIR.glob("*.yaml")):
        if yaml_file.name in _EXCLUDED_FILES:
            continue
        config_name = f"experiments/{yaml_file.stem}"
        params.append(pytest.param(config_name, id=yaml_file.stem))
    return params


# ── Dummy env vars ────────────────────────────────────────────────────────────

# Cluster-specific env vars that lack defaults in at least one config.
# Add entries here if a new required var is introduced.
_DUMMY_ENVS: dict[str, str] = {
    "OUTPUT_DIR": "/dummy/output",
    "SLURM_ACCOUNT": "test_account",
    "HF_HOME": "/dummy/hf_home",
    "DATA_DIR": "/dummy/data",
    "OELLM_DATASETS_TOKENIZED_DIR": "/dummy/datasets",
    "CONTAINER_CACHE_DIR": "/dummy/containers",
    "WORK": "/dummy/work",
    "TORCH_HOME": "/dummy/torch_home",
}


# ── Test ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("config_name", _discover_experiment_configs())
def test_experiment_config_is_feasible(config_name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Config composes under Hydra and passes schema validation without errors.

    Uses hydra.compose (not @hydra.main) so no output directories are
    created.
    """
    for key, val in _DUMMY_ENVS.items():
        monkeypatch.setenv(key, val)

    config_setup = ConfigSetup(
        config_name=config_name,
        config_dir=_CONFIG_DIR,
    )
    load_config_reference(config_setup=config_setup)
