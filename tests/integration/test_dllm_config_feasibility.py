"""Integration tests: dllm experiment configs compose under Hydra and pass schema validation.

Same checks as tests/integration/test_config_feasibility.py, applied to every config under
config/experiments/dllm/, including subfolders such as bd/. The top-level test does not scan
those folders.

These tests run entirely offline. Dummy values are injected for cluster-specific env vars.

Run with:
    pytest tests/integration/test_dllm_config_feasibility.py -v
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from oellm_autoexp.config.loader import load_config_reference
from oellm_autoexp.config.schema import ConfigSetup
from oellm_autoexp.dllm.render import discover_dllm_experiments

# ── Config discovery ──────────────────────────────────────────────────────────

_CONFIG_DIR = Path("config")


def _experiment_id(config_name: str) -> str:
    return config_name.removeprefix("experiments/dllm/")


# ── Dummy env vars ────────────────────────────────────────────────────────────

# Same values as tests/integration/test_config_feasibility.py. The dllm configs also read
# USER (for example in data_cache_path), so it is set here when the environment lacks it.
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


@pytest.mark.parametrize("config_name", discover_dllm_experiments(_CONFIG_DIR), ids=_experiment_id)
def test_dllm_experiment_config_is_feasible(
    config_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Config composes under Hydra and passes schema validation without errors.

    Uses hydra.compose (not @hydra.main) so no output directories are
    created.
    """
    for key, val in _DUMMY_ENVS.items():
        monkeypatch.setenv(key, val)
    if "USER" not in os.environ:
        monkeypatch.setenv("USER", "dummy_user")

    config_setup = ConfigSetup(
        config_name=config_name,
        config_dir=str(_CONFIG_DIR),
    )
    load_config_reference(config_setup=config_setup)
