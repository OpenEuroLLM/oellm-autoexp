"""Test that Megatron backend works when Megatron is available."""

from pathlib import Path

import pytest

from oellm_autoexp.backends.base import BackendJobSpec
from oellm_autoexp.config.loader import load_config_reference
from oellm_autoexp.config.schema import BackendInterface, ConfigSetup

try:
    from oellm_autoexp.backends.megatron_backend import MegatronBackend  # noqa: F401

    _HAS_MEGATRON = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_MEGATRON = False


@pytest.mark.skipif(not _HAS_MEGATRON, reason="Megatron parser not available")
def test_megatron_backend_builds_launch_command(monkeypatch, tmp_path):
    """Test that Megatron backend can build launch commands."""
    monkeypatch.setenv("SLURM_ACCOUNT", "debug")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "outputs"))

    config_setup = ConfigSetup(
        config_name="autoexp",
        config_dir=Path("config"),
        overrides=[
            "backend=megatron",
            "backend/megatron=base",
            "job=default",
            "container=none",
        ],
    )
    cfg = load_config_reference(config_setup=config_setup)

    # Instantiate backend directly
    backend = cfg.backend.instantiate(BackendInterface)

    # Build command with test parameters
    spec = BackendJobSpec(parameters={"megatron.micro_batch_size": 4})
    backend.validate(spec)
    command = backend.build_launch_command(spec)

    # Verify command structure
    assert command.argv[0].endswith("python")
    assert command.argv[1].endswith("submodules/Megatron-LM/pretrain_gpt.py")
    assert any(arg.startswith("--micro-batch-size") for arg in command.argv)
