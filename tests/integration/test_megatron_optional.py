from pathlib import Path

import pytest

from oellm_autoexp.backends.base import BackendJobSpec
from oellm_autoexp.config.evaluator import evaluate
from oellm_autoexp.config.loader import load_config_reference

try:
    from oellm_autoexp.backends.megatron_backend import MegatronBackend  # noqa: F401

    _HAS_MEGATRON = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_MEGATRON = False


@pytest.mark.skipif(not _HAS_MEGATRON, reason="Megatron parser not available")
def test_megatron_backend_builds_launch_command(monkeypatch, tmp_path):
    monkeypatch.setenv("SLURM_ACCOUNT", "debug")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "outputs"))

    cfg = load_config_reference(
        "autoexp", Path("config"), overrides=["backend/megatron=base", "project=default"]
    )
    runtime = evaluate(cfg)

    spec = BackendJobSpec(parameters={"megatron.micro_batch_size": 4})
    runtime.backend.validate(spec)
    command = runtime.backend.build_launch_command(spec)

    assert command.argv[0].endswith("submodules/Megatron-LM/pretrain_gpt.py")
    assert any(arg.startswith("--micro-batch-size") for arg in command.argv)
