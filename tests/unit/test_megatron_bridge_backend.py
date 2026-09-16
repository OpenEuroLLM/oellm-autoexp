from __future__ import annotations

import pytest

from oellm_autoexp.backends.megatron_bridge_backend import (
    MegatronBridgeBackend,
    MegatronBridgeBackendConfig,
)


def _ok_cfg(**overrides) -> MegatronBridgeBackendConfig:
    cfg = dict(
        megatron_path="/run/torch_dist/iter_0001000",
        hf_path="/run/hf/iter_0001000",
        hf_model="Qwen/Qwen3-0.6B",
        tokenizer="Qwen/Qwen3-0.6B",
    )
    cfg.update(overrides)
    return MegatronBridgeBackendConfig(**cfg)


def test_builds_command_with_resources_and_bridge():
    cfg = _ok_cfg()
    cmd = MegatronBridgeBackend(cfg).build_launch_command()
    assert "python -m oellm_autoexp.backends.megatron_bridge.run_export" in cmd
    assert "--megatron-path /run/torch_dist/iter_0001000" in cmd
    assert "--hf-path /run/hf/iter_0001000" in cmd
    assert "--hf-model Qwen/Qwen3-0.6B" in cmd
    assert "--tokenizer Qwen/Qwen3-0.6B" in cmd
    assert "--bridge-root submodules/Megatron-Bridge" in cmd
    assert "--keep-staging" not in cmd


def test_keep_staging_flag():
    cfg = _ok_cfg(keep_staging=True)
    assert "--keep-staging" in MegatronBridgeBackend(cfg).build_launch_command()


def test_extra_cli_args_passthrough():
    cfg = _ok_cfg(extra_cli_args=["--foo=bar"])
    assert "--foo=bar" in MegatronBridgeBackend(cfg).build_launch_command()


def test_full_cmd_overridable():
    cfg = _ok_cfg(full_cmd="echo overridden")
    assert MegatronBridgeBackend(cfg).build_launch_command() == "echo overridden"


@pytest.mark.parametrize("missing", ["megatron_path", "hf_path", "hf_model", "tokenizer"])
def test_validate_requires_fields(missing):
    cfg = _ok_cfg(**{missing: ""})
    with pytest.raises(ValueError, match=f"`{missing}` is required"):
        MegatronBridgeBackend(cfg).validate()


def test_validate_passes_when_complete():
    MegatronBridgeBackend(_ok_cfg()).validate()
