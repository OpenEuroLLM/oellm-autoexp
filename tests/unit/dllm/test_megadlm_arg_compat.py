"""Static guard for invariant I9: every flag a dllm experiment emits is accepted by MegaDLMs.

The MegaDLMs argparse sources are read with ``ast`` and never imported, so this runs without
torch or Megatron. The module is skipped when the MegaDLMs submodule is not checked out.

Run with:
    pytest tests/unit/dllm/test_megadlm_arg_compat.py -v
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from oellm_autoexp.config.loader import load_config_reference
from oellm_autoexp.config.schema import ConfigSetup
from oellm_autoexp.dllm.render import (
    ARGPARSER_FILES,
    discover_dllm_experiments,
    extract_script_flags,
    parse_accepted_flags,
    render_launch_commands,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_DIR = Path("config")
_MEGADLM_ARGUMENTS = REPO_ROOT / "submodules/MegaDLMs/megatron/training/arguments.py"

if not _MEGADLM_ARGUMENTS.exists():
    pytest.skip("submodules/MegaDLMs is not checked out", allow_module_level=True)

# Same dummy values as tests/integration/test_config_feasibility.py.
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

_MEGADLM_SOURCES: tuple[Path, ...] = (
    REPO_ROOT / "submodules/MegaDLMs/megatron/training/arguments.py",
    REPO_ROOT / "submodules/MegaDLMs/custom_args/difflm.py",
)
_ACCEPTED_SOURCES: tuple[Path, ...] = tuple(REPO_ROOT / rel for rel in ARGPARSER_FILES)
_DLLM_CONFIGS: list[str] = discover_dllm_experiments(_CONFIG_DIR)
_SANITY_FLAGS: tuple[str, ...] = (
    "--seq-length",
    "--mask-token",
    "--difflm-varilen-prob",
    "--norm-epsilon",
    "--ckpt-format",
)


def _experiment_id(config_name: str) -> str:
    return config_name.removeprefix("experiments/dllm/")


def _set_render_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Set dummy cluster env vars so rendering never touches real paths."""
    for key, val in _DUMMY_ENVS.items():
        monkeypatch.setenv(key, val)
    if "USER" not in os.environ:
        monkeypatch.setenv("USER", "dummy_user")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))


def test_parser_sources_declare_expected_flags() -> None:
    """The static scan finds flags from both MegaDLMs argparse files."""
    accepted = parse_accepted_flags(_MEGADLM_SOURCES)
    for flag in _SANITY_FLAGS:
        assert flag in accepted, f"{flag} is missing from the MegaDLMs argparse scan"


def test_discovery_finds_dllm_experiments() -> None:
    """Guard against an empty parametrization, which pytest would skip without
    a failure."""
    assert _DLLM_CONFIGS, "no dllm experiments found under config/experiments/dllm"


@pytest.mark.parametrize("config_name", _DLLM_CONFIGS, ids=_experiment_id)
def test_dllm_experiment_flags_are_accepted(
    config_name: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Every flag that a rendered dllm job passes after the launcher is a
    MegaDLMs flag."""
    _set_render_env(monkeypatch, tmp_path)
    commands = render_launch_commands(config_name, _CONFIG_DIR)
    assert commands, f"{config_name}: rendered no jobs"

    root = load_config_reference(
        config_setup=ConfigSetup(config_name=config_name, config_dir=str(_CONFIG_DIR))
    )
    launcher = root.backend.launcher_script
    emitted: set[str] = set()
    for command in commands:
        emitted.update(extract_script_flags(command, launcher))

    unknown = sorted(emitted - parse_accepted_flags(_ACCEPTED_SOURCES))
    assert not unknown, f"{config_name}: flags not accepted by MegaDLMs: {unknown}"


_LAUNCHER = "./submodules/MegaDLMs/pretrain_difflm.py"


def test_extract_script_flags_reads_only_flags_after_launcher() -> None:
    """Continuations are joined, torchrun flags are ignored, and --flag=value
    keeps the name."""
    command = (
        "torchrun --nproc-per-node 1 --rdzv-endpoint $MASTER_ADDR:$MASTER_PORT \\\n"
        f"   {_LAUNCHER} \\\n"
        "   --seq-length 513 --mask-token=50257 --bf16 \\\n"
        "   --attention-mask-type no_mask"
    )
    assert extract_script_flags(command, _LAUNCHER) == [
        "--seq-length",
        "--mask-token",
        "--bf16",
        "--attention-mask-type",
    ]


def test_extract_script_flags_requires_the_launcher_token() -> None:
    """A launcher that is absent from the command raises instead of returning
    no flags."""
    command = f"torchrun --nproc-per-node 1 {_LAUNCHER} --bf16"
    with pytest.raises(ValueError, match="pretrain_block_difflm.py"):
        extract_script_flags(command, "./submodules/MegaDLMs/pretrain_block_difflm.py")
