"""Static guard for invariant I9: every flag a dllm experiment emits is accepted by MegaDLMs.

The MegaDLMs argparse sources are read with ``ast`` and never imported, so this runs without
torch or Megatron. The module is skipped when the MegaDLMs submodule is not checked out.

Run with:
    pytest tests/unit/dllm/test_megadlm_arg_compat.py -v
"""

from __future__ import annotations

import os
import re
import shlex
from pathlib import Path

import pytest

from oellm_autoexp.config.loader import load_config_reference
from oellm_autoexp.config.schema import ConfigSetup
from oellm_autoexp.dllm.render import (
    argparser_files_for,
    discover_dllm_experiments,
    extract_script_flags,
    parse_accepted_flags,
    render_launch_commands,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_DIR = REPO_ROOT / "config"
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
    monkeypatch.setenv("PWD", str(REPO_ROOT))


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
    launcher = getattr(root.backend, "launcher_script", None)
    if launcher is None:
        pytest.skip(f"{config_name}: no launcher script, so no MegaDLMs flags to check")
    emitted: set[str] = set()
    for command in commands:
        emitted.update(extract_script_flags(command, launcher))

    assert emitted, f"{config_name}: no flags found after the launcher {launcher}"
    accepted = parse_accepted_flags(REPO_ROOT / rel for rel in argparser_files_for(launcher))
    unknown = sorted(emitted - accepted)
    assert not unknown, f"{config_name}: flags not accepted by {launcher}: {unknown}"


# Flags that every block-diffusion job must set explicitly, so that a lost or misspelled
# backend.aux.bd_args cannot silently fall back to the launcher defaults. The list holds every
# --bd-* flag of the launcher (checked below); jobs pass "--bd-block-schedule none" for a
# constant block size.
_BD_LAUNCHERS = ("pretrain_block_difflm.py", "convert_chain.sh")
_BD_REQUIRED_FLAGS = (
    "--bd-objective",
    "--bd-block-size",
    "--bd-block-schedule",
    "--bd-token-shift",
    "--bd-complementary-mask",
    "--bd-loss-weighting",
    "--bd-t-min",
    "--bd-t-max",
    "--bd-doc-mask",
    "--bd-val-seed",
    "--mask-token",
    "--difflm-varilen-prob",
)


def _flag_values(command: str, launcher: str) -> dict[str, list[str]]:
    """Map each flag after the launcher to the value tokens that follow it."""
    tokens = shlex.split(re.sub(r"\\\r?\n", " ", command))
    name = os.path.basename(launcher)
    start = next(i for i, token in enumerate(tokens) if token.endswith(name)) + 1
    values: dict[str, list[str]] = {}
    current = None
    for token in tokens[start:]:
        if token.startswith("--"):
            current = token.split("=", 1)[0]
            values.setdefault(current, [])
            if "=" in token:
                values[current].append(token.split("=", 1)[1])
        elif current is not None:
            values[current].append(token)
    return values


@pytest.mark.parametrize("config_name", _DLLM_CONFIGS, ids=_experiment_id)
def test_block_diffusion_jobs_set_their_flags_explicitly(
    config_name: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Block-diffusion jobs pass every core --bd-* flag, the mask token, and
    varilen 0."""
    _set_render_env(monkeypatch, tmp_path)
    root = load_config_reference(
        config_setup=ConfigSetup(config_name=config_name, config_dir=str(_CONFIG_DIR))
    )
    launcher = getattr(root.backend, "launcher_script", None)
    if launcher is None or os.path.basename(launcher) not in _BD_LAUNCHERS:
        return  # no launcher, or MegaDLMs' own launcher: no --bd-* flags
    for command in render_launch_commands(config_name, _CONFIG_DIR):
        values = _flag_values(command, launcher)
        missing = [flag for flag in _BD_REQUIRED_FLAGS if not values.get(flag)]
        assert not missing, f"{config_name}: missing explicit flags {missing}"
        varilen = values["--difflm-varilen-prob"][-1]
        assert float(varilen) == 0.0, f"{config_name}: --difflm-varilen-prob {varilen} != 0"


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


def test_bd_flags_are_accepted_only_by_the_block_diffusion_launcher() -> None:
    """--bd-* flags pass for pretrain_block_difflm.py and fail for MegaDLMs' pretrain_difflm.py.

    The block launcher's --bd-* flags are exactly the --bd-* flags of _BD_REQUIRED_FLAGS, so a
    new flag cannot be left out of the explicit-flag check.
    """
    block = parse_accepted_flags(
        REPO_ROOT / rel
        for rel in argparser_files_for("./oellm_autoexp/dllm/pretrain_block_difflm.py")
    )
    plain = parse_accepted_flags(REPO_ROOT / rel for rel in argparser_files_for(_LAUNCHER))
    bd_flags = {flag for flag in block if flag.startswith("--bd-")}
    assert bd_flags == {flag for flag in _BD_REQUIRED_FLAGS if flag.startswith("--bd-")}
    assert not bd_flags & plain
    assert "--mask-token" in block and "--mask-token" in plain
    with pytest.raises(ValueError):
        argparser_files_for("./some/other_launcher.py")
