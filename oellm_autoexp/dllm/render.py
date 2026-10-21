"""Torch-free helpers that render dllm launch commands and check the flags they
pass.

Shared by tests/unit/dllm and
tests/integration/test_dllm_config_feasibility.py. Nothing in this
module imports torch, writes files, or calls SLURM.
"""

from __future__ import annotations

import ast
import os
import re
import shlex
from collections.abc import Iterable
from pathlib import Path

from oellm_autoexp.config.loader import load_config_reference
from oellm_autoexp.config.schema import BackendInterface, ConfigSetup, RootConfig
from oellm_autoexp.orchestrator import build_execution_plan

# Argparse sources that define the flags a dllm launch may pass. Paths are relative to the
# repository root. oellm_autoexp/dllm/args.py does not exist yet; missing paths are skipped.
ARGPARSER_FILES: tuple[str, ...] = (
    "submodules/MegaDLMs/megatron/training/arguments.py",
    "submodules/MegaDLMs/custom_args/difflm.py",
    "oellm_autoexp/dllm/args.py",
)

_LINE_CONTINUATION = re.compile(r"\\\r?\n")


def parse_accepted_flags(paths: Iterable[Path]) -> set[str]:
    """Collect the option strings that argparse ``add_argument`` calls declare.

    The files are parsed with ``ast`` and never imported, so no dependency is needed. Only
    calls of the form ``<expr>.add_argument(...)`` count. Paths that do not exist are skipped.

    Args:
        paths: Python source files that build argparse parsers.

    Returns:
        The union of every positional string argument that starts with ``-`` across the
        existing files.
    """
    flags: set[str] = set()
    for path in paths:
        source_path = Path(path)
        if not source_path.is_file():
            continue
        tree = ast.parse(source_path.read_bytes(), filename=str(source_path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not (isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"):
                continue
            for arg in node.args:
                if (
                    isinstance(arg, ast.Constant)
                    and isinstance(arg.value, str)
                    and arg.value.startswith("-")
                ):
                    flags.add(arg.value)
    return flags


def discover_dllm_experiments(config_dir: Path) -> list[str]:
    """List every dllm experiment config under ``config_dir``, including
    subfolders.

    Args:
        config_dir: Config root directory, for example ``Path("config")``.

    Returns:
        Config names such as ``"experiments/dllm/qwen3_20m_dlm"`` or
        ``"experiments/dllm/bd/smoke_qwen3_0.6b"``. The list is sorted by file path. Names use
        POSIX separators and omit the ``.yaml`` suffix.
    """
    base = Path(config_dir)
    return [
        path.relative_to(base).with_suffix("").as_posix()
        for path in sorted((base / "experiments" / "dllm").rglob("*.yaml"))
    ]


def render_launch_commands(config_name: str, config_dir: Path) -> list[str]:
    """Render the launch command of every job that a dllm config expands to.

    Follows the ``scripts/run_autoexp.py --dry-run`` path. The config is loaded and the plan
    is built with the same arguments. Each job's backend then produces its launch command, as
    ``oellm_autoexp.orchestrator._build_job_record`` does. ``monitor_state_dir`` keeps its
    schema default, which is also what a plain dry run uses. Nothing is written to disk, and
    SLURM is not called.

    Args:
        config_name: Config name relative to ``config_dir``, for example
            ``"experiments/dllm/qwen3_20m_dlm"``.
        config_dir: Config root directory, for example ``Path("config")``.

    Returns:
        One launch command string per planned job, in plan order.

    Raises:
        ValueError: If a planned job does not carry a ``RootConfig``.
    """
    config_setup = ConfigSetup(
        pwd=os.path.abspath(os.curdir),
        config_name=config_name,
        config_dir=str(config_dir),
    )
    root = load_config_reference(config_setup=config_setup)
    plan = build_execution_plan(root, config_setup=config_setup, subset_indices=None)
    commands: list[str] = []
    for job in plan.jobs:
        if not isinstance(job.config, RootConfig):
            raise ValueError("JobPlan.config must be RootConfig")
        backend = job.config.backend.instantiate(BackendInterface)
        commands.append(backend.build_launch_command())
    return commands


def extract_script_flags(command: str, launcher_script: str) -> list[str]:
    """Return the long flags that a launch command passes to the training
    script.

    Backslash-newline continuations are joined, then the command is split with ``shlex``. The
    launcher is the first token whose value ends with the basename of ``launcher_script``.
    Only tokens after the launcher count, so torchrun's own flags are ignored. A token of the
    form ``--flag=value`` is reported as ``--flag``.

    Args:
        command: Full launch command, as produced by a backend.
        launcher_script: Launcher path from the backend config, for example
            ``"./submodules/MegaDLMs/pretrain_difflm.py"``.

    Returns:
        The ``--`` flags that follow the launcher, in command order. Duplicates are kept.

    Raises:
        ValueError: If ``launcher_script`` has no file name, or if the command does not
            contain the launcher token.
    """
    basename = os.path.basename(launcher_script)
    if not basename:
        raise ValueError(f"launcher_script has no file name: {launcher_script!r}")
    tokens = shlex.split(_LINE_CONTINUATION.sub(" ", command))
    for index, token in enumerate(tokens):
        if token.endswith(basename):
            break
    else:
        raise ValueError(f"launcher {launcher_script!r} not found in command")
    return [token.split("=", 1)[0] for token in tokens[index + 1 :] if token.startswith("--")]
