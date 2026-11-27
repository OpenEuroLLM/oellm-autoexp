"""post-training backend (https://github.com/OpenEuroLLM/post-training): SFT / DPO with TRL.

post-training keeps its own config file. That file is the description of the experiment: model,
data, hyperparameters, run name and output directory. This backend holds only the link to it:

    backend:
      repo_dir: <post-training checkout>
      python:   <interpreter of the post-training environment>
      config:   <post-training YAML, absolute or relative to repo_dir>

When a job record is built (on enqueue), the backend calls post-training's own prepare command in
a subprocess, with that interpreter:

    <python> scripts/prepare_run.py --config <config> --num-nodes N --gpus-per-node G [overrides]

The command freezes the config and the code into the run directory (a new run), or only writes
the launch script again (a run that exists: every enqueue continues it from its last checkpoint).
The launch command of the job is then ``bash <run_dir>/slurm/launch.sh``, one task for each node.
During a dry run the prepare command gets ``--check``: all checks, and no write.

Nothing of post-training is imported here. Its environment and the autoexp environment stay
independent; the interpreter only needs the modules it was built with loaded in the shell that
runs autoexp.

Who owns which value:
  * the run name and the run directory come from the post-training YAML. The resolvers
    ``post_training.run_name`` / ``post_training.run_dir`` read them, because autoexp needs the
    job name and the log directory before the backend runs;
  * the node count and the GPUs for each node come from ``slurm.sbatch``. The prepare command
    stops if the post-training YAML has a different value: the gradient accumulation and the
    checkpoints are valid for one world size only.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from compoconf import MissingValue, NonStrictDataclass, register
from omegaconf import OmegaConf

from oellm_autoexp.backends.base import BaseBackend, BaseBackendConfig
from oellm_autoexp.utils.dry_run import ACTIVE_ENV as DRY_RUN_ENV

LOGGER = logging.getLogger(__name__)

PREPARE_SCRIPT = "scripts/prepare_run.py"
RESULT_PREFIX = "PREPARE_RUN_RESULT "

# Keys that autoexp has already read from the post-training YAML (run name, run directory) or
# that it owns (world size): an override would make the job different from its record.
_FIXED_KEYS = ("run_name", "slurm.num_nodes", "slurm.gpus_per_node")
_FIXED_PREFIXES = ("paths.", "debug.")

# The autoexp environment must not reach the post-training interpreter.
_DROPPED_ENV = ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")


def _read_run_identity(repo_dir: str, config: str) -> tuple[str, str]:
    """Return (run name, run directory) as post-training derives them from its
    YAML ``config``."""
    path = Path(config) if os.path.isabs(config) else Path(repo_dir) / config
    if not path.is_file():
        raise ValueError(f"post-training config {path} not found (backend.config)")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    def _plain(dotted: str) -> Any:
        value: Any = data
        for part in dotted.split("."):
            value = value.get(part) if isinstance(value, dict) else None
        if isinstance(value, str) and "${" in value:
            raise ValueError(f"{dotted} in {path} must be a plain value, got {value!r}")
        return value

    run_name = _plain("run_name")
    if not run_name:
        raise ValueError(
            f"{path} has no run_name. A run that autoexp starts needs a fixed name: "
            "the same name is the same run directory, which a restart continues."
        )
    base_key = "paths.debug_base" if _plain("debug.enabled") else "paths.output_base"
    base = _plain(base_key)
    if not base:
        raise ValueError(f"{path} has no {base_key}. Set it explicitly in the file.")
    # post-training resolves a relative base against its working directory, the repository root.
    run_dir = os.path.join(os.path.realpath(os.path.join(repo_dir, str(base))), str(run_name))
    return str(run_name), run_dir


OmegaConf.register_new_resolver(
    "post_training.run_name",
    lambda repo_dir, config: _read_run_identity(repo_dir, config)[0],
    replace=True,
)
OmegaConf.register_new_resolver(
    "post_training.run_dir",
    lambda repo_dir, config: _read_run_identity(repo_dir, config)[1],
    replace=True,
)


@dataclass(init=False)
class PostTrainingBackendConfig(NonStrictDataclass, BaseBackendConfig):
    """Configuration of the post-training backend.

    ``repo_dir``, ``python`` and ``config`` are
    mandatory and have no default: they are explicit in the YAML or on the command line.
    """

    class_name: str = "PostTrainingBackend"
    env: dict[str, str] = field(default_factory=dict)

    # The post-training checkout. The prepare command runs here.
    repo_dir: str = MissingValue
    # The interpreter of the post-training environment (<venv>/bin/python).
    python: str = MissingValue
    # The post-training YAML of the experiment, absolute or relative to repo_dir.
    config: str = MissingValue
    # post-training overrides (OmegaConf dot list, "key=value"), given to the prepare command.
    overrides: list[str] = field(default_factory=list)

    # Set by config/backend/post_training.yaml from the rest of the config; do not override.
    num_nodes: int = MissingValue  # ${slurm.sbatch.nodes}
    gpus_per_node: int = MissingValue  # ${slurm.sbatch.gpus_per_node}
    run_dir: str = MissingValue  # ${post_training.run_dir:...}


@register
class PostTrainingBackend(BaseBackend):
    config: PostTrainingBackendConfig

    def validate(self) -> None:
        cfg = self.config
        for name in ("repo_dir", "python", "config", "num_nodes", "gpus_per_node", "run_dir"):
            if getattr(cfg, name) is MissingValue or getattr(cfg, name) in ("", None):
                raise ValueError(f"PostTrainingBackend: backend.{name} is mandatory")
        if not (Path(cfg.repo_dir) / PREPARE_SCRIPT).is_file():
            raise ValueError(
                f"PostTrainingBackend: {cfg.repo_dir} has no {PREPARE_SCRIPT} (backend.repo_dir "
                "must be a post-training checkout that has the prepare command)"
            )
        if not (os.path.isfile(cfg.python) and os.access(cfg.python, os.X_OK)):
            raise ValueError(
                f"PostTrainingBackend: backend.python ({cfg.python}) is not an executable file"
            )
        for item in cfg.overrides:
            key = str(item).split("=", 1)[0].lstrip("+")
            if key in _FIXED_KEYS or key.startswith(_FIXED_PREFIXES):
                raise ValueError(
                    f"PostTrainingBackend: {key} cannot be in backend.overrides. The run name and "
                    "the paths come from the post-training YAML, and the node and GPU counts "
                    "from slurm.sbatch."
                )
        if "'" in str(cfg.run_dir):
            # templates/base.sbatch wraps the command in single quotes
            raise ValueError(f"PostTrainingBackend: the run directory {cfg.run_dir} has a quote")

    def build_launch_command(self) -> str:
        """Prepare the run directory (or check it, in a dry run) and return the
        command that runs its launch script.

        The orchestrator calls only this method, so the checks are here
        too.
        """
        self.validate()
        cfg = self.config
        dry_run = os.environ.get(DRY_RUN_ENV) == "1"
        result = self._prepare(check_only=dry_run)

        if os.path.realpath(result["run_dir"]) != os.path.realpath(cfg.run_dir):
            raise ValueError(
                f"PostTrainingBackend: post-training uses the run directory {result['run_dir']}, "
                f"but autoexp derived {cfg.run_dir} from the same config. The job record (name, "
                "log directory) would point to a different directory."
            )
        LOGGER.info(
            "post-training run directory %s: %s%s",
            result["run_dir"],
            "new run" if result.get("new_run") else "continued run",
            " (dry run: checked, nothing written)" if dry_run else "",
        )
        return f"bash {result['launch_script']}"

    def _subprocess_env(self) -> dict[str, str]:
        env = {key: value for key, value in os.environ.items() if key not in _DROPPED_ENV}
        env["PYTHONNOUSERSITE"] = "1"
        return env

    def _prepare(self, *, check_only: bool) -> dict[str, Any]:
        cfg = self.config
        env = self._subprocess_env()

        # A missing module of the interpreter shows as a clear error, not as a failed prepare.
        try:
            probe = subprocess.run(
                [cfg.python, "-c", "pass"], env=env, capture_output=True, text=True, check=False
            )
            probe_error = probe.stderr.strip() if probe.returncode else None
        except OSError as error:
            probe_error = str(error)
        if probe_error is not None:
            raise ValueError(
                f"PostTrainingBackend: cannot start backend.python ({cfg.python}):\n{probe_error}\n"
                "Load the modules that this interpreter needs (for example `module load python`) "
                "in this shell, and try again."
            )

        argv = [
            cfg.python,
            PREPARE_SCRIPT,
            "--config",
            str(cfg.config),
            "--num-nodes",
            str(cfg.num_nodes),
            "--gpus-per-node",
            str(cfg.gpus_per_node),
            *(["--check"] if check_only else []),
            *[str(item) for item in cfg.overrides],
        ]
        # The prepare command prints nothing until it ends, and a new run takes minutes on a
        # parallel file system (check of the model and the datasets, copy of the code).
        if check_only:
            LOGGER.info("post-training: checking the run of %s (dry run) ...", cfg.config)
        else:
            LOGGER.info(
                "post-training: preparing the run directory of %s; a new run can take some "
                "minutes ...",
                cfg.config,
            )
        done = subprocess.run(
            argv, cwd=cfg.repo_dir, env=env, capture_output=True, text=True, check=False
        )
        results = [
            line[len(RESULT_PREFIX) :]
            for line in done.stdout.splitlines()
            if line.startswith(RESULT_PREFIX)
        ]
        if done.returncode != 0 or not results:
            detail = done.stderr.strip() or "\n".join(done.stdout.strip().splitlines()[-20:])
            raise ValueError(
                "PostTrainingBackend: the post-training prepare command failed "
                f"(exit code {done.returncode}):\n  {' '.join(argv)}\n{detail}"
            )
        return json.loads(results[-1])


__all__ = ["PostTrainingBackend", "PostTrainingBackendConfig"]
