"""Oellm-evals evaluation backend.

Wraps ``oellm-eval schedule`` so an evaluation run can be scheduled by
oellm-autoexp like any other backend (Megatron, Titan, ...). The default
mode (``local: true``) runs the eval script directly inside the SLURM
job allocated by oellm-autoexp, so oellm-evals does **not** spawn its own
sbatch and monitoring stays in oellm-autoexp's hands.

See ``submodules/oellm_evals/oellm/main.py::schedule_evals`` for the
underlying CLI; this backend exposes a typed config with the same
surface plus an ``extra_cli_args`` escape hatch.
"""

from __future__ import annotations

import logging
import shlex
from dataclasses import dataclass, field

from compoconf import MissingValue, NonStrictDataclass, register

from oellm_autoexp.backends.base import BaseBackend, BaseBackendConfig

LOGGER = logging.getLogger(__name__)


@dataclass(init=False)
class OELLMEvalBackendConfig(NonStrictDataclass, BaseBackendConfig):
    """Configuration for the oellm-evals evaluation backend."""

    class_name: str = "OELLMEvalBackend"
    env: dict[str, str] = field(default_factory=dict)

    # CLI entrypoint. By default we assume `oellm-eval` is on PATH (e.g.
    # installed via `uv tool install`). Override with an absolute path if
    # needed.
    oellm_cmd: str = "oellm-eval"

    # --- Evaluation target -------------------------------------------------
    models: list[str] = field(default_factory=list)
    tasks: list[str] | None = None
    task_groups: list[str] | None = None
    # oellm-evals accepts an int or list of ints. We allow both.
    n_shot: int | list[int] | None = None
    eval_csv_path: str | None = None

    # --- Runtime / options -------------------------------------------------
    # When ``local`` is true the eval runs sequentially inside the SLURM job
    # allocated by oellm-autoexp. When false, ``oellm-eval schedule`` will
    # submit its own sbatch (not recommended for autoexp pipelines).
    local: bool = True
    venv_path: str | None = None
    limit: int | None = None
    verbose: bool = False
    download_only: bool = False
    skip_checks: bool = False
    trust_remote_code: bool = True
    lm_eval_include_path: str | None = None
    # JSON string passed through to oellm-evals when local=False.
    slurm_template_var: str | None = None
    max_array_len: int = 128
    dry_run: bool = False

    # Anything not covered by the typed fields above (forwarded verbatim).
    extra_cli_args: list[str] = field(default_factory=list)

    # Final assembled launch command (auto-computed in __post_init__,
    # overridable from YAML for full control).
    full_cmd: str = MissingValue

    def __post_init__(self) -> None:
        if self.full_cmd is MissingValue:
            self.full_cmd = _build_oellm_cmd(self)


def _build_oellm_cmd(cfg: OELLMEvalBackendConfig) -> str:
    parts: list[str] = [cfg.oellm_cmd, "schedule"]

    if cfg.eval_csv_path:
        parts.append(f"--eval_csv_path {shlex.quote(cfg.eval_csv_path)}")
    else:
        if cfg.models:
            parts.append(f"--models {shlex.quote(','.join(cfg.models))}")
        if cfg.tasks:
            parts.append(f"--tasks {shlex.quote(','.join(cfg.tasks))}")
        if cfg.task_groups:
            parts.append(f"--task_groups {shlex.quote(','.join(cfg.task_groups))}")
        if cfg.n_shot is not None:
            if isinstance(cfg.n_shot, list):
                parts.append(f"--n_shot {shlex.quote(','.join(str(s) for s in cfg.n_shot))}")
            else:
                parts.append(f"--n_shot {int(cfg.n_shot)}")

    if cfg.venv_path:
        parts.append(f"--venv_path {shlex.quote(cfg.venv_path)}")
    if cfg.local:
        parts.append("--local true")
    if cfg.limit is not None:
        parts.append(f"--limit {int(cfg.limit)}")
    if cfg.verbose:
        parts.append("--verbose true")
    if cfg.download_only:
        parts.append("--download_only true")
    if cfg.skip_checks:
        parts.append("--skip_checks true")
    if not cfg.trust_remote_code:
        parts.append("--trust_remote_code false")
    if cfg.lm_eval_include_path:
        parts.append(f"--lm_eval_include_path {shlex.quote(cfg.lm_eval_include_path)}")
    if cfg.slurm_template_var:
        parts.append(f"--slurm_template_var {shlex.quote(cfg.slurm_template_var)}")
    if cfg.max_array_len != 128:
        parts.append(f"--max_array_len {int(cfg.max_array_len)}")
    if cfg.dry_run:
        parts.append("--dry_run true")

    if cfg.extra_cli_args:
        parts.extend(str(arg) for arg in cfg.extra_cli_args)

    return " ".join(parts)


@register
class OELLMEvalBackend(BaseBackend):
    config: OELLMEvalBackendConfig

    def __init__(self, config: OELLMEvalBackendConfig) -> None:
        super().__init__(config)
        self.config = config

    def validate(self) -> None:
        cfg = self.config
        if cfg.eval_csv_path:
            if cfg.models or cfg.tasks or cfg.task_groups or cfg.n_shot is not None:
                raise ValueError(
                    "OELLMEvalBackend: cannot combine `eval_csv_path` with "
                    "`models`, `tasks`, `task_groups`, or `n_shot`."
                )
        else:
            if not cfg.models:
                raise ValueError(
                    "OELLMEvalBackend: `models` is required unless `eval_csv_path` is set."
                )
            if not cfg.task_groups and not cfg.tasks:
                raise ValueError("OELLMEvalBackend: either `task_groups` or `tasks` must be set.")
            if cfg.tasks and cfg.n_shot is None:
                raise ValueError(
                    "OELLMEvalBackend: `n_shot` is required when using `tasks` "
                    "(ignored when using `task_groups`)."
                )

        if cfg.local and not cfg.venv_path:
            raise ValueError(
                "OELLMEvalBackend: `local: true` requires `venv_path` "
                "(oellm-evals's --local mode needs a venv with lm-eval/lighteval installed)."
            )

    def build_launch_command(self) -> str:
        return self.config.full_cmd


__all__ = [
    "OELLMEvalBackend",
    "OELLMEvalBackendConfig",
]
