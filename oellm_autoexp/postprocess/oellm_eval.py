"""Oellm-cli evaluation step (submits eval as a separate SLURM job)."""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from typing import Literal

from compoconf import ConfigInterface, register

from oellm_autoexp.postprocess.base import PostProcessStepInterface


@dataclass(kw_only=True)
class OELLMEvalStepConfig(ConfigInterface):
    """Config for running oellm-cli evaluation."""

    class_name: str = "OELLMEvalStep"

    model_path: str = ""
    task_groups: list[str] = field(default_factory=lambda: ["open-sci-0.01"])
    dry_run: bool = False


@register
class OELLMEvalStep(PostProcessStepInterface):
    config: OELLMEvalStepConfig

    def get_run_mode(self) -> Literal["same_job", "new_job"]:
        return "new_job"

    def build_command(self) -> str:
        cfg = self.config
        task_groups_args = shlex.join(cfg.task_groups)

        parts = [
            "oellm schedule-eval",
            f"--models {shlex.quote(cfg.model_path)}",
            f"--task_groups {task_groups_args}",
        ]
        if cfg.dry_run:
            parts.append("--dry_run true")

        return " ".join(parts)


__all__ = ["OELLMEvalStep", "OELLMEvalStepConfig"]
