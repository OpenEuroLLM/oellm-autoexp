"""Post-processing step interface used by the orchestrator."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Literal

from compoconf import RegistrableConfigInterface, register_interface


@register_interface
class PostProcessStepInterface(RegistrableConfigInterface):
    """Translates post-processing config into shell commands.

    Subclasses must implement ``build_commands`` and ``get_run_mode``.
    """

    config: Any

    def __init__(self, config: Any) -> None:
        self.config = config

    @abstractmethod
    def build_commands(self) -> list[str]:
        """Return the shell commands for this step (one element per logical command)."""
        ...

    @abstractmethod
    def get_run_mode(self) -> Literal["same_job", "new_job"]:
        """Whether this step runs in the same SLURM job or spawns a new one."""
        ...


def _all_ckpt_steps(save_interval: int, train_iters: int) -> list[int]:
    """Return every checkpoint step: [save_interval, 2*save_interval, ..., train_iters]."""
    steps = list(range(save_interval, train_iters, save_interval))
    if not steps or steps[-1] != train_iters:
        steps.append(train_iters)
    return steps


__all__ = ["PostProcessStepInterface", "_all_ckpt_steps"]
