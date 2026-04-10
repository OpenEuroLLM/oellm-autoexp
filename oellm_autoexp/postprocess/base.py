"""Post-processing step interface used by the orchestrator."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Literal

from compoconf import RegistrableConfigInterface, register_interface


@register_interface
class PostProcessStepInterface(RegistrableConfigInterface):
    """Translates post-processing config into shell commands.

    Subclasses must implement ``build_command`` and ``get_run_mode``.
    """

    config: Any

    def __init__(self, config: Any) -> None:
        self.config = config

    @abstractmethod
    def build_command(self) -> str:
        """Return the shell command string for this step."""
        ...

    @abstractmethod
    def get_run_mode(self) -> Literal["same_job", "new_job"]:
        """Whether this step runs in the same SLURM job or spawns a new one."""
        ...


__all__ = ["PostProcessStepInterface"]
