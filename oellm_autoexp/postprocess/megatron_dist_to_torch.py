"""Megatron torch_dist-to-torch checkpoint conversion step."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from compoconf import ConfigInterface, register

from oellm_autoexp.postprocess.base import PostProcessStepInterface


@dataclass(kw_only=True)
class MegatronDistToTorchStepConfig(ConfigInterface):
    """Config for the torch_dist→torch conversion step.

    ``cmd`` should be the full Megatron launch command including
    ``--ckpt-convert-format torch`` flags (typically ``${backend.full_cmd}``).
    The orchestrator wraps this in its own ``apptainer exec`` so it runs
    inside the container, same as the HF conversion step.
    """

    class_name: str = "MegatronDistToTorchStep"
    cmd: str = ""


@register
class MegatronDistToTorchStep(PostProcessStepInterface):
    config: MegatronDistToTorchStepConfig

    def get_run_mode(self) -> Literal["same_job", "new_job"]:
        return "same_job"

    def build_command(self) -> str:
        return self.config.cmd


__all__ = ["MegatronDistToTorchStep", "MegatronDistToTorchStepConfig"]
