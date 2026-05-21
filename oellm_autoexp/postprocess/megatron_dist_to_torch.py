"""Megatron torch_dist-to-torch checkpoint conversion step."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Literal

from compoconf import ConfigInterface, register

from oellm_autoexp.postprocess.base import PostProcessStepInterface, _all_ckpt_steps


@dataclass(kw_only=True)
class MegatronDistToTorchStepConfig(ConfigInterface):
    """Config for the torch_dist→torch conversion step.

    ``cmd`` should be ``${backend.full_cmd}`` — the full Megatron launch
    command including ``--ckpt-convert-format torch``.

    ``load_dir`` overrides ``--load`` for the conversion. Set it to the
    directory where the just-saved torch_dist checkpoints actually live —
    typically ``${backend.megatron.save}``. For continued-pretraining runs
    where ``--save`` differs from ``--load``, the original ``--load`` points
    at the source-of-resumption (which won't have the new iters), so an
    override is required.

    When ``save_interval`` and ``train_iters`` are both set, every saved
    checkpoint is converted.  ``ckpt_step`` is the fallback for single-step
    use (e.g. tests or one-off runs).
    """

    class_name: str = "MegatronDistToTorchStep"
    cmd: str = ""
    load_dir: str = ""
    ckpt_step: int | None = None
    save_interval: int | None = None
    train_iters: int | None = None


@register
class MegatronDistToTorchStep(PostProcessStepInterface):
    config: MegatronDistToTorchStepConfig

    def get_run_mode(self) -> Literal["same_job", "new_job"]:
        return "same_job"

    def build_commands(self) -> list[str]:
        cfg = self.config
        base = cfg.cmd
        if cfg.load_dir:
            # argparse last-wins, so this overrides any earlier --load.
            base = f"{base} --load {shlex.quote(cfg.load_dir)}"

        if cfg.save_interval is not None and cfg.train_iters is not None:
            steps = _all_ckpt_steps(cfg.save_interval, cfg.train_iters)
        elif cfg.ckpt_step is not None:
            steps = [cfg.ckpt_step]
        else:
            return [base]

        return [f"{base} --load-iteration {step}" for step in steps]


__all__ = ["MegatronDistToTorchStep", "MegatronDistToTorchStepConfig"]
