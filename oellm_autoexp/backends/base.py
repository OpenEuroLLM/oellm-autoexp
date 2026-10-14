"""Backend abstractions used by oellm_autoexp."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from collections.abc import Sequence

from compoconf import ConfigInterface, NonStrictDataclass, register

from oellm_autoexp.config.schema import BackendInterface

LOGGER = logging.getLogger(__name__)


class BaseBackendConfig(ConfigInterface):
    env: dict[str, str]


class BaseBackend(BackendInterface):
    """Base class for backend adapters."""

    config: BaseBackendConfig

    def __init__(self, config: BaseBackendConfig) -> None:
        self.config = config

    def validate(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def build_launch_command(self) -> str:  # pragma: no cover
        raise NotImplementedError


@dataclass(kw_only=True)
class NullBackendConfig(ConfigInterface):
    """Backend that echoes sweep parameters for testing."""

    base_command: Sequence[str] = field(
        default_factory=lambda: [
            "echo",
        ]
    )
    extra_cli_args: Sequence[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    dummy: int = 0


@register
class NullBackend(BaseBackend):
    config: NullBackendConfig

    def validate(self) -> None:  # pragma: no cover - trivial
        pass

    def build_launch_command(self) -> str:
        argv: list[str] = [str(arg) for arg in self.config.base_command]
        argv.extend(str(arg) for arg in self.config.extra_cli_args)
        return " ".join(argv)


# init=False is REQUIRED by NonStrictDataclass (see its docstring); with
# kw_only=True instead, parsing accepts the extra keys and then __init__ rejects
# them: "BashBackendConfig.__init__() got an unexpected keyword argument
# 'megatron'". Matches MegatronBackendConfig / OELLMEvalBackendConfig.
@dataclass(init=False)
class BashBackendConfig(NonStrictDataclass, BaseBackendConfig):
    """Backend that runs an arbitrary bash string instead of a real trainer.

    NON-STRICT, like every other backend config (MegatronBackendConfig,
    MegatronBridgeBackendConfig, OELLMEvalBackendConfig, TitanBackendConfig).
    It has to be, for this backend to be usable as a per-sweep-point STAGE of a
    megatron experiment: swapping `backend: bash` on one sweep point does not
    remove the inherited `backend.megatron` / `backend.aux` /
    `backend.torchrun_args` subtrees -- Hydra merges rather than replaces -- so a
    strict parse dies with
        ValueError: Undefined keys {'megatron', 'torchrun_args', 'aux'} ...
        for <class BashBackendConfig>: ['class_name', 'command', 'env']
    This is exactly why OELLMEvalBackendConfig is non-strict, which is what makes
    the train->eval chain configs work.

    The cost is the usual one: a typo in a bash-backend key is silently accepted
    rather than rejected.

    Useful for exercising the monitor / job-control configs (``auto_cancel``
    etc.) on a real cluster without launching megatron: point ``command`` at a
    sequence of ``echo`` + ``sleep`` statements that emulate a training log, an
    ``srun: error``, a stall, a clean finish, and so on.

    Why not ``NullBackend``? ``NullBackend`` *can* run such a command, but its
    command lives in *list* fields (``base_command``/``extra_cli_args``). The
    staged-sweep serializer (``param_to_cmdlines``) quotes scalar *string*
    overrides but emits list elements raw (``[a,b]``), so a multi-token command
    breaks Hydra override parsing when swept. A single ``command`` *string* is
    quoted and round-trips cleanly (even multi-line) -- which is what lets
    ``config/experiments/tests/auto_cancel_sweep.yaml`` sweep over commands.

    ``command`` is rendered straight into ``srun ... bash -c '<command>'`` (see
    ``templates/base.sbatch``), so it must not contain single quotes; use double
    quotes inside. ``;`` / ``&&`` / shell loops are all fine.
    """

    class_name: str = "BashBackend"
    command: str = 'echo "[bash] no command configured"'
    env: dict[str, str] = field(default_factory=dict)


@register
class BashBackend(BaseBackend):
    config: BashBackendConfig

    def validate(self) -> None:  # pragma: no cover - trivial
        if not self.config.command.strip():
            raise ValueError("BashBackend.command must be a non-empty bash string")
        if "'" in self.config.command:
            # The base sbatch template wraps the command in single quotes.
            raise ValueError(
                "BashBackend.command must not contain single quotes "
                "(it is embedded in `bash -c '...'`); use double quotes instead"
            )

    def build_launch_command(self) -> str:
        return self.config.command


__all__ = [
    "BaseBackend",
    "NullBackendConfig",
    "BashBackendConfig",
    "BashBackend",
]
