"""Configuration schema for slurm_gen.

These types define the configuration interface for SLURM clients and
script generation. Designed for use with compoconf.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field
from typing import Any, Protocol
from pathlib import Path

from compoconf import (
    ConfigInterface,
    NonStrictDataclass,
)


@dataclass(init=False)
class SrunConfig(NonStrictDataclass):
    """Configuration for srun options.

    Used by two fields on :class:`SlurmConfig` with very different status:

    ``srun_args``  RENDERED into the ``{srun_opts}`` template placeholder by
                   ``build_srun_args``. Because it is a mapping, Hydra MERGES it
                   key-by-key across the defaults list, so a cluster group and an
                   experiment can each contribute flags without clobbering each
                   other. Prefer this.
    ``srun``       LEGACY AND DEAD. No template has ever had a placeholder for
                   it and nothing reads it, so every flag in the ``srun:`` blocks
                   of config/slurm/*.yaml (``wait: 60``, ``exclusive: true``,
                   ``kill_on_bad_exit: 1``, ...) has never reached srun. It is
                   left unrendered on purpose: switching it on would silently
                   change every cluster at once, and at least two of those values
                   are actively harmful under ft_launcher (``--wait=60`` kills
                   the step 60 s after the first task exits; Leonardo's
                   ``kill_on_bad_exit: 1`` contradicts the ``--kill-on-bad-exit=0``
                   its experiments pass by hand). Migrate a block to ``srun_args``
                   deliberately, one cluster at a time.
    """

    pass


@dataclass(init=False)
class SbatchConfig(NonStrictDataclass):
    """Configuration for sbatch options."""

    account: str | None = None
    job_name: str | None = None
    nodes: int | None = None
    partition: str | None = None
    qos: str | None = None
    time: str = "0-01:00:00"
    dependency: str | None = None


@dataclass(kw_only=True)
class SlurmConfig(ConfigInterface):
    """Parameters for SBATCH rendering and submission.

    Attributes:
        template_path: Path to the SBATCH template file.
        script_dir: Directory where generated scripts are written.
        log_dir: Directory for SLURM log files.
        name: Optional job name for client tracking.
        script_path: Optional path to the job script for submission.
        log_path: Optional path to the log file for submission.
        array: Whether to use job arrays.
        launcher_cmd: Additional launcher command.
        srun_opts: Additional srun options, as one verbatim string. A STRING is
            replaced wholesale on a Hydra merge, so a cluster group and an
            experiment that both set it will not compose — the last one wins.
            Use ``srun_args`` for anything that needs to compose; keep this for
            flags that cannot be expressed as key/value.
        launcher_env_passthrough: Pass environment to launcher.
        env: Environment variables to set.
        srun_args: srun flags as a MAPPING, merged key-by-key by Hydra and
            rendered ahead of ``srun_opts``. See :class:`SrunConfig`.
        srun: LEGACY, NOT RENDERED. See :class:`SrunConfig`.
        sbatch: sbatch configuration.
        sbatch_extra_directives: Extra sbatch directives.
    """

    class_name: str = "Slurm"
    template_path: str = field(default=MISSING)
    script_dir: str = field(default=MISSING)
    log_dir: str = field(default=MISSING)
    name: str = "job"
    script_path: str | None = None
    log_path: str | None = None
    array: bool = False
    launcher_cmd: str = ""
    srun_opts: str = ""
    launcher_env_passthrough: bool = False
    env: dict[str, Any] = field(default_factory=dict)
    command: list[str] = field(default_factory=list)
    srun_args: SrunConfig = field(default_factory=SrunConfig)
    srun: SrunConfig = field(default_factory=SrunConfig)
    sbatch: SbatchConfig = field(default_factory=SbatchConfig)
    sbatch_extra_directives: list[str] = field(
        default_factory=list
    )  # deprecated, just add to sbatch
    test_only: bool = False

    def __post_init__(self):
        self.script_path = self.script_path or str(Path(self.script_dir) / (self.name + ".sbatch"))
        self.log_path = self.log_path or str(Path(self.log_dir) / (self.name + ".log"))


class SlurmClientInterface(Protocol):  # pragma: no cover - protocol definitions are not executable
    """Protocol for SLURM client implementations."""

    def submit(self, slurm_config: SlurmConfig) -> str:  # pragma: no cover
        ...

    def submit_array(
        self, slurm_config: SlurmConfig, indices: list[int]
    ) -> list[str]:  # pragma: no cover
        ...

    def cancel(self, job_id: str) -> None:  # pragma: no cover
        ...

    def remove(self, job_id: str) -> None:  # pragma: no cover
        ...

    def squeue(self) -> dict[str, str]:  # pragma: no cover
        ...

    def update_excludes(self, job_id: str, nodelist: str) -> None:  # pragma: no cover
        ...

    def get_job(self, job_id: str):  # pragma: no cover
        ...


__all__ = [
    "SlurmConfig",
    "SrunConfig",
    "SbatchConfig",
    "SlurmClientInterface",
]
