"""Configuration schema for AutoExp timeline profiling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from compoconf import ConfigInterface


@dataclass(kw_only=True)
class CaptureDomainsConfig(ConfigInterface):
    kernel: bool = True
    runtime: bool = True
    communication: bool = True
    memory: bool = True
    markers: bool = False


@dataclass(kw_only=True)
class ProfileAnalysisConfig(ConfigInterface):
    enabled: bool = True
    auto_run: bool = False
    execution: str = "dependent_slurm"
    steady_start_iteration: int = 5
    steady_end_iteration: int | None = None
    canvas: bool = False
    partition: str | None = None
    account: str | None = None
    cpus: int = 8
    memory: str = "32G"
    time: str = "00:20:00"

    def __post_init__(self) -> None:
        if self.execution not in {"dependent_slurm", "same_job"}:
            raise ValueError(f"Unsupported profiling analysis execution: {self.execution}")


@dataclass(kw_only=True)
class ProfilingConfig(ConfigInterface):
    provider: str = "none"
    mode: str = "timeline"
    ranks: list[int] = field(default_factory=lambda: [0])
    output_dir: str = "${job.base_output_dir}/profiling"
    capture: CaptureDomainsConfig = field(default_factory=CaptureDomainsConfig)
    analysis: ProfileAnalysisConfig = field(default_factory=ProfileAnalysisConfig)
    provider_options: dict[str, Any] = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        return self.provider != "none"

    def __post_init__(self) -> None:
        if self.provider not in {"none", "rocprofv3", "nsys"}:
            raise ValueError(f"Unsupported profiling provider: {self.provider}")
        if self.mode != "timeline":
            raise ValueError(f"Unsupported profiling mode: {self.mode}")
        if any(rank < 0 for rank in self.ranks):
            raise ValueError("Profiling ranks must be non-negative")


__all__ = [
    "CaptureDomainsConfig",
    "ProfileAnalysisConfig",
    "ProfilingConfig",
]
