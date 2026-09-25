"""Provider-neutral profiling data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


class ProfilingError(RuntimeError):
    """Raised when profiling capture or analysis cannot proceed."""


@dataclass(frozen=True)
class KernelEvent:
    name: str
    start_ns: int
    end_ns: int
    device_id: int | str | None = None
    stream_id: int | str | None = None
    correlation_id: int | str | None = None

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns


@dataclass(frozen=True)
class MemoryEvent:
    name: str
    start_ns: int
    end_ns: int
    bytes: int | None = None
    device_id: int | str | None = None


@dataclass(frozen=True)
class MarkerEvent:
    name: str
    start_ns: int
    end_ns: int
    domain: str | None = None


@dataclass(frozen=True)
class IterationRecord:
    iteration: int
    timestamp: datetime
    elapsed_ms: float
    tflops_per_gpu: float
    tokens_per_second_per_gpu: float


@dataclass(frozen=True)
class SteadyWindow:
    start_iteration: int | None
    end_iteration: int | None
    step_count: int | None
    duration_ns: int | None
    profile_metrics: dict[str, float]


@dataclass(frozen=True)
class ProfileArtifacts:
    provider: str
    root: Path
    primary: Path
    stem: str
    files: dict[str, Path] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CaptureManifest:
    schema_version: int
    provider: str
    mode: str
    rank: int
    hostname: str
    command: list[str]
    tool_path: str | None
    tool_version: str | None
    output_dir: str
    expected_artifacts: list[str]
    provider_options: dict[str, Any]
    analysis_options: dict[str, Any]
