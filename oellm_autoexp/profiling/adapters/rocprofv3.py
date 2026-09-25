"""rocprofv3 CSV adapter."""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from oellm_autoexp.profiling.models import KernelEvent, ProfileArtifacts, ProfilingError


KERNEL_SUFFIX = "_kernel_trace.csv"


def _parse_stats(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            parsed: dict[str, Any] = dict(row)
            for key in ("Calls", "TotalDurationNs", "MinNs", "MaxNs"):
                if key in parsed:
                    try:
                        parsed[key] = int(parsed[key])
                    except (TypeError, ValueError):
                        pass
            for key in ("AverageNs", "Percentage", "StdDev"):
                if key in parsed:
                    try:
                        parsed[key] = float(parsed[key])
                    except (TypeError, ValueError):
                        pass
            rows.append(parsed)
    return rows


class RocprofV3Adapter:
    provider = "rocprofv3"

    def discover(self, path: Path, rank: str = "rank0") -> ProfileArtifacts:
        if path.is_file():
            if not path.name.endswith(KERNEL_SUFFIX):
                raise ProfilingError(f"Expected a {KERNEL_SUFFIX} file, got: {path}")
            kernel_trace = path
        elif path.is_dir():
            candidates = list(path.glob(f"{rank}_pid*{KERNEL_SUFFIX}"))
            if not candidates:
                candidates = list(path.glob(f"*{KERNEL_SUFFIX}"))
            if not candidates:
                raise ProfilingError(f"No rocprof kernel trace CSV found under {path}")
            kernel_trace = max(candidates, key=lambda candidate: candidate.stat().st_size)
        else:
            raise ProfilingError(f"Trace path does not exist: {path}")

        stem = kernel_trace.name[: -len(KERNEL_SUFFIX)]
        directory = kernel_trace.parent
        files: dict[str, Path] = {"kernel_trace": kernel_trace}
        for key in (
            "kernel_stats",
            "rccl_api_stats",
            "hip_api_stats",
            "memory_copy_trace",
            "memory_copy_stats",
            "agent_info",
            "domain_stats",
        ):
            candidate = directory / f"{stem}_{key}.csv"
            if candidate.exists():
                files[key] = candidate
        return ProfileArtifacts(
            provider=self.provider,
            root=directory,
            primary=kernel_trace,
            stem=stem,
            files=files,
        )

    def iter_kernels(self, artifacts: ProfileArtifacts) -> Iterator[KernelEvent]:
        path = artifacts.files["kernel_trace"]
        with path.open("r", newline="", encoding="utf-8", errors="replace") as handle:
            reader = csv.DictReader(handle)
            required = {"Kernel_Name", "Start_Timestamp", "End_Timestamp"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ProfilingError(f"{path} is missing columns: {sorted(missing)}")
            for row in reader:
                try:
                    start_ns = int(row["Start_Timestamp"])
                    end_ns = int(row["End_Timestamp"])
                except (TypeError, ValueError):
                    continue
                if end_ns <= start_ns:
                    continue
                yield KernelEvent(
                    name=" ".join(row["Kernel_Name"].split()),
                    start_ns=start_ns,
                    end_ns=end_ns,
                    device_id=row.get("Agent_Id"),
                    stream_id=row.get("Queue_Id"),
                    correlation_id=row.get("Correlation_Id"),
                )

    def supplemental_stats(self, artifacts: ProfileArtifacts) -> dict[str, Any]:
        return {
            "kernel_stats": _parse_stats(artifacts.files.get("kernel_stats")),
            "communication_api_stats": _parse_stats(
                artifacts.files.get("rccl_api_stats")
            ),
            "runtime_api_stats": _parse_stats(artifacts.files.get("hip_api_stats")),
        }


__all__ = ["RocprofV3Adapter"]
