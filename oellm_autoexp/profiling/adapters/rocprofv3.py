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

    def trace_end_ns(self, artifacts: ProfileArtifacts) -> int:
        """Find the final timestamp from the CSV tail without a second full parse."""

        path = artifacts.files["kernel_trace"]
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            fieldnames = next(csv.reader([handle.readline()]))
        try:
            start_index = fieldnames.index("Start_Timestamp")
            end_index = fieldnames.index("End_Timestamp")
        except ValueError as error:
            raise ProfilingError(f"{path} lacks timestamp columns") from error

        kernel_stats = _parse_stats(artifacts.files.get("kernel_stats"))
        max_duration_ns = max(
            (
                int(row["MaxNs"])
                for row in kernel_stats
                if isinstance(row.get("MaxNs"), int)
            ),
            default=None,
        )
        if max_duration_ns is None:
            return max((event.end_ns for event in self.iter_kernels(artifacts)), default=0)

        file_size = path.stat().st_size
        tail_size = min(file_size, 1024 * 1024)
        while tail_size <= file_size:
            with path.open("rb") as handle:
                handle.seek(file_size - tail_size)
                raw = handle.read().decode("utf-8", errors="replace")
            lines = raw.splitlines()
            if file_size > tail_size and lines:
                lines = lines[1:]  # Discard a potentially partial first record.
            starts: list[int] = []
            ends: list[int] = []
            for line in lines:
                if not line.strip() or line.startswith('"Kind"'):
                    continue
                try:
                    row = next(csv.reader([line]))
                    starts.append(int(row[start_index]))
                    ends.append(int(row[end_index]))
                except (csv.Error, IndexError, ValueError):
                    continue
            if starts:
                monotonic = all(left <= right for left, right in zip(starts, starts[1:]))
                max_end = max(ends)
                if monotonic and (file_size == tail_size or min(starts) + max_duration_ns <= max_end):
                    return max_end
            if tail_size == file_size:
                break
            tail_size = min(file_size, tail_size * 2)
        return max((event.end_ns for event in self.iter_kernels(artifacts)), default=0)

    def supplemental_stats(self, artifacts: ProfileArtifacts) -> dict[str, Any]:
        return {
            "kernel_stats": _parse_stats(artifacts.files.get("kernel_stats")),
            "communication_api_stats": _parse_stats(
                artifacts.files.get("rccl_api_stats")
            ),
            "runtime_api_stats": _parse_stats(artifacts.files.get("hip_api_stats")),
        }


__all__ = ["RocprofV3Adapter"]
