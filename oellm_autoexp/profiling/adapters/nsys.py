"""Nsight Systems SQLite adapter."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from oellm_autoexp.profiling.models import KernelEvent, ProfileArtifacts, ProfilingError


KERNEL_TABLE_CANDIDATES = (
    "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL",
    "CUPTI_ACTIVITY_KIND_KERNEL",
    "CUDA_GPU_KERNEL",
)


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _tables(connection: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }


def _columns(connection: sqlite3.Connection, table: str) -> dict[str, str]:
    return {
        str(row[1]).lower(): str(row[1])
        for row in connection.execute(f"PRAGMA table_info({_quote(table)})")
    }


class NsysAdapter:
    provider = "nsys"

    def discover(self, path: Path, rank: str = "rank0") -> ProfileArtifacts:
        if path.is_file():
            if path.suffix not in {".sqlite", ".db"}:
                raise ProfilingError(f"Expected an nsys SQLite file, got: {path}")
            database = path
        elif path.is_dir():
            candidates = list(path.glob(f"{rank}*.sqlite"))
            if not candidates:
                candidates = list(path.glob("*.sqlite")) + list(path.glob("*.db"))
            if not candidates:
                raise ProfilingError(f"No nsys SQLite export found under {path}")
            database = max(candidates, key=lambda candidate: candidate.stat().st_size)
        else:
            raise ProfilingError(f"Trace path does not exist: {path}")
        return ProfileArtifacts(
            provider=self.provider,
            root=database.parent,
            primary=database,
            stem=database.stem,
            files={"sqlite": database},
        )

    def _kernel_table(self, connection: sqlite3.Connection) -> str:
        present = _tables(connection)
        for candidate in KERNEL_TABLE_CANDIDATES:
            if candidate in present:
                return candidate
        raise ProfilingError(
            "Nsight SQLite has no recognized CUDA kernel table; "
            f"present tables include: {sorted(present)[:20]}"
        )

    def iter_kernels(self, artifacts: ProfileArtifacts) -> Iterator[KernelEvent]:
        database = artifacts.files["sqlite"]
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        try:
            table = self._kernel_table(connection)
            columns = _columns(connection, table)
            try:
                start = columns["start"]
                end = columns["end"]
            except KeyError as error:
                raise ProfilingError(f"Nsight kernel table {table} lacks start/end") from error

            name_column = next(
                (
                    columns[key]
                    for key in ("demangledname", "shortname", "name")
                    if key in columns
                ),
                None,
            )
            if name_column is None:
                raise ProfilingError(f"Nsight kernel table {table} lacks a name column")
            stream = next(
                (columns[key] for key in ("streamid", "stream_id") if key in columns),
                None,
            )
            device = next(
                (columns[key] for key in ("deviceid", "device_id") if key in columns),
                None,
            )
            correlation = next(
                (
                    columns[key]
                    for key in ("correlationid", "correlation_id")
                    if key in columns
                ),
                None,
            )

            string_tables = _tables(connection)
            string_columns = (
                _columns(connection, "StringIds") if "StringIds" in string_tables else {}
            )
            use_string_join = (
                "id" in string_columns
                and "value" in string_columns
                and name_column.lower() != "name"
            )
            selections = [
                f"k.{_quote(start)}",
                f"k.{_quote(end)}",
                (
                    f"s.{_quote(string_columns['value'])}"
                    if use_string_join
                    else f"k.{_quote(name_column)}"
                ),
                f"k.{_quote(device)}" if device else "NULL",
                f"k.{_quote(stream)}" if stream else "NULL",
                f"k.{_quote(correlation)}" if correlation else "NULL",
            ]
            query = f"SELECT {', '.join(selections)} FROM {_quote(table)} k"
            if use_string_join:
                query += (
                    f" LEFT JOIN {_quote('StringIds')} s"
                    f" ON k.{_quote(name_column)} = s.{_quote(string_columns['id'])}"
                )
            query += f" ORDER BY k.{_quote(start)}"

            for start_ns, end_ns, name, device_id, stream_id, correlation_id in connection.execute(
                query
            ):
                if start_ns is None or end_ns is None or int(end_ns) <= int(start_ns):
                    continue
                yield KernelEvent(
                    name=str(name or "<unnamed CUDA kernel>"),
                    start_ns=int(start_ns),
                    end_ns=int(end_ns),
                    device_id=device_id,
                    stream_id=stream_id,
                    correlation_id=correlation_id,
                )
        finally:
            connection.close()

    def supplemental_stats(self, artifacts: ProfileArtifacts) -> dict[str, Any]:
        database = artifacts.files["sqlite"]
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        try:
            return {"sqlite_tables": sorted(_tables(connection))}
        finally:
            connection.close()


__all__ = ["NsysAdapter"]
