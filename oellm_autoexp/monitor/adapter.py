"""Adapter for using monitor library with oellm_autoexp clients."""

from __future__ import annotations

from typing import Any

from monitor.job_client_protocol import JobClientProtocol
from oellm_autoexp.slurm.client import BaseSlurmClient


class SlurmClientAdapter(JobClientProtocol):
    def __init__(self, client: BaseSlurmClient):
        self.client = client

    def submit(
        self,
        name: str,
        command: list[str],
        log_path: str,
        extra_args: list[str] | None = None,
        log_to_file: bool | None = None,
        log_path_current: str | None = None,
        slurm: dict[str, Any] | None = None,
    ) -> str:
        script_path = command[0]
        return self.client.submit(name, script_path, log_path)

    def submit_array(
        self,
        array_name: str,
        command: list[str],
        log_paths: list[str],
        task_names: list[str],
        extra_args: list[str] | None = None,
        start_index: int | None = None,
        log_to_file: bool | None = None,
        log_path_current: str | None = None,
        slurm: dict[str, Any] | None = None,
    ) -> list[str]:
        script_path = command[0]
        if hasattr(self.client, "submit_array"):
            return self.client.submit_array(
                array_name, script_path, log_paths, task_names, start_index=start_index or 0
            )
        raise NotImplementedError("submit_array not supported by this client")

    def cancel(self, job_id: str) -> None:
        self.client.cancel(job_id)

    def remove(self, job_id: str) -> None:
        if hasattr(self.client, "remove"):
            self.client.remove(job_id)

    def squeue(self) -> dict[str, str]:
        return self.client.squeue()
