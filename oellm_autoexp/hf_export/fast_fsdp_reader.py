"""Cached, file-parallel DCP reader for immutable, pinned CPU export inputs.

Conversion arithmetic is unchanged. Parallelism is limited to tensor requests
under PyTorch's default CPU load planner: each request writes a disjoint slice.
Other planners and byte objects retain PyTorch's serial reader behavior.
"""
from __future__ import annotations

import dataclasses
import time
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.planner import LoadItemType


class ExportFileSystemReader(dcp.FileSystemReader):
    def __init__(self, path, *, threads: int = 8):
        if threads < 1:
            raise ValueError("Export read threads must be positive")
        self._metadata_cache = {}
        self.metadata_disk_reads = 0
        self.metadata_seconds = 0.0
        self.read_seconds = 0.0
        self.threads = threads
        super().__init__(path)

    def reset(self, checkpoint_id=None):
        # dcp.load can reset the reader between layers. A reset of the same
        # immutable checkpoint must not invalidate the metadata cache.
        if checkpoint_id is not None and str(checkpoint_id) != str(self.path):
            self._metadata_cache.clear()
        super().reset(checkpoint_id)

    def read_metadata(self, *args, **kwargs):
        key = kwargs.get("rank")
        if key not in self._metadata_cache:
            started = time.perf_counter()
            self._metadata_cache[key] = super().read_metadata(*args, **kwargs)
            self.metadata_seconds += time.perf_counter() - started
            self.metadata_disk_reads += 1
        metadata = self._metadata_cache[key]
        if metadata.storage_meta is not None:
            metadata.storage_meta.load_id = self.load_id
        return metadata

    def read_data(self, plan, planner):
        started = time.perf_counter()
        if (
            self.threads == 1
            or type(planner) is not DefaultLoadPlanner
            or any(item.type == LoadItemType.BYTE_IO for item in plan.items)
        ):
            result = super().read_data(plan, planner)
            self.read_seconds += time.perf_counter() - started
            return result
        per_file = {}
        for item in plan.items:
            storage = self.storage_data[item.storage_index]
            per_file.setdefault(storage.relative_path, []).append(item)
        # No concurrent checkpoint loads or writes to shared planner bookkeeping:
        # DefaultLoadPlanner.commit_tensor is a no-op; target CPU slices are
        # independent and DCP performs all plan construction before this call.
        def read_file(items):
            items.sort(key=lambda item: self.storage_data[item.storage_index].offset)
            subplan = dataclasses.replace(plan, items=items)
            dcp.FileSystemReader.read_data(self, subplan, planner).wait()

        with ThreadPoolExecutor(max_workers=self.threads) as executor:
            list(executor.map(read_file, per_file.values()))
        self.read_seconds += time.perf_counter() - started
        result = torch.futures.Future()
        result.set_result(None)
        return result
