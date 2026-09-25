"""Provider adapter protocol and registry."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from oellm_autoexp.profiling.models import KernelEvent, ProfileArtifacts, ProfilingError


class ProfileAdapter(Protocol):
    provider: str

    def discover(self, path, rank: str = "rank0") -> ProfileArtifacts: ...

    def iter_kernels(self, artifacts: ProfileArtifacts) -> Iterator[KernelEvent]: ...

    def supplemental_stats(self, artifacts: ProfileArtifacts) -> dict: ...


_ADAPTERS: dict[str, ProfileAdapter] = {}


def register_adapter(adapter: ProfileAdapter) -> None:
    _ADAPTERS[adapter.provider] = adapter


def get_adapter(provider: str) -> ProfileAdapter:
    try:
        return _ADAPTERS[provider]
    except KeyError as error:
        raise ProfilingError(
            f"Unknown profiling provider {provider!r}; available: {sorted(_ADAPTERS)}"
        ) from error


def available_adapters() -> tuple[str, ...]:
    return tuple(sorted(_ADAPTERS))


__all__ = [
    "ProfileAdapter",
    "available_adapters",
    "get_adapter",
    "register_adapter",
]
