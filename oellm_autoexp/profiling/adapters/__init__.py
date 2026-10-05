"""Built-in profiling provider adapters."""

from .base import available_adapters, get_adapter, register_adapter
from .nsys import NsysAdapter
from .rocprofv3 import RocprofV3Adapter

register_adapter(RocprofV3Adapter())
register_adapter(NsysAdapter())

__all__ = [
    "NsysAdapter",
    "RocprofV3Adapter",
    "available_adapters",
    "get_adapter",
]
