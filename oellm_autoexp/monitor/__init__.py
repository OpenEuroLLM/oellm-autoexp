from . import app, loop
from .job_client_protocol import JobClientProtocol
from .local_client import LocalCommandClient

# Import SlurmClient optionally (requires slurm_gen)
try:
    from .slurm_client import SlurmClient

    _has_slurm = True
except ImportError:
    SlurmClient = None  # type: ignore
    _has_slurm = False

__all__ = [
    "loop",
    "app",
    "JobClientProtocol",
    "LocalCommandClient",
]

if _has_slurm:
    __all__.append("SlurmClient")
