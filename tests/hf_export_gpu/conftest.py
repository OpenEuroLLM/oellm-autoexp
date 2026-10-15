"""HF-export parity against tiny Megatron models (GPU + Megatron-LM on the
path).

Run with Megatron-LM (and, for the FLA tests, the ComplexKDA ``fla``) on the path::

    PYTHONPATH=.:submodules/Megatron-LM:<ComplexKDA checkout> \\
        torchrun --nproc-per-node 1 -m pytest tests/hf_export_gpu

(FLA tests also need the ComplexKDA ``fla`` on the path.)
"""

import os

import pytest
import torch
import torch.distributed as dist

try:
    import megatron.core  # noqa: F401

    HAVE_MEGATRON = True
except ImportError:
    HAVE_MEGATRON = False


def pytest_collection_modifyitems(config, items):
    skip = None
    if not torch.cuda.is_available():
        skip = pytest.mark.skip(reason="needs a GPU")
    elif not HAVE_MEGATRON:
        skip = pytest.mark.skip(reason="needs Megatron-LM on the path")
    if skip is not None:
        for item in items:
            if "hf_export_gpu" in str(item.fspath):
                item.add_marker(skip)


class Utils:
    @staticmethod
    def initialize_model_parallel(tp: int = 1, pp: int = 1) -> None:
        from megatron.core import parallel_state

        if not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29511")
            dist.init_process_group(
                backend="nccl",
                world_size=int(os.environ.get("WORLD_SIZE", 1)),
                rank=int(os.environ.get("RANK", 0)),
            )
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        parallel_state.destroy_model_parallel()
        parallel_state.initialize_model_parallel(tp, pp)

    @staticmethod
    def destroy_model_parallel() -> None:
        from megatron.core import parallel_state

        parallel_state.destroy_model_parallel()
