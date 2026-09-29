"""Provider-neutral kernel categories applicable to arbitrary models."""

from __future__ import annotations


def classify_generic_kernel(name: str) -> str:
    text = name.lower().replace("-", "_")
    if any(value in text for value in ("nccl", "rccl")):
        return "communication"
    if any(
        value in text
        for value in ("cijk", "gemm", "matmul", "hipblas", "rocblas", "tensile", "cublas")
    ):
        return "gemm_ambiguous"
    if any(value in text for value in ("fmha", "flash_fprop", "flash_bprop", "attention")):
        return "attention"
    if any(value in text for value in ("layer_norm", "layernorm", "rmsnorm", "norm_kernel")):
        return "normalization"
    if any(value in text for value in ("optimizer", "adam", "multi_tensor", "weight_decay")):
        return "optimizer"
    if any(value in text for value in ("reduce", "softmax", "cross_entropy")):
        return "reduction_softmax"
    if any(
        value in text
        for value in ("elementwise", "vectorized", "fillfunctor", "copy_kernel", "cast_kernel")
    ):
        return "elementwise"
    if any(value in text for value in ("memcpy", "memset", "fillbuffer")):
        return "memory"
    return "other"


__all__ = ["classify_generic_kernel"]
