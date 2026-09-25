"""MoE-oriented kernel taxonomy shared by all profiling providers."""

from __future__ import annotations

from collections.abc import Iterable


def _contains(text: str, patterns: Iterable[str]) -> bool:
    return any(pattern in text for pattern in patterns)


def classify_kernel(name: str) -> str:
    """Classify a kernel without claiming semantic ownership for generic GEMMs."""

    text = name.lower().replace("-", "_")
    if _contains(text, ("nccl", "rccl")):
        return "communication"
    if _contains(
        text,
        (
            "sort_chunks",
            "permute",
            "unpermute",
            "router",
            "routing",
            "topk",
            "top_k",
            "radix_sort",
            "histogram",
        ),
    ):
        return "routing_permutation"
    if _contains(
        text,
        (
            "chunk_gated_delta",
            "chunk_fwd_kernel",
            "chunk_bwd_kernel",
            "prepare_wy",
            "recompute_w",
            "causal_conv1d",
            "fused_recurrent",
            "wy_repr",
            "gated_delta",
        ),
    ):
        return "gated_delta_net"
    if _contains(text, ("fmha", "flash_fprop", "flash_bprop", "attention")):
        return "attention"
    if _contains(
        text,
        ("cijk", "gemm", "matmul", "hipblas", "rocblas", "tensile", "grouped_gemm"),
    ):
        return "gemm_ambiguous"
    if _contains(text, ("layer_norm", "layernorm", "rmsnorm", "norm_kernel")):
        return "normalization"
    if _contains(text, ("optimizer", "adam", "multi_tensor", "weight_decay")):
        return "optimizer"
    if _contains(text, ("reduce", "softmax", "cross_entropy")):
        return "reduction_softmax"
    if _contains(
        text,
        ("elementwise", "vectorized", "fillfunctor", "copy_kernel", "cast_kernel"),
    ):
        return "elementwise"
    if _contains(text, ("memcpy", "memset", "fillbuffer")):
        return "memory"
    return "other"


__all__ = ["classify_kernel"]
