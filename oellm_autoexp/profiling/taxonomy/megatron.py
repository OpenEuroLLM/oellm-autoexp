"""Megatron architecture-specific kernel categories."""

from __future__ import annotations


def classify_megatron_kernel(name: str) -> str | None:
    text = name.lower().replace("-", "_")
    if any(
        value in text
        for value in (
            "chunk_gated_delta",
            "chunk_fwd_kernel",
            "chunk_bwd_kernel",
            "prepare_wy",
            "recompute_w",
            "causal_conv1d",
            "fused_recurrent",
            "wy_repr",
            "gated_delta",
        )
    ):
        return "gated_delta_net"
    if any(value in text for value in ("selective_scan", "mamba", "ssd_chunk")):
        return "state_space_model"
    return None


__all__ = ["classify_megatron_kernel"]
