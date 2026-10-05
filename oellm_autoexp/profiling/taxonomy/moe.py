"""Optional MoE-specific kernel categories."""

from __future__ import annotations

def classify_moe_kernel(name: str) -> str | None:
    text = name.lower().replace("-", "_")
    if any(
        value in text
        for value in (
            "sort_chunks",
            "permute",
            "unpermute",
            "router",
            "routing",
            "topk",
            "top_k",
            "radix_sort",
            "histogram",
        )
    ):
        return "routing_permutation"
    return None


__all__ = ["classify_moe_kernel"]
