"""Monkey-patch Megatron's TEGroupedLinear to use Primus-Turbo grouped GEMM.

This replaces the TE-backed per-expert GEMM with primus_turbo's fused
grouped_gemm kernel. On each forward call the per-expert weight{i}
parameters are stacked into [G, out, in] and passed to grouped_gemm.
Gradients flow back through torch.stack to the original TE parameters,
so checkpoint loading, distributed optimizer, and DDP work unchanged.

Usage — set as the launcher script in your experiment config:

    backend:
      launcher_script: ./scripts/turbo_grouped_gemm_patch.py
      megatron:
        moe_grouped_gemm: true

Requires primus_turbo (already in primus_v26.1.sif). TP=1 only.
"""

import sys

import torch


def _apply_turbo_grouped_gemm_patch():
    """Replace TEColumnParallelGroupedLinear and TERowParallelGroupedLinear
    with Primus-Turbo-backed versions in all Megatron modules that reference them."""

    import primus_turbo.pytorch as primus_turbo_torch
    from megatron.core.extensions import transformer_engine as te_ext
    from megatron.core.extensions.transformer_engine import (
        TEGroupedLinear,
        condition_init_method,
    )
    from megatron.core.utils import get_pg_size

    class TurboGroupedLinear(TEGroupedLinear):
        """TEGroupedLinear subclass that uses primus_turbo grouped_gemm.

        Keeps TE's original weight{i} parameters untouched — checkpoint
        loading, distributed optimizer, and DDP all work as before.
        On each forward call we torch.stack the per-expert weights and
        call the fused grouped_gemm kernel."""

        def __init__(
            self,
            num_gemms,
            input_size,
            output_size,
            *,
            parallel_mode,
            config,
            init_method,
            bias,
            skip_bias_add,
            is_expert=False,
            tp_comm_buffer_name=None,
            pg_collection=None,
        ):
            super().__init__(
                num_gemms=num_gemms,
                input_size=input_size,
                output_size=output_size,
                parallel_mode=parallel_mode,
                config=config,
                init_method=init_method,
                bias=bias,
                skip_bias_add=skip_bias_add,
                is_expert=is_expert,
                tp_comm_buffer_name=tp_comm_buffer_name,
                pg_collection=pg_collection,
            )
            tp_size = get_pg_size(self._tp_group)
            assert tp_size == 1, f"TurboGroupedLinear only supports TP=1, got {tp_size}"

        def forward(self, x, m_splits):
            weights = torch.stack(
                [getattr(self, f"weight{i}") for i in range(self.num_gemms)]
            )
            if isinstance(m_splits, list):
                m_splits = torch.tensor(m_splits, dtype=torch.long, device=x.device)
            else:
                m_splits = m_splits.to(x.device)
            out = primus_turbo_torch.ops.grouped_gemm(
                x, weights, m_splits, trans_b=True
            )
            return out, None

    class TurboColumnParallelGroupedLinear(TurboGroupedLinear):

        def __init__(
            self,
            num_gemms,
            input_size,
            output_size,
            *,
            config,
            init_method,
            bias,
            skip_bias_add,
            is_expert,
            tp_comm_buffer_name=None,
            pg_collection=None,
        ):
            super().__init__(
                num_gemms=num_gemms,
                input_size=input_size,
                output_size=output_size,
                parallel_mode="column",
                config=config,
                init_method=condition_init_method(config, init_method),
                bias=bias,
                skip_bias_add=skip_bias_add,
                is_expert=is_expert,
                tp_comm_buffer_name=tp_comm_buffer_name,
                pg_collection=pg_collection,
            )

        def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
            tp_axis_map = {}
            for gemm_idx in range(self.num_gemms):
                tp_axis_map.update(
                    {f"{gemm_idx}.weight": 0, f"{gemm_idx}.bias": 0}
                )
            return super()._sharded_state_dict_grouped(
                tp_axis_map, prefix, sharded_offsets, metadata
            )

    class TurboRowParallelGroupedLinear(TurboGroupedLinear):

        def __init__(
            self,
            num_gemms,
            input_size,
            output_size,
            *,
            config,
            init_method,
            bias,
            skip_bias_add,
            is_expert,
            tp_comm_buffer_name=None,
            pg_collection=None,
        ):
            super().__init__(
                num_gemms=num_gemms,
                input_size=input_size,
                output_size=output_size,
                parallel_mode="row",
                config=config,
                init_method=condition_init_method(config, init_method),
                bias=bias,
                skip_bias_add=skip_bias_add,
                is_expert=is_expert,
                tp_comm_buffer_name=tp_comm_buffer_name,
                pg_collection=pg_collection,
            )

        def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
            tp_axis_map = {
                f"{gemm_idx}.weight": 1 for gemm_idx in range(self.num_gemms)
            }
            return super()._sharded_state_dict_grouped(
                tp_axis_map, prefix, sharded_offsets, metadata
            )

    # --- Monkey-patch all Megatron modules that reference these classes ---

    te_ext.TEColumnParallelGroupedLinear = TurboColumnParallelGroupedLinear
    te_ext.TERowParallelGroupedLinear = TurboRowParallelGroupedLinear

    from megatron.core.extensions import transformer_engine_spec_provider
    transformer_engine_spec_provider.TEColumnParallelGroupedLinear = TurboColumnParallelGroupedLinear
    transformer_engine_spec_provider.TERowParallelGroupedLinear = TurboRowParallelGroupedLinear

    from megatron.core.models import backends as model_backends
    model_backends.TEColumnParallelGroupedLinear = TurboColumnParallelGroupedLinear
    model_backends.TERowParallelGroupedLinear = TurboRowParallelGroupedLinear

    rank = int(__import__("os").environ.get("RANK", 0))
    if rank == 0:
        print(
            "[turbo_grouped_gemm_patch] Patched TEColumnParallelGroupedLinear -> TurboColumnParallelGroupedLinear"
        )
        print(
            "[turbo_grouped_gemm_patch] Patched TERowParallelGroupedLinear -> TurboRowParallelGroupedLinear"
        )
        import primus_turbo
        print(
            f"[turbo_grouped_gemm_patch] primus_turbo version: {primus_turbo.__version__}"
        )


if __name__ == "__main__":
    import os
    import runpy

    megatron_dir = os.path.join(os.path.dirname(__file__), "..", "submodules", "Megatron-LM")
    megatron_dir = os.path.abspath(megatron_dir)
    sys.path.insert(0, megatron_dir)
    # Also add project root for any local imports
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    _apply_turbo_grouped_gemm_patch()

    # Execute pretrain_gpt.py as __main__ (it has no main() function)
    pretrain_script = os.path.join(megatron_dir, "pretrain_gpt.py")
    sys.argv[0] = pretrain_script
    runpy.run_path(pretrain_script, run_name="__main__")
