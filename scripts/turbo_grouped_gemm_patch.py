"""Monkey-patch Megatron's TEGroupedLinear to use Primus-Turbo grouped GEMM.

This replaces the TE-backed per-expert GEMM with primus_turbo's fused
grouped_gemm kernel, which consolidates all expert weights into a single
[G, out_features, in_features] tensor and launches one kernel instead of G.

Usage — set as the launcher script in your experiment config:

    backend:
      megatron:
        moe_grouped_gemm: true
      launcher_script: ./scripts/turbo_grouped_gemm_patch.py

Or run directly:

    python scripts/turbo_grouped_gemm_patch.py --moe-grouped-gemm [other megatron args...]

Requires primus_turbo to be installed in the container (already present in primus_v26.1.sif).
Only supports TP=1 (tensor_model_parallel_size=1), which is the standard MoE configuration.
"""

import gc
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
        """TEGroupedLinear subclass that consolidates per-expert weights into a
        single [G, out, in] tensor and uses primus_turbo grouped_gemm."""

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
            assert tp_size == 1, (
                f"TurboGroupedLinear only supports TP=1, got {tp_size}"
            )
            assert not self.delay_wgrad_compute, (
                "TurboGroupedLinear does not support delay_wgrad_compute"
            )

            w0 = self.weight0
            buffer = torch.empty(
                self.num_gemms,
                self.out_features,
                self.in_features,
                device=w0.device,
                dtype=w0.dtype,
            )

            with torch.no_grad():
                for i in range(self.num_gemms):
                    buffer[i].copy_(getattr(self, f"weight{i}"))

            self.register_parameter("weights", torch.nn.Parameter(buffer.clone()))

            saved_attrs = [
                dict(getattr(self, f"weight{i}").__dict__)
                for i in range(self.num_gemms)
            ]
            for attr_name, attr_val in saved_attrs[0].items():
                setattr(self.weights, attr_name, attr_val)

            for i in range(self.num_gemms):
                if f"weight{i}" in self._parameters:
                    del self._parameters[f"weight{i}"]
            del buffer
            gc.collect()
            torch.cuda.empty_cache()

            for i in range(self.num_gemms):
                weight_i = torch.nn.Parameter(
                    self.weights[i].detach(), requires_grad=False
                )
                for attr_name, attr_val in saved_attrs[i].items():
                    setattr(weight_i, attr_name, attr_val)
                self.register_parameter(f"weight{i}", weight_i)

        def forward(self, x, m_splits):
            if isinstance(m_splits, list):
                m_splits = torch.tensor(m_splits, dtype=torch.long, device=x.device)
            else:
                m_splits = m_splits.to(x.device)
            out = primus_turbo_torch.ops.grouped_gemm(
                x, self.weights, m_splits, trans_b=True
            )
            return out, None

        def backward_dw(self):
            pass

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
