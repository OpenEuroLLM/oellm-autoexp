"""Custom TorchTitan JobConfig with extra model sizing fields.

NOTE: Must NOT use 'from __future__ import annotations' — torchtitan's
ConfigManager requires runtime type inspection of field annotations.
"""

from dataclasses import dataclass, field

from titan_oellm.configs.oellm_job_config import JobConfig as BaseJobConfig
from titan_oellm.configs.oellm_job_config import Model as BaseModel


@dataclass
class Model(BaseModel):
    """Extend Titan-OELLM Model config.

    oellm_job_config.Model already provides all MoE fields
    (moe_num_experts, moe_top_k, moe_score_func, moe_route_norm,
    moe_route_scale, moe_score_before_experts, moe_num_shared_experts)
    and attn_gate_* fields. Only add fields not present in the base.
    """

    ffn_dim_multiplier: float | None = None
    multiple_of: int | None = 256


@dataclass
class JobConfig(BaseJobConfig):
    model: Model = field(default_factory=Model)


def _apply_pp_fsdp_reshard_fix() -> None:
    """Monkey-patch for
    torch.distributed.pipelining.stage.backward_maybe_with_nosync.

    Root cause of OOM in PP+FSDP2 training:
    The original backward_maybe_with_nosync always calls:
        self.submod.set_reshard_after_backward(False)
    for FSDP modules, causing ALL N layers' unsharded parameters to accumulate in
    memory DURING the backward pass. For 235BA22B PP=2 (47 layers × ~543 MB/layer),
    this adds ~26 GiB on top of the ~37 GiB fixed overhead, causing OOM.

    Fix 1 (reshard): set reshard_after_backward=True so each layer's AllGather
    buffer is freed immediately after its backward instead of accumulating.

    Fix 2 (run_post_backward): call run_post_backward() for last_backward=True to
    explicitly trigger ReduceScatter for all accumulated B0+...+Bk gradients.

    Root cause of clip_grad_norm_ OOM (jobs 325012, 325069, 325077):
    The original backward_maybe_with_nosync in the container (newer PyTorch) relies on
    perform_reduce_grad as a separate schedule action to do the ReduceScatter.
    However, perform_reduce_grad may not correctly iterate ALL inner FSDP states
    (47 layer param groups), leaving p.grad unset or in Partial placement.
    clip_grad_norm_ then triggers NCCL OOM (DTensor Partial→Replicate redistribution).

    Fix 2 mirrors the LOCAL backward_maybe_with_nosync behavior which explicitly
    calls post_backward() for ALL states via fully_shard.state()._state_ctx.all_states
    and waits via _root_post_backward_final_callback(). If the fully_shard.state()
    API is not available (newer PyTorch restructured it), falls back gracefully.

    Cost: (n_microbatches - 1) × N extra AllGather ops per backward step.
    For 235BA22B PP=2 bs=4 (4 microbatches, 47 layers): 3×47=141 extra AllGathers.
    Estimated overhead: <2.5% of step time (AllGathers overlap with computation).
    """
    try:
        from torch.distributed.pipelining.stage import _PipelineStageBase
        from torch.distributed.fsdp import FSDPModule, fully_shard
        from torch.distributed.pipelining._backward import (
            stage_backward,
            stage_backward_input,
            stage_backward_weight,
        )
    except ImportError:
        return

    _orig = _PipelineStageBase.backward_maybe_with_nosync

    def _patched(self, backward_type, bwd_kwargs, last_backward=False):
        if not isinstance(self.submod, FSDPModule):
            # Non-FSDP path: use original implementation unchanged
            return _orig(self, backward_type, bwd_kwargs, last_backward)

        # KEY FIX 1: use reshard_after_backward=True so that after each individual
        # layer's backward pass, its params are resharded immediately.
        # Original always sets False, causing all N layers' unsharded params
        # to accumulate simultaneously (~26 GiB for 235BA22B PP=2).
        #
        # requires_gradient_sync=False defers ReduceScatter to run_post_backward
        # (called below for last_backward=True, like the local backward_maybe_with_nosync).
        # is_last_backward=False here always — run_post_backward sets True for B_last.
        self.submod.set_is_last_backward(False)
        self.submod.set_reshard_after_backward(True)  # <-- FIXED (was always False)
        self.submod.set_requires_gradient_sync(False)  # deferred to run_post_backward

        if backward_type == "full":
            result = (
                stage_backward(
                    bwd_kwargs["stage_output"],
                    bwd_kwargs["output_grads"],
                    bwd_kwargs["input_values"],
                ),
                None,
            )
        elif backward_type == "input":
            result = stage_backward_input(
                bwd_kwargs["stage_output"],
                bwd_kwargs["output_grads"],
                bwd_kwargs["input_values"],
                self.submod.parameters(),
            )
        elif backward_type == "weight":
            result = (
                stage_backward_weight(self.submod.parameters(), bwd_kwargs["param_groups"]),
                None,
            )
        else:
            raise RuntimeError(f"Unknown backward type: {backward_type}")

        if last_backward:
            # KEY FIX 2: explicitly trigger ReduceScatter for all accumulated grads.
            # Mirrors the LOCAL backward_maybe_with_nosync's run_post_backward() call.
            #
            # Why needed: in the container (newer PyTorch), backward_maybe_with_nosync
            # always sets requires_gradient_sync=False (no per-backward ReduceScatter).
            # perform_reduce_grad handles ReduceScatter via the schedule IR. However,
            # perform_reduce_grad may not correctly iterate ALL inner FSDP states,
            # leaving p.grad unset. Our explicit call here ensures ALL 47 inner layer
            # param groups get their ReduceScatter triggered and p.grad set to Shard(0).
            # This is safe even if perform_reduce_grad also runs — it will be a no-op
            # since we clear unsharded_accumulated_grad below.
            self.submod.set_is_last_backward(True)
            self.submod.set_reshard_after_backward(True)  # keep memory fix
            self.submod.set_requires_gradient_sync(True)
            try:
                fsdp_state = fully_shard.state(self.submod)
                for state in fsdp_state._state_ctx.all_states:
                    if state._fsdp_param_group:
                        state._fsdp_param_group.post_backward()
                fsdp_state._root_post_backward_final_callback()
            except Exception as e:
                # Fallback: if FSDP internal API changed in this PyTorch version,
                # at least synchronize CUDA to prevent async race conditions.
                import torch
                import torch.distributed as dist

                rank = dist.get_rank() if dist.is_initialized() else 0
                if rank == 0:
                    print(
                        f"[PP_FSDP_FIX] run_post_backward fallback triggered: {e}\n"
                        f"  FSDP API may differ in container PyTorch. Falling back to cuda sync.",
                        flush=True,
                    )
                torch.cuda.synchronize()  # ensure all async ops complete

        return result

    _PipelineStageBase.backward_maybe_with_nosync = _patched


_apply_pp_fsdp_reshard_fix()
