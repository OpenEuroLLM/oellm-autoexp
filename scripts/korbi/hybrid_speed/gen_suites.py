"""Generate the arm lists of the hybrid speed suites (config/experiments/korbi/hybrid_speed).

    python scripts/korbi/hybrid_speed/gen_suites.py

Each arm changes ONE feature of its family's reference model; the shared setup is in
speed_base_mn5.yaml. Mixer arms use the 3:1 hybrid stack (X-X-X-* x 3).
"""

from pathlib import Path

import yaml

OUT = Path(__file__).resolve().parents[3] / "config/experiments/korbi/hybrid_speed"
B = "backend.megatron."


def h3(x):
    return f"{x}-{x}-{x}-*-" * 3


def dense130m():
    arms = []

    def arm(name, comment, *dicts, **kv):
        merged = {}
        for d in dicts:
            merged.update(d)
        merged.update(kv)
        arms.append((name, comment, merged))

    lin = lambda v: {"aux.pattern": h3("L"), B + "experimental_attention_variant": v}  # noqa: E731
    # ---- references
    arm("dense", "reference: HybridModel, attention + SwiGLU MLP x 12")
    arm("dense_gpt", "the same model through GPTModel", {
        "backend.launcher_script": "./submodules/Megatron-LM/pretrain_gpt.py",
        B + "is_hybrid_model": False, B + "spec": None, B + "hybrid_layer_pattern": None,
        B + "num_layers": 12})
    arm("dense_nocg", "reference without CUDA graphs (what the graphs are worth)",
        {B + "cuda_graph_impl": "none"})
    # ---- baseline tuning: one speed knob each on the reference (no model change)
    liger = {B + "liger_fused_linear_cross_entropy": True,
             B + "liger_fused_linear_cross_entropy_chunk_size": 8192}
    arm("base_liger", "fused LM-head + CE (Liger, chunked: no full-vocab logits)", liger)
    arm("base_ce_te", "TE parallel cross entropy", {B + "cross_entropy_fusion_impl": "te"})
    arm("base_flash", "attention backend flash (FA2) instead of auto (cuDNN SDPA)",
        {B + "attention_backend": "flash"})
    arm("base_graph_full", "full-iteration CUDA graph (fwd+bwd, not the optimizer)",
        {B + "cuda_graph_impl": "full_iteration"})
    arm("base_liger_mbs16", "Liger CE + mbs 16 (gbs 64)", liger,
        {B + "micro_batch_size": 16, B + "global_batch_size": 64})
    arm("base_liger_nocg", "Liger CE without CUDA graphs", liger, {B + "cuda_graph_impl": "none"})
    arm("base_liger_nodistopt", "Liger CE, plain all-reduce instead of the distributed optimizer",
        liger, {B + "use_distributed_optimizer": False, B + "overlap_param_gather": False})
    arm("base_liger_teact", "Liger CE + TE's fused SwiGLU (instead of Megatron's)", liger,
        {B + "use_te_activation_func": True, B + "bias_swiglu_fusion": False})
    nomask = {B + "create_attention_mask_in_dataloader": False}
    arm("base_liger_nomask", "Liger CE, no dense attention mask from the dataloader", liger, nomask)
    arm("base_liger_nomask_mbs16_nocg", "Liger CE, no mask, mbs 16, no CUDA graphs", liger,
        nomask, {B + "micro_batch_size": 16, B + "global_batch_size": 64,
                 B + "cuda_graph_impl": "none"})
    arm("base_liger_noqkln", "Liger CE without q/k layernorm (architecture change: prices it)",
        liger, {B + "qk_layernorm": False})
    arm("base_liger_mbs4", "Liger CE, mbs 4 (2 accumulation steps)", liger,
        {B + "micro_batch_size": 4})
    qknr = {B + "fused_qk_norm_rope": True}
    arm("base_qknr", "Liger, no mask + fused q/k norm + RoPE (Triton)", liger, nomask, qknr)
    arm("base_resnorm", "Liger, no mask + TE fused residual/RMSNorm backward", liger, nomask,
        {B + "fused_residual_rmsnorm": True})
    arm("base_opfuser", "Liger, no mask + TE operation fuser", liger, nomask,
        {B + "use_transformer_engine_op_fuser": True})
    arm("base_qknr_resnorm", "Liger, no mask, fused q/k norm + RoPE, fused residual norm",
        liger, nomask, qknr, {B + "fused_residual_rmsnorm": True})
    arm("base_fp8", "FP8 (hybrid, delayed scaling as the 32B runs)",
        {B + "fp8": "hybrid", B + "fp8_amax_compute_algo": "max",
         B + "fp8_amax_history_len": 1024})
    # ---- attention
    arm("nope", "no positional encoding", {B + "position_embedding_type": "none"})
    arm("gqa4", "grouped-query attention, 4 kv heads", {B + "num_query_groups": 4})
    arm("xsa", "exclusive self attention", {B + "exclusive_self_attention": True})
    arm("gate", "sigmoid output gate", {B + "attention_output_gate": True})
    arm("xsa_gate", "XSA + output gate",
        {B + "exclusive_self_attention": True, B + "attention_output_gate": True})
    arm("swa3", "3:1 sliding-window (1024) : global",
        {"aux.pattern": h3("W"), B + "window_size": "1023,0"})
    mla = {B + "multi_latent_attention": True, B + "q_lora_rank": 384, B + "kv_lora_rank": 256,
           B + "qk_head_dim": 64, B + "v_head_dim": 64, B + "qk_pos_emb_head_dim": 0,
           B + "rope_type": "rope", "aux.pattern": "+-" * 12,
           # NoPE MLA has no RoPE to fuse (Megatron rejects the flag)
           B + "apply_rope_fusion": False}
    arm("mla", "multi-latent attention, NoPE", mla)
    arm("mla_gqa_xsa_gate", "MLA + 4 kv groups + XSA + output gate", mla, {
        B + "mla_num_query_groups": 4, B + "exclusive_self_attention": True,
        B + "attention_output_gate": True})
    # ---- linear / recurrent mixers, 3:1 with attention
    arm("mamba2", "Mamba2 (24 heads of 64, state 128)", {
        "aux.pattern": h3("M"), B + "mamba_num_heads": 24, B + "mamba_head_dim": 64,
        B + "mamba_num_groups": 8, B + "mamba_state_dim": 128})
    arm("gdn_native", "Megatron's native Gated DeltaNet", {"aux.pattern": h3("G")})
    for v in ["gated_delta_net", "mlstm", "kda", "signed_kda", "complex_kda"]:
        arm(v, f"linear mixer {v}", lin(v))
    arm("kda_xsa", "KDA with exclusive self attention", lin("kda"), {B + "linear_xsa": True})
    tri128 = {B + "linear_key_head_dim": 128, B + "linear_value_head_dim": 128,
              B + "linear_num_key_heads": 6, B + "linear_num_value_heads": 6}
    arm("kda_hd128", "KDA, 6 heads of 128 (the triadic arms at E=1)", lin("kda"), tri128)
    arm("tri_gdn_e4_cute", "triadic GDN E=4, CuTe kernels (6 heads of 128)", lin("triadic"),
        tri128, {B + "linear_triadic_base": "gdn", B + "linear_triadic_e": 4,
                 B + "linear_triadic_kernel": "cute"})
    arm("tri_kda_e4", "triadic KDA E=4 (6 heads of 128)", lin("triadic"), tri128,
        {B + "linear_triadic_base": "kda", B + "linear_triadic_e": 4})
    arm("tri_ckda_e4", "triadic CKDA E=4 (6 heads of 128)", lin("triadic"), tri128,
        {B + "linear_triadic_base": "ckda", B + "linear_triadic_e": 4})
    # mbs 8 x 4096 runs out of memory (the expanded deg-2 state: 41.7 GiB allocated + 6 GiB
    # more); mbs 4 with 2 accumulation steps keeps the tokens per step.
    arm("power", "power retention deg 2 (Triton baseline), mbs 4", lin("power"),
        {B + "micro_batch_size": 4})
    # ---- token mixing / memory layers added to the dense stack
    arm("shortconv", "ShortConv before every attention", {"aux.pattern": "C*-" * 12})
    arm("canon", "Canon layers ABCD", {B + "canon_layers": "ABCD"})
    arm("engram", "Engram n-gram memory before blocks 2 and 8 (data-parallel tables)", {
        "aux.pattern": "*-N*-*-*-*-*-*-N*-*-*-*-*-", B + "engram_ngram_orders": [2, 3],
        B + "engram_table_parallel": "dp"})
    arm("pkm", "product-key memory in place of 2 MLPs", {
        "aux.pattern": "*-*-*-*P*-*-*-*-*P*-*-*-", B + "pkm_table_parallel": "dp"})
    # ---- residual stream / norms / embeddings
    mhc = {B + "mhc": True, B + "mhc_expansion_rate": 4}
    arm("mhc4", "mHC, 4 residual streams", mhc)
    arm("routed_k2", "routed residual, 2 of 4 streams", mhc, {B + "mhc_routed_topk": 2})
    arm("routed_k2_mix", "routed residual, 2 of 4, mixed, balanced", mhc, {
        B + "mhc_routed_topk": 2, B + "mhc_routed_mix": True,
        B + "mhc_routed_balance_rate": 0.001})
    arm("routed_k2_rev", "routed residual, 2 of 4, reversible", mhc,
        {B + "mhc_routed_topk": 2, B + "mhc_routed_reversible": True})
    arm("attn_res", "block attention residuals (4 blocks)",
        {B + "attn_res": True, B + "attn_res_num_blocks": 4})
    arm("postnorm", "sandwich norm", {B + "post_sublayer_norm": True})
    arm("ngpt", "nGPT", {B + "ngpt": True, B + "weight_normalizer": True})
    arm("ple128", "per-layer embeddings (128)", {B + "per_layer_embedding_dim": 128})
    arm("value_emb", "value embeddings", {B + "value_embedding": True})
    # ---- optimizers
    arm("muon", "Muon", {B + "optimizer": "muon"})
    arm("muown", "MuOwn", {B + "optimizer": "muown"})
    # ---- memory: optimizer-state offloading (compared alone first, combined later)
    arm("adam_precaware", "TE precision-aware Adam (fp32 states; offload's prerequisite)",
        {B + "use_precision_aware_optimizer": True})
    arm("adam_offload", "Adam with the optimizer state on the CPU, D2H/H2D overlapped",
        {B + "use_precision_aware_optimizer": True, B + "optimizer_cpu_offload": True,
         B + "optimizer_offload_fraction": 1.0, B + "overlap_cpu_optimizer_d2h_h2d": True})
    arm("muown_offload", "MuOwn with its momentum streamed from host memory",
        {B + "optimizer": "muown", B + "muown_cpu_offload": True})
    return arms


def moe():
    """50M active / 500M total: hidden 512, 12 blocks of 8 heads; 64 experts of 384, top 4,
    plus a shared expert of 768 (= 2 experts): 6 expert-widths active per token."""
    arms = []

    def arm(name, comment, *dicts, **kv):
        merged = {}
        for d in dicts:
            merged.update(d)
        merged.update(kv)
        arms.append((name, comment, merged))

    arm("moe", "reference: 64 experts top-4 + shared, aux loss, EP 4")
    arm("moe_nocg", "reference without CUDA graphs", {B + "cuda_graph_impl": "none"})
    arm("moe_ep1", "experts replicated (EP 1)", {B + "expert_model_parallel_size": 1})
    arm("dense50M", "dense reference with the MoE's active width (SwiGLU 2304)", {
        "aux.pattern": "*-" * 12, B + "num_experts": None, B + "moe_ffn_hidden_size": None,
        B + "moe_shared_expert_intermediate_size": None, B + "expert_model_parallel_size": 1,
        B + "ffn_hidden_size": 2304})
    arm("noshared", "no shared expert, top-6", {
        B + "moe_shared_expert_intermediate_size": None, B + "moe_router_topk": 6})
    arm("auxfree", "aux-loss-free balancing (sigmoid router + expert bias)", {
        B + "moe_router_load_balancing_type": "none", B + "moe_aux_loss_coeff": 0.0,
        B + "moe_router_enable_expert_bias": True, B + "moe_router_score_function": "sigmoid"})
    arm("latent", "LatentMoE (experts in a 256-dim latent)", {B + "moe_latent_size": 256})
    arm("zero16", "16 zero-computation experts", {B + "moe_zero_expert_num": 16})
    arm("muown", "MuOwn", {B + "optimizer": "muown"})
    arm("muown_offload", "MuOwn with its momentum streamed from host memory",
        {B + "optimizer": "muown", B + "muown_cpu_offload": True})
    arm("adam_offload", "Adam with the optimizer state on the CPU, D2H/H2D overlapped",
        {B + "use_precision_aware_optimizer": True, B + "optimizer_cpu_offload": True,
         B + "optimizer_offload_fraction": 1.0, B + "overlap_cpu_optimizer_d2h_h2d": True})
    arm("gdn3", "3:1 GDN hybrid on the MoE", {
        "aux.pattern": "LELELE*E" * 3, B + "experimental_attention_variant": "gated_delta_net"})
    return arms


HEADER = """# @package _global_
# GENERATED by scripts/korbi/hybrid_speed/gen_suites.py -- edit there.
# {doc}
defaults:
  - speed_base_mn5
  - _self_
{extra}
sweep:
  type: product
  store_sweep_json: true
  filter: "\\\\${{oc.eval:'len(\\\\${{aux.only}}) == 0 or \\"\\\\${{aux.arm}}\\" in \\\\${{aux.only}}'}}"
  groups:
    - type: list
      configs:
"""


def dump(fname, doc, extra, arms):
    lines = [HEADER.format(doc=doc, extra=extra)]
    for name, comment, kv in arms:
        lines.append(f"        # {comment}\n        - aux.arm: {name}\n")
        for k, v in kv.items():
            lines.append(f"          {k}: {yaml.safe_dump(v, default_flow_style=True).strip().removesuffix('...').strip()}\n")
    (OUT / fname).write_text("".join(lines))
    print(f"{fname}: {len(arms)} arms")


if __name__ == "__main__":
    dump("speed_130M_mn5.yaml",
         "Speed of every architecture feature on the 130M dense model (speed_base_mn5.yaml).",
         "\naux:\n  family: dense130M\n", dense130m())
    dump("speed_moe_mn5.yaml",
         "Speed of the MoE features on a 50M-active / 500M-total MoE (speed_base_mn5.yaml).",
         """
aux:
  family: moe50M500M
  pattern: "*E*E*E*E*E*E*E*E*E*E*E*E"

backend:
  megatron:
    hidden_size: 512
    num_attention_heads: 8
    num_query_groups: 8
    linear_num_key_heads: 8
    linear_num_value_heads: 8
    ffn_hidden_size: 2304
    num_experts: 64
    moe_ffn_hidden_size: 384
    moe_router_topk: 4
    moe_shared_expert_intermediate_size: 768
    moe_grouped_gemm: true
    moe_token_dispatcher_type: alltoall
    moe_router_load_balancing_type: aux_loss
    moe_aux_loss_coeff: 0.01
    moe_router_dtype: fp32
    moe_permute_fusion: true
    expert_model_parallel_size: 4
    cuda_graph_modules: [attn, moe_router, moe_preprocess]
""", moe())
