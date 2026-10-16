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
    noqknr = {B + "fused_qk_norm_rope": False}  # arms the fused q/k norm + RoPE cannot run
    # ---- references
    arm("dense", "reference: HybridModel, attention + SwiGLU MLP x 12")
    arm("dense_gpt", "the same model through GPTModel", {
        "backend.launcher_script": "./submodules/Megatron-LM/pretrain_gpt.py",
        B + "is_hybrid_model": False, B + "spec": None, B + "hybrid_layer_pattern": None,
        B + "num_layers": 12})
    arm("dense_nocg", "reference without CUDA graphs (what the graphs are worth)",
        {B + "cuda_graph_impl": "none"})
    # the torchtitan twin's data (titan_dense_mn5.yaml): FineWeb-10BT, gpt-neox tokens
    fineweb = {B + "mock_data": False, B + "tokenizer_type": "HuggingFaceTokenizer",
               B + "tokenizer_model": "/gpfs/projects/ehpc533/datasets/gpt-neox-20b-tokenizer",
               B + "vocab_size": None, B + "make_vocab_size_divisible_by": 128,
               B + "data_path": ["/gpfs/projects/ehpc533/datasets/FineWeb-gpt-neox/fineweb-10BT_text_document"],
               B + "split": "100,0,0"}
    arm("dense_fineweb", "reference on FineWeb-10BT (the torchtitan twin's data)", fineweb)
    arm("dense_fineweb_mbs16", "reference on FineWeb-10BT, mbs 16", fineweb,
        {B + "micro_batch_size": 16, B + "global_batch_size": 64})
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
    nomask = {B + "create_attention_mask_in_dataloader": False}
    arm("base_liger_nomask", "Liger CE, no dense attention mask from the dataloader", liger, nomask)
    arm("base_liger_nomask_mbs16_nocg", "Liger CE, no mask, mbs 16, no CUDA graphs", liger,
        nomask, {B + "micro_batch_size": 16, B + "global_batch_size": 64,
                 B + "cuda_graph_impl": "none"})
    arm("base_liger_noqkln", "Liger CE without q/k layernorm (architecture change: prices it)",
        liger, {B + "qk_layernorm": False}, noqknr)
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
    best = {**liger, **nomask, **qknr, B + "fused_residual_rmsnorm": True}
    arm("base_best", "best stack so far: Liger, no mask, fused q/k norm + RoPE, residual norm",
        best)
    arm("base_best_swiglu", "best stack + Triton SwiGLU", best, {B + "swiglu_triton": True})
    arm("base_best_mbs16_nocg", "best stack, mbs 16 (gbs 64), no CUDA graphs", best,
        {B + "micro_batch_size": 16, B + "global_batch_size": 64, B + "cuda_graph_impl": "none"})
    arm("base_best_hd128", "best stack with 6 heads of 128 (architecture change: attention "
        "kernels at head dim 128)", best, {B + "num_attention_heads": 6, B + "num_query_groups": 6,
                                           B + "kv_channels": 128})
    arm("base_fp8", "FP8 (hybrid, delayed scaling as the 32B runs)",
        {B + "fp8": "hybrid", B + "fp8_amax_compute_algo": "max",
         B + "fp8_amax_history_len": 1024})
    # ---- attention
    arm("nope", "no positional encoding", {B + "position_embedding_type": "none"}, noqknr)
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
           B + "apply_rope_fusion": False, B + "fused_qk_norm_rope": False}
    arm("mla", "multi-latent attention, NoPE", mla)
    arm("mla_gqa_xsa_gate", "MLA + 4 kv groups + XSA + output gate", mla, {
        B + "mla_num_query_groups": 4, B + "exclusive_self_attention": True,
        B + "attention_output_gate": True})
    # ---- linear / recurrent mixers, 3:1 with attention
    arm("mamba2", "Mamba2 (24 heads of 64, state 128)", {
        "aux.pattern": h3("M"), B + "mamba_num_heads": 24, B + "mamba_head_dim": 64,
        B + "mamba_num_groups": 8, B + "mamba_state_dim": 128})
    mamba = {"aux.pattern": h3("M"), B + "mamba_num_heads": 24, B + "mamba_head_dim": 64,
             B + "mamba_num_groups": 8, B + "mamba_state_dim": 128}
    for ck in (64, 256):  # SSD chunk size (default 128)
        arm(f"mamba2_ck{ck}", f"Mamba2, chunk {ck}", mamba, {B + "mamba_chunk_size": ck})
    arm("gdn_native", "Megatron's native Gated DeltaNet", {"aux.pattern": h3("G")})
    for v in ["gated_delta_net", "mlstm", "kda"]:
        arm(v, f"linear mixer {v}", lin(v))
    # complex_kda (signed_kda is the same FLA layer, ComplexKimiDeltaAttention, from the
    # linear_fla_* fields): signed gate, beta in (0, 2), full-rank output gate, no qkv silu
    arm("complex_kda", "linear mixer complex_kda", lin("complex_kda"), {
        B + "linear_beta_max": 2.0, B + "linear_output_gate": "linear",
        B + "linear_drop_qkv_silu": True, B + "linear_gate_init_style": "spread"})
    arm("kda_xsa", "KDA with exclusive self attention", lin("kda"), {B + "linear_xsa": True})
    for ck in (64, 256):  # chunkwise mLSTM (default 128)
        arm(f"mlstm_ck{ck}", f"mLSTM, chunk {ck}", lin("mlstm"), {B + "mlstm_chunk_size": ck})
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
    # At 4096 tokens power retention runs its quadratic (attention) form (faster than any
    # chunk size: linear_power_switch_over_seq_len); the chunked form (mbs 4: its expanded
    # deg-2 state runs out of memory at mbs 8) for comparison.
    arm("power", "power retention deg 2 (attention form at 4096)", lin("power"))
    arm("power_hd128", "power retention deg 2, 6 heads of 128 (attention form)", lin("power"),
        tri128)
    arm("power_chunked", "power retention deg 2, chunked (256), mbs 4", lin("power"),
        {B + "micro_batch_size": 4, B + "linear_power_switch_over_seq_len": 0})
    # ---- token mixing / memory layers added to the dense stack
    arm("shortconv", "ShortConv before every attention", {"aux.pattern": "C*-" * 12})
    arm("canon", "Canon layers ABCD (needs the unfused input norm)", {B + "canon_layers": "ABCD"},
        noqknr, {B + "fused_residual_rmsnorm": False})
    arm("engram", "Engram n-gram memory before blocks 2 and 8 (data-parallel tables)", {
        "aux.pattern": "*-N*-*-*-*-*-*-N*-*-*-*-*-", B + "engram_ngram_orders": [2, 3],
        B + "engram_table_parallel": "dp"})
    # mbs 8 runs out of memory (one 12 GiB value gather); mbs 4 x 2 accumulation steps
    arm("pkm", "product-key memory in place of 2 MLPs, mbs 4", {B + "micro_batch_size": 4}, {
        "aux.pattern": "*-*-*-*P*-*-*-*-*P*-*-*-", B + "pkm_table_parallel": "dp"})
    # ---- residual stream / norms / embeddings
    mhc = {B + "mhc": True, B + "mhc_expansion_rate": 4}
    # routed residual and attention residuals cannot be CUDA-graphed yet (open issue)
    nocg = {B + "cuda_graph_impl": "none"}
    arm("mhc4", "mHC, 4 residual streams", mhc)
    arm("routed_k2", "routed residual, 2 of 4 streams", mhc, nocg, {B + "mhc_routed_topk": 2})
    arm("routed_k2_mix", "routed residual, 2 of 4, mixed, balanced", mhc, nocg, {
        B + "mhc_routed_topk": 2, B + "mhc_routed_mix": True,
        B + "mhc_routed_balance_rate": 0.001})
    arm("routed_k2_rev", "routed residual, 2 of 4, reversible", mhc, nocg,
        {B + "mhc_routed_topk": 2, B + "mhc_routed_reversible": True})
    # the same with CUDA graphs: needs MEGATRON_ROUTED_CUDA_GRAPHS=1 in backend.env (run as
    # their own session, losses compared against the no-graph arms)
    cg = {B + "cuda_graph_impl": "local"}
    arm("routed_k2_cg", "routed residual, 2 of 4, CUDA graphs", mhc, cg, {B + "mhc_routed_topk": 2})
    arm("routed_k2_mix_cg", "routed residual, 2 of 4, mixed, balanced, CUDA graphs", mhc, cg, {
        B + "mhc_routed_topk": 2, B + "mhc_routed_mix": True,
        B + "mhc_routed_balance_rate": 0.001})
    arm("routed_k2_rev_cg", "routed residual, 2 of 4, reversible, CUDA graphs", mhc, cg,
        {B + "mhc_routed_topk": 2, B + "mhc_routed_reversible": True})
    arm("attn_res", "block attention residuals (4 blocks), no CUDA graphs", nocg,
        {B + "attn_res": True, B + "attn_res_num_blocks": 4})
    arm("postnorm", "sandwich norm", {B + "post_sublayer_norm": True})
    arm("ngpt", "nGPT", {B + "ngpt": True, B + "weight_normalizer": True}, noqknr)
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
    """120M active / 1.2B total (the dense 130M model's attention and width): hidden 768,
    12 blocks of 12 heads; 128 experts of 320, top 4, plus a shared expert of 640 (= 2
    experts): 6 expert-widths (1920) active per token."""
    arms = []

    def arm(name, comment, *dicts, **kv):
        merged = {}
        for d in dicts:
            merged.update(d)
        merged.update(kv)
        arms.append((name, comment, merged))

    arm("moe", "reference: 128 experts top-4 + shared, aux loss, EP 4")
    arm("moe_nocg", "reference without CUDA graphs", {B + "cuda_graph_impl": "none"})
    arm("moe_ep1", "experts replicated (EP 1)", {B + "expert_model_parallel_size": 1})
    arm("dense120M", "dense reference with the MoE's active width (SwiGLU 1920)", {
        "aux.pattern": "*-" * 12, B + "num_experts": None, B + "moe_ffn_hidden_size": None,
        B + "moe_shared_expert_intermediate_size": None, B + "expert_model_parallel_size": 1,
        B + "ffn_hidden_size": 1920, B + "cuda_graph_modules": []})
    E = "backend.env."
    cutlass = {E + "NVTE_USE_CUTLASS_GROUPED_GEMM": "1", "aux.array_tag": "_cutlass"}
    noshared = {B + "moe_shared_expert_intermediate_size": None, B + "moe_router_topk": 6}
    arm("noshared", "no shared expert, top-6", noshared)
    arm("noshared_cutlass", "no shared expert, top-6, CUTLASS grouped GEMM", noshared, cutlass)
    arm("auxfree", "aux-loss-free balancing (sigmoid router + expert bias)", {
        B + "moe_router_load_balancing_type": "none", B + "moe_aux_loss_coeff": 0.0,
        B + "moe_router_enable_expert_bias": True, B + "moe_router_score_function": "sigmoid"})
    arm("latent", "LatentMoE (experts in a 384-dim latent)", {B + "moe_latent_size": 384})
    # zero-computation experts cannot be CUDA-graphed yet (open issue)
    arm("zero16", "16 zero-computation experts, no CUDA graphs", {B + "moe_zero_expert_num": 16},
        {B + "cuda_graph_impl": "none"})
    arm("muown", "MuOwn", {B + "optimizer": "muown"})
    arm("muown_offload", "MuOwn with its momentum streamed from host memory",
        {B + "optimizer": "muown", B + "muown_cpu_offload": True})
    arm("adam_offload", "Adam with the optimizer state on the CPU, D2H/H2D overlapped",
        {B + "use_precision_aware_optimizer": True, B + "optimizer_cpu_offload": True,
         B + "optimizer_offload_fraction": 1.0, B + "overlap_cpu_optimizer_d2h_h2d": True})
    arm("gdn3", "3:1 GDN hybrid on the MoE", {
        "aux.pattern": "LELELE*E" * 3, B + "experimental_attention_variant": "gated_delta_net"})
    # The MoE levers of the large-scale MN5 runs (oellm-autoexp exp_irene: Qwen3-30B-A3B,
    # 10B-A0.9B), one at a time on our image, then on NeMo 26.08.01 (TE 2.18 + HybridEP)
    E = "backend.env."
    arm("router_fusion", "TE fused router", {B + "moe_router_fusion": True})
    # TE's CUTLASS grouped GEMM: 1.06x on the reference, but it segfaults (TE 2.10) with
    # top-6 / no shared expert and with LatentMoE -- opt-in only; *_nemo: TE 2.18
    cutlass = {E + "NVTE_USE_CUTLASS_GROUPED_GEMM": "1", "aux.array_tag": "_cutlass"}
    arm("cutlass_gg", "TE CUTLASS grouped GEMM", cutlass)
    # EP all-to-all overlap (HybridModel combined-1F1B schedule plan, ported from upstream
    # #4942/#4944): no CUDA graphs, >= 2 microbatches per step (mbs 4: 2 accumulation steps),
    # more than one device connection. moe_mbs4_nocg is its reference.
    ovl = {B + "overlap_moe_expert_parallel_comm": True, B + "cuda_graph_impl": "none",
           B + "micro_batch_size": 4, E + "CUDA_DEVICE_MAX_CONNECTIONS": "8",
           "aux.array_tag": "_conn8"}
    arm("moe_mbs4_nocg", "reference at mbs 4 without CUDA graphs (the overlap's reference)",
        {B + "cuda_graph_impl": "none", B + "micro_batch_size": 4})
    arm("a2a_overlap", "EP all-to-all overlapped with compute, mbs 4, no CUDA graphs", ovl)
    arm("a2a_overlap_dw", "EP all-to-all overlap + delayed wgrad", ovl,
        {B + "delay_wgrad_compute": True})
    arm("a2a_overlap_mbs8", "EP all-to-all overlap, mbs 8 x 2 (gbs 64)", ovl,
        {B + "micro_batch_size": 8, B + "global_batch_size": 64})
    arm("conn8", "CUDA_DEVICE_MAX_CONNECTIONS 8",
        {E + "CUDA_DEVICE_MAX_CONNECTIONS": "8", "aux.array_tag": "_conn8"})
    arm("gc10", "manual GC every 10 steps", {B + "manual_gc_interval": 10})
    a2a = {B + "overlap_moe_expert_parallel_comm": True, B + "cuda_graph_impl": "none",
           B + "micro_batch_size": 4, E + "CUDA_DEVICE_MAX_CONNECTIONS": "8",
           "aux.array_tag": "_nemo_conn8"}
    nemo = {"container.image": "/gpfs/projects/ehpc533/singularity_images/nemo_26.08.01.sif",
            E + "PATH": "/opt/venv/bin:/usr/local/bin:/usr/bin:/bin", "aux.array_tag": "_nemo"}
    arm("nemo", "reference on NeMo 26.08.01 (TE 2.18)", nemo)
    flex = {B + "moe_token_dispatcher_type": "flex", B + "moe_flex_dispatcher_backend": "hybridep",
            B + "moe_hybridep_num_sms": 16}
    arm("nemo_flex", "NeMo 26.08.01, flex dispatcher with HybridEP (16 SMs)", nemo, flex)
    arm("nemo_flex_nocg", "as nemo_flex, no CUDA graphs", nemo, flex,
        {B + "cuda_graph_impl": "none"})
    arm("nemo_flex_rf", "NeMo, HybridEP + router fusion", nemo, flex,
        {B + "moe_router_fusion": True})
    arm("nemo_flex_a2a", "NeMo, HybridEP + all-to-all overlap (+ delayed wgrad)", nemo, flex, a2a)
    arm("nemo_a2a", "NeMo, alltoall dispatcher + all-to-all overlap (+ delayed wgrad)", nemo, a2a)
    # Drop-and-pad (fixed expert capacity): no host sync in the dispatch, so the whole MoE
    # layer is CUDA-graphable (the reference graphs only attention / router / preprocess and
    # leaves the GPU idle ~half the step, v25prof). Tokens over capacity are dropped.
    for cf in (1.0, 1.25):
        tag = str(cf).replace(".", "p")
        dp = {B + "moe_expert_capacity_factor": cf, B + "moe_pad_expert_input_to_capacity": True}
        arm(f"dp{tag}", f"drop-and-pad, capacity {cf}, router graphs", dp)
        arm(f"dp{tag}_cgmoe", f"drop-and-pad, capacity {cf}, whole MoE layer CUDA-graphed", dp,
            {B + "cuda_graph_modules": ["attn", "moe"]})
        arm(f"dp{tag}_cgfull", f"drop-and-pad, capacity {cf}, full-iteration CUDA graph", dp,
            {B + "cuda_graph_impl": "full_iteration", B + "cuda_graph_modules": []})
    arm("dp1p25_cgmoe_best", "drop-and-pad 1.25, MoE graphed, router fusion, CUTLASS GG",
        {B + "moe_expert_capacity_factor": 1.25, B + "moe_pad_expert_input_to_capacity": True,
         B + "cuda_graph_modules": ["attn", "moe"], B + "moe_router_fusion": True}, cutlass)
    nemo_cutlass = {**nemo, E + "NVTE_USE_CUTLASS_GROUPED_GEMM": "1", "aux.array_tag": "_nemo_cutlass"}
    arm("nemo_cutlass", "NeMo (TE 2.18), CUTLASS grouped GEMM", nemo_cutlass)
    arm("nemo_cutlass_noshared", "NeMo (TE 2.18), CUTLASS grouped GEMM, no shared expert, top-6",
        nemo_cutlass, noshared)
    arm("nemo_cutlass_latent", "NeMo (TE 2.18), CUTLASS grouped GEMM, LatentMoE", nemo_cutlass,
        {B + "moe_latent_size": 384})
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
         "Speed of the MoE features on a 120M-active / 1.2B-total MoE (speed_base_mn5.yaml).",
         """
aux:
  family: moe120M1200M
  pattern: "*E*E*E*E*E*E*E*E*E*E*E*E"

backend:
  megatron:
    ffn_hidden_size: 1920
    num_experts: 128
    moe_ffn_hidden_size: 320
    moe_router_topk: 4
    moe_shared_expert_intermediate_size: 640
    moe_grouped_gemm: true
    moe_token_dispatcher_type: alltoall
    moe_router_load_balancing_type: aux_loss
    moe_aux_loss_coeff: 0.01
    moe_router_dtype: fp32
    moe_permute_fusion: true
    expert_model_parallel_size: 4
    cuda_graph_modules: [attn, moe_router, moe_preprocess]
""", moe())
