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
    arm("dense_fineweb_mbs32", "reference on FineWeb-10BT, mbs 32 (titan's best: graph mbs 32)",
        fineweb, {B + "micro_batch_size": 32, B + "global_batch_size": 128})
    arm("dense_fineweb_mbs32_nocg", "reference on FineWeb-10BT, mbs 32, no CUDA graphs", fineweb,
        {B + "micro_batch_size": 32, B + "global_batch_size": 128, B + "cuda_graph_impl": "none"})
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


def a3b():
    """3B-active hybrid MoE at scale (docs/hybrid_moe_3b_campaign.md 3.2): 48 mixer layers
    (3:1 linear : attention, each followed by a MoE layer), hidden 2048, attention 16 x 128 with
    2 kv groups. The reference is the CKDA hybrid; each arm changes one thing. The shape (experts,
    nodes, EP) comes from the family header (a3b30 / a3b80)."""
    arms = []

    def arm(name, comment, *dicts, **kv):
        merged = {}
        for d in dicts:
            merged.update(d)
        merged.update(kv)
        arms.append((name, comment, merged))

    lin = lambda v: {B + "experimental_attention_variant": v}  # noqa: E731
    arm("ckda", "reference: 3:1 CKDA : attention hybrid MoE")
    arm("kda", "KDA instead of CKDA", lin("kda"), {
        B + "linear_beta_max": 1.0, B + "linear_output_gate": "lowrank",
        B + "linear_drop_qkv_silu": False, B + "linear_gate_init_style": "shipped"})
    arm("gdn", "Gated DeltaNet (FLA) instead of CKDA", lin("gated_delta_net"), {
        B + "linear_beta_max": 1.0, B + "linear_output_gate": "lowrank",
        B + "linear_drop_qkv_silu": False, B + "linear_gate_init_style": "shipped"})
    arm("attn", "all-attention MoE (no linear mixer)",
        {"aux.pattern": "|".join(["*E" * 12] * 4)})
    arm("ckda_xsa", "exclusive self attention in the attention layers",
        {B + "exclusive_self_attention": True})
    arm("ckda_gate", "sigmoid output gate in the attention layers (Qwen3-Next style)",
        {B + "attention_output_gate": True})
    arm("ckda_alltoall", "alltoall dispatcher instead of flex / HybridEP",
        {B + "moe_token_dispatcher_type": "alltoall"})
    arm("ckda_ovl", "EP all-to-all overlap with the combined 1F1B schedule (8 connections)",
        {B + "overlap_moe_expert_parallel_comm": True,
         "backend.env.CUDA_DEVICE_MAX_CONNECTIONS": "8", "aux.array_tag": "_conn8"})
    arm("ckda_mbs2", "micro batch 2", {B + "micro_batch_size": 2})
    arm("ckda_mbs8", "micro batch 8", {B + "micro_batch_size": 8})
    arm("ckda_sel_recompute", "selective recompute (MoE activations + norms) instead of full",
        {B + "recompute_granularity": "selective", B + "recompute_method": None,
         B + "recompute_num_layers": None, B + "recompute_modules": ["moe_act", "layernorm"]})
    arm("ckda_ep_half", "half the expert parallelism", {
        B + "expert_model_parallel_size": "${oc.eval:'${aux.ep} // 2'}"})
    arm("ckda_ep_double", "twice the expert parallelism", {
        B + "expert_model_parallel_size": "${oc.eval:'2 * ${aux.ep}'}"})
    arm("ckda_pp2", "pipeline parallel 2 (two stages)", {
        B + "pipeline_model_parallel_size": 2,
        "aux.pattern": "LELELE*E" * 6 + "|" + "LELELE*E" * 6})
    # exp_irene's balanced layout: VPP2 (8 segments on PP4), 7/7/7/6/7/7/6/1 blocks, the last
    # segment one block + the 262k-vocabulary head and loss (~9 blocks of compute by itself)
    vpp = {"aux.pattern": vpp_pattern("LLL*" * 12, (7, 7, 7, 6, 7, 7, 6, 1))}
    vpp_attn = {"aux.pattern": vpp_pattern("*" * 48, (7, 7, 7, 6, 7, 7, 6, 1))}
    tuned = {B + "moe_router_fusion": True, B + "overlap_p2p_comm": True,
             "backend.env.CUDA_DEVICE_MAX_CONNECTIONS": "8", "aux.array_tag": "_conn8"}
    flb = {B + "moe_router_force_load_balancing": True}
    arm("ckda_vpp", "balanced VPP2 pipeline layout (head + loss on their own segment)", vpp)
    arm("ckda_tuned", "VPP2 layout + router fusion, P2P overlap, 8 CUDA connections", vpp, tuned)
    arm("ckda_tuned_flb", "tuned + forced load balancing (benchmark only, as exp_irene's 185.8)",
        vpp, tuned, flb)
    arm("attn_tuned", "all-attention, VPP2 layout + router fusion, P2P overlap, 8 connections "
        "(the torchtitan twin's reference)", vpp_attn, tuned)
    arm("attn_tuned_flb", "all-attention, tuned + forced load balancing (exp_irene's setting)",
        vpp_attn, tuned, flb)
    # PP8 (A3B-80B's memory needs it at EP4): VPP2 = 16 segments, the head + loss segment one block
    pp8_blocks = (4,) + (3,) * 13 + (4, 1)
    arm("ckda_pp8_vpp_tuned", "PP8 x VPP2 (16 segments, head + loss on its own), tuned flags",
        {"aux.pattern": vpp_pattern("LLL*" * 12, pp8_blocks),
         B + "pipeline_model_parallel_size": 8}, tuned)
    arm("ckda_pp8_vpp_tuned_mbs2", "PP8 x VPP2 tuned, micro batch 2 (A3B-80B: OOM at mbs 4)",
        {"aux.pattern": vpp_pattern("LLL*" * 12, pp8_blocks),
         B + "pipeline_model_parallel_size": 8, B + "micro_batch_size": 2}, tuned)
    arm("ckda_tuned_gbs4096", "tuned, global batch 4096 (exp_irene's; 4x fewer bubbles)", vpp, tuned,
        {"aux.gbs": 4096, B + "train_iters": 10})
    # the rounds-3 / 3B recipe (docs/hybrid_moe_3b_campaign.md §10.5): CKDA 5:1 with the reference
    # CKDA (standard inits, low-rank output gate) and no XSA in the CKDA layers; attention NoPE +
    # qk-norm + XSA + output gate; MoE router sigmoid + expert bias + aux 1e-4 with a shared expert
    # (top-k - 1 routed + one shared of the expert width: same active FFN); + mHC-4 (its cost at
    # scale); the all-attention twin with the same MoE recipe
    ckda_std = {B + "linear_beta_max": 2.0, B + "linear_gate_activation": "signed_sigmoid2",
                B + "linear_gate_init_style": "shipped", B + "linear_output_gate": "lowrank",
                B + "linear_drop_qkv_silu": True}
    attn_r3 = {B + "position_embedding_type": "none", B + "attention_output_gate": True,
               B + "exclusive_self_attention": True}
    router_r3 = {B + "moe_router_score_function": "sigmoid", B + "moe_router_enable_expert_bias": True,
                 B + "moe_router_bias_update_rate": 1e-3, B + "moe_aux_loss_coeff": 1e-4}
    shared = {B + "moe_router_topk": "${oc.eval:'${aux.topk} - 1'}",
              B + "moe_shared_expert_intermediate_size": "${aux.expert_ffn}"}
    vpp5 = {"aux.pattern": vpp_pattern("LLLLL*" * 8, (7, 7, 7, 6, 7, 7, 6, 1))}
    mhc4 = {B + "mhc": True, B + "mhc_expansion_rate": 4}
    arm("r3_ckda5", "round-3/3B recipe: CKDA 5:1 + attention NoPE/XSA/gate + MoE sigmoid/bias/shared, "
        "tuned VPP2", vpp5, tuned, ckda_std, attn_r3, router_r3, shared)
    arm("r3_ckda5_mhc4", "r3_ckda5 + mHC, 4 residual streams", vpp5, tuned, ckda_std, attn_r3, router_r3,
        shared, mhc4)
    arm("r3_ckda5_mhc4_mbs2", "r3_ckda5 + mHC-4, micro batch 2 (4x residual activations)", vpp5, tuned,
        ckda_std, attn_r3, router_r3, shared, mhc4, {B + "micro_batch_size": 2})
    routed8 = {B + "mhc": True, B + "mhc_expansion_rate": 8, B + "mhc_routed_topk": 2}
    arm("r3_ckda5_routed8", "r3_ckda5 + routed residual, 8 streams, 2 active", vpp5, tuned, ckda_std,
        attn_r3, router_r3, shared, routed8)
    # full recompute with mHC / routed (Megatron: hybrid_block full-recompute path for the
    # widened residual, 2026-10-18): each checkpoint saves n streams, so also micro batch 2,
    # against the recipe at micro batch 2
    arm("r3_ckda5_mbs2", "r3_ckda5, micro batch 2", vpp5, tuned, ckda_std, attn_r3, router_r3, shared,
        {B + "micro_batch_size": 2})
    arm("r3_ckda5_routed8_mbs2", "r3_ckda5 + routed residual (8 streams, 2 active), micro batch 2", vpp5,
        tuned, ckda_std, attn_r3, router_r3, shared, routed8, {B + "micro_batch_size": 2})
    # the saved checkpoint inputs are n streams wide: checkpoint every r layers (r = 2: one
    # mixer + MoE block) to divide them by r; the baseline at the same r for the comparison
    for r, mbs in ((2, 4), (2, 2), (4, 2), (4, 4)):
        rc = {B + "recompute_num_layers": r, B + "micro_batch_size": mbs}
        arm(f"r3_ckda5_mhc4_rc{r}_mbs{mbs}", f"r3_ckda5 + mHC-4, full recompute every {r} layers, "
            f"micro batch {mbs}", vpp5, tuned, ckda_std, attn_r3, router_r3, shared, mhc4, rc)
        arm(f"r3_ckda5_rc{r}_mbs{mbs}", f"r3_ckda5, full recompute every {r} layers, micro batch {mbs}",
            vpp5, tuned, ckda_std, attn_r3, router_r3, shared, rc)
    arm("r3_ckda5_routed8_rc4_mbs2", "r3_ckda5 + routed (8 of 2), full recompute every 4 layers, mbs 2",
        vpp5, tuned, ckda_std, attn_r3, router_r3, shared, routed8,
        {B + "recompute_num_layers": 4, B + "micro_batch_size": 2})
    # routed residual with sublayer recompute (Megatron, 2026-10-18: under full recompute each
    # routed sublayer checkpoints its body between read and write: nothing n streams wide is
    # saved per layer) and the fused router; reversible keeps one stream state per stage
    rev = {B + "mhc_routed_reversible": True}
    arm("r3_ckda5_routed8_rev", "r3_ckda5 + routed (8 of 2), reversible, sublayer recompute, mbs 4",
        vpp5, tuned, ckda_std, attn_r3, router_r3, shared, routed8, rev)
    arm("r3_ckda5_routed8_rev_mbs2", "r3_ckda5 + routed (8 of 2), reversible, sublayer recompute, "
        "mbs 2", vpp5, tuned, ckda_std, attn_r3, router_r3, shared, routed8, rev,
        {B + "micro_batch_size": 2})
    arm("r3_ckda5_routed8_sub", "r3_ckda5 + routed (8 of 2), not reversible, sublayer recompute, "
        "mbs 4", vpp5, tuned, ckda_std, attn_r3, router_r3, shared, routed8)
    # diagnostics for the hang at mbs 2 (a PP send of the 8-stream payload timed out): without
    # interleaving (PP4, 4 segments), and non-reversible with interleaving
    pp4_plain = {"aux.pattern": vpp_pattern("LLLLL*" * 8, (12, 12, 12, 12))}
    arm("r3_ckda5_routed8_rev_mbs2_novpp", "routed (8 of 2), reversible, mbs 2, PP4 without VPP",
        pp4_plain, {B + "moe_router_fusion": True, "backend.env.CUDA_DEVICE_MAX_CONNECTIONS": "8",
                    "aux.array_tag": "_conn8"}, ckda_std, attn_r3, router_r3, shared, routed8, rev,
        {B + "micro_batch_size": 2})
    arm("r3_ckda5_routed8_sub_mbs2", "routed (8 of 2), not reversible, mbs 2, VPP2", vpp5, tuned,
        ckda_std, attn_r3, router_r3, shared, routed8, {B + "micro_batch_size": 2})
    # the VPP failure is NCCL "unhandled cuda error" in the overlapped grad reduce-scatter (OOM
    # inside NCCL, then the other stages hang on P2P): VPP keeps more microbatches in flight,
    # each with its 8-stream state. mbs 1 tests the memory hypothesis
    arm("r3_ckda5_routed8_rev_mbs1", "routed (8 of 2), reversible, mbs 1, VPP2", vpp5, tuned, ckda_std,
        attn_r3, router_r3, shared, routed8, rev, {B + "micro_batch_size": 1})
    # LatentMoE (moe_latent_size): routed experts in a latent of hidden / 2 = 1024; equal cost
    # (2x experts, 2x routed top-k: same active expert compute, total and dispatch volume) and
    # half cost (same experts and top-k); the shared expert stays at full width
    lat = {B + "moe_latent_size": 1024}
    arm("r3_ckda5_lat2", "r3_ckda5 + LatentMoE 1024, 2x experts and routed top-k (equal cost)", vpp5,
        tuned, ckda_std, attn_r3, router_r3, shared, lat,
        {B + "num_experts": "${oc.eval:'2 * ${backend.megatron.num_experts}'}",
         B + "moe_router_topk": "${oc.eval:'2 * (${aux.topk} - 1)'}"})
    arm("r3_ckda5_lat2_half", "r3_ckda5 + LatentMoE 1024, same experts / top-k (half cost)", vpp5,
        tuned, ckda_std, attn_r3, router_r3, shared, lat)
    arm("r3_attn", "all-attention twin of r3_ckda5 (RoPE + qk-norm + XSA, same MoE recipe)", vpp_attn,
        tuned, {B + "exclusive_self_attention": True}, router_r3, shared)
    # mHC / routed residual / AttnRes do not support full activation recompute (hybrid_block.py
    # raises): selective recompute (MoE activations + norms), which OOMs the plain hybrid at mbs 4,
    # so mbs 2 and 1, each against the recipe under the same recompute / mbs (the mHC cost alone)
    sel = {B + "recompute_granularity": "selective", B + "recompute_method": None,
           B + "recompute_num_layers": None, B + "recompute_modules": ["moe_act", "layernorm"]}
    for mbs in (2, 1):
        m = {B + "micro_batch_size": mbs}
        arm(f"r3_ckda5_sel_mbs{mbs}", f"r3_ckda5, selective recompute, micro batch {mbs}", vpp5, tuned,
            ckda_std, attn_r3, router_r3, shared, sel, m)
        arm(f"r3_ckda5_mhc4_sel_mbs{mbs}", f"r3_ckda5 + mHC-4, selective recompute, micro batch {mbs}",
            vpp5, tuned, ckda_std, attn_r3, router_r3, shared, mhc4, sel, m)
        arm(f"r3_ckda5_routed8_sel_mbs{mbs}", f"r3_ckda5 + routed (8 streams, 2 active), selective "
            f"recompute, micro batch {mbs}", vpp5, tuned, ckda_std, attn_r3, router_r3, shared, routed8,
            sel, m)
    # the same recipe arms on the PP8 x VPP2 layout at micro batch 2 (A3B-80B: its memory needs
    # PP8, and VPP2 on PP4 OOMs at mbs 4)
    pp8 = {B + "pipeline_model_parallel_size": 8, B + "micro_batch_size": 2}
    arm("r3_ckda5_pp8", "r3_ckda5 on PP8 x VPP2, micro batch 2",
        {"aux.pattern": vpp_pattern("LLLLL*" * 8, pp8_blocks)}, pp8, tuned, ckda_std, attn_r3, router_r3,
        shared)
    arm("r3_ckda5_mhc4_pp8", "r3_ckda5 + mHC-4 on PP8 x VPP2, micro batch 2",
        {"aux.pattern": vpp_pattern("LLLLL*" * 8, pp8_blocks)}, pp8, tuned, ckda_std, attn_r3, router_r3,
        shared, mhc4)
    arm("r3_ckda5_routed8_pp8", "r3_ckda5 + routed residual (8 streams, 2 active) on PP8 x VPP2, mbs 2",
        {"aux.pattern": vpp_pattern("LLLLL*" * 8, pp8_blocks)}, pp8, tuned, ckda_std, attn_r3, router_r3,
        shared, routed8)
    arm("r3_attn_pp8", "all-attention twin on PP8 x VPP2, micro batch 2",
        {"aux.pattern": vpp_pattern("*" * 48, pp8_blocks)}, pp8, tuned,
        {B + "exclusive_self_attention": True}, router_r3, shared)
    return arms


def round4b():
    """Round 4B (docs/hybrid_moe_3b_campaign.md §11.1): speed and memory feasibility of the
    best-bet model at production layouts (exp_irene's MN5 setup: TP1, EP4 HybridEP, full
    recompute), bf16 vs fp8 (the 32B-dense production recipe: hybrid, tensorwise, first / last
    layer in bf16). Recipe: CKDA 5:1 (reference CKDA, no linear XSA), attention NoPE + qk-norm +
    XSA + gate, routed residual 2 of 8 (reversible, sublayer recompute, fused router), MoE with a
    shared expert of the expert width and the sigmoid + expert-bias router. No VPP: routed + VPP
    hangs at A3B (§7.2, being debugged), so PP stages are whole 5:1 periods."""
    arms = []

    def arm(name, comment, *dicts):
        merged = {}
        for d in dicts:
            merged.update(d)
        arms.append((name, comment, merged))

    recipe = {B + "linear_beta_max": 2.0, B + "linear_gate_activation": "signed_sigmoid2",
              B + "linear_gate_init_style": "shipped", B + "linear_output_gate": "lowrank",
              B + "linear_drop_qkv_silu": True,
              B + "position_embedding_type": "none", B + "attention_output_gate": True,
              B + "exclusive_self_attention": True,
              B + "moe_router_score_function": "sigmoid", B + "moe_router_enable_expert_bias": True,
              B + "moe_router_bias_update_rate": 1e-3, B + "moe_aux_loss_coeff": 1e-4,
              B + "mhc": True, B + "mhc_expansion_rate": 8, B + "mhc_routed_topk": 2,
              B + "mhc_routed_reversible": True,
              B + "moe_router_fusion": True, "backend.env.CUDA_DEVICE_MAX_CONNECTIONS": "8",
              "aux.array_tag": "_conn8", B + "micro_batch_size": 2}
    fp8 = {B + "fp8": "hybrid", B + "fp8_recipe": "tensorwise", B + "first_last_layers_bf16": True,
           B + "num_layers_at_start_in_bf16": 1, B + "num_layers_at_end_in_bf16": 1,
           B + "fp8_param_gather": False}
    families = {
        # name: (experts, width, routed top-k, PP, nodes, comment)
        "m48a3": (336, 768, 7, 8, 16, "48x at 3B active: 336 x 768, top-7 + shared (~79B total)"),
        "m32a3": (224, 768, 7, 8, 16, "32x at 3B active: 224 x 768, top-7 + shared (~55B total)"),
        "m48a2": (192, 512, 4, 4, 8, "48x at ~2.9B active: 192 x 512, top-4 + shared (~31B total)"),
    }
    for fam, (e, w, k, pp, nodes, comment) in families.items():
        shape = {B + "num_experts": e, B + "moe_ffn_hidden_size": w, B + "moe_router_topk": k,
                 B + "moe_shared_expert_intermediate_size": w, B + "pipeline_model_parallel_size": pp,
                 "aux.nodes": nodes, "aux.ep": 4, "aux.topk": k, "aux.expert_ffn": w,
                 "aux.pattern": vpp_pattern("LLLLL*" * 8, (48 // pp,) * pp),
                 # one array per family: an array has one sbatch header (node count)
                 "aux.array_tag": f"_conn8_{fam}"}
        arm(f"{fam}_bf16", comment + ", bf16", recipe, shape)
        arm(f"{fam}_fp8", comment + ", fp8 (hybrid, tensorwise, first/last layer bf16)", recipe, shape,
            fp8)
    return arms


def prod_width():
    """Expert width for the production shape (docs/inference_model_size.md §0/§5/§8: ~39B total /
    ~2.74B active, hidden 2048 x 48 blocks, CKDA 5:1): 160 x 768, top-5 + shared 768 against
    240 x 512, top-8 + shared 512 -- equal total and active parameters (active FFN 6 x 768 =
    9 x 512 = 4608), granularity 2 h / w 5.3 vs 8. Production-like: 8k sequence with document
    masking (dataloader_inter_document_masking, as oellm_32b_dense), routed residual 2 of 8
    (reversible, sublayer recompute), full recompute, bf16, HybridEP EP4, PP4 on 16 nodes (DP16,
    expert DP 4; gbs 512 = 32 micro-batches per pipeline at mbs 1). 4 nodes OOM: expert DP 1 holds
    the full fp32 optimizer state of 2.3B expert parameters per GPU (~50 GB before activations).
    The 512 shape moves 1.6x the dispatch volume (top-8 vs top-5) through smaller grouped GEMMs
    (2048 x 512)."""
    arms = []

    def arm(name, comment, *dicts):
        merged = {}
        for d in dicts:
            merged.update(d)
        arms.append((name, comment, merged))

    recipe = {B + "linear_beta_max": 2.0, B + "linear_gate_activation": "signed_sigmoid2",
              B + "linear_gate_init_style": "shipped", B + "linear_output_gate": "lowrank",
              B + "linear_drop_qkv_silu": True,
              B + "position_embedding_type": "none", B + "attention_output_gate": True,
              B + "exclusive_self_attention": True,
              B + "moe_router_score_function": "sigmoid", B + "moe_router_enable_expert_bias": True,
              B + "moe_router_bias_update_rate": 1e-3, B + "moe_aux_loss_coeff": 1e-4,
              B + "mhc": True, B + "mhc_expansion_rate": 8, B + "mhc_routed_topk": 2,
              B + "mhc_routed_reversible": True, B + "moe_router_fusion": True,
              B + "seq_length": 8192, B + "max_position_embeddings": 8192,
              B + "dataloader_inter_document_masking": True,
              # 2048 x 262k bf16 logits per chunk (8192: 4 GiB, OOM on the last stage at 8k)
              B + "liger_fused_linear_cross_entropy_chunk_size": 2048,
              "backend.env.CUDA_DEVICE_MAX_CONNECTIONS": "8"}
    shapes = {"w768": (160, 768, 5), "w512": (240, 512, 8)}
    # the masking control: same shape and layout, plain causal 8k
    arm("w512_mbs1_nomask", "240 x 512, top-8 + shared 512, mbs 1, 8k without document masking",
        recipe, {B + "num_experts": 240, B + "moe_ffn_hidden_size": 512, B + "moe_router_topk": 8,
                 B + "moe_shared_expert_intermediate_size": 512, "aux.topk": 8,
                 "aux.expert_ffn": 512, B + "micro_batch_size": 1,
                 B + "dataloader_inter_document_masking": False})
    for mbs in (1, 2):  # run mbs 1 first (aux.only), mbs 2 if its memory allows
        for name, (e, w, k) in shapes.items():
            shape = {B + "num_experts": e, B + "moe_ffn_hidden_size": w, B + "moe_router_topk": k,
                     B + "moe_shared_expert_intermediate_size": w, "aux.topk": k,
                     "aux.expert_ffn": w, B + "micro_batch_size": mbs}
            arm(f"{name}_mbs{mbs}", f"{e} x {w}, top-{k} + shared {w}, mbs {mbs}, 8k docmask",
                recipe, shape)
    return arms


def a3b_slice():
    """One-node "A3B slice" (docs/hybrid_moe_3b_campaign.md §11.2): one 5:1 period (6 blocks)
    at m48a2's per-layer shapes (hidden 2048, 192 experts x 512, top-4 + shared, EP4, mbs 2; m48a3's
    336 x 768 is 2.4B expert parameters per GPU over 6 layers: OOM without recompute), the
    best-bet recipe with routed 2 of 8. Small enough for no recompute, so per-layer CUDA graphs
    are allowed. Rank 0 profiles steps 15-17 (torch profiler: CPU + CUDA) -> how host-bound the
    A3B layers are, in bf16 and fp8, and what CUDA graphs recover."""
    arms = []

    def arm(name, comment, *dicts):
        merged = {}
        for d in dicts:
            merged.update(d)
        arms.append((name, comment, merged))

    base = {B + "linear_beta_max": 2.0, B + "linear_gate_activation": "signed_sigmoid2",
            B + "linear_gate_init_style": "shipped", B + "linear_output_gate": "lowrank",
            B + "linear_drop_qkv_silu": True, B + "position_embedding_type": "none",
            B + "attention_output_gate": True, B + "exclusive_self_attention": True,
            B + "moe_router_score_function": "sigmoid", B + "moe_router_enable_expert_bias": True,
            B + "moe_router_bias_update_rate": 1e-3, B + "moe_aux_loss_coeff": 1e-4,
            B + "mhc": True, B + "mhc_expansion_rate": 8, B + "mhc_routed_topk": 2,
            B + "moe_router_fusion": True, B + "micro_batch_size": 2,
            B + "pipeline_model_parallel_size": 1, "aux.pattern": "LELELELELE*E",
            B + "recompute_granularity": None, B + "recompute_method": None,
            B + "recompute_num_layers": None, B + "train_iters": 20,
            B + "profile": True, B + "use_pytorch_profiler": True, B + "profile_ranks": [0],
            B + "profile_step_start": 15, B + "profile_step_end": 17,
            # PP1 without recompute trips the overlapped param all-gather's ordering assert
            # (param_gather_handle is None) at step 2; the A3B layouts (PP > 1, full recompute)
            # do not. Off for this host-overhead measurement.
            B + "overlap_param_gather": False}
    # attn only: the MoE partial-graph path (moe_router / moe_preprocess) does not thread the
    # mHC / routed stream read through the MLP yet (4-D streams reach the shared expert)
    local = {B + "cuda_graph_impl": "local", B + "cuda_graph_modules": ["attn"]}
    # routed residual + full-iteration graphs is not verified: opt in to see whether capture works
    full = {B + "cuda_graph_impl": "full_iteration", B + "cuda_graph_modules": [],
            "backend.env.MEGATRON_ROUTED_CUDA_GRAPHS": "1", "aux.array_tag": "_routedgraphs"}
    fp8 = {B + "fp8": "hybrid", B + "fp8_recipe": "tensorwise", B + "fp8_param_gather": False,
           B + "moe_router_padding_for_quantization": True}
    arm("bf16", "bf16, no CUDA graphs", base)
    arm("bf16_graph_local", "bf16, per-layer graphs on the mixer layers (attn scope)", base, local)
    arm("bf16_graph_full", "bf16, full-iteration graph", base, full)
    arm("fp8", "fp8 tensorwise + router padding, no graphs", base, fp8)
    arm("fp8_graph_local", "fp8 tensorwise + router padding, per-layer graphs", base, fp8, local)
    arm("fp8_delayed_graph_local", "fp8 delayed + router padding, per-layer graphs", base, fp8, local,
        {B + "fp8_recipe": "delayed"})
    # CKDA's fla nn.Linear projections as TE linears: fused wgrad accumulation, fp8-eligible
    te = {B + "linear_te_projections": True}
    arm("bf16_te", "bf16, TE projections in the CKDA mixers", base, te)
    arm("bf16_te_graph_local", "bf16, TE projections, per-layer graphs", base, te, local)
    arm("fp8_te", "fp8 tensorwise, TE projections (all mixer GEMMs fp8)", base, fp8, te)
    arm("fp8_delayed_te_graph_local", "fp8 delayed, TE projections, per-layer graphs", base, fp8,
        te, local, {B + "fp8_recipe": "delayed"})
    return arms


def vpp_pattern(mixers: str, blocks: tuple[int, ...]) -> str:
    """Pipe-separated hybrid pattern, each mixer followed by a MoE layer, ``blocks`` mixers per
    segment."""
    assert sum(blocks) == len(mixers)
    out, i = [], 0
    for n in blocks:
        out.append("".join(m + "E" for m in mixers[i:i + n]))
        i += n
    return "|".join(out)


A3B_COMMON = """
backend:
  megatron:
    hidden_size: 2048
    num_attention_heads: 16
    num_query_groups: 2
    kv_channels: 128
    ffn_hidden_size: 6144
    linear_key_head_dim: 128
    linear_value_head_dim: 128
    linear_num_key_heads: 16
    linear_num_value_heads: 16
    experimental_attention_variant: complex_kda
    linear_beta_max: 2.0
    linear_output_gate: linear
    linear_drop_qkv_silu: true
    linear_gate_init_style: spread
    untie_embeddings_and_output_weights: true
    vocab_size: 262143
    moe_grouped_gemm: true
    # exp_irene's best MN5 layout for Qwen3-30B-A3B / Qwen3.5-35B-A3B (185.8 TFLOP/s/GPU, 8 nodes):
    # TP1 / PP4 / EP4, flex dispatcher with HybridEP, full recompute, mbs 4
    moe_token_dispatcher_type: flex
    moe_flex_dispatcher_backend: hybridep
    moe_router_load_balancing_type: aux_loss
    moe_aux_loss_coeff: 0.001
    moe_router_dtype: fp32
    moe_permute_fusion: true
    expert_model_parallel_size: ${aux.ep}
    tensor_model_parallel_size: 1
    pipeline_model_parallel_size: 4
    recompute_granularity: full
    recompute_method: uniform
    recompute_num_layers: 1
    micro_batch_size: 4
    global_batch_size: ${aux.gbs}
    cuda_graph_impl: none
    cuda_graph_modules: []
    train_iters: 25
    log_interval: 1
{shape}job:
  est_step_time_s: 20.0
  est_startup_min: 10.0
slurm:
  sbatch:
    nodes: ${aux.nodes}
    time: "00:40:00"
"""


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
    pattern48 = "|".join(["LELELE*E" * 3] * 4)  # PP4: 12 mixer layers per stage
    dump("speed_a3b30_mn5.yaml",
         "Speed of the 3B-active / ~30B-total hybrid MoE (128 experts x 768, top-8) on 8 nodes, TP1 / PP4 / EP4.",
         f"""
aux:
  family: a3b30
  pattern: "{pattern48}"
  nodes: 8
  ep: 4
  gbs: 1024
  topk: 8
  expert_ffn: 768
""" + A3B_COMMON.replace("{shape}", """    num_experts: 128
    moe_ffn_hidden_size: 768
    moe_router_topk: 8
"""), a3b())
    dump("speed_prodw_mn5.yaml",
         "Production shape (~39B / 2.74B active): expert width 768 vs 512 at equal total / active "
         "size, 8k sequence with document masking (docs/inference_model_size.md).",
         f"""
aux:
  family: prodw
  pattern: "{vpp_pattern("LLLLL*" * 8, (12,) * 4)}"
  # 16 nodes (user 2026-10-18: production runs many more, with gradient accumulation): DP16,
  # gbs 512 = 32 micro-batches per pipeline at mbs 1 (8 nodes / gbs 128: 16)
  nodes: 16
  ep: 4
  gbs: 512
  topk: 5
  expert_ffn: 768
""" + A3B_COMMON.replace("{shape}", """    num_experts: 160
    moe_ffn_hidden_size: 768
    moe_router_topk: 5
    moe_shared_expert_intermediate_size: 768
"""), prod_width())
    dump("speed_a3bslice_mn5.yaml",
         "One-node A3B slice: host-boundness and CUDA graphs at A3B layer shapes, bf16 vs fp8 "
         "(docs/hybrid_moe_3b_campaign.md §11.2).",
         f"""
aux:
  family: a3bslice
  pattern: "LELELELELE*E"
  nodes: 1
  ep: 4
  gbs: 64
  topk: 4
  expert_ffn: 512
""" + A3B_COMMON.replace("{shape}", """    num_experts: 192
    moe_ffn_hidden_size: 512
    moe_router_topk: 4
    moe_shared_expert_intermediate_size: 512
"""), a3b_slice())
    dump("speed_r4b_mn5.yaml",
         "Round 4B: speed / memory feasibility of the best-bet MoE hybrid at production layouts, "
         "bf16 vs fp8 (docs/hybrid_moe_3b_campaign.md §11.1).",
         f"""
aux:
  family: r4b
  pattern: "{pattern48}"
  nodes: 16
  ep: 4
  gbs: 1024
  topk: 7
  expert_ffn: 768
""" + A3B_COMMON.replace("{shape}", """    num_experts: 336
    moe_ffn_hidden_size: 768
    moe_router_topk: 7
    moe_shared_expert_intermediate_size: 768
"""), round4b())
    dump("speed_a3b80_mn5.yaml",
         "Speed of the 3B-active / ~80B-total hybrid MoE (512 experts x 512, top-10 + shared) on 16 "
         "nodes, TP1 / PP4 / EP8.",
         f"""
aux:
  family: a3b80
  pattern: "{pattern48}"
  nodes: 16
  ep: 8
  gbs: 1024
  topk: 10
  expert_ffn: 512
""" + A3B_COMMON.replace("{shape}", """    num_experts: 512
    moe_ffn_hidden_size: 512
    moe_router_topk: 10
    moe_shared_expert_intermediate_size: 512
"""), a3b())

