# Serving switches

The existing `flag_eval_fast_mn5` config and `run_flag_fast_mn5.sh` wrapper expose:

- `aux.fusion_impl` / `OELLM_FUSION_IMPL`: `reference` or `triton`. Controls complete mHC read/write and FP32 RMSNorm fusion. The separate existing `mhc_impl` switch controls the legacy Sinkhorn-only path when complete fusion is disabled.
- `aux.decode_graphs` / `OELLM_DECODE_GRAPHS`: 0 or1. Enables bounded local CUDA graph islands: mHC/normalization, TP1 CKDA, local routing/shared experts, and Engram hashes. Host table lookups and expert collectives remain eager boundaries.
- `aux.graph_max_rows` / `OELLM_GRAPH_MAX_ROWS`: upper bound on padded decode rows, also bounded by the local scheduler sequence cap. Powers of two are eligible; mixed/prefill paths remain eager. Each island caches at most8 signatures (Engram row/sign hashes16); misses after the limit fall back to eager. Cache signatures include actual CKDA token counts, cache addresses and tensor layouts.

The vLLM engine's `enforce_eager=True` stays in place: it disables the stock whole-model graph wrapper, not these model-local `torch.cuda.CUDAGraph.replay` regions. Graph capture records an update but does not replay it on the first call; first-call state is advanced only by the initial eager execution. Replay refreshes changing input values and slot indices. Graph scratch shares the vLLM graph pool; outputs retain strong references and islands execute sequentially in model order.

The reference runner/config remain available. Runtime directories and response-cache identities hash the model, fusion and graph implementation files; cache identity also includes the three switches. This experiment is scoped to the vLLM serving path. The HF path does not automatically acquire these fusions/graphs.

Step5000 uses existing `hf_export_v2_mn5` with `aux.model=ambitious_v4 aux.step=5000 aux.expert_dtype=fp8`, then existing `flag_eval_fast_mn5` with `aux.export=ambitious_v4_2ph_fp8_5000`. The standard21 FLAG rows and four shards are inherited. On pool p129, new pool runs use `slurm.sbatch.account=cns144` to match the existing allocation. This does not change the allocation itself.
