## Notes (hand-written; the tables below are regenerated)

Setup: hidden 768, 12 blocks, seq 4096, mbs 8, gbs 32 (DP 4), mock data, bf16, CUDA graphs
`local` unless an arm says otherwise. TFLOP/s is Megatron's analytic count (per-architecture
formula); tok/s/GPU is the formula-free comparison. torchtitan's MFU count gives nearly the
same FLOPs/token for this model (~965M), so a "400 TFLOP/s" target means the same in both.

### Baseline (target >= 400 TFLOP/s)
- `dense` 186 -> `base_liger` 269 (Liger fused LM-head CE: no 50k-vocab logits; memory 26.6 -> 12.9 GB)
- Profile of `base_liger` (steady steps): GPU busy 98% -- kernel-bound, not host-bound
  (an earlier "55% busy" was a double-counting bug in kernel_summary.py, fixed).
  Per step: GEMM 37 ms (~65% of peak), elementwise 27 (SwiGLU 8.6, residual adds 5,
  RoPE 3.8, q/k/v concat 2.6), attention 21 (cuDNN bwd ~355 TFLOP/s), norms 11,
  comm 8, copies 7 (2.4 ms H2D = the dense attention mask), loss 4.6.
- No gain: TE CE (disabled in this Megatron), flash backend (0.92x), FP8 (0.97x),
  full-iteration graph (1.03x), no distributed optimizer (0.98x); qk-norm costs 7%.
- In flight: no dataloader mask, fused q/k norm + RoPE (new Triton kernel, Megatron
  `fused_qk_norm_rope`), TE fused residual/RMSNorm, TE op fuser, TE SwiGLU, mbs 16.
- Next: attention (FA3 / newer cuDNN in the 26.09 container), SwiGLU fusion, the titan
  comparison (needs the 26.09 container: this torchtitan needs torch > 2.10).

### Features (vs the old 186 TFLOP/s reference; to be re-run on the optimized baseline)
Below 80%: mamba2, GDN, KDA family, triadic, power (Triton baseline), mHC, nGPT; MLA's
low TFLOP/s is Megatron's FLOP formula (its tok/s matches dense).

Open failures:
- `routed_k2*`: routed residual refuses CUDA graphs (needs graph support, or run without)
- `attn_res`: CUDA-graph runner mismatch at capture
- `pkm`, `tri_ckda_e4`: out of memory at mbs 8 (rerun at mbs 4)

Fixed: GDN under CUDA graphs, NoPE MLA (RoPE fusion flag), SWA window value, power at mbs 4.
