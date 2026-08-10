# Muon 50M smoke test — RESULT

## Verdict: PASS (2026-07-15)

Muon runs on the LUMI ROCm stack. Standalone GPU smoke test passed end to end.

| Check | Result |
|-------|--------|
| torch + ROCm, GPU visible | OK — AMD Instinct MI250X |
| Muon Newton-Schulz on GPU | OK — orthogonalized (singular values 0.978–1.032) |
| AITER kernel JIT compile | OK — 50s, one-time, now cached in ~/.aiter |
| Full Megatron v0.17 import | OK |
| emerging_optimizers detected | HAVE_EMERGING_OPTIMIZERS = True; coeff types: simple, quintic, polar_express, aol, custom |

## What made it work (LUMI recipe — reuse for the pipeline)
- Container: MegatronTrainingLumi_x86_64.sif + overlay muon-overlay.img (emerging_optimizers 0.2.0)
- Megatron submodule pinned to v0.17 (4a81356)
- singularity exec, NO `--rocm`; binds: /pfs /scratch /flash /opt/cray /var/spool/slurmd /appl
- `bash -c` (not -lc); `python -u`
- **CC=gcc-12 CXX=g++-12** — required so AITER's JIT finds a compiler
  (container has g++-12/gcc-12/hipcc/clang++, but NOT bare c++/g++/gcc)
- Clear ~/.aiter/jit/build/lock_module_aiter_enum if a run is killed mid-compile

## Notes
- transformer_engine in this container lacks `multi_tensor_scale_tensor` (v0.17
  wants newer TE); Megatron falls back to apex's multi_tensor_applier — works fine.
- apex uses its native RoPE kernel (UserWarning, harmless).

## Next: wire muon into the real pipeline
1. Regenerate v0.17 schema (so `optimizer: muon` is a valid autoexp config key).
2. Dry-run render + short training step with config experiments/laingsam/muon_50M_50BT.
3. Data: config reads /scratch/project_462000963/... (needs read access).
All in a SLURM allocation, with the same CC/CXX + binds recipe above.
