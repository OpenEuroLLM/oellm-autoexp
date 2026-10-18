# Kernel optimization — fuse the causal conv into the GDN & mLSTM kernels

**Status:** deferred (we are focusing on the scaling sweep). This is a
self-contained runbook to pick up later. Goal: make the short depthwise causal
conv (xLSTM/GDN conv branch) nearly free for GDN and mLSTM by fusing it into
their chunk kernels — the way Mamba2 already does.

---

## 1. Why

The conv over q,k,v is arithmetically nothing (depthwise, kernel 4 → ~4 MACs per
channel-element) but currently runs as a **separate** FLA Triton kernel wedged
between the `in_proj` GEMM and the recurrence kernel. So it is 100%
memory-bandwidth bound: a full HBM read+write of the largest activation in the
block, plus a launch/epilogue, with no buffer reuse. Measured cost on 0.1B /
MBS32 (jobs 1174837 k4, 1174838 k8):

| variant | Tok/s/GPU | mem |
|---|---:|---:|
| mLSTM no-conv | ~191k | ~0.76 |
| mLSTM conv k4 | ~172k | ~0.94 |
| mLSTM conv k8 | ~164k | ~0.94 |

So ~10–14% throughput + a large memory bump for a ~0-FLOP op.

**Existence proof it can be free:** Mamba2 fuses split+conv+scan into ONE kernel
(`mamba_split_conv1d_scan_combined`, used by way of `use_mamba_mem_eff_path`) and pays
almost nothing. GDN (`chunk_gated_delta_rule`) and mLSTM
(`get_mlstm_kernel("chunkwise--triton_xl_chunk")`) do **not** fuse the conv — the
conv is a global pre-pass. This is a kernel-integration gap, not a fundamental
cost.

**Target:** conv-on ≈ conv-off throughput (within a few %), memory back near the
no-conv level, numerics identical to the current un-fused path.

## 2. Current state (exact anchors)

Container (has all kernels): `MegatronTraining-JUPITER-linattn-mamba_aarch64_202608010917.sif`
in `$CONTAINER_CACHE_DIR`. Pinned versions inside it:

| package | version | kind |
|---|---|---|
| flash-linear-attention (`fla`) | 0.3.2 | pure Python + Triton (editable, no build) |
| mlstm_kernels | 2.0.4 | pure Python + Triton (editable, no build) |
| mamba-ssm | 2.3.2.post1 | **compiled CUDA** (reference only) |
| causal-conv1d | 1.6.2.post1 | compiled CUDA |
| triton | 3.7.1 | |
| torch | 2.11 (nv26.2) | |

Un-fused conv call sites (in `submodules/Megatron-LM`, host-mounted by way of PYTHONPATH):
- GDN: `megatron/core/ssm/gated_delta_net.py:439` — `causal_conv1d(...)` (FLA) then
  `chunk_gated_delta_rule` (`fla.ops.gated_delta_rule`).
- mLSTM: `megatron/core/ssm/mlstm.py:420` — `causal_conv1d(...)` (FLA) then
  `self.mlstm_fn` = `mlstm_kernels.torch.get_mlstm_kernel("chunkwise--triton_xl_chunk")`.
- The conv weight is stored TP-local, CP-sliced at runtime by way of
  `get_parameter_local_cp(split_sections=[qk_dim_local_tp, qk_dim_local_tp, v_dim_local_tp])`,
  applied in the post-all-to-all full-sequence head-sharded regime (so at the
  kernel level it is a full sequence — no CP boundary inside the kernel), silu
  activation, depthwise, layout `[q|k|v]`.
- FLA conv itself: `fla.modules.convolution.causal_conv1d` (Triton path; not
  width-limited, unlike Tri Dao's CUDA `causal_conv1d`).

Fused reference to study: mamba-ssm `mamba_split_conv1d_scan_combined` and
`causal_conv1d` Triton kernel (how it threads conv state through the chunked scan).

Config knobs: `mlstm_conv1d` (bool) + `linear_conv_kernel_dim` (width). GDN
always has the conv. FLOP term already added to both branches in
`megatron/training/training.py`.

---

## Phase 0 — Download the kernels for editable development

FLA and mlstm_kernels are **pure Python + Triton** (kernels JIT-compile at
runtime), so they can be edited and used **without rebuilding the container** —
just shadow the container's installed copies by way of `PYTHONPATH`. (mamba-ssm /
causal-conv1d are compiled CUDA; treat them as read-only references.)

```bash
# On JUPITER, in the repository root.
cd /e/project1/e-sta-openeurollm/poeppel1/Projects/oellm-autoexp-hybrid
mkdir -p kernels && cd kernels

# Pin to the versions baked into the container so behaviour matches BEFORE editing.
git clone https://github.com/fla-org/flash-linear-attention.git
git -C flash-linear-attention checkout v0.3.2      # match container fla 0.3.2

git clone https://github.com/NX-AI/mlstm_kernels.git
git -C mlstm_kernels checkout v2.0.4               # match container mlstm_kernels 2.0.4

# Reference only (do NOT need to build — read the fused kernel source):
git clone https://github.com/state-spaces/mamba.git          # mamba_ssm/ops/triton/ssd_combined.py
git clone https://github.com/Dao-AILab/causal-conv1d.git     # causal_conv1d Triton kernel
```

Make jobs use the editable clones (they must win over the container's
`dist-packages`). Prepend the repository roots to `PYTHONPATH` — the same mechanism
already used for `submodules/Megatron-LM`:

```yaml
# in the experiment config (or a dedicated container override), extend:
container:
  env:
    PYTHONPATH: "kernels/flash-linear-attention:kernels/mlstm_kernels:.:submodules/Megatron-LM"
```

Verify the override took (inside the container, with the repository mounted):
```bash
apptainer exec --bind /e $CONTAINER_CACHE_DIR/MegatronTraining-JUPITER-linattn-mamba_*.sif \
  env PYTHONPATH="kernels/flash-linear-attention:kernels/mlstm_kernels:.:submodules/Megatron-LM" \
  python -c "import fla, mlstm_kernels; print(fla.__file__); print(mlstm_kernels.__file__)"
# Both paths MUST point into ./kernels/, not /usr/local/lib/.../dist-packages/.
```

**Gate 0:** before touching anything, run the two 0.1B throughput smokes
(`throughput_native_mamba2_0.1B` pattern, or a 100-iter mlstm/gdn smoke) with the
editable clones and confirm identical loss + throughput to the container copies.
This proves the PYTHONPATH shadow is clean and gives the un-fused baseline.

---

## Phase 1 — Study the reference, isolate the target

1. Read `kernels/mamba/mamba_ssm/ops/triton/ssd_combined.py` —
   `_mamba_chunk_scan_combined_fwd` / the split-conv-scan wrapper — to see how the
   causal conv is (a) applied inside the chunk prologue and (b) how the conv state
   (last `kernel-1` tokens) is carried across chunk boundaries.
2. Read the FLA chunk kernel to fuse into:
   - GDN: `kernels/flash-linear-attention/fla/ops/gated_delta_rule/chunk.py`
     (`chunk_gated_delta_rule`).
   - mLSTM: `kernels/mlstm_kernels/mlstm_kernels/triton/chunkwise/` (the
     `triton_xl_chunk` backend) — find the fwd/bwd entry the Megatron wrapper calls.
3. Read the FLA conv it will absorb:
   `kernels/flash-linear-attention/fla/modules/convolution.py` (`causal_conv1d`,
   Triton kernel) — depthwise, silu, weight `[d, w]`.

**Key challenges to design around** (document decisions in the PR):
- **Cross-chunk conv state:** the conv at chunk boundary needs the previous
  chunk's last `kernel-1` tokens. Either process the conv in the kernel's chunk
  prologue with a carried state (like Mamba), or (simpler first cut) keep the conv
  as a fused *epilogue of `in_proj`* / *prologue of the chunk loop* that still saves
  one HBM round-trip + the extra buffer even if not fully inside the scan.
- **Layout:** conv is depthwise over concatenated `[q|k|v]`; the recurrence wants
  per-head reshaped q,k,v. Reconcile inside the kernel (conv on the packed layout,
  then reshape) so no extra materialization.
- **CP:** none needed inside the kernel — the Megatron wrappers already hand the
  kernel a full sequence for a head-subset (post all-to-all). Keep it that way.
- **Backward:** the conv is inside autograd today (nn.Conv1d weight + FLA fn). The
  fused kernel must provide grads w.r.t. q,k,v AND the conv weight/bias.

## Phase 2 — Implement the fusion

Do GDN first (FLA `chunk_gated_delta_rule`), then mLSTM (mlstm_kernels), since GDN
is closest to Mamba's structure.

Per architecture:
1. Add an optional conv (weight, bias, width, activation) to the chunk kernel's
   fwd/bwd, applied to q,k,v before the recurrence, state carried across chunks.
2. Expose a Python entry that takes q,k,v + conv params and returns the same
   result as `causal_conv1d(...)` followed by the existing recurrence.
3. Wire into Megatron: in `gated_delta_net.py` / `mlstm.py`, when the fused kernel
   is available and a new flag is set, call the fused entry with
   `conv1d_weight`/`conv1d_bias` (the same CP-sliced tensors) INSTEAD of the
   separate `causal_conv1d(...)` + kernel calls. **Keep the un-fused path as the
   default fallback** behind the flag (for example `mlstm_conv1d_fused` / GDN equivalent),
   so nothing regresses if the fused kernel is unavailable or numerics drift.

## Phase 3 — Test & verify

Correctness first, then performance. Nothing merges until numerics match.

### 3a. Kernel-level numerics (fwd + bwd)
Write `tests/kernels/test_fused_conv.py` (host, GPU; run inside the container):
```python
# For each of {gdn, mlstm}, random q,k,v (b,s,h,d), random depthwise conv weight,
# widths {4, 8}, dtype bf16 (+ fp32 ref):
#   ref  = recurrence(causal_conv1d(qkv))         # current un-fused path
#   test = fused_conv_recurrence(qkv, conv_w)      # new kernel
#   assert torch.allclose(test, ref, atol=..., rtol=...)          # forward
#   ref.sum().backward(); test.sum().backward()
#   assert allclose on grads w.r.t. q,k,v AND conv weight/bias    # backward
# Also assert the fused == unfused with conv DISABLED (identity guard).
```
Run the upstream repository suites too:
```bash
apptainer exec --bind /e $CONTAINER_CACHE_DIR/...-mamba_*.sif \
  bash -lc 'cd kernels/flash-linear-attention && pytest tests/ -k "gated_delta or conv" -q'
apptainer exec --bind /e $CONTAINER_CACHE_DIR/...-mamba_*.sif \
  bash -lc 'cd kernels/mlstm_kernels && pytest tests/ -k "chunkwise or conv" -q'
```

### 3b. End-to-end (loss parity + speed/mem win) — must go through SLURM
Same config as the current conv runs, seed fixed, ~200 iters:
- **Loss parity:** the fused run's per-iter loss must match the un-fused conv run
  (jobs 1174837 k4 / 1174838 k8) to fp noise for the first N iters (same seed,
  same data order). This is the real correctness gate at model scale.
- **Throughput:** Tok/s/GPU should recover most of the conv tax (target: within a
  few % of no-conv mLSTM ~191k; compare against the k4/k8 numbers above).
- **Memory:** peak `mem usages` should drop from ~0.94 back toward the no-conv
  ~0.76.
- **k8:** confirm width 8 still works fused (FLA path is not width-limited).

### 3c. Context parallelism
If a ≥2-node allocation is available, run with `context_parallel_size=2` and
assert the loss matches `cp=1` (fused conv must be CP-invariant; the Megatron
wrapper already guarantees full-sequence input per rank).

### Acceptance criteria
- [ ] fwd + bwd allclose vs un-fused (3a), widths 4 and 8, both archs.
- [ ] upstream fla / mlstm_kernels test suites pass.
- [ ] E2E loss parity vs the un-fused conv 6B runs (same seed).
- [ ] Tok/s within a few % of no-conv; mem back near no-conv.
- [ ] CP=2 loss == CP=1 loss.
- [ ] `hydra_staged_sweep` + `test_config_feasibility` suites still green.

## Phase 4 — Integration & container

- Land the kernel changes as forks/branches of fla + mlstm_kernels; keep the
  editable-clone + PYTHONPATH setup for iteration.
- Once stable, bake into the container: add the forks to
  `container/megatron/MegatronTraining-JUPITER-linattn-mamba.def.in` by way of
  `pip install -e kernels/flash-linear-attention kernels/mlstm_kernels`
  (or pinned fork commits), rebuild with `container/build_container.sh`, bump
  `config/container/jupiter_liger_mamba.yaml` to the new image, and drop the
  PYTHONPATH override.
- Flip the fused flag on for the scaling sweep's mLSTM/GDN variants
  (`main_scaling_all.yaml`) once acceptance passes; re-check MBS headroom at the
  larger sizes (the conv memory bump was what pushed 0.1B to ~0.94).

---

## File map
- Conv call sites: `submodules/Megatron-LM/megatron/core/ssm/{gated_delta_net.py:439, mlstm.py:420}`
- mLSTM conv integration + flag: `mlstm.py` (`use_conv1d`, `mlstm_conv1d`), `transformer_config.py` (`mlstm_conv1d`)
- FLOP terms: `submodules/Megatron-LM/megatron/training/training.py` (gdn & mlstm branches)
- Fused reference: mamba-ssm `mamba_split_conv1d_scan_combined`
- 6B conv runs (un-fused baseline for loss parity): `mlstm7_conv_0.1B_50BT.yaml` (jobs 1174837 k4, 1174838 k8)
- Container def: `container/megatron/MegatronTraining-JUPITER-linattn-mamba.def.in`
- Background/measurements: `experiment/003_multilingual_throughput_tuning/log.md`
