# Megatron → Hugging Face export: engram_scaling (Clariden)

Tools to convert the `engram_scaling` runs (configs under
`config/experiments/swagatam/alps/engram_scaling/`, trained with `submodules/Megatron-LM` on
branch `oellm/v0.19-hybrid`) from Megatron `torch_dist` checkpoints to a Hugging Face model
that loads with `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`.

The work is built in stages, each one checked before the next:

| Stage | What | Status |
|---|---|---|
| 1 | Inspect the checkpoint: what is saved, how each layer is stored | done (`inspect_megatron_ckpt.py`) |
| 2 | HF config builder: `config-<job>.yaml` → `config.json` | done (`hf_config.py`, `check_hf_config.py`) |
| 3 | HF modeling code, one HF layer per Megatron pattern symbol | done (`oellm_engram/`, `check_modeling.py`) |
| 4 | Weight conversion: DCP → safetensors + config + code + tokenizer | done (`export_hf.py`, `check_hf_export.py`) |
| 5 | Parity with Megatron's own forward: per layer, logits, NLL, Engram ablations, tokenizer, batched generation | done (`megatron_parity.py`) |
| 6 | Batch export of all rho arms and both scales (100m_1b, 230m_2b3) | next |

## Quick start: convert and verify a run

From the `oellm-autoexp` root (each command runs in the training container on a debug node):

```bash
D=oellm_autoexp/hf_export/engram_scaling_clariden
RUN=output/<run>          # e.g. output/engram_230m_2b3_E40_rho40_N1_clariden
N=<iteration>             # e.g. 10842; omit --iter for the latest checkpoint
OUT=$RUN/hf/iter_$(printf %07d $N)

# 1. convert (no GPU, 1-2 minutes)
TIME=00:30:00 $D/run_in_container.sh export_hf --run-dir $RUN --iter $N --out-dir $OUT
# 2. check the export: structure, every checkpoint element exported once, loading, forward, generation
GPUS=1 TIME=00:40:00 $D/run_in_container.sh check_hf_export $OUT
# 3. parity with the trained Megatron model on the run's validation data (the proof)
LAUNCHER="torchrun --standalone --nproc-per-node 1" GPUS=1 TIME=01:00:00 \
    $D/run_in_container.sh megatron_parity --export-dir $OUT --out $OUT/parity.json
```

All checks print `[ok]` / `[FAIL]` lines and end with `N/M checks passed` (exit code 1 on a failure).
Load the export with `AutoModelForCausalLM.from_pretrained($OUT, trust_remote_code=True,
dtype="bfloat16")` (stage 4).

## The conversion in brief

What a Megatron → HF conversion has to do, and what this one does (details in the stage sections):

1. **Read the checkpoint without Megatron.** A `torch_dist` checkpoint is PyTorch DCP: `.metadata`
   indexes every tensor's global shape, so the model tensors load whole on one CPU. 80% of the
   checkpoint is optimizer state, which is skipped (stage 1).
2. **Describe the architecture** in `config.json`, including what is not in the checkpoint: the layer
   types, and for Engram the table sizes and the hash multipliers, recomputed from Megatron's seeds
   (stage 2).
3. **Write the forward pass** as HF model code, transcribing each Megatron module (Gated DeltaNet,
   gated attention, MoE, Engram) with the same numerics and kernels (stage 3).
4. **Map the weights**: renames, the GDN projection sections concatenated in Megatron's order, dtypes
   (bf16; router fp32), the Engram token map from its `.npy`, the tokenizer (stage 4).
5. **Prove it** by running the trained Megatron model and the export on the same tokens, layer by layer
   (stage 5).

What would have gone wrong *silently* (the model loads, runs and scores far below chance, but is not
the trained model), and is now checked:

| Pitfall | Effect | Where it is handled |
|---|---|---|
| Engram hash multipliers without the per-layer seed offset (`convert_megatron_to_hf.py`) | +0.16-0.19 nats, worse than no Engram | stage 2 (vs Megatron's own function), stage 5 (ablation) |
| HybridModel's RoPE: interleaved pairs, but non-interleaved angles | +0.38 nats, 71% top-1 agreement | stage 5 found it; stages 2-3 now model and test it |
| Positions under left padding (this RoPE is not shift-invariant) | different logits for padded prompts | stage 3 (left-padding check) |
| Zero-centred backbone norms (`1 + w`) vs Engram's plain norms (`w`) | every norm off by one | stages 1 and 3 |
| `fix_mistral_regex=True` on the tokenizer | every document tokenized differently from training | stage 5 (re-encoding check) |
| Tokenizer `<pad>` id 262144 outside the 262144-row embedding | crash on any padded batch | stage 4 (pad → `<eos>`) |
| Router / expert bias cast to bf16 | near-tie expert choices change | stage 3 (kept fp32) |
| Non-persistent buffers under `from_pretrained` (meta device) | RoPE / hash tables silently zero | stage 3 (no such buffers) |

## Background

Why a new exporter: the existing ones in `hf_export/` do not fit these runs.
`export_ambitious.py` / `export_q35.py` read `fsdp_dtensor` checkpoints and expect mHC/LDM + NoPE
(ambitious) or strict mixer+MoE pairs (q35); `convert_megatron_to_hf.py` has no MoE (`E`) and ignores
`engram_hash_per_layer`. None of them implements HybridModel's partial RoPE (`rotary_percent: 0.25`,
`rotary_interleaved: true`, which in HybridModel means interleaved pairs with non-interleaved angles,
stage 5).

Why no Megatron-Bridge: Bridge converts between Megatron and architectures it has a bridge for, by
building the Megatron model from its own model provider in a (distributed) Megatron process and
mapping parameters one by one. This model (GDN + gated attention + sigmoid MoE + Engram, HybridModel's
partial RoPE) exists neither in transformers nor in Bridge, so Bridge would need the HF model code
written here *plus* a bridge class and a provider for the fork's HybridModel. What Bridge saves
elsewhere -- gathering TP / PP / EP shards -- the DCP reader does here: `.metadata` records the global
shape of every tensor, so the full tensors load on one CPU. Bridge is the tool for the other direction
(HF → Megatron, e.g. to start training from HF weights) and for architectures transformers already has.

Why a modeling script: an HF checkpoint is weights + `config.json`; the forward pass is Python code
chosen by `model_type`. No transformers class computes this model (Qwen3-Next is the closest: no
Engram, non-interleaved RoPE, a softmax router), so the export ships its own
`configuration_oellm_engram.py` / `modeling_oellm_engram.py` next to the weights, referenced from
`config.json`'s `auto_map` and loaded with `trust_remote_code=True` (stage 3).

## Files

| File | Purpose |
|---|---|
| `inspect_megatron_ckpt.py` | Read-only listing of a `torch_dist` checkpoint against the run's layer pattern |
| `hf_config.py` | Builds `config.json` / `generation_config.json` from the run's resolved config |
| `check_hf_config.py` | Checks `hf_config.py` against Megatron's Engram code and the checkpoint shapes |
| `oellm_engram/` | The HF model code (`configuration_oellm_engram.py`, `modeling_oellm_engram.py`), copied next to the weights by the export |
| `check_modeling.py` | Checks the HF model code against Megatron's functions, the FLA kernels, caching / padding / loading |
| `export_hf.py` | Converts a run's checkpoint into an HF model directory |
| `check_hf_export.py` | Checks an export: structure, conservation of every checkpoint element, strict load, forward, generation |
| `megatron_parity.py` | Runs the trained Megatron model and the export on the same validation tokens: per layer, logits, NLL, ablations |
| `run_in_container.sh` | Runs a module of this folder in the Clariden container on a debug node |
| `results/` | Outputs of the example runs below |

## Prerequisites

- The login nodes have no torch. Everything runs inside the training container
  (`container/alps/nemo-26.08.toml`), which `run_in_container.sh` starts via `srun --environment=...`
  on the `debug` partition (no GPU by default; set `GPUS=1` for later stages).
- The tokenizer (`openeurollm/tokenizer-256k`) is read offline from the training runs' cache
  `HF_HOME=/iopsstor/scratch/cscs/$USER/hf_home` (the wrapper's default).
- The run's resolved config `output/<run>/config-<jobid>.yaml` (written by oellm-autoexp at submission)
  is the source of truth for the architecture, not the experiment yaml.
- Run from the `oellm-autoexp` root.

Versions everything here was written and checked against (2026-11-21):

| Component | Version |
|---|---|
| oellm-autoexp | branch `feat/hybrid-v0.19-alps`, commit `dbfbfce7` |
| `submodules/Megatron-LM` | branch `oellm/v0.19-hybrid`, commit `1441d16cb` (the HF code transcribes this version's modules) |
| container | `nemo-26.08` (`/capstor/store/cscs/swissai/a0248/images/nemo-26.08.sqsh`) |
| torch / transformers / flash-linear-attention | `2.13.0a0+8145d630e8.nv26.06` / `5.12.1` / `0.4.2` |

A later Megatron commit can change a module the HF code copies; re-run `check_hf_config.py` and
`check_modeling.py` after updating the submodule.

## Stage 1: inspect a checkpoint

```bash
RUN=output/engram_100m_1b_cons_E40_rho40_N1_clariden
oellm_autoexp/hf_export/engram_scaling_clariden/run_in_container.sh inspect_megatron_ckpt \
    --ckpt-dir $RUN/ckpt/iter_0001000 --config $RUN/config-3994876.yaml
# add --stats to also load the model tensors and print mean / std / min / max (and flag NaN / Inf)
```

Environment variables of the wrapper: `PARTITION` (default `debug`), `TIME` (default `00:20:00`),
`GPUS` (default `0`), `HF_HOME` (default `/iopsstor/scratch/cscs/$USER/hf_home`), `HF_HUB_OFFLINE`
(default `1`), `LAUNCHER` (default `python`; `torchrun --standalone --nproc-per-node 1` for
`megatron_parity`). Inside the container it sets `PYTHONPATH=.:submodules/Megatron-LM`, so the checks can
import the submodule's Megatron (not the image's) as the reference.

What it prints:

1. saved entries by category: model weights, optimizer state, Transformer Engine `_extra_state`, the rest;
2. the layer pattern, one Megatron layer per symbol, with its index;
3. the tensors of every layer, each distinct (symbol, names / shapes / dtypes) signature once with the
   layers that share it; `!!` flags a symbol stored in more than one way, or layer indices that do not
   match the pattern;
4. the non-layer tensors;
5. the Engram tables: the prime sizes derived from the config against the saved table rows, and the
   per-layer hash seed offset;
6. with `--stats`, per-tensor weight statistics.

Only `.metadata` is read unless `--stats` is given (seconds); `--stats` reads the 3.45 GB of model
tensors (about a minute), never the optimizer state.

### Example: `engram_100m_1b_cons_E40_rho40_N1_clariden`, iteration 1000

Run on 2026-11-21 on a Clariden debug node. Full outputs:
[`results/inspect_rho40_iter1000.txt`](results/inspect_rho40_iter1000.txt) and
[`results/inspect_rho40_iter1000_stats.txt`](results/inspect_rho40_iter1000_stats.txt).

```text
== saved entries
  model                325 entries      3.45 GB
  optimizer           1400 entries     14.08 GB
  te_extra_state      1382 entries      0.00 GB
  other                  2 entries      0.00 GB
  model dtypes: {'float32': 17, 'bfloat16': 308}

== layer pattern 'LENLELE*ELELELE*ELELELE*ELELELE*E': 33 Megatron layers
  0:L 1:E 2:N 3:L 4:E 5:L 6:E 7:* 8:E 9:L 10:E ... 31:* 32:E

-- 'L' linear mixer: layers [0, 3, 5, 9, 11, 13, 17, 19, 21, 25, 27, 29]
     self_attention.A_log                                         [20]                   bfloat16
     self_attention.conv1d.weight.{query,key}                     [640, 1, 4]            bfloat16
     self_attention.conv1d.weight.value                           [1280, 1, 4]           bfloat16
     self_attention.dt_bias                                       [20]                   bfloat16
     self_attention.in_proj.layer_norm_weight                     [640]                  bfloat16
     self_attention.in_proj.weight.{alpha,beta}                   [20, 640]              bfloat16
     self_attention.in_proj.weight.{query,key}                    [640, 640]             bfloat16
     self_attention.in_proj.weight.{value,z}                      [1280, 640]            bfloat16
     self_attention.out_norm.weight                               [64]                   bfloat16
     self_attention.out_proj.weight                               [640, 1280]            bfloat16

-- 'E' MoE: layers [1, 4, 6, 8, ..., 32]
     mlp.experts.experts.linear_fc1.weight                        [40, 640, 640]         bfloat16
     mlp.experts.experts.linear_fc2.weight                        [40, 640, 320]         bfloat16
     mlp.router.expert_bias                                       [40]                   float32
     mlp.router.weight                                            [40, 640]              bfloat16
     mlp.shared_experts.gate_weight                               [1, 640]               bfloat16
     mlp.shared_experts.linear_fc1.weight                         [640, 640]             bfloat16
     mlp.shared_experts.linear_fc2.weight                         [640, 320]             bfloat16
     pre_mlp_layernorm.weight                                     [640]                  bfloat16

-- 'N' Engram: layers [2]
     conv1d.weight                                                [640, 1, 4]            bfloat16
     {conv_norm,hidden_norm,key_norm}.weight                      [640]                  bfloat16
     {key_proj,value_proj}.weight                                 [640, 1024]            bfloat16
     table                                                        [8610592, 64]          float32

-- '*' attention: layers [7, 15, 23, 31]
     self_attention.{q,k}_layernorm.weight                        [128]                  bfloat16
     self_attention.linear_proj.weight                            [640, 1280]            bfloat16
     self_attention.linear_qkv.layer_norm_weight                  [640]                  bfloat16
     self_attention.linear_qkv.weight                             [3072, 640]            bfloat16

== non-layer tensors
  decoder.final_norm.weight                                      [640]                  bfloat16
  embedding.word_embeddings.weight                               [262144, 640]          bfloat16

== Engram: layers [2], 16 tables (2 orders x 8 heads), sizes 538079 .. 538259, sum 8610592
  layer 2: table rows 8610592 (no padding); hash seed offset 10007 * layer_number = 10007 * 3
```

(Lines condensed with `{a,b}` here; the file lists each tensor.)

Excerpt of `--stats` (no tensor was flagged non-finite):

```text
  decoder.final_norm.weight                         mean +3.7736e-01 std 1.1047e-01 ...
  decoder.layers.0.self_attention.in_proj.layer_norm_weight   mean -5.4537e-02 ...
  decoder.layers.0.self_attention.out_norm.weight   mean -1.4613e-01 std 1.5659e-02 ...
  decoder.layers.0.self_attention.A_log             mean +1.9943e+00 std 8.0192e-01 ...
  decoder.layers.1.mlp.router.expert_bias           mean +1.7920e-01 std 7.7562e-02 ...
  decoder.layers.2.hidden_norm.weight               mean +9.4025e-01 std 5.2552e-02 ...
  decoder.layers.2.conv_norm.weight                 mean +9.1110e-01 std 7.3959e-02 ...
  decoder.layers.2.table                            mean +4.8431e-05 std 1.2441e-01 ...
```

### How to read the example

- **The model is a fifth of the checkpoint.** 3.45 GB of model tensors against 14.1 GB of optimizer
  state (Muon / Adam moments and fp32 master copies `optimizer.state.fp32_param.*`). The export uses the
  bf16 model weights, the ones the training forward ran with. The 17 fp32 tensors are the 16 router
  `expert_bias` vectors and the Engram table (`engram_table_fp32`).
- **One Megatron layer per pattern symbol**, and every symbol has a single signature, so the exporter
  maps per symbol. HF will get the same 33 layers.
- **Shapes follow from the config:**
  - `L` (Gated DeltaNet): q, k = 10 key heads × 64 = 640; v, z (output gate) = 20 value heads × 64 = 1280;
    beta, alpha, `A_log`, `dt_bias` one per value head (20); the causal conv covers q, k, v only;
    `out_norm` [64] is one norm over the head dim, shared by the heads. The checkpoint stores `in_proj`
    and `conv1d` as named sections; the forward uses them concatenated in the order q, k, v, z, beta, alpha.
  - `*` (gated attention): `linear_qkv` [3072, 640] = 2 KV groups × (5 q + 5 gate + 1 k + 1 v) heads × 128,
    interleaved per group; a plain 3-way split would mix them up.
  - `E` (MoE): `linear_fc1` [40 experts, 640 = gate 320 | up 320, 640 hidden] — both 640s, so the
    orientation cannot be guessed from the shape: the gate/up split is on dim 1. `linear_fc2` is
    [40, 640 hidden, 320].
  - `N` (Engram): `key_proj` / `value_proj` [640, 1024], 1024 = 16 lookups × 64 dims
    (2 n-gram orders × 8 hash heads).
- **Engram table:** 8,610,592 rows = exactly the 16 primes 538079 … 538259 back to back. No padding,
  because `engram_table_parallel: dp` splits the unpadded rows over the data-parallel ranks (the other
  modes pad to a multiple of 4096). The hash multipliers are not in the checkpoint: the exporter
  recomputes them from seed `0x5EED_E46A + 10007 × layer_number`, `layer_number` = pattern index + 1 = 3.
- **Norm placement:** in `L` and `*` the pre-norm is fused into the first projection
  (`in_proj.layer_norm_weight`, `linear_qkv.layer_norm_weight`) and becomes a separate
  `input_layernorm` in HF; `E` has its own `pre_mlp_layernorm`; `N` has no pre-norm.
- **Two kinds of RMSNorm (seen in `--stats`):** the backbone norms (`final_norm`, the fused pre-norms,
  `pre_mlp_layernorm`, `q/k_layernorm`, GDN `out_norm`) are zero-centred (`layernorm_zero_centered_gamma`):
  stored near 0, applied as `1 + w`. The Engram norms (`hidden_norm`, `key_norm`, `conv_norm`) are
  Megatron's own `engram._RMSNorm`: initialised to ones, stored near 0.94 here, applied as `w`. The HF
  model has to keep this distinction; adding 1 to the Engram norms would be silently wrong.
- **Tied embedding:** no `output_layer.weight`; the LM head reuses `embed_tokens`.

## Stage 2: the HF config

`config.json` is the contract between the exporter and the HF model code: it fixes every shape, and
it carries the parts of the model that are not in the checkpoint. A wrong value here gives a model that
loads and runs but is a different model, so this stage is checked on its own.

```bash
RUN=output/engram_100m_1b_cons_E40_rho40_N1_clariden
D=oellm_autoexp/hf_export/engram_scaling_clariden
# config.json + generation_config.json only (no weights)
$D/run_in_container.sh hf_config --config $RUN/config-3994876.yaml \
    --ckpt-dir $RUN/ckpt/iter_0001000 --out-dir <out dir>
# checks on any number of runs (latest config and checkpoint of each)
TIME=00:30:00 $D/run_in_container.sh check_hf_config output/engram_*
```

Inputs: the resolved `config-<job>.yaml` (`backend.megatron`), the vocabulary size (rows of the saved
embedding, from `.metadata`) and the tokenizer (`--tokenizer`, default the run's `tokenizer_model`,
resolved to its local snapshot) for the special token ids.

### What goes into `config.json`

Example: [`results/hf_config_rho40_cons/config.json`](results/hf_config_rho40_cons/config.json).

| Kind | Fields | Notes |
|---|---|---|
| Copied | `hidden_size`, `num_attention_heads`, `head_dim`, `linear_*` (GDN heads and dims, conv kernel), `num_experts`, `num_experts_per_tok`, `moe_intermediate_size`, `shared_expert_intermediate_size`, `rms_norm_eps`, `vocab_size`, `max_position_embeddings` | sizes |
| Translated | `block_types` | one per pattern symbol: `L`→`gdn`, `E`→`moe`, `N`→`engram`, `*`→`attention`; HF layer `i` = `decoder.layers.i`. Not named `layer_types`: transformers 5 validates that name against its own attention kinds |
| | `num_key_value_heads` | `num_query_groups` only under `group_query_attention`, else = heads |
| | `rope_theta`, `partial_rotary_factor`, `rotary_dim`, `rope_interleaved`, `rope_interleaved_freqs` | RoPE on the first `128 × 0.25 = 32` channels of each head, `inv_freq = 1 / 1e7^(2i/32)`. The rotation pairs `(2i, 2i+1)` (`rope_interleaved` = `rotary_interleaved`), but the per-channel angles keep the non-interleaved layout `[f0 … f15, f0 … f15]` (`rope_interleaved_freqs: false`): HybridModel builds its `RotaryEmbedding` without `rotary_interleaved`. Found in stage 5, see there |
| | `layernorm_zero_centered_gamma` | backbone norms applied as `1 + w`; the Engram norms are plain |
| | `router_*`, `shared_expert_gate` | sigmoid scores in fp32; top-k on scores + expert bias; weights = unbiased scores normalised over the top-k; sigmoid-gated shared expert |
| Derived (not in the checkpoint) | `engram_layers`, `engram_table_sizes` | 16 primes ≥ `engram_table_size`, in lookup order (order 2 heads 0-7, then order 3) |
| | `engram_hash_multipliers` | per Engram layer: `[16 lookups][max order = 3]` odd ints from `numpy.random.default_rng(0x5EED_E46A + 10007 × layer_number)`; a 2-gram lookup uses the first 2 of its 3 |
| | `engram_conv_dilation` | = max n-gram order (3), as in Megatron |
| | `engram_token_map` | true for the 230m runs: the canonical-id map (`/capstor/scratch/.../engram_token_map_256k.npy`) is a non-persistent Megatron buffer, not in the checkpoint. The export (stage 4) must copy it into the weights -- scratch is purged |
| Tokens | `bos/eos/pad_token_id`, `generation_config.json` | the tokenizer's `<pad>` is id 262144, past the 262144-row embedding, so pad → eos (2) |

### Guards

The resolved config has 1101 fields, most of them optimizer, logging, data, kernel or disabled-feature
settings. `hf_config.check_supported` checks the ones that change the forward pass: `_REQUIRED` (e.g.
`experimental_attention_variant: gated_delta_net`, sigmoid router with expert bias,
`linear_beta_max: 1.0`) and `_OFF` (mHC, depth memory, MTP, MLA, sliding windows, Engram sign hash /
auxiliary lookups, SwiGLU clamping, router groups, ...). It reports every violation at once and stops.
Kernel-only options (`fused_short_conv`, `gdn_fused_out_norm`, `apply_rope_fusion`, `moe_router_fusion`,
...) do not change the result and are not checked.

### Checks (`check_hf_config.py`) and results

Full output: [`results/check_hf_config_all_runs.txt`](results/check_hf_config_all_runs.txt). All 22
engram runs in `output/` (both scales, consolidated and two-bank, with and without token map, rho 100
baselines), at their latest checkpoints: **22/22 OK**.

1. **Engram geometry against Megatron's own code** (`megatron.core.ssm.engram.engram_table_sizes` /
   `engram_hash_multipliers` on the same arguments): equal for every Engram layer of every run. The
   same check with the per-layer seed offset left out (what `convert_megatron_to_hf.py` computes) gets
   48 of 48 multipliers wrong in every layer.
2. **Every layer's parameter count from the config alone** equals the checkpoint's, layer by layer.
   Without the router `expert_bias` buffers the totals are the grand totals in the experiment yaml
   headers (1,174,547,680 for rho 40, 1,174,538,208 for rho 90, 1,173,230,432 for rho 100).
3. **Guards:** `mhc`, `engram_sign_hash`, `mtp_num_layers`, a softmax router and a `-` (dense MLP) symbol
   are each refused.

```text
== engram_100m_1b_cons_E40_rho40_N1_clariden  (config-3994876.yaml, iter_0003000)
    33 layers, Engram layers [2], token map False
    layer 2: multipliers == Megatron (48 values); without the per-layer seed offset 48/48 would differ
    parameters: config 1,174,548,320 == checkpoint 1,174,548,320 (incl. 640 router expert_bias; without: 1,174,547,680)
    OK

== engram_100m_1b_E40_rho40_N1_clariden  (config-3844092.yaml, iter_0004819)
    34 layers, Engram layers [2, 17], token map False
    layer 2: multipliers == Megatron (48 values); without the per-layer seed offset 48/48 would differ
    layer 17: multipliers == Megatron (48 values); without the per-layer seed offset 48/48 would differ
    parameters: config 1,175,866,336 == checkpoint 1,175,866,336 (incl. 640 router expert_bias; without: 1,175,865,696)
    OK

== engram_230m_2b3_E40_rho40_N1_clariden  (config-3983440.yaml, iter_0010842)
    74 layers, Engram layers [2, 37], token map True
    layer 2: multipliers == Megatron (48 values); without the per-layer seed offset 48/48 would differ
    layer 37: multipliers == Megatron (48 values); without the per-layer seed offset 48/48 would differ
    parameters: config 2,432,694,552 == checkpoint 2,432,694,552 (incl. 1,440 router expert_bias; without: 2,432,693,112)
    OK

22/22 runs OK
```

## Stage 3: the HF model code

`oellm_engram/` holds the code the export ships next to the weights: `configuration_oellm_engram.py`
(`OellmEngramConfig`, `model_type: oellm_engram`) and `modeling_oellm_engram.py`
(`OellmEngramForCausalLM`). `config.json`'s `auto_map` points at them, and
`AutoModelForCausalLM.from_pretrained(dir, trust_remote_code=True)` imports them from the export
directory. The code is a transcription of the Megatron modules, one class per module:

| HF class | Megatron module (submodules/Megatron-LM) | What it computes |
|---|---|---|
| `OellmEngramDecoderLayer` | `TransformerLayer` / `MoETransformerLayer` / `EngramLayer` of `hybrid_stack_spec` | one pattern symbol: `h + sublayer(input_layernorm(h))`; Engram: `h + Engram(h, ids)` |
| `OellmGatedDeltaNet` | `ssm/gated_delta_net/gdn.py`, `common.py` | `in_proj` → causal conv + SiLU → l2-norm q, k → repeat to 20 value heads → `g = -exp(A_log)·softplus(alpha + dt_bias)`, `beta = sigmoid` → gated delta rule (FLA kernel) → `RMSNorm(o)·silu(z)` → `out_proj` |
| `OellmGatedAttention` | `transformer/attention.py` `SelfAttention` | `linear_qkv` split per KV group into q, gate, k, v → per-head q/k RMSNorm → RoPE on 32 of 128 dims (interleaved pairs, non-interleaved angles) → causal SDPA (GQA 10:2) → `· sigmoid(gate)` (fp32) → `linear_proj` |
| `OellmRouter` | `moe/router.py`, `moe_utils.topk_routing_with_score_function` | fp32 `sigmoid(x Wᵀ)`, top-4 of `scores + expert_bias`, weights = unbiased scores / their sum |
| `OellmMoE` | `moe/moe_layer.py`, `experts.py` (TEGroupedMLP), `shared_experts.py` | per expert `fc2((silu(gate)·up)·weight)`, summed; plus `fc2(silu(gate)·up)·sigmoid(x gᵀ)` (shared expert) |
| `OellmEngramMemory` | `ssm/engram.py` `EngramLayer` | hash rows → table lookup → `k, v` → `alpha = sigmoid(signed-sqrt(RMSNorm(h)·RMSNorm(k)/√d))` → `v~ = alpha·v` → `silu(conv(RMSNorm(v~))) + v~` |
| `OellmRotaryEmbedding`, `_apply_rope` | `models/common/embeddings/rotary_pos_embedding.py` (as built by `hybrid_model.py`), `rope_utils.apply_rotary_pos_emb` → TE fused RoPE | `inv_freq = 1/θ^(2i/32)`; per channel d: `t[d]·cos F[d] + rot(t)[d]·sin F[d]`, `F = [f0 … f15, f0 … f15]`, `rot` pairing `(2i, 2i+1)` |
| `OellmEngramCache` | (Megatron has no inference path for GDN / Engram) | per-layer decoding state: attention KV; GDN conv inputs + delta-rule state; Engram last ids + conv inputs |

Design choices:

- **Fused layouts kept.** Attention `linear_qkv` (per KV group `[5 q | 5 gate | k | v] × 128`), GDN
  `in_proj` (`[q | k | v | z | beta | alpha]`) and the stacked experts `[E, out, in]` stay one tensor
  each, split at run time the way Megatron splits them. The conversion (stage 4) is then renames, one
  concatenation (the GDN sections, which the checkpoint stores split) and no re-layout. These are the
  HF names it has to produce (from `check_modeling.py`, rho 40):

  | HF tensor | Megatron tensor |
  |---|---|
  | `model.embed_tokens.weight` | `embedding.word_embeddings.weight` (tied: also the LM head) |
  | `model.norm.weight` | `decoder.final_norm.weight` |
  | `layers.i.input_layernorm.weight` | `L`: `self_attention.in_proj.layer_norm_weight`; `*`: `self_attention.linear_qkv.layer_norm_weight`; `E`: `pre_mlp_layernorm.weight` |
  | `layers.i.linear_attn.in_proj.weight` [3880, 640] | `self_attention.in_proj.weight.{query,key,value,z,beta,alpha}` concatenated |
  | `layers.i.linear_attn.conv1d.weight` [2560, 1, 4] | `self_attention.conv1d.weight.{query,key,value}` concatenated |
  | `layers.i.linear_attn.{A_log, dt_bias, out_norm.weight, out_proj.weight}` | `self_attention.{same}` |
  | `layers.i.self_attn.qkv_proj.weight` / `o_proj` / `q_norm` / `k_norm` | `self_attention.linear_qkv.weight` / `linear_proj` / `q_layernorm` / `k_layernorm` |
  | `layers.i.mlp.router.{weight, expert_bias}` | `mlp.router.{weight, expert_bias}` |
  | `layers.i.mlp.experts_gate_up` [40, 640, 640] / `experts_down` [40, 640, 320] | `mlp.experts.experts.linear_fc1.weight` / `linear_fc2.weight` |
  | `layers.i.mlp.shared_gate_up.weight` / `shared_down.weight` / `shared_gate.weight` | `mlp.shared_experts.linear_fc1.weight` / `linear_fc2.weight` / `gate_weight` |
  | `layers.i.engram.{table, key_proj, value_proj, hidden_norm, key_norm, conv_norm, conv1d}` | `{same}` |
  | `layers.i.engram.token_map` (230m runs) | not in the checkpoint: the `engram_token_map` `.npy` |

- **Numerics as in training.** RMSNorms in fp32 (backbone zero-centred `1 + w`, Engram plain `w`); the
  router in fp32 and kept fp32 under any model dtype (`OellmRouter._apply`, `_keep_in_fp32_modules`): a
  bf16 expert bias would move near-tie top-k choices; the attention gate, RoPE and the GDN output norm
  in fp32; routing weights applied after the expert activation, as TEGroupedMLP does. The Engram table
  may be stored in bf16 without loss for inference: Megatron casts the looked-up rows to bf16
  (`self._memory(...).to(h.dtype)`) before using them.
- **Same kernels.** On the GPU the gated delta rule runs FLA's `chunk_gated_delta_rule` (prefill) /
  `fused_recurrent_gated_delta_rule` (one-token steps), the kernels Megatron trains with; on the CPU a
  reference recurrence in fp32 (used by the tests).
- **Padding and caching.** A left-padded batch (`attention_mask` zeros) gives the logits of the
  unpadded sequences: padded keys are masked in attention, a padded token's GDN input is zeroed (adds
  nothing to the conv / delta-rule state), and in Engram a padded token hashes as "before the start"
  and adds nothing to the conv. Positions are counted from each sequence's first real token
  (`attention_mask.cumsum - 1`): this model's RoPE is not shift-invariant (stage 5), so a left-padded
  prompt must see the positions an unpadded one sees. `generate` uses `OellmEngramCache`; prefill and
  decode share one code path.
- **No hidden buffers.** RoPE frequencies, Engram hash multipliers and table offsets are recomputed or
  plain Python values, never non-persistent buffers: `from_pretrained` builds on the meta device and
  would leave those uninitialised (silently wrong, no error).

### Checks (`check_modeling.py`) and results

```bash
GPUS=1 $D/run_in_container.sh check_modeling \
    --config-json $D/results/hf_config_rho40_cons --expect-numel 1174548320
```

Full output: [`results/check_modeling.txt`](results/check_modeling.txt). **21/21 passed** on a GH200
(after the stage 5 RoPE fix):

| Check | Result |
|---|---|
| RoPE vs HybridModel's: its `RotaryEmbedding` construction + `apply_rotary_pos_emb`, TE fused (training) and unfused, `rotary_interleaved` true and false, bf16 | fused: max diff 0; unfused: 1.6e-2 (bf16 rounding of N(0,1) inputs) |
| negative control: the textbook interleaved angles `[f0, f0, f1, f1, …]` | differ by up to 5.3 (the stage 3 version used these) |
| Engram hash rows vs Megatron's `engram_hash` (token map, 2 orders × 2 heads) | 600 / 600 rows equal |
| Router vs Megatron's `topk_routing_with_score_function` (sigmoid, expert bias) | max diff 0 |
| GDN on FLA's recurrent kernel (70 tokens) vs the fp32 reference recurrence | 2.3e-7 relative |
| GDN on FLA's chunk kernel vs the reference | 4.7e-4 relative (TF32 tensor cores) |
| tiny model (pattern `LENLE*ELENE*E`, fp32): cached decoding / chunked prefill / left padding vs one full forward | ≤ 4.2e-6 |
| `save_pretrained` + `from_pretrained(trust_remote_code=True)` | identical logits |
| `from_pretrained(dtype=bf16)` | router fp32, experts bf16 |
| greedy `generate`, cached vs uncached | same 12 tokens |
| whole model on the GPU (fp32, recurrent GDN kernel) vs the CPU | max log-prob diff 2.5e-5 |
| whole model on the GPU (fp32, chunk GDN kernel, TF32 inside) vs the CPU | KL 7.5e-4 nats, below the bf16 baseline |
| whole model on the GPU in bf16 vs CPU fp32 | KL 4.4e-3 nats, vs 4.2e-3 for bf16 rounding alone (CPU bf16) |
| model built from `results/hf_config_rho40_cons/config.json` | 1,174,548,320 elements = the checkpoint |

These check the code against Megatron's functions and against itself; that the converted *weights*
reproduce Megatron's *outputs* is stage 5 (layer-by-layer parity against a Megatron forward).

Lessons from writing the checks, worth knowing for any port:

- **TF32 in NVIDIA containers.** The NGC images set `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`: every fp32
  matmul / conv on the GPU runs in TF32 (relative error 2.5e-4 instead of 3e-7 for a 512×512 matmul).
  An fp32 GPU-vs-CPU comparison measures that unless it is turned off
  (`torch.backends.cuda.matmul.allow_tf32 = False`, `torch.backends.cudnn.allow_tf32 = False`). FLA's
  Triton chunk kernel uses TF32 in its own matmuls regardless.
- **`model.to(bf16)` then `.to(fp32)` is not a round trip:** the weights stay rounded to bf16. Compare
  dtypes on copies of the model.
- **`layer_types`** is a reserved config name in transformers 5 (validated against its attention kinds),
  hence `block_types`.
- **Test a function the way the model calls it.** The first RoPE check built `RotaryEmbedding(
  rotary_interleaved=True)` itself and passed; HybridModel builds it *without* that argument. Only the
  stage 5 comparison with the real model found it. The check now copies HybridModel's construction.

## Stage 4: converting a checkpoint

```bash
D=oellm_autoexp/hf_export/engram_scaling_clariden
RUN=output/engram_100m_1b_cons_E40_rho40_N1_clariden
TIME=00:30:00 $D/run_in_container.sh export_hf --run-dir $RUN --iter 3000 --out-dir $RUN/hf/iter_0003000
GPUS=1 TIME=00:40:00 $D/run_in_container.sh check_hf_export $RUN/hf/iter_0003000
```

`export_hf.py` options: `--iter` (default: the run's `latest_checkpointed_iteration.txt`),
`--engram-table-dtype bf16|fp32` (default bf16), `--tokenizer` (default: the run's `tokenizer_model`),
`--overwrite`. The resolved config is the run's newest `config-*.yaml`. No GPU needed; a 100m run
takes about 40 s, a 230m run about a minute.

Use the result like any HF model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
d = "output/engram_230m_2b3_E40_rho40_N1_clariden/hf/iter_0010842"
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(d, trust_remote_code=True, dtype="bfloat16").cuda()
```

(It needs `flash-linear-attention` on the GPU, as in the training container; the CPU path is plain torch.)

### What the exporter does, step by step

1. **Architecture**: `hf_config.hf_config` on the run's resolved config, with its guards (stage 2).
2. **Read**: the model tensors from the DCP checkpoint, one Megatron layer at a time
   (`iter_model_tensors`): from `.metadata`, an empty tensor of each tensor's global shape and dtype,
   filled by `torch.distributed.checkpoint.load(no_dist=True)` -- one process, no GPU, the TP / EP / DP
   shards gathered by DCP. The optimizer state (80% of the checkpoint) is never read.
3. **Map** (`convert_layer`): the README stage 3 table. Renames, plus one transform: the GDN `in_proj`
   sections concatenated as `query, key, value, z, beta, alpha` and the `conv1d` sections as
   `query, key, value` -- the order Megatron's forward splits them in. Every Megatron tensor of a layer
   must be consumed, and the layers must match the pattern, or the export stops.
4. **Dtypes**: the bf16 model weights as they are -- the training forward's weights, not the fp32
   master copies in the optimizer state; the router weight and `expert_bias` in fp32 (bf16 → fp32 is
   exact); the Engram table, stored fp32 in the checkpoint, in bf16 by default (lossless for inference,
   see stage 3) or fp32.
5. **Token map** (230m runs): `engram_token_map` loaded from its `.npy` and stored per Engram layer as
   `engram.token_map` (int64), its sha256 recorded. The export no longer depends on the scratch file.
6. **Write**: sharded safetensors (`model-0000N-of-0000M.safetensors`, 5 GB shards) and
   `model.safetensors.index.json` (`fsdp_dtensor.ShardedSafetensorsWriter`, which also refuses NaN / Inf:
   a checkpoint read while it is being written deserialises to garbage without an error);
   `config.json` + `generation_config.json`; the model code; the tokenizer files; `megatron_source.json`
   (run, checkpoint, iteration, config, token map path + sha256, table dtype, the oellm-autoexp and
   Megatron commits, date). Files are made group-readable (safetensors writes them owner-only).
7. **Tokenizer pad token**: the tokenizer's `<pad>` is id 262144, one past the 262144-row embedding, so
   any padded batch (lm-eval's, `generate` on several prompts) would index past the embedding (a CUDA
   device-side assert, found in stage 5). `pad_with_eos` sets the pad token of the copied
   `tokenizer_config.json` / `special_tokens_map.json` to `<eos>` (id 2), as `config.json` already does;
   the vocabulary is unchanged, and `megatron_source.json` records the replacement.

### Checks (`check_hf_export.py`) and results

1. **Structure** (no weights loaded): config loads, model builds on the meta device, names and shapes
   equal the safetensors headers; every file dtype follows the policy of step 4.
2. **Conservation**: the source checkpoint is read again and every Megatron tensor is looked up in the
   export through an inverse of the name table written separately from `convert_layer` (the GDN
   sections at their row offsets). Values must be bitwise equal after the dtype cast, and every exported
   element covered exactly once. The token map is compared with its `.npy` and sha256.
3. **Strict load**: `from_pretrained(dtype=bf16, output_loading_info=True)` reports nothing missing,
   unexpected or mismatched; the router is fp32 after loading.
4. **Forward**: mean NLL of an 84-token multilingual text (`validate_hf_export.PROBE_TEXT`) far below
   chance (`log V = 12.48`); a layout mistake lands near chance.
5. **Generation**: greedy continuations of three prompts. Cached decoding and a full forward round
   differently in bf16, so tokens may differ at near ties: the check compares each cached step's
   distribution with a full forward over the same tokens (mean KL < 5e-3) and requires every argmax flip
   to be a near tie (log-prob gap < 0.5).

Exports (re-made on 2026-11-21 after the stage 5 RoPE fix; each under `output/<run>/hf/iter_<N>`),
full outputs in `results/`:

| Run | Iteration | Layers / Engram | Elements (all conserved) | Size | Probe NLL | Cached vs full: KL, flips |
|---|---|---|---|---|---|---|
| `engram_100m_1b_cons_E40_rho40_N1_clariden` | 3000 | 33 / [2] | 1,174,548,320 | 2.3 GB | 3.041 | 2.7e-4, 0 |
| `engram_100m_1b_E40_rho40_N1_clariden` (final) | 4819 | 34 / [2, 17] | 1,175,866,336 | 2.3 GB | 2.547 | 4.8e-4, 0 |
| `engram_230m_2b3_E40_rho40_N1_clariden` (final, token map) | 10842 | 74 / [2, 37] | 2,432,694,552 | 4.6 GB | 1.759 | 4.8e-4, 0 |

7/7, 7/7 and 8/8 checks passed (the 230m one adds the token map check: 262,144 token ids → 175,230
canonical ids, equal to the `.npy`). Example continuations of the final 230m model:

```text
'The capital of France is' -> ' Paris.\n\nThe capital of France is Paris.\n\nThe capital of France is Paris. ...'
'Die Hauptstadt von Deutschland ist' -> ' die Hauptstadt der Bundesrepublik Deutschland. Die Hauptstadt ist die Hauptstadt ...'
'def fibonacci(n):' -> '\n    """\n    Returns the Fibonacci sequence.\n    """\n    return fibonacci(n)\n\n\n'
```

**These checks passed before the RoPE fix too** (probe NLL 3.06 / 2.59 / 2.00 with the wrong RoPE,
3.04 / 2.55 / 1.76 with the right one): the export was conserved, loadable and far below chance, and
still a different model from the trained one. The probe NLL is a smoke test; stage 5 is the proof.

Two bugs the first runs found:

- **Shared memory in safetensors.** The token map tensor was added to both Engram layers as the same
  object; `safetensors` refuses to write two names that share memory. Each layer now gets a copy.
- **Exact cached-vs-uncached equality is the wrong test in bf16.** The first version compared greedy
  tokens and failed on the final 100m model at a 0.06-nat near tie; the fp32 tiny model of stage 3 had
  already shown the cache logic is exact. The check now compares distributions and allows near-tie flips.

## Stage 5: parity with Megatron

The question of this stage: does the export compute what the trained model computes? Stages 1-4 check
the code against Megatron's functions, and the export against the checkpoint's numbers; neither runs
the trained model. `megatron_parity.py` does: it builds the Megatron model exactly as training did,
loads the checkpoint, and runs it and the export on the same validation tokens.

```bash
LAUNCHER="torchrun --standalone --nproc-per-node 1" GPUS=1 TIME=01:00:00 \
    $D/run_in_container.sh megatron_parity --export-dir $RUN/hf/iter_0003000 \
    --out $D/results/parity_rho40_cons_iter3000.json
```

Options: `--samples` (logits / NLL, default 64 = 262,144 tokens), `--layer-samples` (per layer, default
2), `--ablation-samples` (default 32), `--start` (first validation sample), `--out` (a JSON of all
numbers). About 6 minutes for a 100m run on one GH200 (most of it building Megatron and loading the
datasets), somewhat longer for a 230m run.

### How Megatron is run outside training

1. **The training's command line**, from the run's `job.sbatch` (everything after `pretrain_hybrid.py`),
   minus wandb / tensorboard / `--save`, plus `--no-load-optim --no-load-rng`; the tokenizer as its local
   snapshot (offline, the hub id needs a `config.json` the tokenizer repo does not have).
2. `parse_and_validate_args` + `initialize_megatron` (one process: `torchrun --nproc-per-node 1`), the
   model from `pretrain_hybrid`'s own `model_provider` / `hybrid_builder`
   (`get_model(..., wrap_with_ddp=False)`), and `load_checkpoint(strict=True)` at the export's iteration
   (`args.ckpt_step`). The checkpoint was saved with 4 data-parallel ranks (Engram table rows sharded over
   them); DCP reshards it to one.
3. **The validation samples** from `pretrain_hybrid.train_valid_test_datasets_provider` with the
   training's sample counts (`get_train_valid_test_num_samples`), so the same cached index and samples.
   These runs use no inter-document masking (`dataloader_inter_document_masking: false`,
   `reset_position_ids` / `reset_attention_mask: false`, `eod_mask_loss: false`): a sample is one
   4096-token sequence with documents separated by EOD, positions 0 … 4095, every label counted -- what
   the HF model computes on the same 4096 ids, with no per-document splitting.

### What is compared

1. **Every layer's output** (forward hooks on `decoder.layers[i]` / `model.layers[i]` and the final
   norms), Megatron bf16 and HF bf16, each as a relative L2 error against an HF fp32 reference (TF32
   off). If Megatron's error equals the HF model's own bf16 error, both compute the same function and
   differ only by rounding; a mapping mistake shows as a jump at its layer.
2. **Logits** on 262,144 tokens: KL(Megatron ‖ HF), top-1 agreement (pass: KL < 2e-3, agreement ≥ 98%,
   the calibration of `validate_hf_export.py`), and the NLL of both on the labels (pass: |Δ| < 0.01).
   Megatron's logged validation loss of that iteration is shown for scale (2560 other samples).
3. **Engram ablations** (HF): the NLL with the Engram outputs zeroed, and with the hash multipliers
   `convert_megatron_to_hf.py` would compute (no per-layer seed offset); both must be worse.
4. **Tokenizer**: complete documents of the samples, decoded and re-encoded by the export's tokenizer
   with and without `fix_mistral_regex`.
5. **Batched generation**: four prompts of 6-12 tokens, left-padded in one batch, against each alone:
   identical tokens, or a divergence at a near tie (step distributions compared up to the divergence).

### The bug it found: HybridModel's RoPE

The first run (before the fix) matched to bf16 rounding through layer 6 -- GDN, MoE, Engram -- and then:

```text
      layer  type        Megatron bf16   HF bf16      Megatron vs HF   (relative L2 error vs HF fp32)
          6  moe         5.36e-03        5.33e-03     4.46e-03
          7  attention   2.74e-01        5.52e-03     2.74e-01
          8  moe         2.93e-01        6.60e-03     2.93e-01
          ...
  [FAIL] logits: mean KL 3.82e-01 nats, top-1 agreement 71.15%
  [FAIL] NLL: Megatron 2.8805, HF 3.2619
```

Layer 7 is the first attention layer. The cause, in `megatron/core/models/hybrid/hybrid_model.py`:
HybridModel builds `RotaryEmbedding(kv_channels, rotary_percent, ..., rotary_base)` **without**
`rotary_interleaved`, so the per-channel angle table keeps the non-interleaved layout
`F = [f0 … f15, f0 … f15]`. The attention then applies it with `config.rotary_interleaved = True`
(`rope_utils.apply_rotary_pos_emb` → TE's fused RoPE, `apply_rope_fusion`), which computes per channel
`t[d]·cos F[d] + rot(t)[d]·sin F[d]` with `rot` pairing `(2i, 2i+1)`. So the two channels of a pair turn
by *different* angles (`f_{2i mod 16}` and `f_{(2i+1) mod 16}`): not the textbook interleaved RoPE
(`[f0, f0, f1, f1, …]`) the HF port first implemented -- and not a rotation at all. The model was trained
with it, so the export must reproduce it. Two consequences:

- The fix: the HF config separates the rotation's pairing (`rope_interleaved: true`) from the angle
  layout (`rope_interleaved_freqs: false`, always for HybridModel); the stage 3 RoPE check now builds RoPE
  the way HybridModel does and calls Megatron's `apply_rotary_pos_emb` (TE fused and unfused), with the
  textbook layout as a negative control (off by up to 5.3).
- Attention is no longer shift-invariant: scores depend on absolute positions, not only on distances.
  Training always starts positions at 0 at the sample start, so the HF model now counts positions from
  each sequence's first real token (left padding would otherwise shift them; the stage 3 left-padding
  check caught this right after the fix).

The stage 3 RoPE check had passed because it built `RotaryEmbedding(rotary_interleaved=True)` itself.

### Results (after the fix)

Full logs and JSON: `results/parity_*.log`, `results/parity_*.json`. **7/7 checks passed on all three
exports.**

| | 100m consolidated, iter 3000 | 100m two-bank, iter 4819 | 230m two-bank + token map, iter 10842 |
|---|---|---|---|
| per layer: max (Megatron error / HF bf16 error) vs fp32 | 1.03 (34 outputs) | 1.05 (35) | 1.02 (75) |
| logits: mean KL(Megatron ‖ HF) | 4.5e-4 nats | 5.2e-4 | 6.5e-4 |
| logits: top-1 agreement | 98.94% | 98.82% | 98.63% |
| NLL, Megatron / HF (262,144 tokens) | 2.8805 / 2.8806 | 2.7261 / 2.7262 | 2.5094 / 2.5094 |
| Megatron's logged validation loss (other samples) | 2.956 (iter 3000) | 2.769 (iter 4819) | 2.568 (iter 10842) |
| NLL with Engram zeroed | +0.123 | +0.115 | +0.151 |
| NLL with hashes lacking the per-layer offset | +0.162 | +0.155 | +0.189 |
| tokenizer re-encodes 141 documents exactly: fix off / on | 100% / 0% | 100% / 0% | 100% / 0% |
| batched left-padded generate vs single prompts | 3/4 identical, 1 split at an exact bf16 tie (gap 0.000) | 4/4 identical | 4/4 identical |

How to read it:

- **Per layer**, every row of every model has Megatron as close to the fp32 reference as the HF model's
  own bf16 rounding (ratio ≤ 1.05); the error grows smoothly with depth (2e-3 → 1.4e-2), the same for both.
  Excerpt (100m consolidated):

  ```text
      layer  type        Megatron bf16   HF bf16      Megatron vs HF
          0  gdn         1.86e-03        1.84e-03     1.09e-03
          2  engram      3.37e-03        3.38e-03     1.95e-03
          7  attention   5.52e-03        5.48e-03     5.06e-03
         33  final_norm  1.38e-02        1.37e-02     1.59e-02
  ```
- **Logits**: KL 4.5-6.5e-4 and 98.6-98.9% top-1 agreement are the bf16 noise floor (two bf16
  implementations with different kernels); the NLL agrees to 1e-4.
- **The logged validation loss** is on other samples (the 2560 that evaluation drew from the validation
  stream, not the first 64 used here); the 0.04-0.08 gap to the 64-sample NLL is sample variation -- the
  paired Megatron-vs-HF NLL on the same tokens is the parity measure.
- **Engram matters, and its hashing is the trained one**: removing it costs 0.12-0.15 nats; the hashes
  without the per-layer offset are *worse than no Engram at all* (rows looked up at random add noise).
- **Tokenizer**: the training data was tokenized with the regex as shipped. Leave `fix_mistral_regex`
  off (the default) and ignore transformers' warning; turning it on changes every document's ids.
- **Batched generation** with left padding gives the single-prompt results, now that the tokenizer pads
  with `<eos>` (stage 4, step 7) and positions start at each prompt's first token. Where one prompt
  split (100m consolidated), the two candidate tokens had the same bf16 log-probability: a padded batch
  runs other kernel shapes, and an exact tie can go either way. Before the split the step distributions
  agree to KL ≤ 5.2e-4.

## Open items

- **Token map file:** the exports contain the token map, but exporting more 230m checkpoints still needs
  `/capstor/scratch/cscs/swagatam/engram_token_map_256k.npy`, and scratch is purged: keep a copy outside
  scratch.
- **Coverage so far:** three exports, all rho 40 (one Engram bank / two banks, both scales, with and
  without token map), verified through stage 5. The other rho arms (88-56 experts) and the rho 100
  baselines (no Engram layer) use the same code paths, but are not exported or checked yet (stage 6).
- **Downstream evaluation** (lm-eval / oellm-eval) of the exports.
- **Not committed:** `engram_scaling_clariden/` is untracked in git.
- **The conversion depends on this Megatron version's HybridModel.** In particular the RoPE angle layout
  (`rope_interleaved_freqs: false`) is hard-coded from `hybrid_model.py`; if a later Megatron passes
  `rotary_interleaved` to the HybridModel `RotaryEmbedding`, `hf_config.py` must follow, and
  `megatron_parity.py` would show it at the first attention layer.
