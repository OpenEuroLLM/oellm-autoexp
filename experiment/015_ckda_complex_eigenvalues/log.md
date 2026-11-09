# 015: are complex eigenvalues used in v4's CKDA layers?

## Question (user, 2026-11-09)

The ambitious v4_2ph main run has 42 CKDA layers (16 heads, head dim 128; signed_sigmoid2 alpha in
[-1, 1], beta_max 2). Does the trained model use complex eigenvalues in its state transitions, and to
what extent: which layers, heads, positions, tokens, domains?

## Math

Per head and token, CKDA applies M_t = (I - beta k k^T) Diag(alpha) (k unit, alpha in [-1, 1]^128 per
channel, beta in [0, 2]). With f(lam) = 1 - beta sum_i alpha_i k_i^2 / (alpha_i - lam),
det(M - lam I) = prod_i (alpha_i - lam) f(lam). f has a real root between every two consecutive
same-sign alphas, so:

- **at most one complex-conjugate pair per step** (the two roots not bracketed by same-sign poles);
- **a pair needs beta > 1 AND alpha of both signs**. For beta <= 1, M is similar to the symmetric
  P^1/2 D P^1/2. For alpha of one sign, M is similar to a symmetric matrix as well.

So the per-step spectrum is 126 real eigenvalues (near the alphas: negative alphas give reflections)
plus at most one rotation r e^{+-i theta}. The pair is computed exactly: bisection for the bracketed
roots, then sum = trace - sum(roots) and product = det / prod(roots). Checks:
- against torch.linalg.eigvals on 512 random signed transitions: 0 disagreements, max error 5e-13;
- in situ on real steps in the job (verify.json).

## Method

`scripts/ckda_spectra.py` wraps FLA's `chunk_kda` inside `fla.layers.complex_kda_layer` (the exact
kernel inputs). A forward pre-hook on each CKDA module tells the wrapper which layer runs.

Data: v8ml fixed validation sets. 16 documents each of web_en, code, math, multilingual, reasoning and
long_64k; BOS + up to 8191 tokens (phase-1 context); documents of >= 256 tokens.

Statistics per (layer, head), per (layer, head, log2 position), per (layer, position), per token id:
- complex fraction; complex with |lam| > 0.5;
- |Im lam|, arg and |lam| of the pair;
- beta, the beta > 1 fraction;
- the fraction of channels with alpha < 0, the key mass on them, mean |alpha|.

Plus a complex-plane histogram per layer.

Ablations (NLL change on the same documents):
- abs_alpha: alpha -> |alpha| everywhere;
- beta_le1: beta -> min(beta, 1);
- cplx_abs: alpha -> |alpha| only at the complex steps;
- cplx_beta1: beta -> min(beta, 1) only at the complex steps;
- ctrl_abs: alpha -> |alpha| at as many random mixed-sign real steps (control);
- cplx_abs per group of 7 CKDA layers.

Job: config/experiments/korbi/production/ckda_spectra_mn5.yaml, a 1-node pool run in p264 (priority
60, may use excluded nodes), snapshot autoexp_ckdaspec. Output in
/gpfs/scratch/ehpc533/hybrid_nucleus/analysis/ckda_spectra/<export>. `scripts/summarize.py <out>`
writes summary.json + figures.

Local test (tiny random model of the HF class, dt_bias zeroed to get mixed signs): pipeline, dumps
and ablations run; in-situ check 36 steps, 0 disagreements, max error 1e-15.

## Runs

| export | pool run | status |
|---|---|---|
| ambitious_v4_2ph_hd256_fp8_21000 | p264.0155 (17:50, snapshot autoexp_ckdaspec) | FAILED 18:03: index_add_ across cuda:0 / cuda:1 (position bins on the embedding's GPU; fixed 9aeda1ae; the model loaded in 60 s, self-check passed on H100) |
| ambitious_v4_2ph_hd256_fp8_21000 | p264.0157 (job _249bad, snapshot autoexp_ckdaspec2) | FAILED 18:10 at srun: placed on p264 host003 (Slurm 48488261), an allocation at 128 of 264 nodes (NumTasks 264) whose every step fails with 'Requested node configuration is not available'; reported to the main-run session |
| ambitious_v4_2ph_hd256_fp8_21000 | acc_debug Slurm 48830030 (autoexp session 1794244334, monitor ckda_spectra_dbg, snapshot autoexp_ckdaspec2; account ehpc533, 2 h) | running since 18:16 |
| checkpoint sweep 1000 / 2000 / 3700 / 6000 / 9814 / hd256 14000 / 18000 | p264.0208-0214 (user 2026-11-09: experiments on host003), snapshot autoexp_ckdaspec3, pool_use_excluded_nodes false | all on host003 except 18000 (host002). 18000 was cancelled by me at 20:08 after stats + 3 ablations: it held host002's only free regular node into the phase-2 start (~20:50) |
| 21000 layer groups | p264.0247 (--no-stats, snapshot autoexp_ckdaspec4) | done; redundant with the debug job (which finished in 1:55); identical baselines |
| acc_debug 48830030 | | COMPLETED 1:55 (all ablations) |

## Results (step 21000 unless stated; 96 docs, 277k tokens x 42 layers x 16 heads)

- Complex pairs are common but small:
  - 25.5 % of (token, head) steps have one;
  - mean |lam| 0.16; only 0.16 % of steps have |lam| > 0.5;
  - mean arg 0.59 pi.

  The pair is the damped key direction (the delta-rule erase), coupled across the sign boundary.
- Conditions: beta > 1 on 54 % of steps (mean beta 1.07); alpha with both signs on 59 %. Only 1.1 % of
  channels have alpha < 0, yet 16 % of the key mass falls on them.
- Layers: the first CKDA layers are at ~45 %, the last at 8-16 %, falling together with beta > 1.
- Heads: strongly specialized; within a layer, from 0-10 % to 40-96 %.
- Positions: BOS 62 %, the first tokens ~50 %, ~25 % beyond 1k.
- Domains: web_en 31 > multilingual 29 > long 26 > math 25 > reasoning 23 > code 20 %.
- Tokens: highest on clause-initial words and BOS / EOS; lowest on predictable code / format continuations.
- Ablations at 21000, mean over the 6 domains, nats:

  | ablation | delta |
  |---|---|
  | all alpha -> abs(alpha) | +0.0031 |
  | beta <= 1 | +0.031 |
  | complex steps: alpha -> abs(alpha) | +0.0018 |
  | complex steps: beta -> 1 | +0.016 |
  | random control | +0.0008 |
  | per group of 7 layers | <= +0.001 each, spread across depth |
- Over training, complex fraction / beta > 1 / mean beta:

  | step | complex | beta > 1 | mean beta |
  |---|---|---|---|
  | 1000 | 1.1 % | 4 % | 0.30 |
  | 2000 | 2.2 % | 6 % | 0.34 |
  | 3700 | 4.8 % | 10 % | 0.44 |
  | 6000 | 9.7 % | 20 % | 0.58 |
  | 21000 | 25.5 % | 54 % | 1.07 |

  The beta <= 1 ablation grows from +0.0008 to +0.031. The sign / complex ablations stay below +0.004.
- Interpretation: the extended beta range (reflection, an overshooting erase) is what the model uses.
  The complex rotations exist, but they are heavily damped and contribute little.
- Limits: per-step spectra only (multi-step products can rotate even with alpha >= 0). The
  complex-steps beta -> 1 ablation also changes the write strength.

Report: scripts/collect.py + build_report.py (report_template.html) -> published artifact.
