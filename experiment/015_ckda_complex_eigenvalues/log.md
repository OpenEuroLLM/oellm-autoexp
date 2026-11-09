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
| ambitious_v4_2ph_hd256_fp8_21000 | job _249bad (18:10, snapshot autoexp_ckdaspec2) | pending |

## Results

(pending)
