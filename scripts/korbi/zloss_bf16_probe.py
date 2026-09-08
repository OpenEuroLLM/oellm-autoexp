#!/usr/bin/env python3
"""How much of the LM-head z-loss gradient survives bf16 rounding?

Megatron's fused ("native") vocab-parallel cross entropy rounds the CE logit gradient to
bf16 BEFORE the z-loss gradient is added to it:

    megatron/core/fusions/fused_cross_entropy.py
        calculate_gradients(...)  ->  grad_input = grad_input.to(torch.bfloat16)   # line 82
        backward(...)             ->  grad_input = grad_input + logsumexp_grad.to(bf16)

For a non-target vocab entry both terms are proportional to the same softmax probability:

    d(CE)/d(logit_v)      = w * softmax_v
    d(z_loss)/d(logit_v)  = w * 2 * coeff * logZ * softmax_v

so their RATIO is the constant  r = 2 * coeff * logZ,  independent of the token and of the
vocabulary entry. bf16 keeps 8 significant bits, so a value already ON the bf16 grid moves
only if perturbed by more than half an ulp, i.e. by more than 2**-9 .. 2**-8 in relative
terms. When r falls below 2**-9 = 1.95e-3 the addition CANNOT change a single element and
the regularizer contributes exactly nothing to the weights.

The flagship runs coeff=1e-4 at logZ~8 (the logged mean(logZ**2) sits at 59-67), i.e.
r ~ 1.6e-3 -- below the floor. This script measures that rather than asserting it, and
reports what changes when the two terms are summed in fp32 before rounding, which is what
`--output-z-loss-fp32-grad-accum` does.

Note logZ is a free parameter here: shifting every logit by a constant moves logZ without
touching the softmax or the cross entropy, and r depends on logZ alone. That is the whole
point -- the z-loss exists to control logZ, and logZ is what decides whether it can.

    python3 scripts/korbi/zloss_bf16_probe.py
    python3 scripts/korbi/zloss_bf16_probe.py --logz 8.0 --coeff 1e-4 1e-3
"""

import argparse

import torch


def build_logits(vocab: int, tokens: int, target_ce: float, target_logz: float, seed: int):
    """Logits with a prescribed cross entropy AND a prescribed log-
    normalizer."""
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(tokens, vocab, generator=g, dtype=torch.float32) * 2.0
    target = torch.randint(0, vocab, (tokens,), generator=g)
    rows = torch.arange(tokens)

    # Boost the target entry until the cross entropy matches. Fixed point in ~10 passes.
    for _ in range(60):
        logZ = torch.logsumexp(logits, dim=-1)
        logits[rows, target] += (logZ - logits[rows, target]) - target_ce

    # A uniform per-token shift moves logZ and leaves softmax (hence CE) untouched.
    logits += (target_logz - torch.logsumexp(logits, dim=-1)).unsqueeze(-1)
    return logits, target, rows


def probe(logits, target, rows, coeff: float) -> dict:
    """Compare the two accumulation orders against the fp32 truth."""
    tokens = logits.size(0)
    logZ = torch.logsumexp(logits, dim=-1)
    softmax = torch.softmax(logits, dim=-1)

    # Per-token loss weight w = d(total)/d(loss_token). It multiplies both terms equally, so
    # it cannot affect the ratio; it is kept explicit so the magnitudes are realistic.
    w = torch.full((tokens, 1), 1.0 / tokens, dtype=torch.float32)

    g_ce = softmax.clone()
    g_ce[rows, target] -= 1.0
    g_ce = g_ce * w
    g_z = (2.0 * coeff * logZ).unsqueeze(-1) * softmax * w

    baseline = g_ce.bfloat16()  # outcome if the z-loss did nothing
    off = (g_ce.bfloat16() + g_z.bfloat16()).bfloat16()  # current code path
    on = (g_ce + g_z).bfloat16()  # --output-z-loss-fp32-grad-accum

    # The aggregate that actually reaches the weights: the LM-head weight gradient is a
    # contraction of the logit gradient, so what matters is how much of the intended
    # perturbation survives in SUM, not element by element. 1.0 = fully delivered.
    def delivered(x):
        return ((x.float() - baseline.float()).sum() / g_z.sum()).item()

    return {
        "logZ": logZ.mean().item(),
        "ratio": (2.0 * coeff * logZ.mean()).item(),
        "moved_off": (off != baseline).float().mean().item(),
        "moved_on": (on != baseline).float().mean().item(),
        "delivered_off": delivered(off),
        "delivered_on": delivered(on),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vocab", type=int, default=256000, help="full vocabulary size")
    ap.add_argument("--tokens", type=int, default=512, help="tokens to average over")
    ap.add_argument(
        "--coeff",
        type=float,
        nargs="+",
        default=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
        help="output_z_loss_coeff sweep",
    )
    ap.add_argument("--ce", type=float, default=1.5, help="target cross entropy in nats")
    ap.add_argument("--logz", type=float, default=8.0, help="target log-normalizer")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    logits, target, rows = build_logits(args.vocab, args.tokens, args.ce, args.logz, args.seed)
    ce = (torch.logsumexp(logits, -1) - logits[rows, target]).mean().item()

    print(f"bf16 half-ulp (relative): 2**-9 = {2**-9:.3e} .. 2**-8 = {2**-8:.3e}")
    print(f"vocab={args.vocab}  tokens={args.tokens}  CE={ce:.4f} nats  logZ={args.logz}\n")
    print(
        f"{'coeff':>8} {'ratio r':>9}  {'r vs floor':>11} | "
        f"{'% logit-grad elements moved':>29} | {'z-grad delivered':>25}"
    )
    print(
        f"{'':>8} {'':>9}  {'':>11} | {'bf16 accum':>14}{'fp32 accum':>15} | "
        f"{'bf16 accum':>12}{'fp32 accum':>13}"
    )
    print("-" * 100)
    for coeff in args.coeff:
        r = probe(logits, target, rows, coeff)
        verdict = "BELOW" if r["ratio"] < 2**-9 else "above"
        print(
            f"{coeff:>8.0e} {r['ratio']:>9.2e}  {verdict:>11} | "
            f"{100 * r['moved_off']:>13.4f}%{100 * r['moved_on']:>14.4f}% | "
            f"{r['delivered_off']:>12.4f}{r['delivered_on']:>13.4f}"
        )


if __name__ == "__main__":
    main()
