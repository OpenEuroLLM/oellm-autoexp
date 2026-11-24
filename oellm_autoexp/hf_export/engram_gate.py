"""Scale-invariant Engram gate query: offline rescore of the per-token diagnosis arrays.

The diagnosis (``scripts/rluukkon/engram_gate_diag.py``) stores, per Engram layer and token, the raw gate
logit ``x = <RMSNorm(h), RMSNorm(k)> / sqrt(d)``, the normalized-hidden RMS ``|h~| = RMS(RMSNorm(h))``, and
the current gate ``alpha = sigmoid(signed_sqrt(x))``. The proposed fix replaces the eps-limited query norm
with a scale-invariant one (eps ~ 1e-12), which divides the query by ``|h~|``, and adds a learnable
``logit_scale`` before the signed square root::

    x_fixed = x * logit_scale / |h~|,   gate = sigmoid(signed_sqrt(x_fixed))

At ``logit_scale = |h~|`` this reproduces the current logit exactly per token; at the layer mean it
reproduces it on average. This module recomputes the gate and its rank correlation with the layer's
per-token utility ``dNLL = NLL(zeroed) - NLL(base)`` from the stored arrays, without a model forward.

The arrays are float32 and the current gate uses ``clamp_min(1e-6)`` before the square root, so the same
clamp is applied here.
"""

from __future__ import annotations

import numpy as np

CLAMP = 1e-6
HN_FLOOR = 1e-12


def signed_sqrt(x, clamp: float = CLAMP):
    """``sign(x) * sqrt(max(|x|, clamp))``, as the Megatron / HF Engram gate computes it."""
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.sqrt(np.maximum(np.abs(x), clamp))


def gate_from_logit(x, clamp: float = CLAMP):
    """``sigmoid(signed_sqrt(x))``."""
    return 1.0 / (1.0 + np.exp(-signed_sqrt(x, clamp)))


def fixed_query_logit(x, hn_rms, logit_scale, hn_floor: float = HN_FLOOR):
    """The fixed query's logit: ``x * logit_scale / |h~|``.

    ``hn_rms`` is the RMS of the eps-limited normalized hidden (the stored ``hn``). Dividing by it turns
    that query into a unit-RMS one, so ``logit_scale`` sets the size instead of the hidden magnitude.
    """
    x = np.asarray(x, dtype=np.float64)
    hn = np.maximum(np.asarray(hn_rms, dtype=np.float64), hn_floor)
    return x * float(logit_scale) / hn


def fixed_query_gate(x, hn_rms, logit_scale, clamp: float = CLAMP, hn_floor: float = HN_FLOOR):
    """``sigmoid(signed_sqrt(x * logit_scale / |h~|))``, the gate of the proposed fix."""
    return gate_from_logit(fixed_query_logit(x, hn_rms, logit_scale, hn_floor), clamp)


def align_to_targets(values, doc_lengths):
    """Cut a per-token record to the positions that predict a token.

    The record has ``doc_length + 1`` entries per document (BOS first); the NLL arrays have
    ``doc_length`` entries per document. Keep the first ``doc_length`` entries of each document and drop
    the final token.
    """
    values = np.asarray(values)
    out = []
    offset = 0
    for n in np.asarray(doc_lengths, dtype=np.int64):
        n = int(n)
        out.append(values[offset:offset + n])
        offset += n + 1
    if not out:
        return np.empty(0, dtype=values.dtype)
    return np.concatenate(out)


def ranks(a):
    """Average-free ordinal ranks, ties broken by position (stable), as the report script uses."""
    a = np.asarray(a, dtype=np.float64)
    r = np.empty(len(a), dtype=np.float64)
    r[np.argsort(a, kind="stable")] = np.arange(len(a))
    return r


def spearman(a, b):
    """Spearman rank correlation; ``nan`` when either side is constant or shorter than three."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 3 or len(a) != len(b):
        return float("nan")
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    ra, rb = ranks(a), ranks(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    if denom <= 0:
        return float("nan")
    return float((ra * rb).sum() / denom)


def gate_summary(alpha, utility):
    """Spread of the gate and its relation to the layer's per-token utility."""
    alpha = np.asarray(alpha, dtype=np.float64)
    utility = np.asarray(utility, dtype=np.float64)
    finite = np.isfinite(alpha) & np.isfinite(utility)
    a, u = alpha[finite], utility[finite]
    if a.size == 0:
        return {"tokens": 0}
    lo, mid, hi = np.percentile(a, [10, 50, 90])
    top = u[a >= hi]
    bottom = u[a <= lo]
    return {
        "tokens": int(a.size),
        "gate_mean": float(a.mean()),
        "gate_p10": float(lo),
        "gate_p50": float(mid),
        "gate_p90": float(hi),
        "gate_spread": float(hi - lo),
        "rho": spearman(a, u),
        "utility_top_decile": float(top.mean()) if top.size else float("nan"),
        "utility_bottom_decile": float(bottom.mean()) if bottom.size else float("nan"),
    }


def engram_layers(npz) -> list[int]:
    """The Engram layers present in a diagnosis ``.npz`` (keys ``alpha_L<i>``)."""
    layers = []
    for key in npz.files:
        if key.startswith("alpha_L"):
            layers.append(int(key[len("alpha_L"):]))
    return sorted(layers)


def rescore(npz, logit_scales, sweep_layer: int | None = None) -> dict:
    """Recompute the gate and its gate/utility relation at each ``logit_scale``.

    ``sweep_layer`` is the layer whose ``logit_scale`` takes each value in ``logit_scales``; the other
    layers keep ``logit_scale = mean(|h~|)`` (their function-preserving init). With ``sweep_layer=None``
    every layer takes each value.

    Returns ``{"layers", "sweep_layer", "hn_mean", "detail"}``; ``detail[i]`` holds ``current`` and
    ``fixed`` (per logit_scale) summaries.
    """
    layers = engram_layers(npz)
    doc_lengths = np.asarray(npz["doc_lengths"], dtype=np.int64)
    detail: dict[int, dict] = {}
    hn_mean: dict[int, float] = {}
    for i in layers:
        x = align_to_targets(npz[f"x_L{i}"], doc_lengths)
        hn = align_to_targets(npz[f"hn_L{i}"], doc_lengths)
        alpha = align_to_targets(npz[f"alpha_L{i}"], doc_lengths)
        utility = np.asarray(npz[f"zero_L{i}"], dtype=np.float64) - np.asarray(npz["base"], dtype=np.float64)
        hn_mean[i] = float(np.asarray(npz[f"hn_L{i}"], dtype=np.float64).mean())
        current = gate_summary(alpha, utility)
        fixed: dict[float, dict] = {}
        for scale in logit_scales:
            if sweep_layer is not None and i != sweep_layer:
                scale_i = hn_mean[i]
            else:
                scale_i = float(scale)
            gate = fixed_query_gate(x, hn, scale_i)
            summary = gate_summary(gate, utility)
            summary["logit_scale"] = scale_i
            summary["max_abs_gate_change_vs_current"] = float(np.max(np.abs(gate - alpha)))
            nll_key = f"fixed_ls{scale:g}"
            if nll_key in npz.files and (sweep_layer is None or i == sweep_layer):
                # the pass changes every layer, but only the swept one moves; report the total on its row
                summary["nll_jump"] = float(
                    np.mean(np.asarray(npz[nll_key], dtype=np.float64) - np.asarray(npz["base"], dtype=np.float64))
                )
            fixed[float(scale)] = summary
        detail[i] = {"current": current, "fixed": fixed}
    return {"layers": layers, "sweep_layer": sweep_layer, "hn_mean": hn_mean, "detail": detail}


def rescore_report(result: dict, source: str = "") -> str:
    """Markdown report of :func:`rescore`."""
    lines = []
    if source:
        lines.append(f"# Fixed-query rescore: {source}\n")
    sweep = result["sweep_layer"]
    lines.append(
        "The fixed query is `x * logit_scale / |h~|`; the current gate is `x` with `logit_scale = |h~|` "
        "per token. "
        + (f"Layer {sweep} is swept; the other layers keep `logit_scale = mean(|h~|)`.\n" if sweep is not None else "\n")
    )
    for i in result["layers"]:
        d = result["detail"][i]
        cur = d["current"]
        lines.append(f"## Engram layer {i}  (mean |h~| {result['hn_mean'][i]:.4g})\n")
        lines.append(
            "| logit_scale | gate p10/50/90 | spread | rho(gate, dNLL) | dNLL top vs bottom decile | max |Δgate| | ΔNLL |"
        )
        lines.append("|---:|---|---:|---:|---|---:|---:|")
        lines.append(
            f"| current | {cur['gate_p10']:.3f}/{cur['gate_p50']:.3f}/{cur['gate_p90']:.3f} | "
            f"{cur['gate_spread']:.3f} | {cur['rho']:+.3f} | "
            f"{cur['utility_top_decile']:+.4f} vs {cur['utility_bottom_decile']:+.4f} | - | - |"
        )
        for scale, s in sorted(d["fixed"].items()):
            jump = f"{s['nll_jump']:+.4f}" if "nll_jump" in s else "-"
            lines.append(
                f"| {scale:g} | {s['gate_p10']:.3f}/{s['gate_p50']:.3f}/{s['gate_p90']:.3f} | "
                f"{s['gate_spread']:.3f} | {s['rho']:+.3f} | "
                f"{s['utility_top_decile']:+.4f} vs {s['utility_bottom_decile']:+.4f} | "
                f"{s['max_abs_gate_change_vs_current']:.3f} | {jump} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"
