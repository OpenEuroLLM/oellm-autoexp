"""Fail-case MoE metrics — how *our* metrics look on synthetic router failures.

Reproduces the 11 synthetic failure scenarios from the colleague's notebook
(github.com/luukkonenr/moe_metric_examples / dashboard.ipynb) and then runs the
metrics *we* compute for the OLMoE analysis on each one:

  * token-count-per-expert      (compute_moe_metrics_hf._save_token_counts)
  * expert co-activation         (adjacent-rank, _aggregate_coactivation)
  * router saturation vs final   (top-k overlap, aggregate_hf_revisions.compute_router_saturation)

plus the colleague's scalar router-health metrics (cv / entropy / max-prob /
dead-count) for cross-reference.

Pseudo-time via `severity`
--------------------------
Router saturation is intrinsically temporal: it asks "what fraction of a
token's top-k experts already match the *final* checkpoint's choice for that
same token?". A static fail-case snapshot has no trajectory, so we use each
scenario's `severity` knob as pseudo-time: with a *fixed* token-noise seed we
sweep severity 0->1 and compare every step's routing to the severity=1
endpoint. This mirrors the real "vs final revision" plots (curve ends at 100%).

Key structural finding
-----------------------
Top-k-overlap saturation is INVARIANT TO LOGIT SCALING (argsort of a positively
scaled vector is unchanged), so it is blind to `untrained_router`,
`saturated_balanced` and `logit_explosion` -- exactly the pure-scaling failures
that per-token entropy / max-prob catch. The metrics are complementary.

Run headless to dump PNGs:
    python scripts/moe/failcase_metrics.py --out results/moe_failcases
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Config (OLMoE-1B-7B-0924 routing config so coactivation/saturation are real) #
# --------------------------------------------------------------------------- #
E_DEFAULT = 64        # num experts (colleague used 16; OLMoE uses 64)
TOPK_DEFAULT = 8      # experts per token (colleague used k=1; OLMoE uses 8)
N_DEFAULT = 4096      # tokens
SEED_DEFAULT = 0
N_SEVERITY = 9        # pseudo-time grid points in [0, 1]

_BASE_NOISE = 1.0     # per-token logit std for a "normal" router


# --------------------------------------------------------------------------- #
# Scenario generators (verbatim logic from the colleague's cell 3)            #
# --------------------------------------------------------------------------- #
def _healthy_base(E, N, rng):
    return rng.standard_normal((N, E)) * _BASE_NOISE


def scenario_healthy_balanced(E, N, severity, rng):
    return _healthy_base(E, N, rng)


def scenario_untrained_router(E, N, severity, rng):
    sigma = 0.01 + (1.0 - severity) * _BASE_NOISE
    return rng.standard_normal((N, E)) * sigma


def scenario_saturated_balanced(E, N, severity, rng):
    sigma = _BASE_NOISE + severity * 25.0
    return rng.standard_normal((N, E)) * sigma


def scenario_full_collapse(E, N, severity, rng):
    base = _healthy_base(E, N, rng)
    bias = np.zeros(E)
    bias[0] = severity * 15.0
    return base + bias[None, :]


def scenario_partial_collapse(E, N, severity, rng):
    base = _healthy_base(E, N, rng)
    n_winners = max(1, int(np.ceil(severity * E / 4)))
    bias = np.zeros(E)
    bias[:n_winners] = severity * 5.0
    return base + bias[None, :]


def scenario_dead_experts(E, N, severity, rng):
    base = _healthy_base(E, N, rng)
    n_dead = int(np.ceil(severity * E / 2))
    bias = np.zeros(E)
    if n_dead > 0:
        bias[-n_dead:] = -30.0
    return base + bias[None, :]


def scenario_bimodal_specialization(E, N, severity, rng):
    base = _healthy_base(E, N, rng)
    half = E // 2
    bias = np.zeros(E)
    bias[:half] = +severity * 1.0
    bias[half:] = -severity * 1.0
    return base + bias[None, :]


def scenario_heavy_tail(E, N, severity, rng):
    base = _healthy_base(E, N, rng)
    rank = np.arange(E, dtype=float)
    bias = -severity * 3.0 * np.log1p(rank)
    return base + bias[None, :]


def scenario_bias_drift(E, N, severity, rng):
    base = _healthy_base(E, N, rng)
    bias = np.zeros(E)
    bias[0] = +severity * 0.5
    bias[1] = -severity * 0.5
    return base + bias[None, :]


def scenario_premature_specialization(E, N, severity, rng):
    base = _healthy_base(E, N, rng)
    chosen = rng.integers(0, E, size=N)
    boost = np.zeros((N, E))
    boost[np.arange(N), chosen] = severity * 8.0
    return base + boost


def scenario_logit_explosion(E, N, severity, rng):
    base = _healthy_base(E, N, rng)
    return base * (1.0 + severity * 20.0)


SCENARIOS: Dict[str, Callable] = {
    "healthy_balanced": scenario_healthy_balanced,
    "untrained_router": scenario_untrained_router,
    "saturated_balanced": scenario_saturated_balanced,
    "full_collapse": scenario_full_collapse,
    "partial_collapse": scenario_partial_collapse,
    "dead_experts": scenario_dead_experts,
    "bimodal_specialization": scenario_bimodal_specialization,
    "heavy_tail": scenario_heavy_tail,
    "bias_drift": scenario_bias_drift,
    "premature_specialization": scenario_premature_specialization,
    "logit_explosion": scenario_logit_explosion,
}

SCENARIO_DESCRIPTIONS: Dict[str, str] = {
    "healthy_balanced": "Baseline; balanced load, moderate sharpness.",
    "untrained_router": "Router still random; softmax ~ uniform (scale-only).",
    "saturated_balanced": "Per-token one-hot but random winner; load balances (scale-only).",
    "full_collapse": "One expert absorbs all tokens.",
    "partial_collapse": "A few winner experts soak up most traffic.",
    "dead_experts": "Bottom-N experts starved to zero.",
    "bimodal_specialization": "Half preferred / half rarely picked.",
    "heavy_tail": "Smooth Zipf-like usage gradient.",
    "bias_drift": "Growing per-expert bias; token noise still dominates top-1.",
    "premature_specialization": "Sharp per-token top-k but balanced load.",
    "logit_explosion": "Logits scaled up; over-confident softmax (scale-only).",
}


# --------------------------------------------------------------------------- #
# Metric implementations (match our repo's definitions)                       #
# --------------------------------------------------------------------------- #
def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def topk_indices(logits: np.ndarray, k: int) -> np.ndarray:
    """Top-k expert indices per token, sorted by descending logit.

    Sorting matters: our coactivation is adjacent-rank within the top-k list,
    and torch.topk (used by the hooks) returns indices sorted descending.
    """
    k = min(k, logits.shape[1])
    part = np.argpartition(-logits, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(logits.shape[0])[:, None]
    vals = logits[rows, part]
    order = np.argsort(-vals, axis=1)
    return part[rows, order]  # [N, k] descending


def token_counts(idx: np.ndarray, E: int) -> np.ndarray:
    """Per-expert token count == bincount over the flattened top-k indices.
    Matches compute_moe_metrics_hf._save_token_counts (the `singles` vector)."""
    return np.bincount(idx.ravel(), minlength=E).astype(np.int64)


def coactivation(idx: np.ndarray, E: int) -> Tuple[np.ndarray, np.ndarray]:
    """Adjacent-rank co-activation, matching _aggregate_coactivation /
    _normalize_coact: co[i,j] += 1 for every adjacent (rank r, r+1) pair in a
    token's sorted top-k; normalised by single-counts; diagonal zeroed."""
    sg = np.bincount(idx.ravel(), minlength=E).astype(np.float64)
    co = np.zeros((E, E), dtype=np.float64)
    if idx.shape[1] >= 2:
        a = idx[:, :-1].ravel()
        c = idx[:, 1:].ravel()
        np.add.at(co, (a, c), 1.0)
        np.add.at(co, (c, a), 1.0)
    norm = co / np.maximum(sg, 1.0)[:, None]
    np.fill_diagonal(norm, 0.0)
    return norm, sg


def routing_map(idx: np.ndarray, E: int) -> np.ndarray:
    """Bool [N, E] map: which experts are in each token's top-k.
    Matches the per-token routing arrays stored in routing_maps.npz."""
    R = np.zeros((idx.shape[0], E), dtype=bool)
    R[np.arange(idx.shape[0])[:, None], idx] = True
    return R


def saturation(R: np.ndarray, R_final: np.ndarray, k: int) -> float:
    """Mean per-token top-k overlap fraction vs the final-snapshot routing.
    Matches aggregate_hf_revisions.compute_router_saturation."""
    inter = (R & R_final).sum(axis=1)
    return float((inter / k).mean())


# colleague's scalar metrics (formulas from common.py) -----------------------
def load_metrics(counts: np.ndarray) -> Dict[str, float]:
    counts = counts.astype(np.float64)
    tot = counts.sum()
    mean = counts.mean()
    p = counts / tot if tot > 0 else np.zeros_like(counts)
    nz = p[p > 0]
    ent_pct = (-(nz * np.log(nz)).sum() / np.log(len(counts)) * 100.0) if len(nz) else 0.0
    return {
        "cv_pct": (counts.std() / mean * 100.0) if mean > 0 else 0.0,
        "max_frac": (counts.max() / tot) if tot > 0 else 0.0,
        "entropy_pct": ent_pct,
        "dead_count": int((counts == 0).sum()),
    }


def router_metrics(logits: np.ndarray) -> Dict[str, float]:
    P = softmax(logits, axis=1)
    E = logits.shape[1]
    Htok = -(P * np.log(P + 1e-12)).sum(axis=1)
    mean_logit = logits.mean(axis=0)                       # [E]
    spread = mean_logit.std()                              # unbiased=False
    spread_norm = float(spread / (np.abs(mean_logit).mean() + 1e-8))
    return {
        "mean_token_entropy_pct": float(Htok.mean() / np.log(E) * 100.0),
        "mean_max_prob": float(P.max(axis=1).mean()),
        "logit_spread_norm": spread_norm,
    }


# --------------------------------------------------------------------------- #
# Severity sweep                                                              #
# --------------------------------------------------------------------------- #
class SweepResult:
    def __init__(self, name: str):
        self.name = name
        self.sev: List[float] = []
        self.sat: List[float] = []          # saturation vs final
        self.cv: List[float] = []
        self.entropy: List[float] = []
        self.max_prob: List[float] = []
        self.spread: List[float] = []
        self.dead: List[int] = []
        # final-snapshot artefacts (severity = 1.0)
        self.final_counts: np.ndarray = None
        self.final_coact: np.ndarray = None


def run_scenario(name: str, E: int, N: int, topk: int, seed: int,
                 n_sev: int = N_SEVERITY) -> SweepResult:
    """Sweep severity 0->1 (fixed seed => same tokens) and collect metrics.
    Reference for saturation is the severity = 1.0 snapshot."""
    gen = SCENARIOS[name]
    sevs = np.linspace(0.0, 1.0, n_sev)

    def routing_at(sev: float) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)          # same seed => same tokens
        logits = gen(E, N, float(sev), rng)
        idx = topk_indices(logits, topk)
        return logits, idx

    _, idx_final = routing_at(1.0)
    R_final = routing_map(idx_final, E)
    k_eff = min(topk, E)

    res = SweepResult(name)
    for sev in sevs:
        logits, idx = routing_at(sev)
        R = routing_map(idx, E)
        counts = token_counts(idx, E)
        lm = load_metrics(counts)
        rm = router_metrics(logits)
        res.sev.append(float(sev))
        res.sat.append(saturation(R, R_final, k_eff) * 100.0)
        res.cv.append(lm["cv_pct"])
        res.entropy.append(lm["entropy_pct"])
        res.max_prob.append(rm["mean_max_prob"])
        res.spread.append(rm["logit_spread_norm"])
        res.dead.append(lm["dead_count"])

    res.final_counts = token_counts(idx_final, E)
    res.final_coact, _ = coactivation(idx_final, E)
    return res


# --------------------------------------------------------------------------- #
# Plotting                                                                    #
# --------------------------------------------------------------------------- #
def plot_scenario(res: SweepResult, E: int, topk: int, fig=None):
    import matplotlib.pyplot as plt

    if fig is None:
        fig = plt.figure(figsize=(15, 3.4))
    axes = fig.subplots(1, 4)
    sev = np.asarray(res.sev)

    # 1. token count per expert (final snapshot)
    ax = axes[0]
    ax.bar(np.arange(E), res.final_counts, color="#4c72b0", width=1.0)
    ax.axhline(res.final_counts.sum() / E, color="k", ls="--", lw=0.8, alpha=0.6)
    ax.set_title("token count / expert", fontsize=9)
    ax.set_xlabel("expert"); ax.set_ylabel("tokens")

    # 2. coactivation heatmap (final snapshot, top-32 by activity)
    ax = axes[1]
    top = np.argsort(res.final_counts)[::-1][:min(32, E)]
    top = np.sort(top)
    sub = res.final_coact[np.ix_(top, top)] * 100.0
    im = ax.imshow(sub, cmap="magma", vmin=0)
    ax.set_title("co-activation (%) top-32", fontsize=9)
    ax.set_xlabel("expert"); ax.set_ylabel("expert")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 3. saturation vs final, over pseudo-time
    ax = axes[2]
    ax.plot(sev, res.sat, "-o", color="#c44e52", ms=3)
    ax.scatter([sev[-1]], [res.sat[-1]], marker="*", s=120,
               color="#c44e52", zorder=5)
    ax.set_ylim(0, 108)
    ax.set_title("router saturation vs final", fontsize=9)
    ax.set_xlabel("severity (pseudo-time)"); ax.set_ylabel("saturation (%)")
    flat = max(res.sat) - min(res.sat) < 1.0
    if flat:
        ax.text(0.5, 50, "scale-invariant:\nblind to this failure",
                ha="center", va="center", fontsize=8, color="gray",
                transform=ax.transData)

    # 4. colleague's scalar metrics over pseudo-time
    ax = axes[3]
    ax.plot(sev, res.entropy, "-o", ms=3, color="#55a868", label="load entropy %")
    ax.plot(sev, np.asarray(res.max_prob) * 100, "-o", ms=3, color="#8172b3",
            label="mean max-prob %")
    ax.plot(sev, np.minimum(res.cv, 200), "-o", ms=3, color="#ccb974",
            label="cv % (clip 200)")
    ax.plot(sev, np.asarray(res.spread) * 50, "-o", ms=3, color="#937860",
            label="logit-spread x50")
    ax.set_ylim(0, 210)
    ax.set_title("scalar router-health", fontsize=9)
    ax.set_xlabel("severity"); ax.legend(fontsize=6, loc="best")

    fig.suptitle(f"{res.name}  —  {SCENARIO_DESCRIPTIONS.get(res.name, '')}",
                 fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def summary_table(results: List[SweepResult], E: int) -> str:
    """One row per scenario: does each metric react to the failure?"""
    hdr = f"{'scenario':<26} {'sat_range':>10} {'final_cv%':>9} " \
          f"{'final_ent%':>10} {'final_mxp':>9} {'spread':>7} {'dead':>5}"
    lines = [hdr, "-" * len(hdr)]
    for r in results:
        sat_range = max(r.sat) - min(r.sat)
        lines.append(
            f"{r.name:<26} {sat_range:>10.1f} {r.cv[-1]:>9.1f} "
            f"{r.entropy[-1]:>10.1f} {r.max_prob[-1]:>9.3f} "
            f"{r.spread[-1]:>7.2f} {r.dead[-1]:>5d}"
        )
    return "\n".join(lines)


def run_all(E=E_DEFAULT, N=N_DEFAULT, topk=TOPK_DEFAULT, seed=SEED_DEFAULT,
            n_sev=N_SEVERITY) -> List[SweepResult]:
    return [run_scenario(name, E, N, topk, seed, n_sev) for name in SCENARIOS]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/moe_failcases"))
    ap.add_argument("--experts", type=int, default=E_DEFAULT)
    ap.add_argument("--tokens", type=int, default=N_DEFAULT)
    ap.add_argument("--topk", type=int, default=TOPK_DEFAULT)
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--n-severity", type=int, default=N_SEVERITY)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.out.mkdir(parents=True, exist_ok=True)
    results = run_all(args.experts, args.tokens, args.topk, args.seed, args.n_severity)
    for r in results:
        fig = plot_scenario(r, args.experts, args.topk)
        fp = args.out / f"failcase_{r.name}.png"
        fig.savefig(fp, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {fp}")

    table = summary_table(results, args.experts)
    (args.out / "summary.txt").write_text(table + "\n")
    print("\n" + table)
    print(f"\nConfig: E={args.experts} top_k={args.topk} N={args.tokens} "
          f"seed={args.seed} severity_steps={args.n_severity}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
