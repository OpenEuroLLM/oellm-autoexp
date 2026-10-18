"""Ground-truth tests on synthetic data (no cluster data needed).

Run:  OMP_NUM_THREADS=1 python -m pytest optimization_procedure/tests -q -o addopts=""
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from optimization_procedure import curves, design, scheduler  # noqa: E402
from optimization_procedure.forms import features, get_form  # noqa: E402
from optimization_procedure.ledger import Ledger  # noqa: E402
from optimization_procedure.model import HierarchicalModel  # noqa: E402

TAGS = {"ref": (), "lin_a": ("lin",), "lin_b": ("lin",), "lin_new": ("lin",)}
TRUE_TAG_B = -0.15  # "lin" tag lowers the data term B_log: a shared, learnable effect
FORM = get_form("chinchilla_bowl")
TRUTH = {s.name: s.init for s in FORM.params}
TRUTH.update({"E_log": 0.5, "A_log": -0.8, "alpha": 0.3, "B_log": -0.6, "beta": 0.28,
              "lr_c": np.log(2e-3), "h_lr": -4.0, "h_bs": -4.0})


def _truth(arch, N, D, lr, B):
    p = {k: np.full(len(N), v, float) for k, v in TRUTH.items()}
    if "lin" in TAGS[arch]:
        p["B_log"] = p["B_log"] + TRUE_TAG_B
    return FORM.predict(p, features(N, D, lr, B))


@pytest.fixture(scope="module")
def synth():
    rng = np.random.default_rng(0)
    rows = []
    for arch in ("ref", "lin_a", "lin_b"):
        for N in (5e7, 1e8, 2e8, 4e8):
            for D in (5e9, 1e10, 2e10, 5e10, 1e11):
                for m in ((0.5, 1.0, 2.0) if arch == "ref" else (1.0,)):
                    rows.append({"arch": arch, "study": "s", "N": N, "D": D,
                                 "lr": 2e-3 * m, "B": 2.0**20})
    df = pd.DataFrame(rows)
    df["y"] = [float(_truth(r.arch, [r.N], [r.D], [r.lr], [r.B])[0]) for r in df.itertuples()]
    df["y"] += 0.002 * rng.standard_normal(len(df))
    df["obs_id"] = [f"o{i}" for i in range(len(df))]
    outlier = df.index[(df.arch == "lin_a") & (df.N == 2e8) & (df.D == 2e10)][0]
    df.loc[outlier, "y"] += 0.05  # a broken run: 5% too high
    model = HierarchicalModel(FORM, TAGS, ["s"], "s")
    fit = model.fit(df)
    return df, fit, df.obs_id[outlier]


def test_outlier_is_held_out(synth):
    df, fit, outlier_id = synth
    held = [h.obs_id for h in fit.held_out]
    assert outlier_id in held
    assert len(held) <= 3  # and not much else


def test_shared_tag_effect_is_learned(synth):
    """The 'lin' effect is identified *functionally*: predicted envelope losses of the
    tagged arms match the truth in and beyond the data range (individual deltas can
    trade off between correlated envelope parameters, so they are not the test)."""
    _, fit, _ = synth
    N = np.array([1e8, 4e8, 1e9, 3e9])  # up to 7.5x beyond the largest trained size
    D = 100 * N
    mus, truths = {}, {}
    for arch in ("lin_a", "lin_b", "ref"):
        q = pd.DataFrame({"arch": arch, "study": "s", "N": N, "D": D, "lr": 2e-3,
                          "B": 2.0**20, "y": 0.0})
        mu, sd = fit.predict(q, what="envelope")
        p = {k: np.full(len(N), v, float) for k, v in TRUTH.items()}
        if "lin" in TAGS[arch]:
            p["B_log"] = p["B_log"] + TRUE_TAG_B
        mus[arch], truths[arch] = mu, FORM.envelope(p, features(N, D))
        # absolute extrapolation is weakly identified (E/A/alpha trade off over an 8x
        # range) -- but the posterior must say so: calibrated, not accurate
        assert np.all(np.abs(mu - truths[arch]) < 3 * sd), (arch, (mu - truths[arch]) / sd)
    # the *architecture difference* is what the hierarchy pins down, far beyond the data
    for arch in ("lin_a", "lin_b"):
        d_hat = mus[arch] - mus["ref"]
        d_true = truths[arch] - truths["ref"]
        assert np.all(np.abs(d_hat - d_true) < 0.005), (arch, d_hat - d_true)


def test_new_arch_borrows_from_its_tags(synth):
    """lin_new has no data: its prediction must come from the 'lin' tag (close to the
    truth), with a larger sd than the observed arms."""
    _, fit, _ = synth
    q = pd.DataFrame({"arch": ["lin_new", "lin_a"], "study": "s", "N": 3e8, "D": 3e10,
                      "lr": 2e-3, "B": 2.0**20, "y": 0.0})
    mu, sd = fit.predict(q)
    truth = _truth("lin_new", [3e8], [3e10], [2e-3], [2.0**20])[0]
    assert abs(mu[0] - truth) < 0.01
    assert sd[0] > sd[1]


def test_design_prefers_unobserved_arm_and_diverse_batch(synth):
    _, fit, _ = synth
    arms = ["ref", "lin_a", "lin_new"]
    cand = design.candidate_grid(fit, arms, {"s": 1e8, "m": 4e8}, budgets_bt=(10, 50))
    Q, _ = design.make_Q(fit, arms, reference="ref", N_T=3e9, ratios=(100.0,))
    batch = design.select_batch(fit, cand, Q, k=4, cost_power=0.0)
    assert batch.iloc[0].arch == "lin_new"
    assert batch.cand_index.is_unique
    # information is sub-modular: later picks gain less than the first
    assert batch.IG.iloc[-1] <= batch.IG.iloc[0] + 1e-9


def test_scheduler_rule_extremes(synth):
    _, fit, _ = synth
    sizes = {"s": 1e8, "m": 4e8}

    def mk_cand(f, archs):
        return design.candidate_grid(f, [a for a in archs if a in TAGS], sizes, budgets_bt=(10, 50))

    def mk_Q(f, archs):
        archs = [a for a in archs if a in TAGS]
        return design.make_Q(f, archs, reference="ref", N_T=3e9, ratios=(100.0,))

    pending = mk_cand(fit, ["lin_new"]).iloc[:1]
    # patch the arch registry expected by decision/eliminate: reference is "ref"
    from optimization_procedure import decision
    orig = decision.compute_multipliers

    def cm(f, **kw):
        return orig(f, archs=["lin_a", "lin_b", "lin_new"], **kw)

    decision.compute_multipliers = cm
    orig_tp = decision._data.throughput
    decision._data.throughput = lambda a, fallback="fullattn": (2e5 * 1e8**0.7, 0.7)
    try:
        free = scheduler.launch_decisions(fit, mk_cand, mk_Q, pending, k=2, K=3,
                                          clock=scheduler.Clock(lam=0.0), reference="ref")
        rush = scheduler.launch_decisions(fit, mk_cand, mk_Q, pending, k=2, K=3,
                                          clock=scheduler.Clock(lam=1e9, dt_hours=1.0),
                                          reference="ref")
    finally:
        decision.compute_multipliers = orig
        decision._data.throughput = orig_tp
    # no wall-clock price: launch only what is wanted in every fantasy (waste 0 < 0 is False)
    assert not free.launch_now.any()
    # infinite wall-clock price: launch everything now
    assert rush.launch_now.all()
    assert ((free.P_wanted_after_pending >= 0) & (free.P_wanted_after_pending <= 1)).all()


# ------------------------------------------------------------------ curves
def _curve(n=4000, every=20, noise=0.003, seed=1):
    rng = np.random.default_rng(seed)
    eta = curves.wsd_schedule(n, warmup=100, decay_fraction=0.2)
    S1, S2 = curves._features(eta, 0.999)
    L = 2.2 + 3.0 * (S1 + 10) ** (-0.45) - 2e-4 * S2
    steps = np.arange(every, n + 1, every)
    return steps, L[steps - 1] + noise * rng.standard_normal(len(steps)), eta


def test_clean_curve_passes():
    steps, y, eta = _curve()
    qa = curves.assess_run(steps, y, eta)
    assert qa.weight == 1.0, qa.events
    assert not [e for e in qa.events if e.kind in ("spike", "bump", "stall")]


def test_spike_detected_with_scar():
    steps, y, eta = _curve()
    i = len(steps) // 3
    y = y.copy()
    y[i:i + 3] += 0.3  # spike
    y[i + 3:] += 0.03  # ...that leaves a scar
    qa = curves.assess_run(steps, y, eta, check_stall=False)
    spikes = [e for e in qa.events if e.kind == "spike"]
    assert spikes and abs(spikes[0].start - steps[i]) <= 2 * (steps[1] - steps[0])
    assert qa.weight < 1.0


def test_bump_detected():
    steps, y, eta = _curve()
    i = len(steps) // 2
    y = y.copy()
    y[i:i + 30] += 0.02 * np.hanning(30)  # sustained, smooth excursion
    qa = curves.assess_run(steps, y, eta, check_stall=False)
    assert "bump" in qa.flags


def test_stall_detected():
    steps, y, eta = _curve()
    y = y.copy()
    half = len(steps) // 2
    y[half:] = y[half] + 0.003 * np.random.default_rng(2).standard_normal(len(y) - half)
    qa = curves.assess_run(steps, y, eta)
    assert "stall" in qa.flags and qa.weight == 0.0


def test_nan_and_divergence():
    steps, y, eta = _curve()
    y2 = y.copy()
    y2[-5:] = np.nan
    assert curves.assess_run(steps, y2, eta).weight == 0.0
    y3 = y.copy()
    y3[-20:] = 12.0
    assert "divergence" in curves.assess_run(steps, y3, eta).flags


# ------------------------------------------------------------------ ledger
def test_ledger_calibration(tmp_path):
    led = Ledger(tmp_path)
    cand = pd.DataFrame({"arch": ["a"] * 20, "study": "s", "N": 1e8, "D": np.arange(1, 21) * 1e9,
                         "lr": 1e-3, "B": 1e6})
    rng = np.random.default_rng(0)
    truth = rng.normal(0.0, 0.03, 20)
    led.register_predictions(cand, np.zeros(20), np.full(20, 0.01), "m:0")  # sd 3x too small
    obs = cand.assign(y=truth, weight=1.0, obs_id=[f"o{i}" for i in range(20)])
    assert led.add_observations(obs) == 20
    assert led.add_observations(obs) == 0  # dedup
    cal = led.calibration()
    assert cal["n"] == 20 and cal["inflation"] > 2.0
