"""Layer A: per-run loss-curve model and run-quality assessment.

Curve model (Tissue et al. 2024, "Scaling Law with LR Annealing"), explicit in the
LR schedule so one fit covers the WSD trunk *and* its cooldown branches:

    L(t) = L0 + A (S1(t) + S_w)^(-alpha) - C S2(t)
    S1(t) = sum_{i<=t} eta_i                              (forward area)
    S2(t) = sum_{i<=t} m_i,  m_i = lam m_{i-1} + (eta_{i-1} - eta_i)   (annealing momentum)

eta is normalised by its peak, so the parameters are comparable across runs.

Run QA works on robust residuals against a *robust* fit (soft-L1), so an anomaly
does not bend the fit around itself:

  spike  transient positive excursion (z > z_spike), with height, duration and "scar"
         (persistent offset after recovery)
  bump   sustained positive shift (one-sided CUSUM on z)
  stall  prequential: fit on the first part of the curve, predict the rest; the
         observed decrease falls short of the predicted one by more than ``stall_q``

``assess_run`` turns these into a weight in [0, 1] for the loss-law fit plus a
list of events, so a run is never silently discarded.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares


def wsd_schedule(total_steps: int, warmup: int = 2000, decay_fraction: float = 0.2,
                 stable_until: int | None = None, min_ratio: float = 0.0) -> np.ndarray:
    """Normalised (peak = 1) WSD schedule; ``stable_until`` for a trunk that never decays."""
    t = np.arange(1, total_steps + 1, dtype=float)
    eta = np.minimum(1.0, t / max(warmup, 1))
    d0 = stable_until if stable_until is not None else int(total_steps * (1 - decay_fraction))
    dec = t > d0
    if dec.any() and d0 < total_steps:
        frac = (t[dec] - d0) / (total_steps - d0)
        eta[dec] = 1.0 - (1.0 - min_ratio) * frac
    return eta


def _features(eta: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    S1 = np.cumsum(eta)
    d = np.concatenate([[0.0], eta[:-1] - eta[1:]])
    m = np.zeros_like(eta)
    acc = 0.0
    for i, di in enumerate(d):  # short python loop; curves have ~1e5 steps at most
        acc = lam * acc + di
        m[i] = acc
    return S1, np.cumsum(m)


@dataclass
class CurveFit:
    params: dict
    steps: np.ndarray
    eta: np.ndarray
    lam: float

    def predict(self, steps: np.ndarray | None = None) -> np.ndarray:
        S1, S2 = _features(self.eta, self.lam)
        p = self.params
        full = p["L0"] + p["A"] * (S1 + p["Sw"]) ** (-p["alpha"]) - p["C"] * S2
        idx = (self.steps if steps is None else np.asarray(steps)) - 1
        return full[idx]


def fit_curve(steps: np.ndarray, loss: np.ndarray, eta: np.ndarray, lam: float = 0.999,
              robust: bool = True, fit_upto: int | None = None) -> CurveFit:
    """Fit the annealing law to (steps, loss) observed under schedule eta (per step,
    1-indexed steps). ``fit_upto`` restricts the fit to steps <= fit_upto."""
    steps = np.asarray(steps, int)
    loss = np.asarray(loss, float)
    m = np.ones(len(steps), bool) if fit_upto is None else steps <= fit_upto
    S1, S2 = _features(eta, lam)
    s1, s2 = S1[steps[m] - 1], S2[steps[m] - 1]
    y = loss[m]

    def unpack(x):
        return {"L0": x[0], "A": np.exp(x[1]), "alpha": x[2], "Sw": np.exp(x[3]), "C": x[4]}

    def res(x):
        p = unpack(x)
        return p["L0"] + p["A"] * (s1 + p["Sw"]) ** (-p["alpha"]) - p["C"] * s2 - y

    x0 = np.array([y.min() * 0.8, np.log(max(y[0] - y.min() * 0.8, 1e-3)) + 0.5 * np.log(s1[0] + 1),
                   0.5, np.log(10.0), 1e-4])
    best = None
    for a0 in (0.3, 0.5, 0.8):
        x0[2] = a0
        sol = least_squares(res, x0, loss="soft_l1" if robust else "linear",
                            f_scale=max(1e-3, 1.4826 * np.median(np.abs(np.diff(y))) + 1e-6),
                            bounds=([0, -20, 0.01, -5, -1], [np.inf, 20, 3.0, 15, 1]), max_nfev=2000)
        if best is None or sol.cost < best.cost:
            best = sol
    return CurveFit(unpack(best.x), steps, eta, lam)


@dataclass
class Event:
    kind: str  # spike | bump | stall | divergence | nan
    start: int
    end: int
    size: float  # spike height / bump mean shift / stall shortfall fraction
    extra: dict = field(default_factory=dict)


@dataclass
class RunQA:
    weight: float
    events: list[Event]
    sigma: float
    fit: CurveFit | None

    @property
    def flags(self) -> list[str]:
        return sorted({e.kind for e in self.events})


def _robust_sigma(r: np.ndarray) -> float:
    # scale from first differences: insensitive to slow misfit and to level shifts
    return max(1.4826 * np.median(np.abs(np.diff(r))) / np.sqrt(2), 1e-6)


def detect_spikes(steps, z, z_spike=6.0, recover_z=2.0, scar_window=5, r=None, sigma=1.0):
    events = []
    i, n = 0, len(z)
    while i < n:
        if z[i] > z_spike:
            j = i
            while j + 1 < n and z[j + 1] > recover_z:
                j += 1
            pre = r[max(0, i - scar_window):i]
            post = r[j + 1:j + 1 + scar_window]
            scar = (np.median(post) - np.median(pre)) if len(pre) and len(post) else np.nan
            events.append(Event("spike", int(steps[i]), int(steps[j]), float(z[i:j + 1].max() * sigma),
                                {"scar": float(scar), "n_points": j - i + 1}))
            i = j + 1
        else:
            i += 1
    return events


def detect_bumps(steps, z, k=0.5, h=8.0, min_len=5):
    """One-sided CUSUM for a sustained positive shift in the residual z-scores."""
    events, s, start = [], 0.0, None
    for i, zi in enumerate(z):
        s_new = max(0.0, s + zi - k)
        if s == 0.0 and s_new > 0.0:
            start = i
        s = s_new
        if s > h and start is not None:
            j = i
            while j + 1 < len(z) and z[j + 1] > 0:
                j += 1
            if j - start + 1 >= min_len:
                events.append(Event("bump", int(steps[start]), int(steps[j]),
                                    float(np.mean(z[start:j + 1])), {}))
            s, start = 0.0, None
    return events


def detect_stall(steps, loss, eta, lam=0.999, frac_fit=0.5, stall_q=0.6, min_points=10,
                 z_sig=3.0):
    """Prequential check: fit on the first ``frac_fit`` of the observed steps, then
    compare the observed decrease over the rest with the predicted one.

    Flags only if the shortfall is (a) large relative to the predicted drop and
    (b) significant against the eval noise (``z_sig`` standard errors) -- late,
    nearly flat constant-lr segments otherwise look "stalled" by noise alone."""
    steps = np.asarray(steps)
    cut = steps[0] + frac_fit * (steps[-1] - steps[0])
    if (steps <= cut).sum() < min_points or (steps > cut).sum() < 3:
        return []
    f = fit_curve(steps, loss, eta, lam, fit_upto=int(cut))
    tail = steps > cut
    pred = f.predict(steps[tail])
    k = max(3, tail.sum() // 5)
    obs_drop = np.median(loss[tail][:k]) - np.median(loss[tail][-k:])
    pred_drop = np.median(pred[:k]) - np.median(pred[-k:])
    head = steps <= cut
    noise = _robust_sigma(loss[head] - f.predict(steps[head]))
    se = 1.2533 * noise * np.sqrt(2.0 / k)  # sd of a difference of two medians of k points
    if pred_drop > 0 and obs_drop < stall_q * pred_drop and (pred_drop - obs_drop) > z_sig * se:
        return [Event("stall", int(steps[tail][0]), int(steps[-1]), float(1 - obs_drop / pred_drop),
                      {"observed_drop": float(obs_drop), "predicted_drop": float(pred_drop)})]
    return []


def assess_run(steps, loss, eta, lam=0.999, z_spike=6.0, scar_tol=0.005,
               check_stall=True) -> RunQA:
    """Full QA of one run. Returns a weight for the loss-law fit:

      NaN / divergence              -> 0
      stall                         -> 0 (the run is not measuring the architecture)
      spike with scar > scar_tol*L  -> 0.5 per such spike (floored at 0)
      bump                          -> 0.7
      clean recovered spikes        -> 1 (logged only)
    """
    steps = np.asarray(steps, int)
    loss = np.asarray(loss, float)
    events: list[Event] = []
    bad = ~np.isfinite(loss)
    if bad.any():
        events.append(Event("nan", int(steps[bad][0]), int(steps[-1]), float(bad.mean())))
        return RunQA(0.0, events, np.nan, None)
    if loss[-1] > loss[0] or loss.max() > 1.5 * loss[: max(3, len(loss) // 20)].mean():
        events.append(Event("divergence", int(steps[np.argmax(loss)]), int(steps[-1]),
                            float(loss.max())))
        return RunQA(0.0, events, np.nan, None)
    f = fit_curve(steps, loss, eta, lam)
    r = loss - f.predict()
    sig = _robust_sigma(r)
    z = r / sig
    events += detect_spikes(steps, z, z_spike=z_spike, r=r, sigma=sig)
    events += detect_bumps(steps, np.clip(z, -5, 5))
    if check_stall:
        events += detect_stall(steps, loss, eta, lam)
    w = 1.0
    for e in events:
        if e.kind == "stall":
            w = 0.0
        elif e.kind == "spike" and np.isfinite(e.extra.get("scar", np.nan)) \
                and e.extra["scar"] > scar_tol * np.median(loss):
            w -= 0.5
        elif e.kind == "bump":
            w = min(w, 0.7)
    return RunQA(max(0.0, w), events, sig, f)
