"""Parametric functional forms (the plug-in registry of the procedure).

A form maps per-row *form parameters* ``p`` (dict name -> array) and per-row
*features* ``X`` (dict name -> array) to a predicted **log loss**. Everything is
plain numpy with broadcasting, so the same code evaluates

  - one parameter vector on many rows     p[k].shape == (n,)
  - many posterior samples on many rows   p[k].shape == (n, S), X[k].shape == (n, 1)

Features are normalised logs so the parameters are O(1) and well conditioned:

  logN  = ln(N / 1e8)        N   non-embedding parameters
  logD  = ln(D / 1e10)       D   training tokens
  loglr = ln(lr)
  logB  = ln(B / 2**20)      B   global batch in tokens

New forms subclass :class:`Form` and register with :func:`register`; the
hierarchical model (``model.py``) is agnostic to what the parameters mean.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

LOGN0 = np.log(1e8)
LOGD0 = np.log(1e10)
LOGB0 = np.log(2.0**20)


@dataclass(frozen=True)
class ParamSpec:
    """One form parameter.

    ``prior_mean``/``prior_sd`` is the (weak) prior on the *population* value.
    ``fixed`` parameters are not fitted at their base level (they can still carry
    hierarchical deltas, e.g. a per-study loss offset whose base is pinned at 0).
    """

    name: str
    init: float
    prior_mean: float
    prior_sd: float
    fixed: bool = False


@dataclass
class Form:
    name: str
    params: tuple[ParamSpec, ...]
    description: str = ""
    # predicted log loss at the given hyperparameters
    predict: Callable[[dict, dict], np.ndarray] = field(repr=False, default=None)
    # predicted log loss at the (form-implied) optimal hyperparameters
    envelope: Callable[[dict, dict], np.ndarray] = field(repr=False, default=None)
    # log-loss penalty for being off-optimum (0 if the form has no HP dependence)
    bowl: Callable[[dict, dict], np.ndarray] = field(repr=False, default=None)
    # optimal (log) hyperparameters: dict name -> array
    optimum: Callable[[dict, dict], dict] = field(repr=False, default=None)

    @property
    def names(self) -> list[str]:
        return [s.name for s in self.params]

    def spec(self, name: str) -> ParamSpec:
        for s in self.params:
            if s.name == name:
                return s
        raise KeyError(name)


REGISTRY: dict[str, Callable[[], Form]] = {}


def register(fn: Callable[[], Form]) -> Callable[[], Form]:
    REGISTRY[fn.__name__] = fn
    return fn


def get_form(name: str) -> Form:
    return REGISTRY[name]()


def features(N, D, lr=None, B=None) -> dict[str, np.ndarray]:
    """Build normalised features from raw N, D, lr, B (tokens)."""
    X = {
        "logN": np.log(np.asarray(N, float)) - LOGN0,
        "logD": np.log(np.asarray(D, float)) - LOGD0,
    }
    X["loglr"] = np.full_like(X["logN"], np.nan) if lr is None else np.log(np.asarray(lr, float))
    X["logB"] = np.full_like(X["logN"], np.nan) if B is None else np.log(np.asarray(B, float)) - LOGB0
    return X


# --------------------------------------------------------------------------- #
# Chinchilla envelope + asymmetric (lr, batch) bowl with power-law optima
# --------------------------------------------------------------------------- #
@register
def chinchilla_bowl() -> Form:
    r"""log L = log(E + A N^-a + B D^-b) + bowl(lr, B) + off.

    Optimal HPs follow power laws (log-linear in the normalised features):

      u_B*  = bs_c + bs_pN logN + bs_pD logD                         (batch, tokens)
      u_lr* = lr_c + lr_pN logN + lr_pD logD + lr_pB (logB - u_B*)   (lr at batch B)

    The lr optimum is *conditional on the batch actually used* (lr_pB ~ 0.5 is the
    sqrt-batch rule), so fixed-batch literature exponents (Bjorck: lr ~ D^-0.32) and
    batch-co-scaled ones (Step Law: lr ~ D^+0.31 with B ~ D^0.57) are both special
    cases. The bowl is quadratic in log space with an lr-asymmetry (steeper wall at
    high lr, where runs diverge):

      bowl = e^{h_lr} d_lr^2 exp(lr_asym tanh(2 d_lr)) + e^{h_bs} d_B^2
    """
    P = (
        ParamSpec("E_log", 0.55, 0.55, 0.5),
        ParamSpec("A_log", -1.0, -1.0, 1.5),
        ParamSpec("alpha", 0.35, 0.35, 0.2),
        ParamSpec("B_log", -1.0, -1.0, 1.5),
        ParamSpec("beta", 0.30, 0.30, 0.2),
        ParamSpec("lr_c", np.log(1.5e-3), np.log(1.5e-3), 1.5),
        # HP-law exponents: literature-informed priors (Bjorck: lr ~ D^-0.32 at fixed B;
        # sqrt-batch rule lr ~ B^0.5; Power Lines: B* ~ D^0.38, no N term). The (lr, B)
        # loss surface is flat, so with wide priors the exponents drift to unphysical
        # values; on the repo's data the tight priors extrapolate better to a held-out
        # size (0.9B: 0.27% vs 0.41% error, 90%-coverage 0.80 vs 0.37) even though the
        # in-sample evidence prefers wide ones -- out-of-sample decides.
        ParamSpec("lr_pN", -0.3, -0.3, 0.15),
        ParamSpec("lr_pD", -0.3, -0.3, 0.15),
        ParamSpec("lr_pB", 0.5, 0.5, 0.15),
        ParamSpec("bs_c", 0.0, 0.0, 1.5),
        ParamSpec("bs_pN", 0.0, 0.0, 0.15),
        ParamSpec("bs_pD", 0.4, 0.4, 0.15),
        ParamSpec("h_lr", -4.5, -4.5, 1.5),
        ParamSpec("h_bs", -4.5, -4.5, 1.5),
        ParamSpec("lr_asym", 0.0, 0.0, 0.5),
        ParamSpec("off", 0.0, 0.0, 0.1, fixed=True),
    )

    def envelope(p, X):
        t1 = p["A_log"] - p["alpha"] * X["logN"]
        t2 = p["B_log"] - p["beta"] * X["logD"]
        return np.log(np.exp(p["E_log"]) + np.exp(t1) + np.exp(t2)) + p["off"]

    def optimum(p, X, logB=None):
        uB = p["bs_c"] + p["bs_pN"] * X["logN"] + p["bs_pD"] * X["logD"]
        lb = uB if logB is None else logB
        ulr = p["lr_c"] + p["lr_pN"] * X["logN"] + p["lr_pD"] * X["logD"] + p["lr_pB"] * (lb - uB)
        return {"logB": uB, "loglr": ulr}

    def bowl(p, X):
        opt = optimum(p, X, logB=X["logB"])
        d_lr = X["loglr"] - opt["loglr"]
        d_b = X["logB"] - opt["logB"]
        pen = np.exp(p["h_lr"]) * d_lr**2 * np.exp(p["lr_asym"] * np.tanh(2.0 * d_lr))
        pen = pen + np.exp(p["h_bs"]) * d_b**2
        return np.where(np.isfinite(pen), pen, 0.0)  # rows without HPs: on-optimum

    def predict(p, X):
        return envelope(p, X) + bowl(p, X)

    return Form(
        "chinchilla_bowl",
        P,
        description="Chinchilla envelope (shared-E friendly) + asymmetric log-quadratic HP bowl",
        predict=predict,
        envelope=envelope,
        bowl=bowl,
        optimum=optimum,
    )


@register
def chinchilla() -> Form:
    """Envelope only (rows are assumed on-optimum). Useful as a baseline form."""
    base = chinchilla_bowl()
    keep = {"E_log", "A_log", "alpha", "B_log", "beta", "off"}
    P = tuple(s for s in base.params if s.name in keep)
    zero = lambda p, X: np.zeros(np.broadcast(X["logN"], p["E_log"]).shape)  # noqa: E731
    return Form(
        "chinchilla",
        P,
        description="Chinchilla envelope, no HP dependence",
        predict=base.envelope,
        envelope=base.envelope,
        bowl=zero,
        optimum=lambda p, X, logB=None: {},
    )


@register
def fixed_ratio_slice() -> Form:
    """L*(N; r) = E + A N^-alpha at a fixed D/N ratio (Marin hero-ladder style).

    ``logD`` is ignored; fit one per ratio bucket. Robust when only the overtrained
    regime matters and the compute-optimal frontier is not needed.
    """
    P = (
        ParamSpec("E_log", 0.55, 0.55, 0.5),
        ParamSpec("A_log", -1.0, -1.0, 1.5),
        ParamSpec("alpha", 0.3, 0.3, 0.2),
        ParamSpec("off", 0.0, 0.0, 0.1, fixed=True),
    )

    def envelope(p, X):
        t1 = p["A_log"] - p["alpha"] * X["logN"]
        return np.log(np.exp(p["E_log"]) + np.exp(t1)) + p["off"]

    zero = lambda p, X: np.zeros(np.broadcast(X["logN"], p["E_log"]).shape)  # noqa: E731
    return Form("fixed_ratio_slice", P, predict=envelope, envelope=envelope, bowl=zero,
                optimum=lambda p, X, logB=None: {})
