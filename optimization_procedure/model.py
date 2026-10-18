"""Hierarchical, robust, self-auditing fit of a parametric loss form.

Every form parameter k is resolved per row as

    p_k(row) = base_k + sum_{tag in tags(arch)} delta[tag]_k + delta[arch]_k + delta[study]_k

with Gaussian shrinkage priors  delta[level]_k ~ N(0, (tau_level * s_k)^2).  This is
how the procedure *learns similarities across architectures*:

  - tags ("linear", "gdn", "nope", "ratio7", ...) carry effects shared by every
    architecture that has them, so a new GDN variant starts from what all GDN
    variants taught us, not from zero;
  - per-architecture deltas absorb what the tags do not explain;
  - the level scales ``tau`` are learned by empirical Bayes (Laplace evidence), so
    the data decides how similar architectures are allowed to be.

Robustness / hold-out (the "points that don't match" requirement):

  - Student-t IRLS weights down-weight moderate outliers during the fit;
  - approximate leave-one-out z-scores (hat-matrix) flag observations the model
    cannot predict *from the other points*; the worst is held out and the model is
    refit, until nothing exceeds ``z_holdout``. Held-out points stay in the result,
    with their z-score, and are re-admitted if a later fit explains them;
  - per-architecture RMS of LOO z flags *structural* misfit (the form, not a
    point, is wrong for that architecture).

Uncertainty is the Laplace approximation at the MAP (Gauss-Newton Hessian). It is
cheap enough to sit inside the design / scheduling loops, where it is used in
linearised form. Use ``bootstrap`` in ``validate.py`` as an independent check.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .forms import Form, features

REQUIRED = ("arch", "study", "N", "D", "y")


@dataclass
class Level:
    """A level of the hierarchy.

    ``params`` maps form-parameter name -> base delta scale ``s_k``; the effective
    prior sd is ``tau * s_k`` where ``tau`` is learned when ``learn_tau``.
    """

    name: str  # "tag" | "arch" | "study" (how groups are looked up per row)
    params: dict[str, float]
    tau: float = 1.0
    learn_tau: bool = True


def default_levels() -> list[Level]:
    # Envelope shape and the lr optimum may differ by architecture; E (irreducible
    # loss of the data) is shared; the study level only shifts the loss (different
    # validation sets / clusters) and is not shrinkage-tuned.
    shape = {"A_log": 0.3, "alpha": 0.05, "B_log": 0.3, "beta": 0.05, "lr_c": 0.5}
    return [
        Level("tag", dict(shape)),
        Level("arch", dict(shape)),
        Level("study", {"off": 0.1}, learn_tau=False),
    ]


@dataclass
class Layout:
    """Maps the flat parameter vector theta onto per-row form parameters."""

    form: Form
    levels: list[Level]
    arch_tags: dict[str, tuple[str, ...]]
    studies: list[str]
    reference_study: str
    names: list[str] = field(default_factory=list)
    prior_mean: np.ndarray = None
    prior_sd_base: np.ndarray = None  # sd before tau scaling
    level_of: np.ndarray = None  # index into levels, -1 for base
    param_of: list[str] = field(default_factory=list)
    group_of: list[str] = field(default_factory=list)

    def __post_init__(self):
        names, mean, sd, lvl, pk, grp = [], [], [], [], [], []
        for s in self.form.params:
            if not s.fixed:
                names.append(f"base/{s.name}")
                mean.append(s.prior_mean)
                sd.append(s.prior_sd)
                lvl.append(-1)
                pk.append(s.name)
                grp.append("")
        for li, level in enumerate(self.levels):
            for g in self._groups(level):
                for k, s_k in level.params.items():
                    if k not in self.form.names:
                        continue
                    names.append(f"{level.name}/{g}/{k}")
                    mean.append(0.0)
                    sd.append(s_k)
                    lvl.append(li)
                    pk.append(k)
                    grp.append(g)
        self.names = names
        self.prior_mean = np.asarray(mean, float)
        self.prior_sd_base = np.asarray(sd, float)
        self.level_of = np.asarray(lvl, int)
        self.param_of = pk
        self.group_of = grp
        self.index = {n: i for i, n in enumerate(names)}

    def _groups(self, level: Level) -> list[str]:
        if level.name == "tag":
            return sorted({t for tags in self.arch_tags.values() for t in tags})
        if level.name == "arch":
            return sorted(self.arch_tags)
        if level.name == "study":
            return [s for s in self.studies if s != self.reference_study]
        raise ValueError(level.name)

    @property
    def P(self) -> int:
        return len(self.names)

    def prior_sd(self, taus: dict[str, float]) -> np.ndarray:
        sd = self.prior_sd_base.copy()
        for li, level in enumerate(self.levels):
            sd[self.level_of == li] *= taus.get(level.name, level.tau)
        return sd

    def row_groups(self, arch: str, study: str) -> dict[str, list[str]]:
        if arch not in self.arch_tags:
            raise KeyError(f"unknown architecture {arch!r}: add it to arch_tags")
        return {
            "tag": list(self.arch_tags[arch]),
            "arch": [arch],
            "study": [] if study == self.reference_study else [study],
        }

    def matrices(self, arch, study) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Per form-parameter design matrices M_k (n x P) and constants c_k (n,)
        such that p_k = M_k @ theta + c_k."""
        arch = np.asarray(arch)
        study = np.asarray(study)
        n = len(arch)
        M = {k: np.zeros((n, self.P)) for k in self.form.names}
        c = {k: np.zeros(n) for k in self.form.names}
        for s in self.form.params:
            if s.fixed:
                c[s.name][:] = s.init
            else:
                M[s.name][:, self.index[f"base/{s.name}"]] = 1.0
        cache: dict[tuple[str, str], list[tuple[int, str]]] = {}
        for i in range(n):
            key = (str(arch[i]), str(study[i]))
            if key not in cache:
                cols = []
                for lname, groups in self.row_groups(*key).items():
                    for g in groups:
                        for k in self.form.names:
                            j = self.index.get(f"{lname}/{g}/{k}")
                            if j is not None:
                                cols.append((j, k))
                cache[key] = cols
            for j, k in cache[key]:
                M[k][i, j] = 1.0
        return M, c


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns {missing}")
    df = df.copy()
    for c, v in (("lr", np.nan), ("B", np.nan), ("weight", 1.0)):
        if c not in df.columns:
            df[c] = v
    if "obs_id" not in df.columns:
        df["obs_id"] = [f"obs{i}" for i in range(len(df))]
    return df.reset_index(drop=True)


class _Rows:
    """Pre-computed design for a set of rows."""

    def __init__(self, layout: Layout, df: pd.DataFrame):
        self.layout = layout
        self.M, self.c = layout.matrices(df["arch"].values, df["study"].values)
        self.X = features(df["N"].values, df["D"].values, df["lr"].values, df["B"].values)
        self.n = len(df)

    def params(self, theta: np.ndarray) -> dict[str, np.ndarray]:
        """theta: (P,) -> p_k (n,);  (P, S) -> p_k (n, S)."""
        return {k: self.M[k] @ theta + (self.c[k] if theta.ndim == 1 else self.c[k][:, None])
                for k in self.M}

    def Xs(self, theta: np.ndarray) -> dict[str, np.ndarray]:
        return self.X if theta.ndim == 1 else {k: v[:, None] for k, v in self.X.items()}

    def predict(self, theta: np.ndarray, what: str = "predict") -> np.ndarray:
        fn = getattr(self.layout.form, what)
        return fn(self.params(theta), self.Xs(theta))

    def jac(self, theta: np.ndarray, what: str = "predict", eps: float = 1e-6) -> np.ndarray:
        """d f / d theta (n x P) by the chain rule through the form parameters
        (central differences on the ~16 form parameters, exact linear map after)."""
        fn = getattr(self.layout.form, what)
        p = self.params(theta)
        J = np.zeros((self.n, self.layout.P))
        for k in p:
            if not self.M[k].any():
                continue
            hi = dict(p)
            lo = dict(p)
            hi[k] = p[k] + eps
            lo[k] = p[k] - eps
            g = (fn(hi, self.X) - fn(lo, self.X)) / (2 * eps)
            J += g[:, None] * self.M[k]
        return J


@dataclass
class HeldOut:
    obs_id: str
    arch: str
    z: float
    reason: str


@dataclass
class FitResult:
    model: "HierarchicalModel"
    df: pd.DataFrame
    theta: np.ndarray
    cov: np.ndarray
    sigma: dict[str, float]  # observation noise (log-loss sd) per study
    taus: dict[str, float]
    active: np.ndarray  # bool per row (False = held out)
    robust_w: np.ndarray
    z_loo: np.ndarray  # LOO z (active rows) / predictive z (held-out rows)
    held_out: list[HeldOut]
    log_evidence: float
    arch_misfit: pd.DataFrame

    # ------------------------------------------------------------------ utils
    @property
    def layout(self) -> Layout:
        return self.model.layout

    def rows(self, df: pd.DataFrame) -> _Rows:
        return _Rows(self.layout, prepare(df))

    def inflate(self, factor: float) -> "FitResult":
        """Copy with parameter covariance and noise scaled by a calibration factor
        (e.g. the RMS z of leave-one-scale-out validation, or Ledger.calibration)."""
        from dataclasses import replace
        f = max(1.0, float(factor))
        return replace(self, cov=self.cov * f**2, sigma={k: v * f for k, v in self.sigma.items()})

    def noise_sd(self, study) -> np.ndarray:
        worst = max(self.sigma.values())
        return np.array([self.sigma.get(s, worst) for s in np.atleast_1d(study)])

    def samples(self, S: int = 400, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        L = _chol(self.cov)
        return self.theta[:, None] + L @ rng.standard_normal((len(self.theta), S))

    def predict(self, df: pd.DataFrame, what: str = "predict", noise: bool = False):
        """Posterior mean and sd of log loss (linearised)."""
        df = prepare(df)
        r = _Rows(self.layout, df)
        mu = r.predict(self.theta, what)
        J = r.jac(self.theta, what)
        var = np.einsum("ij,jk,ik->i", J, self.cov, J)
        if noise:
            var = var + self.noise_sd(df["study"].values) ** 2
        return mu, np.sqrt(np.maximum(var, 0.0))

    def param_table(self) -> pd.DataFrame:
        sd = np.sqrt(np.diag(self.cov))
        lay = self.layout
        lvl = ["base" if li < 0 else lay.levels[li].name for li in lay.level_of]
        return pd.DataFrame({
            "name": lay.names, "level": lvl, "group": lay.group_of, "param": lay.param_of,
            "mean": self.theta, "sd": sd, "z": self.theta / np.maximum(sd, 1e-12),
        })

    def arch_effects(self) -> pd.DataFrame:
        """Total (tag + arch) delta per architecture and parameter, with sd."""
        lay = self.layout
        out = []
        for a in lay.arch_tags:
            for k in {p for lv in lay.levels if lv.name in ("tag", "arch") for p in lv.params}:
                if k not in lay.form.names:
                    continue
                v = np.zeros(lay.P)
                for t in lay.arch_tags[a]:
                    j = lay.index.get(f"tag/{t}/{k}")
                    if j is not None:
                        v[j] = 1
                j = lay.index.get(f"arch/{a}/{k}")
                if j is not None:
                    v[j] = 1
                out.append({"arch": a, "param": k, "effect": v @ self.theta,
                            "sd": float(np.sqrt(v @ self.cov @ v)),
                            "n_obs": int(((self.df.arch == a) & self.active).sum())})
        return pd.DataFrame(out)

    def arch_similarity(self, N_grid, D_grid, study: str | None = None) -> pd.DataFrame:
        """Functional distance between architectures: RMS over an (N, D) grid of the
        posterior z-score of the envelope log-loss difference. ~0 means "the data
        cannot tell them apart", large means clearly different scaling behaviour."""
        study = study or self.layout.reference_study
        archs = list(self.layout.arch_tags)
        NN, DD = np.meshgrid(N_grid, D_grid)
        grid = pd.DataFrame({"N": NN.ravel(), "D": DD.ravel()})
        mus, Js = {}, {}
        for a in archs:
            g = grid.assign(arch=a, study=study, y=0.0)
            r = self.rows(g)
            mus[a] = r.predict(self.theta, "envelope")
            Js[a] = r.jac(self.theta, "envelope")
        D = np.zeros((len(archs), len(archs)))
        for i, a in enumerate(archs):
            for j, b in enumerate(archs):
                if j <= i:
                    continue
                dJ = Js[a] - Js[b]
                var = np.einsum("ij,jk,ik->i", dJ, self.cov, dJ)
                z = (mus[a] - mus[b]) / np.sqrt(np.maximum(var, 1e-12))
                D[i, j] = D[j, i] = float(np.sqrt(np.mean(z**2)))
        return pd.DataFrame(D, index=archs, columns=archs)

    def to_prior(self, inflate: float = 2.0) -> dict[str, tuple[float, float]]:
        """Posterior of base and tag effects as priors for the next study
        ("learn from every experiment"). ``inflate`` widens the sd to allow drift."""
        tab = self.param_table()
        keep = tab[tab.level.isin(["base", "tag"])]
        return {r.name: (float(r["mean"]), float(r["sd"] * inflate)) for _, r in keep.iterrows()}


def _chol(C: np.ndarray) -> np.ndarray:
    C = 0.5 * (C + C.T)
    jitter = 1e-12 * np.trace(C) / len(C)
    for _ in range(8):
        try:
            return np.linalg.cholesky(C + jitter * np.eye(len(C)))
        except np.linalg.LinAlgError:
            jitter *= 10
    w, V = np.linalg.eigh(C)
    return V * np.sqrt(np.maximum(w, 0))


class HierarchicalModel:
    def __init__(
        self,
        form: Form,
        arch_tags: dict[str, tuple[str, ...]],
        studies: list[str],
        reference_study: str,
        levels: list[Level] | None = None,
        priors: dict[str, tuple[float, float]] | None = None,
        sigma_floor: float = 1.5e-3,
        nu: float = 4.0,
    ):
        self.form = form
        self.layout = Layout(form, levels or default_levels(), dict(arch_tags), list(studies),
                             reference_study)
        if priors:  # transferred posterior from an earlier study
            for name, (m, s) in priors.items():
                j = self.layout.index.get(name)
                if j is not None:
                    self.layout.prior_mean[j] = m
                    self.layout.prior_sd_base[j] = s
        self.sigma_floor = sigma_floor
        self.nu = nu

    # ------------------------------------------------------------ core solve
    def _solve(self, rows: _Rows, y, w, sigma, taus, theta0):
        """``sigma`` is per row (noise level of the row's study)."""
        lay = self.layout
        psd = lay.prior_sd(taus)
        sw = np.sqrt(w)

        def fun(th):
            r_d = sw * (y - rows.predict(th)) / sigma
            r_p = (th - lay.prior_mean) / psd
            return np.concatenate([r_d, r_p])

        def jac(th):
            J_d = -(sw / sigma)[:, None] * rows.jac(th)
            return np.vstack([J_d, np.diag(1.0 / psd)])

        sol = least_squares(fun, theta0, jac=jac, method="trf", x_scale="jac", max_nfev=200,
                            ftol=1e-7, xtol=1e-7)
        return sol.x, psd

    def _laplace(self, rows, y, w, sigma, psd, theta):
        J_d = (np.sqrt(w) / sigma)[:, None] * rows.jac(theta)
        H = J_d.T @ J_d + np.diag(1.0 / psd**2)
        cov = np.linalg.inv(H)
        r_d = np.sqrt(w) * (y - rows.predict(theta)) / sigma
        r_p = (theta - self.layout.prior_mean) / psd
        sign, logdet = np.linalg.slogdet(H)
        log_ev = (-0.5 * np.sum(r_d**2) - 0.5 * np.sum(r_p**2) - np.sum(np.log(psd))
                  - 0.5 * logdet - np.sum((w > 0) * np.log(sigma)))
        return cov, J_d, log_ev

    def _row_sigma(self, df, sig: dict[str, float]) -> np.ndarray:
        return df["study"].map(sig).astype(float).values

    def _fit_once(self, df, rows, active, taus, theta0, sig0: dict, n_irls=4):
        """IRLS: MAP solve, then re-estimate per-study noise (robust MAD) and
        Student-t weights. Returns sigma as a per-row array (+ the per-study dict)."""
        y = df["y"].values
        base_w = df["weight"].values * active
        theta, sig, rob = theta0, dict(sig0), np.ones(len(df))
        for _ in range(n_irls):
            w = base_w * rob
            sigma = self._row_sigma(df, sig)
            theta, psd = self._solve(rows, y, w, sigma, taus, theta)
            res = y - rows.predict(theta)
            for st in sig:
                a = active & (base_w > 0) & (df["study"].values == st)
                if a.sum() >= 5:
                    sig[st] = max(self.sigma_floor, 1.4826 * np.median(np.abs(res[a])))
            sigma = self._row_sigma(df, sig)
            z = res / sigma
            rob = (self.nu + 1) / (self.nu + z**2)
        w = base_w * rob
        cov, J_d, log_ev = self._laplace(rows, y, w, sigma, psd, theta)
        self._last_sig = sig
        return theta, cov, sig, rob, J_d, log_ev

    # -------------------------------------------------------------- public
    def fit(
        self,
        df: pd.DataFrame,
        learn_taus: bool = True,
        taus: dict[str, float] | None = None,
        z_holdout: float = 4.0,
        z_readmit: float = 2.5,
        max_holdout_frac: float = 0.1,
        tau_grid=(1 / 32, 1 / 16, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0),
        exclude: set[str] | None = None,
        verbose: bool = False,
    ) -> FitResult:
        df = prepare(df)
        rows = _Rows(self.layout, df)
        taus = dict(taus or {lv.name: lv.tau for lv in self.layout.levels})
        active = np.ones(len(df), bool)
        # rows excluded up front: zero run-quality weight (diverged / broken runs) or
        # explicitly excluded ids. They are reported, never silently dropped.
        pre: dict[int, str] = {}
        flags = df["flag"].values if "flag" in df.columns else [""] * len(df)
        for i in np.where(df["weight"].values <= 0)[0]:
            pre[i] = f"run QA weight 0 ({flags[i] or 'flagged'})"
        if exclude:
            for i in np.where(df["obs_id"].isin(exclude).values)[0]:
                pre[i] = "excluded by caller"
        active[list(pre)] = False
        theta0 = self.layout.prior_mean.copy()
        for s in self.form.params:
            j = self.layout.index.get(f"base/{s.name}")
            if j is not None:
                theta0[j] = s.init
        sigma0 = {st: 5e-3 for st in df["study"].unique()}

        # 1) empirical Bayes over the level scales (coordinate ascent on the grid)
        theta, cov, sigma, rob, J_d, log_ev = self._fit_once(df, rows, active, taus, theta0, sigma0)
        if learn_taus:
            for _ in range(2):
                for lv in self.layout.levels:
                    if not lv.learn_tau:
                        continue
                    best = (log_ev, taus[lv.name], theta, sigma)
                    for t in tau_grid:
                        cand = dict(taus, **{lv.name: t})
                        th, _, sg, _, _, le = self._fit_once(df, rows, active, cand, theta, sigma,
                                                             n_irls=2)
                        if verbose:
                            print(f"  tau[{lv.name}]={t:<6} log_ev={le:10.2f}")
                        if le > best[0]:
                            best = (le, t, th, sg)
                    taus[lv.name] = best[1]
                    theta, sigma = best[2], best[3]
                theta, cov, sigma, rob, J_d, log_ev = self._fit_once(df, rows, active, taus, theta,
                                                                     sigma)

        # 2) leave-one-out hold-out loop
        held: dict[int, str] = {}
        max_out = int(np.floor(max_holdout_frac * active.sum()))
        y = df["y"].values
        while True:
            z = self._loo_z(rows, y, theta, cov, self._row_sigma(df, sigma), active)
            cand = np.where(active & (np.abs(z) > z_holdout))[0]
            if len(cand) == 0 or len(held) >= max_out:
                break
            i = cand[np.argmax(np.abs(z[cand]))]
            active[i] = False
            held[i] = f"LOO |z|={abs(z[i]):.1f} > {z_holdout}"
            theta, cov, sigma, rob, J_d, log_ev = self._fit_once(df, rows, active, taus, theta, sigma)
        # re-admit points the final model explains (hysteresis)
        z = self._loo_z(rows, y, theta, cov, self._row_sigma(df, sigma), active)
        readmit = [i for i in held if abs(z[i]) < z_readmit]
        if readmit:
            active[readmit] = True
            for i in readmit:
                held.pop(i)
            theta, cov, sigma, rob, J_d, log_ev = self._fit_once(df, rows, active, taus, theta, sigma)
            z = self._loo_z(rows, y, theta, cov, self._row_sigma(df, sigma), active)

        held_out = [HeldOut(df.obs_id[i], df.arch[i], float(z[i]), r)
                    for i, r in {**pre, **held}.items()]
        mis = (pd.DataFrame({"arch": df.arch, "z": z, "active": active})
               .query("active").groupby("arch")["z"]
               .agg(n="size", mean_z="mean", rms_z=lambda s: float(np.sqrt(np.mean(s**2))))
               .reset_index())
        mis["structural_misfit"] = (mis.rms_z > 2.0) & (mis.n >= 3)
        return FitResult(self, df, theta, cov, sigma, taus, active, rob, z, held_out, log_ev, mis)

    def _loo_z(self, rows, y, theta, cov, sigma, active):
        """Active rows: approximate LOO residual z = r / sqrt(1 - h_ii) (whitened).
        Held-out rows: predictive z against the fit that excluded them."""
        res = y - rows.predict(theta)
        J = rows.jac(theta)
        var_f = np.einsum("ij,jk,ik->i", J, cov, J)
        h = var_f / sigma**2  # leverage of an unweighted observation
        z = np.empty(len(y))
        a = active
        z[a] = res[a] / (sigma[a] * np.sqrt(np.clip(1.0 - h[a], 1e-3, None)))
        z[~a] = res[~a] / np.sqrt(sigma[~a] ** 2 + var_f[~a])
        return z
