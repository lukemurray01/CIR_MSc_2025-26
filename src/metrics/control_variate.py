"""Analytical martingale control variates for the CIR weak-error benchmark.

For a test functional g with value function

    u(t, x) = E[ g(X_T) | X_t = x ],

the martingale representation of the *exact* CIR process gives

    g(X_T) = E[g(X_T)] + int_0^T sigma sqrt(X_s) d_x u(s, X_s) dW_s,

so the discrete analogue

    C_h = sum_n sigma sqrt(X_n^+) d_x u(t_n, X_n^+) dW_n                    (*)

is a near-perfect hedge for the scheme's payoff.  Two properties matter.

1.  C_h has mean EXACTLY zero for any scheme whose state X_n is adapted and
    whose step is chosen before dW_n is drawn: each summand is
    F_{t_n}-measurable times a conditionally centred increment.  The
    control therefore cannot bias the weak-error estimator, whatever
    d_x u is -- accuracy in d_x u buys variance reduction only.  This is
    what makes the g2 grid interpolation below safe.

2.  For an order-one scheme the residual P_h - C_h is itself O(h), so the
    signal-to-noise ratio |bias| / s.e. is asymptotically LEVEL-INDEPENDENT.
    The experiment stops being resolution-limited at fine h, which is what
    makes the deep ladder reachable at the existing path count.

Because the hedge is exact in the h -> 0 limit, the optimal coefficient
beta -> 1.  We use beta = 1 by default: any fixed beta leaves the estimator
unbiased, so this keeps the batch confidence intervals exactly valid without
a pilot-estimation correction.

Derivatives supplied here (tau = T - t is the remaining time):

    g1(x) = x                  d_x u = exp(-kappa tau)
    g2(x) = (x - K)_+^2        d_x u = ncx2 lambda-derivative, tabulated
    g3(x) = exp(-x)            d_x u = -(e^{-kappa tau}/D) u3,  D = 1 + q(tau)
    g4    = exp(-int_0^T X)    d_x u = -B(tau) P(tau, x) exp(-A_n)

where A_n is the running trapezoidal integral already used to evaluate g4.
"""

import numpy as np
from scipy.stats import ncx2

__all__ = [
    "dxu_g1",
    "dxu_g3",
    "dxu_g4",
    "cir_bond_AB",
    "G2Table",
    "control_from_paths",
    "running_trapezoid",
]


# ---------------------------------------------------------------- g1, g3 ---

def dxu_g1(tau, kappa):
    """d/dx E[X_T | X_t = x] = exp(-kappa (T - t))."""
    return np.exp(-kappa * tau)


def _laplace_D(tau, kappa, sigma, u=1.0):
    """Denominator 1 + u q(tau) of the CIR Laplace transform."""
    q = sigma**2 * (1.0 - np.exp(-kappa * tau)) / (2.0 * kappa)
    return 1.0 + u * q


def u_g3(tau, x, kappa, theta, sigma):
    """E[exp(-X_T) | X_t = x], the u = 1 CIR Laplace transform."""
    delta = 4.0 * kappa * theta / sigma**2
    D = _laplace_D(tau, kappa, sigma)
    return D ** (-0.5 * delta) * np.exp(-x * np.exp(-kappa * tau) / D)


def dxu_g3(tau, x, kappa, theta, sigma):
    """d/dx E[exp(-X_T) | X_t = x]."""
    D = _laplace_D(tau, kappa, sigma)
    return -(np.exp(-kappa * tau) / D) * u_g3(tau, x, kappa, theta, sigma)


# -------------------------------------------------------------------- g4 ---

def cir_bond_AB(tau, kappa, theta, sigma):
    """Affine coefficients of P(tau, x) = A(tau) exp(-B(tau) x).

    Same closed form as src.metrics.weak_error.affine_cir_bond_price, but
    exposed as (A, B) so the x-derivative is available.
    """
    gamma = np.sqrt(kappa**2 + 2.0 * sigma**2)
    exp_gamma_tau = np.exp(gamma * tau)
    denominator = (gamma + kappa) * (exp_gamma_tau - 1.0) + 2.0 * gamma

    B = 2.0 * (exp_gamma_tau - 1.0) / denominator
    A = (
        2.0 * gamma * np.exp(0.5 * (kappa + gamma) * tau) / denominator
    ) ** (2.0 * kappa * theta / sigma**2)

    return A, B


def dxu_g4(tau, x, discount, kappa, theta, sigma):
    """d/dx E[exp(-int_0^T X) | F_t], with `discount` = exp(-A_n).

    u4(t, x, A_n) = exp(-A_n) P(T - t, x), so d_x u4 = -B P exp(-A_n).
    """
    A, B = cir_bond_AB(tau, kappa, theta, sigma)
    return -B * A * np.exp(-B * x) * discount


def running_trapezoid(paths, dt):
    """Running trapezoidal integral A_n = int_0^{t_n} X ds, shape as `paths`.

    Matches src.metrics.weak_error.trapezoidal_integral at n = N:
        A_n = dt (cumsum_n - x_0/2 - x_n/2).
    """
    cum = np.cumsum(paths, axis=1)
    return dt * (cum - 0.5 * paths[:, :1] - 0.5 * paths)


# -------------------------------------------------------------------- g2 ---
#
# X_T | X_t = x  ~  Z / c(tau),   Z ~ ncx2(delta, lambda),
#     c(tau) = 4 kappa / (sigma^2 (1 - e^{-kappa tau})),
#     lambda = c x e^{-kappa tau},   delta = 4 kappa theta / sigma^2.
#
# u2(tau, x) = c^{-2} G(delta, lambda; z*),  z* = c K,
#     G(delta, lambda; z) = E[(Z - z)_+^2].
#
# The Poisson-mixture derivative identity for the noncentral chi-square,
#     dG/dlambda = (1/2) [ G(delta + 2, lambda) - G(delta, lambda) ],
# together with dlambda/dx = c e^{-kappa tau}, gives
#
#     d_x u2 = (e^{-kappa tau} / (2 c)) [ G(delta+2, lambda) - G(delta, lambda) ].
#
# G itself follows from the truncated-moment identities (derived by expanding
# the Poisson mixture over central chi-squares and re-indexing):
#     P(Z > z)       = sf(z; d, l)
#     E[Z 1_{Z>z}]   = d sf(z; d+2, l) + l sf(z; d+4, l)
#     E[Z^2 1_{Z>z}] = d(d+2) sf(z; d+4, l) + 2 l (d+2) sf(z; d+6, l)
#                      + l^2 sf(z; d+8, l)
# Both are validated against quadrature in tests.


def _G_squared_call(z, df, nc):
    """E[(Z - z)_+^2] for Z ~ ncx2(df, nc), z >= 0."""
    m0 = ncx2.sf(z, df, nc)
    m1 = df * ncx2.sf(z, df + 2, nc) + nc * ncx2.sf(z, df + 4, nc)
    m2 = (
        df * (df + 2.0) * ncx2.sf(z, df + 4, nc)
        + 2.0 * nc * (df + 2.0) * ncx2.sf(z, df + 6, nc)
        + nc**2 * ncx2.sf(z, df + 8, nc)
    )
    return m2 - 2.0 * z * m1 + z**2 * m0


def dxu_g2_exact(tau, x, kappa, theta, sigma, strike=0.02):
    """d/dx E[(X_T - K)_+^2 | X_t = x], evaluated directly (slow)."""
    tau = np.asarray(tau, dtype=float)
    x = np.asarray(x, dtype=float)

    delta = 4.0 * kappa * theta / sigma**2
    c = 4.0 * kappa / (sigma**2 * (1.0 - np.exp(-kappa * tau)))
    lam = c * x * np.exp(-kappa * tau)
    z_star = c * strike

    return (np.exp(-kappa * tau) / (2.0 * c)) * (
        _G_squared_call(z_star, delta + 2.0, lam)
        - _G_squared_call(z_star, delta, lam)
    )


class G2Table:
    """Tabulated d_x u2 on a (log tau) x (x) grid, with bilinear lookup.

    Evaluating ncx2 survival functions per path per step is far too slow, so
    d_x u2 is tabulated once per regime and interpolated.  Interpolation
    error costs variance-reduction efficiency only -- the control stays
    exactly mean-zero, so the weak-error estimate is unaffected.
    """

    def __init__(self, kappa, theta, sigma, T, tau_min, strike=0.02,
                 n_tau=96, n_x=384, x_max=None):
        self.kappa, self.theta, self.sigma = kappa, theta, sigma
        self.strike = strike

        if x_max is None:
            # Cover the terminal law generously; d_x u2 is smooth and grows
            # linearly beyond the strike, so the top of the grid is cheap.
            delta = 4.0 * kappa * theta / sigma**2
            c_T = 4.0 * kappa / (sigma**2 * (1.0 - np.exp(-kappa * T)))
            lam_T = c_T * theta * np.exp(-kappa * T)
            x_max = float(max(20.0 * theta, ncx2.ppf(1.0 - 1e-9, delta, lam_T) / c_T))
        self.x_max = x_max

        tau_min = max(float(tau_min), 1e-9)
        self.log_tau = np.linspace(np.log(tau_min), np.log(T), n_tau)
        self.tau = np.exp(self.log_tau)

        # As tau -> 0, d_x u2 -> 2 (x - K)_+, which has a kink at the strike.
        # Grade the grid so most nodes sit in [0, 4K] where that kink and the
        # bulk of the terminal law both live.
        n_near = 3 * n_x // 4
        self.x = np.unique(np.concatenate([
            np.linspace(0.0, 4.0 * strike, n_near),
            np.linspace(4.0 * strike, x_max, n_x - n_near),
        ]))

        TAU, X = np.meshgrid(self.tau, self.x, indexing="ij")
        self.table = dxu_g2_exact(TAU, X, kappa, theta, sigma, strike)

    def rows_for_grid(self, taus):
        """Interpolate the table onto an arbitrary tau ladder -> (len, n_x)."""
        taus = np.asarray(taus, dtype=float)
        lt = np.log(np.clip(taus, self.tau[0], self.tau[-1]))
        idx = np.clip(np.searchsorted(self.log_tau, lt) - 1, 0, len(self.tau) - 2)
        lo, hi = self.log_tau[idx], self.log_tau[idx + 1]
        w = ((lt - lo) / (hi - lo))[:, None]
        return (1.0 - w) * self.table[idx] + w * self.table[idx + 1]

    def interpolate(self, tau, x):
        """d_x u2 at per-path (tau, x) pairs, vectorised.

        Bilinear from four indexed table entries per path, so storage is
        O(len(x)) rather than O(len(x) * n_x).  The adaptive schemes need
        this: they present a different tau for every active path on every
        round, and materialising a full table row per path allocated
        ~600 MB and took ~3 s per round at 200k paths.
        """
        tau = np.asarray(tau, dtype=float)
        x = np.asarray(x, dtype=float)

        lt = np.log(np.clip(tau, self.tau[0], self.tau[-1]))
        i = np.clip(np.searchsorted(self.log_tau, lt) - 1, 0, len(self.tau) - 2)
        wt = (lt - self.log_tau[i]) / (self.log_tau[i + 1] - self.log_tau[i])

        xc = np.clip(x, self.x[0], self.x[-1])
        j = np.clip(np.searchsorted(self.x, xc) - 1, 0, len(self.x) - 2)
        wx = (xc - self.x[j]) / (self.x[j + 1] - self.x[j])

        lo = (1.0 - wx) * self.table[i, j] + wx * self.table[i, j + 1]
        hi = (1.0 - wx) * self.table[i + 1, j] + wx * self.table[i + 1, j + 1]
        return (1.0 - wt) * lo + wt * hi

    def __call__(self, tau, x):
        """d_x u2(tau, x) for scalar tau and array x."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        return self.interpolate(np.full(x.shape, float(tau)), x)


# ------------------------------------------------------------ assembly -----

def control_from_paths(payoff, paths, dW, dt, T, kappa, theta, sigma,
                       g2_rows=None):
    """Discrete control (*) for a path-based scheme.

    `paths` is the (n_paths, n_steps+1) non-negative read-off returned by the
    samplers -- exactly the state whose square root each scheme used in its
    diffusion coefficient -- so no sampler change is required.

    `g2_rows` is the precomputed (n_steps, n_x) table from
    G2Table.rows_for_grid; required only for payoff == "g2".
    """
    n_steps = dW.shape[1]
    x = paths[:, :-1]                       # states at t_0 .. t_{n-1}
    taus = T - dt * np.arange(n_steps)      # remaining time at each step

    if payoff == "g1":
        dxu = dxu_g1(taus, kappa)[None, :]
    elif payoff == "g3":
        dxu = dxu_g3(taus[None, :], x, kappa, theta, sigma)
    elif payoff == "g4":
        discount = np.exp(-running_trapezoid(paths, dt)[:, :-1])
        dxu = dxu_g4(taus[None, :], x, discount, kappa, theta, sigma)
    elif payoff == "g2":
        if g2_rows is None:
            raise ValueError("g2 control requires a precomputed g2_rows table")
        dxu = np.empty_like(x)
        for n in range(n_steps):
            dxu[:, n] = np.interp(x[:, n], g2_rows["x"], g2_rows["rows"][n])
    else:
        raise ValueError(f"unknown payoff {payoff!r}")

    return np.sum(sigma * np.sqrt(x) * dxu * dW, axis=1)
