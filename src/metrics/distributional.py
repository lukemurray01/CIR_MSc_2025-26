# Distributional diagnostics against the exact CIR terminal law.
#
# The exact transition law is  X_T | X_0 = x0  ~  Z / c  with
# Z ~ ncx2(df, nc) and (c, df, nc) as in src.samplers.exact.cir_ncx2_params,
# so the exact CDF is  F(x) = ncx2.cdf(c * x, df, nc).
#
# Both diagnostics compare LAWS, not Brownian-coupled paths, so exact
# transition sampling is a valid ground truth here (thesis background
# chapter, error-notion definitions).

from functools import lru_cache

import numpy as np
from scipy.stats import ncx2

from src.samplers.exact import cir_ncx2_params


def exact_terminal_cdf(
    x: np.ndarray,
    x0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
) -> np.ndarray:
    c, df, nc = cir_ncx2_params(x0, kappa, theta, sigma, T)
    return ncx2.cdf(c * np.asarray(x), df, nc)


def exact_terminal_quantile(
    q: np.ndarray,
    x0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
) -> np.ndarray:
    c, df, nc = cir_ncx2_params(x0, kappa, theta, sigma, T)
    return ncx2.ppf(np.asarray(q), df, nc) / c


def ks_statistic_vs_exact(
    samples: np.ndarray,
    x0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
) -> float:
    """One-sample Kolmogorov--Smirnov statistic sup_x |F_n(x) - F(x)|."""
    x = np.sort(np.asarray(samples, dtype=float))
    n = x.size
    if n == 0:
        raise ValueError("samples must be non-empty")

    cdf = exact_terminal_cdf(x, x0, kappa, theta, sigma, T)

    upper = np.arange(1, n + 1) / n - cdf
    lower = cdf - np.arange(0, n) / n

    return float(np.max(np.maximum(upper, lower)))


def _ncx2_partial_expectation(t: np.ndarray, df: float, nc: float) -> np.ndarray:
    """E[Z 1{Z <= t}] for Z ~ ncx2(df, nc).

    From the Poisson-mixture representation of the noncentral chi-square
    together with the central identity x f_m(x) = m f_{m+2}(x):

        E[Z 1{Z<=t}] = df * F_{df+2,nc}(t) + nc * F_{df+4,nc}(t),

    which tends to df + nc = E[Z] as t -> infinity, as it must.
    """
    return df * ncx2.cdf(t, df + 2, nc) + nc * ncx2.cdf(t, df + 4, nc)


@lru_cache(maxsize=32)
def _crossing_points(
    n: int, x0: float, kappa: float, theta: float, sigma: float, T: float
) -> np.ndarray:
    """The exact quantiles F^{-1}(i/n), i = 1..n-1.

    These depend only on the sample size and the law, not on the sample, so
    repeated calls across schemes, levels and replicates reuse one array.
    """
    c, df, nc = cir_ncx2_params(x0, kappa, theta, sigma, T)
    return ncx2.ppf(np.arange(1, n) / n, df, nc) / c


def wasserstein1_vs_exact(
    samples: np.ndarray,
    x0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
) -> float:
    """Wasserstein-1 distance  W1 = integral |F_n(x) - F(x)| dx, in closed form.

    F_n is piecewise constant, so between consecutive order statistics the
    integrand is |k - F(x)| with k = i/n fixed.  F is monotone, so it crosses
    that level at most once, at F^{-1}(k); splitting there removes the
    absolute value.  What remains is int F dx on each piece, and integration
    by parts turns that into the ncx2 partial expectation above.  The result
    is exact up to scipy's ncx2 routines: no quadrature grid, no truncation
    of the upper tail, and no tuning parameters.

    An earlier version integrated on a 4096-point uniform grid.  That is
    accurate to well under a percent for smooth samples, but the integrand
    has a jump wherever a scheme places an atom -- at zero for full
    truncation, at sigma^2 h / 4 for the floored maps -- and a trapezoid
    across a jump is only first-order accurate.  In regime E, where the floor
    sits about two grid cells from the origin and the exact CDF rises like
    x^{delta/2}, that produced errors of tens of percent whose sign depended
    on where the atom sat, biasing exactly the comparison this diagnostic
    exists to make.
    """
    x = np.sort(np.asarray(samples, dtype=float))
    n = x.size
    if n == 0:
        raise ValueError("samples must be non-empty")

    c, df, nc = cir_ncx2_params(x0, kappa, theta, sigma, T)
    mean = (df + nc) / c

    def cdf(v):
        return ncx2.cdf(c * v, df, nc)

    def partial(v):
        return _ncx2_partial_expectation(c * v, df, nc) / c

    if n == 1:
        # Degenerate empirical law: W1 = E|X - x| about the single atom.
        return float(mean - 2.0 * partial(x[0]) + x[0] * (2.0 * cdf(x[0]) - 1.0))

    # Below the smallest observation F_n = 0; above the largest it is 1.
    total = (x[0] * cdf(x[0]) - partial(x[0])) + (
        mean - partial(x[-1]) - x[-1] * (1.0 - cdf(x[-1]))
    )

    a, b = x[:-1], x[1:]
    k = np.arange(1, n) / n
    m = np.clip(_crossing_points(n, x0, kappa, theta, sigma, T), a, b)

    Fa, Fb, Fm = cdf(a), cdf(b), cdf(m)
    Pa, Pb, Pm = partial(a), partial(b), partial(m)

    area_ab = b * Fb - a * Fa - (Pb - Pa)  # int_a^b F dx
    area_am = m * Fm - a * Fa - (Pm - Pa)
    area_mb = b * Fb - m * Fm - (Pb - Pm)

    segments = np.where(
        Fa >= k,
        area_ab - k * (b - a),
        np.where(
            Fb <= k,
            k * (b - a) - area_ab,
            (k * (m - a) - area_am) + (area_mb - k * (b - m)),
        ),
    )

    return float(total + segments.sum())


def lower_tail_mass(samples: np.ndarray, epsilon: float) -> float:
    """P(X_T <= epsilon) under the empirical law; boundary-mass diagnostic."""
    x = np.asarray(samples, dtype=float)
    return float(np.mean(x <= epsilon))
