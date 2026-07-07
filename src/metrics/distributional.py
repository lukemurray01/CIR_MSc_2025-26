# Distributional diagnostics against the exact CIR terminal law.
#
# The exact transition law is  X_T | X_0 = x0  ~  Z / c  with
# Z ~ ncx2(df, nc) and (c, df, nc) as in src.samplers.exact.cir_ncx2_params,
# so the exact CDF is  F(x) = ncx2.cdf(c * x, df, nc).
#
# Both diagnostics compare LAWS, not Brownian-coupled paths, so exact
# transition sampling is a valid ground truth here (thesis background
# chapter, error-notion definitions).

import numpy as np
from scipy.integrate import trapezoid
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


def wasserstein1_vs_exact(
    samples: np.ndarray,
    x0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
    n_grid: int = 4096,
    tail_quantile: float = 0.9999,
) -> float:
    """Wasserstein-1 distance  W1 = integral |F_n(x) - F(x)| dx.

    Evaluated by trapezoidal quadrature on a grid from 0 to the max of the
    exact tail quantile and the sample maximum, which captures the lower-tail
    mass near the boundary that motivates this diagnostic.
    """
    x = np.asarray(samples, dtype=float)
    if x.size == 0:
        raise ValueError("samples must be non-empty")

    upper = max(
        float(exact_terminal_quantile(tail_quantile, x0, kappa, theta, sigma, T)),
        float(np.max(x)),
    )
    grid = np.linspace(0.0, upper, n_grid)

    exact_cdf = exact_terminal_cdf(grid, x0, kappa, theta, sigma, T)
    empirical_cdf = np.searchsorted(np.sort(x), grid, side="right") / x.size

    return float(trapezoid(np.abs(empirical_cdf - exact_cdf), grid))


def lower_tail_mass(samples: np.ndarray, epsilon: float) -> float:
    """P(X_T <= epsilon) under the empirical law; boundary-mass diagnostic."""
    x = np.asarray(samples, dtype=float)
    return float(np.mean(x <= epsilon))
