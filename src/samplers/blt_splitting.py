# BLT splitting scheme for CIR (Brehier--Cohen--Herzwurm--Kelly--Neuenkirch,
# in preparation, 2026).
#
# The thesis CIR SDE  dX = kappa*(theta - X) dt + sigma*sqrt(X) dW  is mapped
# by  x = 4*X/sigma^2  onto the paper's normal form
#
#   dx = (a - b*x) dt + 2*sqrt(x) dW,     a = 4*kappa*theta/sigma^2 = delta,
#                                          b = kappa,
#
# driven by the SAME Brownian motion W, so Brownian coupling with the other
# thesis schemes is direct.  Note a equals the Bessel dimension delta of the
# regime grid, and the paper's Feller index is nu = a/2 = delta/2.
#
# Lie--Trotter splitting: dx = (a-1-b*x) dt + [1 dt + 2*sqrt(x) dW].  The
# bracket is a squared Bessel process of dimension one, solved EXACTLY by the
# reflection representation
#
#   Psi_R(h; x) = | sqrt(x) + dW + (sqrt(x) + m)^- |^2,
#
# where m = inf_{u in [t, t+h]} (W_u - W_t) is the step's running infimum
# (nonlinear Brownian information -- this is what places BLT outside the
# Hefter--Jentzen equidistant-increment information class).  The linear ODE
# part is also solved exactly:
#
#   Psi_O(h; o) = o*exp(-b*h) + (a-1)*(1 - exp(-b*h))/b       (b > 0),
#               = o + (a-1)*h                                  (b = 0).
#
# One step is x_{k+1} = Psi_O(Psi_R(x_k)); the only error is splitting error.
# Proven L1 rates: order 1 up to log^2 for a > 2 (regimes A, B); order
# a/2 - eps for a in (1, 2] (regime C).  For a < 1 (regimes D, E) the paper's
# MODIFIED scheme applies the positive part before the Bessel flow,
# x_{k+1} = Psi_O(Psi_R(x_k^+)); no theory yet, observed rate min(1, a/2).
# With a < 1 the ODE flow can output negative values, so the carried state
# may be negative between steps (FTE-spirit); read-off is max(x, 0).

import numpy as np

from src.utils.cir_params import cir_delta
from src.utils.rng import make_brownian_increments_with_infima


def blt_ab_from_cir(kappa: float, theta: float, sigma: float) -> tuple[float, float]:
    """Normal-form parameters (a, b) of the scaled SDE; a is the Bessel dim."""
    return cir_delta(kappa, theta, sigma), kappa


def besq1_flow(x_pos: np.ndarray, dW: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Exact squared-Bessel(1) flow over one step.

    Requires x_pos >= 0 and m <= min(0, dW) (a valid running infimum).
    Output is >= 0 by construction.
    """
    root = np.sqrt(x_pos)
    reflection = np.maximum(0.0, -(root + m))

    return (root + dW + reflection) ** 2


def linear_ode_flow(o: np.ndarray, a: float, b: float, dt: float) -> np.ndarray:
    """Exact flow of o' = (a - 1) - b*o over one step."""
    if b == 0.0:
        return o + (a - 1.0) * dt

    decay = np.exp(-b * dt)
    return o * decay + (a - 1.0) * (1.0 - decay) / b


def blt_step(
    x: np.ndarray,
    a: float,
    b: float,
    dt: float,
    dW: np.ndarray,
    m: np.ndarray,
) -> np.ndarray:
    """One (modified) BLT step in normal-form coordinates.

    For a >= 1 the positive part is inactive (the carried state stays
    nonnegative) and this is exactly the paper's scheme (8); for a < 1 it is
    the modified scheme, whose output may be negative and is clipped at the
    NEXT step's input, matching the full-truncation reading of the paper.
    """
    x_pos = np.maximum(x, 0.0)
    return linear_ode_flow(besq1_flow(x_pos, dW, m), a, b, dt)


def blt_terminal_from_noise(
    X0: float,
    kappa: float,
    theta: float,
    sigma: float,
    dt: float,
    dW: np.ndarray,
    m: np.ndarray,
) -> np.ndarray:
    """Terminal CIR values from pre-generated (increment, infimum) pairs.

    dW and m have shape (n_paths, n_steps); m must be the running infimum of
    the same Brownian steps (see make_brownian_increments_with_infima and
    aggregate_brownian_increments_and_infima).  Read-off is max(x, 0) mapped
    back to CIR coordinates; for a >= 1 the clip never binds.
    """
    if dW.shape != m.shape:
        raise ValueError("dW and m must have the same shape")

    a, b = blt_ab_from_cir(kappa, theta, sigma)
    scale = sigma**2 / 4.0

    n_paths, n_steps = dW.shape
    x = np.full(n_paths, X0 / scale, dtype=float)

    for n in range(n_steps):
        x = blt_step(x, a, b, dt, dW[:, n], m[:, n])

    return scale * np.maximum(x, 0.0)


def blt_paths_from_noise(
    X0: float,
    kappa: float,
    theta: float,
    sigma: float,
    dt: float,
    dW: np.ndarray,
    m: np.ndarray,
) -> np.ndarray:
    """Full CIR paths (read-off values) from pre-generated noise pairs."""
    if dW.shape != m.shape:
        raise ValueError("dW and m must have the same shape")

    a, b = blt_ab_from_cir(kappa, theta, sigma)
    scale = sigma**2 / 4.0

    n_paths, n_steps = dW.shape
    x = np.empty((n_paths, n_steps + 1), dtype=float)
    x[:, 0] = X0 / scale

    for n in range(n_steps):
        x[:, n + 1] = blt_step(x[:, n], a, b, dt, dW[:, n], m[:, n])

    return scale * np.maximum(x, 0.0)


def blt_terminal(
    X0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Free-running BLT terminal values (scheme draws its own noise)."""
    dt = T / n_steps
    dW, m = make_brownian_increments_with_infima(rng, n_paths, n_steps, dt)

    return blt_terminal_from_noise(
        X0=X0, kappa=kappa, theta=theta, sigma=sigma, dt=dt, dW=dW, m=m,
    )
