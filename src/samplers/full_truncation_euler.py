import numpy as np

from src.utils.rng import make_brownian_increments

# Full truncation Euler (FTE), Lord--Koekkoek--van Dijk (2010).
#
# The scheme carries a possibly NEGATIVE auxiliary variable x_tilde forward:
#
#   x_tilde_{n+1} = x_tilde_n + kappa*(theta - x_tilde_n^+)*dt
#                   + sigma*sqrt(x_tilde_n^+)*dW_{n+1}
#
# and the CIR approximation is the read-off  X_n = max(x_tilde_n, 0).
#
# Clipping the STORED state each step instead (max(x_hat, 0)) collapses the
# scheme to absorption-at-zero with FT coefficient evaluation, which is a
# different scheme near the boundary and is not covered by the
# Cozma--Reisinger (2020) order-1/2 result. See notes/diagnostics.md
# (01.07.2026) for the regime-E mean bias this distinction produces.


def fte_step(
    x_aux: np.ndarray,
    kappa: float,
    theta: float,
    sigma: float,
    dt: float,
    dW: np.ndarray,
) -> np.ndarray:
    """One FTE update of the auxiliary variable. Input and output may be negative."""
    x_pos = np.maximum(x_aux, 0.0)

    return x_aux + kappa * (theta - x_pos) * dt + sigma * np.sqrt(x_pos) * dW


def fte_paths_from_dW(
    X0: float,
    kappa: float,
    theta: float,
    sigma: float,
    dt: float,
    dW: np.ndarray,
    return_auxiliary: bool = False,
) -> np.ndarray:
    """Full paths from pre-generated Brownian increments.

    Returns the non-negative read-off process max(x_tilde, 0) by default;
    set return_auxiliary=True for the raw auxiliary paths (diagnostics).
    """
    n_paths, n_steps = dW.shape

    X = np.empty((n_paths, n_steps + 1), dtype=float)
    X[:, 0] = X0

    for n in range(n_steps):
        X[:, n + 1] = fte_step(
            x_aux=X[:, n],
            kappa=kappa,
            theta=theta,
            sigma=sigma,
            dt=dt,
            dW=dW[:, n],
        )

    if return_auxiliary:
        return X

    return np.maximum(X, 0.0)


def fte_terminal_from_dW(
    X0: float,
    kappa: float,
    theta: float,
    sigma: float,
    dt: float,
    dW: np.ndarray,
) -> np.ndarray:
    """Terminal read-off values max(x_tilde_N, 0) only."""
    n_paths, n_steps = dW.shape
    x_aux = np.full(n_paths, X0, dtype=float)

    for n in range(n_steps):
        x_aux = fte_step(
            x_aux=x_aux,
            kappa=kappa,
            theta=theta,
            sigma=sigma,
            dt=dt,
            dW=dW[:, n],
        )

    return np.maximum(x_aux, 0.0)


def fte_paths(
    X0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
    return_auxiliary: bool = False,
) -> np.ndarray:
    dt = T / n_steps
    dW = make_brownian_increments(rng, n_paths, n_steps, dt)

    return fte_paths_from_dW(
        X0=X0,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        dt=dt,
        dW=dW,
        return_auxiliary=return_auxiliary,
    )


def fte_terminal(
    X0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    dt = T / n_steps
    dW = make_brownian_increments(rng, n_paths, n_steps, dt)

    return fte_terminal_from_dW(
        X0=X0,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        dt=dt,
        dW=dW,
    )
