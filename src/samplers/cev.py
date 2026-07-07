# Mean-reverting CEV samplers.
#
#   dF = kappa*(theta - F) dt + sigma * F^beta dW,   beta in [1/2, 1).
#
# The unnormalised Lamperti transform  Y = F^{1-beta} / (1-beta)  gives
#
#   dY = [ kappa*theta*((1-beta)*Y)^{-beta/(1-beta)}
#          - kappa*(1-beta)*Y
#          - beta*sigma^2 / (2*(1-beta)*Y) ] dt + sigma dW,
#
# which keeps the diffusion coefficient equal to sigma (thesis background
# chapter, CEV section).  At beta = 1/2 this reduces, with Y = 2*sqrt(F),
# to exactly twice the CIR Lamperti update used by projected Euler; the
# beta = 1/2 consistency test asserts that equality pathwise.

import numpy as np

from src.utils.rng import make_brownian_increments

# As in src/samplers/projected_euler.py, the default floor must scale with the
# one-step noise (diffusion coefficient sigma here), not with dt: a dt-scaled
# floor becomes an absorbing trap near the boundary.  sigma*sqrt(dt) also
# preserves the beta = 1/2 reduction: Y_cev = 2*Y_cir, so this floor is
# exactly twice the CIR default 0.5*sigma*sqrt(dt).


def default_cev_y_floor(sigma: float, dt: float) -> float:
    return sigma * np.sqrt(dt)


def cev_exact_mean(F0: float, kappa: float, theta: float, T: float) -> float:
    """First moment; closes for every beta in the mean-reverting family."""
    return float(theta + (F0 - theta) * np.exp(-kappa * T))


def cev_lamperti_drift(
    y: np.ndarray,
    kappa: float,
    theta: float,
    sigma: float,
    beta: float,
) -> np.ndarray:
    one_minus_beta = 1.0 - beta

    mean_reversion = kappa * theta * (one_minus_beta * y) ** (-beta / one_minus_beta)
    linear = -kappa * one_minus_beta * y
    ito_correction = -beta * sigma**2 / (2.0 * one_minus_beta * y)

    return mean_reversion + linear + ito_correction


def y_from_f(f: np.ndarray, beta: float) -> np.ndarray:
    return f ** (1.0 - beta) / (1.0 - beta)


def f_from_y(y: np.ndarray, beta: float) -> np.ndarray:
    return ((1.0 - beta) * y) ** (1.0 / (1.0 - beta))


def cev_projected_step(
    y: np.ndarray,
    kappa: float,
    theta: float,
    sigma: float,
    beta: float,
    dt: float,
    dW: np.ndarray,
    y_floor: float,
) -> np.ndarray:
    """Explicit Lamperti Euler step projected onto a strictly positive floor."""
    if y_floor <= 0.0:
        raise ValueError("y_floor must be positive")

    y_safe = np.maximum(y, y_floor)
    y_hat = y_safe + cev_lamperti_drift(y_safe, kappa, theta, sigma, beta) * dt + sigma * dW

    return np.maximum(y_hat, y_floor)


def cev_projected_terminal_from_dW(
    F0: float,
    kappa: float,
    theta: float,
    sigma: float,
    beta: float,
    dt: float,
    dW: np.ndarray,
    y_floor: float | None = None,
) -> np.ndarray:
    if not 0.5 <= beta < 1.0:
        raise ValueError("beta must lie in [1/2, 1)")

    if y_floor is None:
        y_floor = default_cev_y_floor(sigma, dt)

    n_paths, n_steps = dW.shape
    y = np.full(n_paths, max(y_from_f(F0, beta), y_floor), dtype=float)

    for n in range(n_steps):
        y = cev_projected_step(
            y=y,
            kappa=kappa,
            theta=theta,
            sigma=sigma,
            beta=beta,
            dt=dt,
            dW=dW[:, n],
            y_floor=y_floor,
        )

    return f_from_y(y, beta)


def cev_projected_paths_from_dW(
    F0: float,
    kappa: float,
    theta: float,
    sigma: float,
    beta: float,
    dt: float,
    dW: np.ndarray,
    y_floor: float | None = None,
) -> np.ndarray:
    if not 0.5 <= beta < 1.0:
        raise ValueError("beta must lie in [1/2, 1)")

    if y_floor is None:
        y_floor = default_cev_y_floor(sigma, dt)

    n_paths, n_steps = dW.shape

    Y = np.empty((n_paths, n_steps + 1), dtype=float)
    Y[:, 0] = max(y_from_f(F0, beta), y_floor)

    for n in range(n_steps):
        Y[:, n + 1] = cev_projected_step(
            y=Y[:, n],
            kappa=kappa,
            theta=theta,
            sigma=sigma,
            beta=beta,
            dt=dt,
            dW=dW[:, n],
            y_floor=y_floor,
        )

    return f_from_y(Y, beta)


def cev_fte_step(
    f_aux: np.ndarray,
    kappa: float,
    theta: float,
    sigma: float,
    beta: float,
    dt: float,
    dW: np.ndarray,
) -> np.ndarray:
    """LKvD-style full truncation in the original coordinate with F^beta diffusion.

    Carries the possibly negative auxiliary variable; read off max(f_aux, 0).
    """
    f_pos = np.maximum(f_aux, 0.0)

    return f_aux + kappa * (theta - f_pos) * dt + sigma * f_pos**beta * dW


def cev_fte_terminal_from_dW(
    F0: float,
    kappa: float,
    theta: float,
    sigma: float,
    beta: float,
    dt: float,
    dW: np.ndarray,
) -> np.ndarray:
    if not 0.5 <= beta < 1.0:
        raise ValueError("beta must lie in [1/2, 1)")

    n_paths, n_steps = dW.shape
    f_aux = np.full(n_paths, F0, dtype=float)

    for n in range(n_steps):
        f_aux = cev_fte_step(
            f_aux=f_aux,
            kappa=kappa,
            theta=theta,
            sigma=sigma,
            beta=beta,
            dt=dt,
            dW=dW[:, n],
        )

    return np.maximum(f_aux, 0.0)


def cev_projected_terminal(
    F0: float,
    kappa: float,
    theta: float,
    sigma: float,
    beta: float,
    T: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
    y_floor: float | None = None,
) -> np.ndarray:
    dt = T / n_steps
    dW = make_brownian_increments(rng, n_paths, n_steps, dt)

    return cev_projected_terminal_from_dW(
        F0=F0,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        beta=beta,
        dt=dt,
        dW=dW,
        y_floor=y_floor,
    )


def cev_fte_terminal(
    F0: float,
    kappa: float,
    theta: float,
    sigma: float,
    beta: float,
    T: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    dt = T / n_steps
    dW = make_brownian_increments(rng, n_paths, n_steps, dt)

    return cev_fte_terminal_from_dW(
        F0=F0,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        beta=beta,
        dt=dt,
        dW=dW,
    )
