# Drift implicit lamperti scheme (IF) for CIR

# Used as a reference solution in KLM paper

import numpy as np
from src.utils.cir_params import kl_alpha
from src.utils.cir_params import kl_coefficients
from src.utils.rng import make_brownian_increments

def if_step(
        y: np.ndarray,
        alpha: float,
        beta: float,
        gamma: float,
        dt: float,
        dW: np.ndarray,
) -> np.ndarray:
    a = 1.0 - beta * dt
    b = -(y + gamma * dW)
    c = -alpha * dt

    # Pos root of y^2 + by + c = 0
    discriminant = b**2 - 4.0 * a * c
    return (-b + np.sqrt(discriminant)) / (2.0 * a)


def if_paths_from_dW(
        X0: float,
        kappa: float,
        theta: float,
        sigma: float,
        dt: float,
        dW: np.ndarray,
) -> np.ndarray:
    alpha, beta, gamma = kl_coefficients(kappa, theta, sigma)

    n_paths, n_steps = dW.shape

    Y = np.empty( (n_paths, n_steps + 1), dtype = float )
    Y[:, 0] = np.sqrt(X0)

    for n in range(n_steps):
        Y[:, n+1] = if_step(
            y = Y[:, n],
            alpha = alpha,
            beta = beta,
            gamma = gamma,
            dt = dt,
            dW = dW[:, n],
        )

    return Y**2 # as X = Y^2

def if_terminal_from_dW(
        X0: float,
        kappa: float,
        theta: float,
        sigma: float,
        dt: float,
        dW: np.ndarray,
) -> np.ndarray:
    alpha = kl_alpha(kappa, theta, sigma)
    beta = -kappa / 2.0
    gamma = sigma / 2.0

    n_paths, n_steps = dW.shape

    y = np.full(n_paths, np.sqrt(X0), dtype=float)

    for n in range(n_steps):
        y = if_step(
            y=y,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            dt=dt,
            dW=dW[:, n],
        )

    return y**2


def if_paths(X0, kappa, theta, sigma, T, n_steps, n_paths, rng):
    dt = T / n_steps
    dW = make_brownian_increments(rng, n_paths, n_steps, dt)
    return if_paths_from_dW(X0, kappa, theta, sigma, dt, dW)


def if_terminal(X0, kappa, theta, sigma, T, n_steps, n_paths, rng):
    dt = T / n_steps
    dW = make_brownian_increments(rng, n_paths, n_steps, dt)
    return if_terminal_from_dW(X0, kappa, theta, sigma, dt, dW)