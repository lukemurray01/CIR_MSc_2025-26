import numpy as np

from src.utils.rng import make_brownian_increments


def hh_milstein_step(
    x: np.ndarray,
    kappa: float,
    theta: float,
    sigma: float,
    dt: float,
    dW: np.ndarray,
) -> np.ndarray:
    floor = 0.25 * sigma**2 * dt

    r1 = np.maximum(
        0.5 * sigma * np.sqrt(dt),
        np.sqrt(np.maximum(floor, x)) + 0.5 * sigma * dW,
    )

    x_hat = r1**2 + dt * (kappa * (theta - x) - 0.25 * sigma**2)

    return np.maximum(x_hat, 0.0)


def hh_milstein_paths_from_dW(
    X0: float,
    kappa: float,
    theta: float,
    sigma: float,
    dt: float,
    dW: np.ndarray,
) -> np.ndarray:
    n_paths, n_steps = dW.shape

    X = np.empty((n_paths, n_steps + 1), dtype=float)
    X[:, 0] = X0

    for n in range(n_steps):
        X[:, n + 1] = hh_milstein_step(
            x=X[:, n],
            kappa=kappa,
            theta=theta,
            sigma=sigma,
            dt=dt,
            dW=dW[:, n],
        )

    return X


def hh_milstein_terminal_from_dW(
    X0: float,
    kappa: float,
    theta: float,
    sigma: float,
    dt: float,
    dW: np.ndarray,
) -> np.ndarray:
    n_paths, n_steps = dW.shape
    x = np.full(n_paths, X0, dtype=float)

    for n in range(n_steps):
        x = hh_milstein_step(
            x=x,
            kappa=kappa,
            theta=theta,
            sigma=sigma,
            dt=dt,
            dW=dW[:, n],
        )

    return x


def hh_milstein_paths(
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

    return hh_milstein_paths_from_dW(
        X0=X0,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        dt=dt,
        dW=dW,
    )


def hh_milstein_terminal(
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

    return hh_milstein_terminal_from_dW(
        X0=X0,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        dt=dt,
        dW=dW,
    )