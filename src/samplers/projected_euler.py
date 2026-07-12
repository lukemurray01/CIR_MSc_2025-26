import numpy as np

from src.utils.cir_params import kl_alpha
from src.utils.rng import make_brownian_increments

# The default floor must scale like the one-step noise, y_floor ~ gamma*sqrt(dt)
# with gamma = sigma/2, so that the drift kick at the floor (alpha/y_floor)*dt
# stays O(sqrt(dt)).  A dt-scaled floor makes that kick O(1); for alpha < 0
# (regimes D/E) escaping the floor then requires a Gaussian tail event
# N > 2|alpha|/(sigma*sqrt(dt)), so paths absorb at the floor as dt -> 0 and
# the strong error diverges instead of converging.  In X-coordinates the
# default floor is y_floor^2 = sigma^2*dt/4, the Hefter--Herzwurm truncation
# level.


def default_y_floor(sigma: float, dt: float) -> float:
    return 0.5 * sigma * np.sqrt(dt)


def projected_euler_step(
    y: np.ndarray,
    alpha: float,
    kappa: float,
    sigma: float,
    dt: float,
    dW: np.ndarray,
    y_floor: float,
) -> np.ndarray:
    if y_floor <= 0.0:
        raise ValueError("y_floor must be positive")

    y_safe = np.maximum(y, y_floor)

    y_hat = (
        y_safe
        + (alpha / y_safe - 0.5 * kappa * y_safe) * dt
        + 0.5 * sigma * dW
    )

    return np.maximum(y_hat, y_floor)


def projected_euler_paths_from_dW(
    X0: float,
    kappa: float,
    theta: float,
    sigma: float,
    dt: float,
    dW: np.ndarray,
    y_floor: float | None = None,
) -> np.ndarray:
    alpha = kl_alpha(kappa, theta, sigma)

    if y_floor is None:
        y_floor = default_y_floor(sigma, dt)

    n_paths, n_steps = dW.shape

    Y = np.empty((n_paths, n_steps + 1), dtype=float)
    Y[:, 0] = max(np.sqrt(X0), y_floor)

    for n in range(n_steps):
        Y[:, n + 1] = projected_euler_step(
            y=Y[:, n],
            alpha=alpha,
            kappa=kappa,
            sigma=sigma,
            dt=dt,
            dW=dW[:, n],
            y_floor=y_floor,
        )

    return Y**2


def projected_euler_terminal_from_dW(
    X0: float,
    kappa: float,
    theta: float,
    sigma: float,
    dt: float,
    dW: np.ndarray,
    y_floor: float | None = None,
) -> np.ndarray:
    alpha = kl_alpha(kappa, theta, sigma)

    if y_floor is None:
        y_floor = default_y_floor(sigma, dt)

    n_paths, n_steps = dW.shape
    y = np.full(n_paths, max(np.sqrt(X0), y_floor), dtype=float)

    for n in range(n_steps):
        y = projected_euler_step(
            y=y,
            alpha=alpha,
            kappa=kappa,
            sigma=sigma,
            dt=dt,
            dW=dW[:, n],
            y_floor=y_floor,
        )

    return y**2


def projected_euler_terminal_with_stats_from_dW(
    X0: float,
    kappa: float,
    theta: float,
    sigma: float,
    dt: float,
    dW: np.ndarray,
    y_floor: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Projected Euler terminal value plus projection diagnostics.

    The numerical method is unchanged.  The counters record how often the
    Lamperti state is already at the floor before an update and how often the
    unconstrained explicit update would be projected back to the floor.
    """
    alpha = kl_alpha(kappa, theta, sigma)

    if y_floor is None:
        y_floor = default_y_floor(sigma, dt)
    if y_floor <= 0.0:
        raise ValueError("y_floor must be positive")

    n_paths, n_steps = dW.shape
    y = np.full(n_paths, max(np.sqrt(X0), y_floor), dtype=float)

    pre_floor_count = 0
    post_projection_count = 0
    total_updates = n_paths * n_steps

    for n in range(n_steps):
        pre_floor_count += int(np.count_nonzero(y <= y_floor))

        y_safe = np.maximum(y, y_floor)
        y_hat = (
            y_safe
            + (alpha / y_safe - 0.5 * kappa * y_safe) * dt
            + 0.5 * sigma * dW[:, n]
        )

        post_projection_count += int(np.count_nonzero(y_hat <= y_floor))
        y = np.maximum(y_hat, y_floor)

    stats = {
        "y_floor": float(y_floor),
        "pre_floor_fraction": pre_floor_count / max(total_updates, 1),
        "post_projection_fraction": post_projection_count / max(total_updates, 1),
    }

    return y**2, stats


def projected_euler_paths(
    X0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
    y_floor: float | None = None,
) -> np.ndarray:
    dt = T / n_steps
    dW = make_brownian_increments(rng, n_paths, n_steps, dt)

    return projected_euler_paths_from_dW(
        X0=X0,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        dt=dt,
        dW=dW,
        y_floor=y_floor,
    )


def projected_euler_terminal(
    X0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
    y_floor: float | None = None,
) -> np.ndarray:
    dt = T / n_steps
    dW = make_brownian_increments(rng, n_paths, n_steps, dt)

    return projected_euler_terminal_from_dW(
        X0=X0,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        dt=dt,
        dW=dW,
        y_floor=y_floor,
    )
