import numpy as np
from scipy.stats import ncx2

from src.samplers.exact import cir_ncx2_params


def terminal_zero_mass(paths: np.ndarray) -> float:
    if paths.ndim != 2:
        raise ValueError("paths must be a 2D array")

    return float(np.mean(paths[:, -1] == 0.0))


def path_hit_zero_fraction(paths: np.ndarray) -> float:
    """
    Fraction of paths that hit zero at least once on the simulated grid.
    """
    if paths.ndim != 2:
        raise ValueError("paths must be a 2D array")

    return float(np.mean(np.any(paths == 0.0, axis=1)))


def zero_time_fraction(
    paths: np.ndarray,
    include_initial: bool = False,
) -> float:

    if paths.ndim != 2:
        raise ValueError("paths must be a 2D array")

    values = paths if include_initial else paths[:, 1:]

    return float(np.mean(values == 0.0))


def terminal_near_zero_mass(
    terminal_values: np.ndarray,
    epsilon: float,
) -> float:

    if epsilon < 0.0:
        raise ValueError("epsilon must be nonnegative")

    return float(np.mean(terminal_values <= epsilon))


def bernoulli_standard_error(
    p_hat: float,
    n_paths: int,
) -> float:

    if n_paths <= 0:
        raise ValueError("n_paths must be positive")

    return float(np.sqrt(p_hat * (1.0 - p_hat) / n_paths))


def exact_terminal_cdf(
    epsilon: float,
    x0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
) -> float:

    if epsilon < 0.0:
        raise ValueError("epsilon must be nonnegative")

    c, df, nc = cir_ncx2_params(
        x=x0,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        dt=T,
    )

    c = float(np.asarray(c))
    df = float(np.asarray(df))
    nc = float(np.asarray(nc))

    return float(ncx2.cdf(c * epsilon, df=df, nc=nc))