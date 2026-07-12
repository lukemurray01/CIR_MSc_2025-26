from collections.abc import Callable

import numpy as np
from scipy.integrate import quad
from scipy.stats import ncx2

from src.samplers.exact import cir_ncx2_params


ArrayPayoff = Callable[[np.ndarray], np.ndarray]
PathFunctional = Callable[[np.ndarray, float], np.ndarray]


def g1_identity(x: np.ndarray) -> np.ndarray:

    return x


def g2_squared_call(
    x: np.ndarray,
    strike: float = 0.02,
) -> np.ndarray:

    return np.maximum(x - strike, 0.0) ** 2


def g3_exp_minus_x(x: np.ndarray) -> np.ndarray:

    return np.exp(-x)


def trapezoidal_integral(paths: np.ndarray, dt: float) -> np.ndarray:

    if paths.ndim != 2:
        raise ValueError("paths must be a 2D array")

    if dt <= 0.0:
        raise ValueError("dt must be positive")

    return dt * (
        0.5 * paths[:, 0]
        + np.sum(paths[:, 1:-1], axis=1)
        + 0.5 * paths[:, -1]
    )


def g4_bond_discount_from_paths(paths: np.ndarray, dt: float) -> np.ndarray:

    integral = trapezoidal_integral(paths, dt)

    return np.exp(-integral)


TERMINAL_PAYOFFS: dict[str, ArrayPayoff] = {
    "g1": g1_identity,
    "g2": g2_squared_call,
    "g3": g3_exp_minus_x,
}


PATH_FUNCTIONALS: dict[str, PathFunctional] = {
    "g4": g4_bond_discount_from_paths,
}


def exact_cir_mean(
    x0: float,
    kappa: float,
    theta: float,
    T: float,
) -> float:
    """
    Exact CIR first moment E[X_T | X_0 = x0].
    """
    return float(theta + (x0 - theta) * np.exp(-kappa * T))


def exact_cir_laplace_transform(
    x0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
    u: float,
) -> float:

    if u < 0.0:
        raise ValueError("u must be nonnegative")

    exp_term = np.exp(-kappa * T)
    delta = 4.0 * kappa * theta / sigma**2
    q = sigma**2 * (1.0 - exp_term) / (2.0 * kappa)

    denominator = 1.0 + u * q

    value = denominator ** (-0.5 * delta) * np.exp(
        -u * x0 * exp_term / denominator
    )

    return float(value)


def exact_g3_exp_minus_x_expectation(
    x0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
) -> float:
    """
    Exact E[exp(-X_T)].
    """
    return exact_cir_laplace_transform(
        x0=x0,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        T=T,
        u=1.0,
    )


def exact_terminal_pdf(
    x: float | np.ndarray,
    x0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
) -> float | np.ndarray:

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

    x_array = np.asarray(x)

    density = c * ncx2.pdf(c * x_array, df=df, nc=nc)

    return density


def exact_g2_squared_call_expectation(
    x0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
    strike: float = 0.02,
) -> float:

    def integrand(x: float) -> float:
        return float((x - strike) ** 2 * exact_terminal_pdf(
            x=x,
            x0=x0,
            kappa=kappa,
            theta=theta,
            sigma=sigma,
            T=T,
        ))

    value, _error = quad(
        integrand,
        strike,
        np.inf,
        epsabs=1.0e-12,
        epsrel=1.0e-10,
        limit=200,
    )

    return float(value)


def affine_cir_bond_price(
    x0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
) -> float:
    gamma = np.sqrt(kappa**2 + 2.0 * sigma**2)
    exp_gamma_T = np.exp(gamma * T)

    denominator = (gamma + kappa) * (exp_gamma_T - 1.0) + 2.0 * gamma

    B = 2.0 * (exp_gamma_T - 1.0) / denominator

    A = (
        2.0 * gamma * np.exp(0.5 * (kappa + gamma) * T)
        / denominator
    ) ** (2.0 * kappa * theta / sigma**2)

    return float(A * np.exp(-B * x0))


def terminal_exact_expectations(
    x0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
    strike: float = 0.02,
) -> dict[str, float]:

    return {
        "g1": exact_cir_mean(
            x0=x0,
            kappa=kappa,
            theta=theta,
            T=T,
        ),
        "g2": exact_g2_squared_call_expectation(
            x0=x0,
            kappa=kappa,
            theta=theta,
            sigma=sigma,
            T=T,
            strike=strike,
        ),
        "g3": exact_g3_exp_minus_x_expectation(
            x0=x0,
            kappa=kappa,
            theta=theta,
            sigma=sigma,
            T=T,
        ),
    }


def weak_error_from_values(
    approx_values: np.ndarray,
    exact_expectation: float,
) -> float:

    approx_mean = float(np.mean(approx_values))

    return abs(approx_mean - exact_expectation)


def monte_carlo_standard_error(values: np.ndarray) -> float:

    return float(np.std(values, ddof=1) / np.sqrt(values.size))