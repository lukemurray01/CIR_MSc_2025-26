import numpy as np

from src.samplers.hh_milstein import hh_milstein_terminal_from_dW
from src.utils.brownian import aggregate_brownian_increments
from src.utils.rng import make_brownian_increments


def strong_errors_fixed_step(
        scheme_terminal_from_dW,  
        X0: float,
        kappa: float,
        theta: float,
        sigma: float,
        T: float,
        reference_n_steps: int,
        coarse_n_steps_list: list[int],
        n_paths: int,
        rng: np.random.Generator,
) -> dict:
    dt_fine = T / reference_n_steps

    dW_fine = make_brownian_increments(rng, n_paths, reference_n_steps, dt_fine)

    # HH truncated Milstein: proven L1 rate min(1, delta)/2 over the full
    # parameter range (Hefter--Herzwurm 2018, arXiv:1608.00410) -- i.e. only
    # delta/2 in the accessible-boundary regimes D/E, so the reference itself
    # converges slowly there.  Chosen over the drift-implicit Lamperti scheme,
    # whose quadratic loses its real root for alpha < 0 and returns NaN.
    # Matches the reference used by experiments/run_strong_error.py.
    reference_terminal = hh_milstein_terminal_from_dW(
        X0=X0, kappa=kappa, theta=theta, sigma=sigma, dt=dt_fine, dW=dW_fine,
    )

    dt_list, l1_list, l2_list = [], [], []

    for n_steps in coarse_n_steps_list:
        if reference_n_steps % n_steps != 0:
            raise ValueError(
                f"reference_n_steps ({reference_n_steps}) must be divisible "
                f"by coarse n_steps ({n_steps})"
            )
        factor = reference_n_steps // n_steps
        dt_coarse = dt_fine * factor

        dW_coarse = aggregate_brownian_increments(dW_fine, factor)

        scheme_terminal = scheme_terminal_from_dW(
            X0=X0, kappa=kappa, theta=theta, sigma=sigma,
            dt=dt_coarse, dW=dW_coarse,
        )

        diff = scheme_terminal - reference_terminal
        dt_list.append(dt_coarse)
        l1_list.append(np.mean(np.abs(diff)))
        l2_list.append(np.sqrt(np.mean(diff**2)))

    return {
        "dt": np.array(dt_list),
        "l1": np.array(l1_list),
        "l2": np.array(l2_list),
    }


def fit_loglog_order(step_sizes: np.ndarray, errors: np.ndarray) -> float:
    
    log_h = np.log(np.asarray(step_sizes))
    log_e = np.log(np.asarray(errors))
    slope, _intercept = np.polyfit(log_h, log_e, 1)
    return slope