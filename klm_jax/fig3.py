# KLM Fig.-3-style diagnostic: strong convergence of the backstopped
# adaptive scheme across the Feller ratio  a = sigma^2 / (2*kappa*lambda).
#
# For each (kappa, a) the coupled JAX kernel is run at several h_max levels
# against the drift-implicit reference on a shared fine Brownian path, and
# the observed L2 order is fitted from the RMSE-vs-h_max slope.  Note that
# a < 2 corresponds to delta = 2/a > 1, so alpha > 0 across the sweep and
# the implicit backstop is admissible.

import time

import jax
import jax.numpy as jnp
import numpy as np

from klm_jax.coefficients import sigma_from_a
from src.jax_schemes import (
    if_terminal_from_fine_dW_jax,
    klm_backstop_terminal_from_fine_dW_jax,
)

jax.config.update("jax_enable_x64", True)

# Fine grids above 2^18 steps are too large for this compact in-memory driver.
# The historical paper-scale Kaggle route uses chunked generation, so use that
# notebook for 2^-25 runs and this cap for reference-sensitivity checks.
MAX_IN_MEMORY_REFERENCE_POWER = 18


def make_a_values(config):
    paper_values = np.linspace(
        config["a_min"],
        config["a_max"],
        config["n_paper_a_values"],
        dtype=np.float64,
    )

    if config.get("include_a0_diagnostic", False):
        return np.concatenate(([0.0], paper_values))

    return paper_values


def make_coefficients(kappa, a_values, lambda_value):
    a = jnp.asarray(a_values, dtype=jnp.float64)
    sigma = jnp.asarray(sigma_from_a(a, kappa, lambda_value), dtype=jnp.float64)

    alpha = (4.0 * kappa * lambda_value - sigma * sigma) / 8.0
    beta = -kappa / 2.0
    gamma = sigma / 2.0

    return a, sigma, alpha, beta, gamma


def implicit_lamperti_step(y_old, brownian_increment, step_size, alpha, beta, gamma):
    u = y_old + gamma * brownian_increment
    denominator = 1.0 - beta * step_size

    return (
        u / (2.0 * denominator)
        + jnp.sqrt(
            u * u / (4.0 * denominator * denominator)
            + alpha * step_size / denominator
        )
    )


def choose_adaptive_step_size(y_values, current_times, hmax_by_level, rho, final_time):
    proposed = hmax_by_level[None, :, None] * jnp.minimum(1.0, jnp.abs(y_values))
    minimum = (hmax_by_level / rho)[None, :, None]

    used_minimum_step = proposed <= minimum
    chosen = jnp.where(used_minimum_step, minimum, proposed)

    remaining_time = final_time - current_times
    return jnp.minimum(chosen, remaining_time), used_minimum_step


def root_mean_square_error(scheme_x_values, reference_x_values):
    errors = scheme_x_values - reference_x_values[:, None, :]
    return jnp.sqrt(jnp.mean(errors * errors, axis=2))


def fit_orders(step_sizes, errors):
    log_h = jnp.log(step_sizes)
    log_e = jnp.log(errors)

    centered_h = log_h - jnp.mean(log_h, axis=1, keepdims=True)
    centered_e = log_e - jnp.mean(log_e, axis=1, keepdims=True)

    numerator = jnp.sum(centered_h * centered_e, axis=1)
    denominator = jnp.sum(centered_h * centered_h, axis=1)

    return numerator / denominator


def run_fig3_experiment(config, outdir=None):
    """Simulate the Fig.-3 sweep and return one row per (kappa, a, level).

    Each row carries the measured RMSE against the shared-path reference,
    the backstop-usage statistics, and the wall-clock runtime of the
    scheme call, so the CSV alone reproduces the convergence figure.
    """
    lambda_value = config["lambda_value"]
    final_time = config["final_time"]
    initial_x = config["initial_y"]
    rho = config["rho"]
    n_paths = int(config["n_paths"])
    reference_power = int(config["reference_power"])

    if reference_power > MAX_IN_MEMORY_REFERENCE_POWER:
        raise ValueError(
            f"reference_power={reference_power} exceeds the in-memory cap "
            f"{MAX_IN_MEMORY_REFERENCE_POWER}; lower it in the config"
        )

    reference_step = 2.0 ** (-reference_power)
    number_of_fine_steps = int(round(final_time / reference_step))

    a_values = make_a_values(config)
    hmax_by_level = np.array([2.0 ** (-level) for level in config["levels"]])

    seed = int(config["seed"])
    rows = []

    for kappa in config["kappas"]:
        # One fine Brownian path per kappa, shared across a-values and levels
        # so that comparisons along the sweep are not clouded by Monte Carlo
        # variation.
        rng = np.random.default_rng(seed)
        dW_fine = np.sqrt(reference_step) * rng.standard_normal(
            (n_paths, number_of_fine_steps)
        )

        _, sigmas, alphas, beta, gammas = make_coefficients(
            kappa, a_values, lambda_value
        )

        for a_i, sigma_i in zip(np.asarray(a_values), np.asarray(sigmas), strict=True):
            reference = if_terminal_from_fine_dW_jax(
                X0=initial_x,
                kappa=kappa,
                theta=lambda_value,
                sigma=float(sigma_i),
                T=final_time,
                dW_fine=dW_fine,
            )
            reference = np.asarray(reference)

            for level, hmax in zip(config["levels"], hmax_by_level, strict=True):
                start = time.perf_counter()
                terminal, stats = klm_backstop_terminal_from_fine_dW_jax(
                    X0=initial_x,
                    kappa=kappa,
                    theta=lambda_value,
                    sigma=float(sigma_i),
                    T=final_time,
                    h_max=float(hmax),
                    dW_fine=dW_fine,
                    rho=rho,
                    backstop="implicit",
                )
                runtime = time.perf_counter() - start

                terminal = np.asarray(terminal)
                rmse = float(np.sqrt(np.mean((terminal - reference) ** 2)))

                rows.append(
                    {
                        "kappa": float(kappa),
                        "a": float(a_i),
                        "sigma": float(sigma_i),
                        "level": int(level),
                        "hmax": float(hmax),
                        "reference_power": reference_power,
                        "reference_step": float(reference_step),
                        "number_of_fine_steps": number_of_fine_steps,
                        "n_paths": n_paths,
                        "rmse": rmse,
                        "backstop_fraction": stats["backstop_fraction"],
                        "mean_steps_per_path": stats["n_steps_total"] / n_paths,
                        "runtime_s": runtime,
                    }
                )

    return rows


def fitted_orders_by_group(rows):
    """Fit the L2 order per (kappa, a) across h_max levels."""
    orders = []
    groups = sorted({(r["kappa"], r["a"]) for r in rows})
    for kappa, a in groups:
        group = [r for r in rows if r["kappa"] == kappa and r["a"] == a]
        if len(group) < 2:
            continue
        h = np.array([r["hmax"] for r in group])
        e = np.maximum(np.array([r["rmse"] for r in group]), 1e-300)
        slope = np.polyfit(np.log(h), np.log(e), 1)[0]
        orders.append({"kappa": kappa, "a": a, "fitted_order": float(slope)})
    return orders
