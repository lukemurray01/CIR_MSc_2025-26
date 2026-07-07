import math

import jax.numpy as jnp
import numpy as np

from klm_jax.fig3 import (
    choose_adaptive_step_size,
    fit_orders,
    implicit_lamperti_step,
    make_a_values,
    make_coefficients,
    root_mean_square_error,
    run_fig3_experiment,
)


def quick_config():
    return {
        "lambda_value": 0.05,
        "final_time": 1.0,
        "initial_y": 0.02,
        "n_paths": 64,
        "reference_power": 12,
        "n_paper_a_values": 5,
        "a_min": 0.04,
        "a_max": 1.6,
        "include_a0_diagnostic": True,
        "kappas": [2.0, 0.2],
        "levels": [4, 5],
        "rho": 64,
        "fine_steps_per_chunk": 1024,
        "a_batch_size": 5,
        "fixed_step_source": "hmax",
        "fit_x_source": "hmax",
        "plot_ef_y_diagnostic": False,
        "seed": 120339106,
    }


def test_make_a_values_includes_a0_diagnostic():
    values = make_a_values(quick_config())

    assert values[0] == 0.0
    assert len(values) == 6
    assert np.allclose(values[1:], np.linspace(0.04, 1.6, 5))


def test_make_coefficients_match_notebook_formula():
    a_values = np.array([0.0, 0.04, 1.6])
    a, sigma, alpha, beta, gamma = make_coefficients(
        kappa=2.0,
        a_values=a_values,
        lambda_value=0.05,
    )

    expected_sigma = np.sqrt(2.0 * 2.0 * 0.05 * a_values)

    assert np.allclose(np.asarray(a), a_values)
    assert np.allclose(np.asarray(sigma), expected_sigma)
    assert np.allclose(np.asarray(alpha), (4.0 * 2.0 * 0.05 - expected_sigma**2) / 8.0)
    assert math.isclose(float(beta), -1.0)
    assert np.allclose(np.asarray(gamma), expected_sigma / 2.0)


def test_implicit_lamperti_step_returns_positive_values():
    y_new = implicit_lamperti_step(
        y_old=jnp.array([0.02, 0.1]),
        brownian_increment=jnp.array([0.0, -0.01]),
        step_size=0.01,
        alpha=0.04,
        beta=-1.0,
        gamma=0.1,
    )

    assert np.all(np.asarray(y_new) > 0.0)


def test_choose_adaptive_step_size_respects_bounds():
    y_values = jnp.array([[[0.02, 2.0]], [[0.5, 1.5]]])
    current_times = jnp.zeros_like(y_values)
    hmax_by_level = jnp.array([0.25])
    rho = 64
    final_time = 1.0

    steps, used_minimum = choose_adaptive_step_size(
        y_values,
        current_times,
        hmax_by_level,
        rho,
        final_time,
    )

    assert steps.shape == y_values.shape
    assert used_minimum.shape == y_values.shape
    assert np.all(np.asarray(steps) > 0.0)
    assert np.all(np.asarray(steps) <= 0.25)


def test_root_mean_square_error_shape_and_values():
    reference = jnp.array([[1.0, 2.0], [3.0, 5.0]])
    scheme = jnp.array(
        [
            [[1.0, 4.0], [2.0, 2.0]],
            [[4.0, 5.0], [3.0, 7.0]],
        ]
    )

    rmse = root_mean_square_error(scheme, reference)

    assert rmse.shape == (2, 2)
    assert np.all(np.asarray(rmse) >= 0.0)


def test_fit_orders_recovers_known_slope():
    step_sizes = jnp.array([[0.25, 0.125, 0.0625]])
    errors = 3.0 * step_sizes**0.5

    orders = fit_orders(step_sizes, errors)

    assert np.allclose(np.asarray(orders), np.array([0.5]))


def tiny_config():
    # Small enough that the simulating run stays in test-suite time budgets.
    config = quick_config()
    config.update(
        n_paths=16,
        reference_power=10,
        n_paper_a_values=2,
        levels=[4, 5],
        kappas=[2.0],
    )
    return config


def test_run_fig3_experiment_simulates_and_reports_rmse():
    config = tiny_config()
    rows = run_fig3_experiment(config)

    # kappas * (a0 diagnostic + paper a values) * levels
    assert len(rows) == 1 * 3 * 2

    expected_keys = {
        "kappa",
        "a",
        "sigma",
        "level",
        "hmax",
        "reference_power",
        "reference_step",
        "number_of_fine_steps",
        "n_paths",
        "rmse",
        "backstop_fraction",
        "mean_steps_per_path",
        "runtime_s",
    }
    assert set(rows[0]) == expected_keys

    for row in rows:
        assert np.isfinite(row["rmse"])
        assert row["rmse"] >= 0.0
        assert 0.0 <= row["backstop_fraction"] <= 1.0
        assert row["mean_steps_per_path"] > 0.0

    # Stochastic rows (a > 0) must show a genuinely positive coupled error.
    stochastic = [r for r in rows if r["a"] > 0.0]
    assert all(r["rmse"] > 0.0 for r in stochastic)


def test_run_fig3_experiment_rejects_oversized_reference_power():
    import pytest

    config = tiny_config()
    config["reference_power"] = 25

    with pytest.raises(ValueError):
        run_fig3_experiment(config)