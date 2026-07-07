import numpy as np
import pytest

from src.metrics.weak_error import (
    TERMINAL_PAYOFFS,
    PATH_FUNCTIONALS,
    affine_cir_bond_price,
    exact_cir_laplace_transform,
    exact_cir_mean,
    exact_g3_exp_minus_x_expectation,
    g1_identity,
    g2_squared_call,
    g3_exp_minus_x,
    g4_bond_discount_from_paths,
    monte_carlo_standard_error,
    terminal_exact_expectations,
    trapezoidal_integral,
    weak_error_from_values,
)


def test_g1_identity() -> None:
    x = np.array([0.0, 0.02, 0.05])

    np.testing.assert_allclose(g1_identity(x), x)


def test_g2_squared_call_default_strike() -> None:
    x = np.array([0.0, 0.02, 0.03, 0.05])

    expected = np.array([0.0, 0.0, 0.0001, 0.0009])

    np.testing.assert_allclose(g2_squared_call(x), expected)


def test_g2_squared_call_custom_strike() -> None:
    x = np.array([0.01, 0.03, 0.05])

    expected = np.array([0.0, 0.0001, 0.0009])

    np.testing.assert_allclose(g2_squared_call(x, strike=0.02), expected)


def test_g3_exp_minus_x() -> None:
    x = np.array([0.0, 1.0])

    expected = np.exp(-x)

    np.testing.assert_allclose(g3_exp_minus_x(x), expected)


def test_terminal_payoff_registry_contains_only_g1_to_g3() -> None:
    assert list(TERMINAL_PAYOFFS.keys()) == ["g1", "g2", "g3"]


def test_path_functional_registry_contains_g4() -> None:
    assert list(PATH_FUNCTIONALS.keys()) == ["g4"]


def test_trapezoidal_integral_constant_paths() -> None:
    paths = np.array(
        [
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
        ]
    )

    dt = 0.5

    expected = np.array([2.0, 3.0])

    np.testing.assert_allclose(trapezoidal_integral(paths, dt), expected)


def test_trapezoidal_integral_rejects_non_2d_array() -> None:
    paths = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="paths must be a 2D array"):
        trapezoidal_integral(paths, dt=0.5)


def test_trapezoidal_integral_rejects_nonpositive_dt() -> None:
    paths = np.array([[1.0, 2.0, 3.0]])

    with pytest.raises(ValueError, match="dt must be positive"):
        trapezoidal_integral(paths, dt=0.0)


def test_g4_bond_discount_from_constant_path() -> None:
    paths = np.array([[2.0, 2.0, 2.0]])

    dt = 0.5

    expected = np.array([np.exp(-2.0)])

    np.testing.assert_allclose(g4_bond_discount_from_paths(paths, dt), expected)


def test_exact_cir_mean_at_initial_mean() -> None:
    x0 = 0.02
    kappa = 2.0
    theta = 0.02
    T = 1.0

    expected = 0.02

    np.testing.assert_allclose(
        exact_cir_mean(x0=x0, kappa=kappa, theta=theta, T=T),
        expected,
    )


def test_exact_cir_laplace_transform_at_zero_is_one() -> None:
    value = exact_cir_laplace_transform(
        x0=0.02,
        kappa=2.0,
        theta=0.02,
        sigma=0.2,
        T=1.0,
        u=0.0,
    )

    np.testing.assert_allclose(value, 1.0)


def test_exact_g3_matches_laplace_transform_at_one() -> None:
    params = {
        "x0": 0.02,
        "kappa": 2.0,
        "theta": 0.02,
        "sigma": 0.2,
        "T": 1.0,
    }

    expected = exact_cir_laplace_transform(**params, u=1.0)
    actual = exact_g3_exp_minus_x_expectation(**params)

    np.testing.assert_allclose(actual, expected)


def test_affine_cir_bond_price_is_between_zero_and_one() -> None:
    price = affine_cir_bond_price(
        x0=0.02,
        kappa=2.0,
        theta=0.02,
        sigma=0.2,
        T=1.0,
    )

    assert 0.0 < price < 1.0


def test_terminal_exact_expectations_has_expected_keys() -> None:
    values = terminal_exact_expectations(
        x0=0.02,
        kappa=2.0,
        theta=0.02,
        sigma=0.2,
        T=1.0,
    )

    assert list(values.keys()) == ["g1", "g2", "g3"]

    assert np.isfinite(values["g1"])
    assert np.isfinite(values["g2"])
    assert np.isfinite(values["g3"])

    assert values["g2"] >= 0.0
    assert 0.0 < values["g3"] < 1.0


def test_weak_error_from_values() -> None:
    values = np.array([1.0, 2.0, 3.0])

    error = weak_error_from_values(values, exact_expectation=2.5)

    np.testing.assert_allclose(error, 0.5)


def test_monte_carlo_standard_error() -> None:
    values = np.array([1.0, 2.0, 3.0])

    expected = 1.0 / np.sqrt(3.0)

    np.testing.assert_allclose(monte_carlo_standard_error(values), expected)