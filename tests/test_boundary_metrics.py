import numpy as np
import pytest

from src.metrics.boundary import (
    bernoulli_standard_error,
    exact_terminal_cdf,
    path_hit_zero_fraction,
    terminal_near_zero_mass,
    terminal_zero_mass,
    zero_time_fraction,
)


def test_terminal_zero_mass() -> None:
    paths = np.array(
        [
            [0.02, 0.01, 0.00],
            [0.02, 0.03, 0.04],
            [0.02, 0.00, 0.00],
        ]
    )

    assert terminal_zero_mass(paths) == pytest.approx(2.0 / 3.0)


def test_path_hit_zero_fraction() -> None:
    paths = np.array(
        [
            [0.02, 0.01, 0.00],
            [0.02, 0.03, 0.04],
            [0.02, 0.00, 0.05],
        ]
    )

    assert path_hit_zero_fraction(paths) == pytest.approx(2.0 / 3.0)


def test_zero_time_fraction_excludes_initial_by_default() -> None:
    paths = np.array(
        [
            [0.02, 0.00, 0.00],
            [0.02, 0.03, 0.04],
            [0.02, 0.00, 0.05],
        ]
    )

    assert zero_time_fraction(paths) == pytest.approx(3.0 / 6.0)


def test_terminal_near_zero_mass() -> None:
    terminal_values = np.array([0.0, 1.0e-8, 1.0e-4, 0.01])

    assert terminal_near_zero_mass(terminal_values, 1.0e-6) == pytest.approx(2.0 / 4.0)


def test_terminal_near_zero_mass_rejects_negative_epsilon() -> None:
    terminal_values = np.array([0.0, 0.01])

    with pytest.raises(ValueError, match="epsilon must be nonnegative"):
        terminal_near_zero_mass(terminal_values, -1.0)


def test_bernoulli_standard_error() -> None:
    se = bernoulli_standard_error(p_hat=0.25, n_paths=100)

    expected = np.sqrt(0.25 * 0.75 / 100)

    assert se == pytest.approx(expected)


def test_exact_terminal_cdf_is_between_zero_and_one() -> None:
    value = exact_terminal_cdf(
        epsilon=1.0e-4,
        x0=0.02,
        kappa=2.0,
        theta=0.02,
        sigma=0.2,
        T=1.0,
    )

    assert 0.0 <= value <= 1.0


def test_exact_terminal_cdf_is_monotone_in_epsilon() -> None:
    params = {
        "x0": 0.02,
        "kappa": 2.0,
        "theta": 0.02,
        "sigma": 0.8,
        "T": 1.0,
    }

    small = exact_terminal_cdf(epsilon=1.0e-8, **params)
    large = exact_terminal_cdf(epsilon=1.0e-3, **params)

    assert small <= large