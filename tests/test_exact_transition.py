# Tests for the exact CIR sampler implementation

import numpy as np
import pytest

from src.samplers.exact import cir_ncx2_params, simulate_paths

# From textbook check the parameters give the expected output
def test_cir_ncx2_params_match():
    # Parameters from 
    x = 0.05
    kappa = 0.05
    theta = 0.05
    sigma = 0.02
    dt = 0.01

    c, df, nc = cir_ncx2_params(x, kappa, theta, sigma, dt)

    expected_c = 4.0 * kappa / (sigma**2 * (1.0 - np.exp(-kappa * dt)))
    expected_df = 4.0 * kappa * theta / sigma**2
    expected_nc = expected_c * x * np.exp(-kappa * dt)

    assert c == pytest.approx(expected_c)
    assert df == pytest.approx(expected_df)
    assert nc == pytest.approx(expected_nc)


def test_simulate_paths_has_correct_shape():
    rng = np.random.default_rng(10)

    X = simulate_paths(
        X0=0.05,
        kappa=0.05,
        theta=0.05,
        sigma=0.02,
        dt=0.01,
        n_steps=100,
        n_paths=5,
        rng=rng,
    )

    assert X.shape == (5, 101)


def test_simulate_paths_starts_at_X0():
    rng = np.random.default_rng(10)

    X0 = 0.05

    X = simulate_paths(
        X0=X0,
        kappa=0.05,
        theta=0.05,
        sigma=0.02,
        dt=0.01,
        n_steps=100,
        n_paths=5,
        rng=rng,
    )

    np.testing.assert_allclose(X[:, 0], X0)


def test_simulate_paths_are_nonnegative():
    rng = np.random.default_rng(10)

    X = simulate_paths(
        X0=0.05,
        kappa=0.05,
        theta=0.05,
        sigma=0.02,
        dt=0.01,
        n_steps=100,
        n_paths=100,
        rng=rng,
    )

    assert np.all(X >= 0.0)


def test_simulate_paths_reproducible_with_same_seed():
    rng1 = np.random.default_rng(10)
    rng2 = np.random.default_rng(10)

    args = dict(
        X0=0.05,
        kappa=0.05,
        theta=0.05,
        sigma=0.02,
        dt=0.01,
        n_steps=100,
        n_paths=5,
    )

    X1 = simulate_paths(**args, rng=rng1)
    X2 = simulate_paths(**args, rng=rng2)

    np.testing.assert_array_equal(X1, X2)


def test_simulate_paths_change_with_different_seeds():
    rng1 = np.random.default_rng(10)
    rng2 = np.random.default_rng(11)

    args = dict(
        X0=0.05,
        kappa=0.05,
        theta=0.05,
        sigma=0.02,
        dt=0.01,
        n_steps=100,
        n_paths=5,
    )

    X1 = simulate_paths(**args, rng=rng1)
    X2 = simulate_paths(**args, rng=rng2)

    assert not np.array_equal(X1, X2)