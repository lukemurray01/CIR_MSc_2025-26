import numpy as np


# Pull the samplers for FTE scheme
from src.samplers.full_truncation_euler import (
    fte_paths,
    fte_terminal,
    fte_paths_from_dW,
    fte_terminal_from_dW,
)

from src.utils.rng import make_brownian_increments

# Test that the FTE Paths have the correct shape
def test_fte_paths_has_correct_shape():
    rng = np.random.default_rng(123)

    X = fte_paths(
        X0=0.02,
        kappa=2.0,
        theta=0.02,
        sigma=0.2,
        T=1.0,
        n_steps=100,
        n_paths=1000,
        rng=rng,
    )

    assert X.shape == (1000, 101)


def test_fte_paths_start_at_X0():
    rng = np.random.default_rng(123)

    X = fte_paths(
        X0=0.02,
        kappa=2.0,
        theta=0.02,
        sigma=0.2,
        T=1.0,
        n_steps=100,
        n_paths=1000,
        rng=rng,
    )

    # For floats, we use assert_allclose instead of == as this works better with floating point values due to machine precision.
    np.testing.assert_allclose(X[:, 0], 0.02)


def test_fte_paths_are_nonnegative():
    rng = np.random.default_rng(123)

    X = fte_paths(
        X0=0.02,
        kappa=2.0,
        theta=0.02,
        sigma=0.8,
        T=1.0,
        n_steps=50,
        n_paths=1000,
        rng=rng,
    )

    assert np.all(X >= 0.0)


def test_same_dW_gives_same_paths():
    rng = np.random.default_rng(123)

    dt = 1.0 / 100
    dW = np.sqrt(dt) * rng.standard_normal((1000, 100))

    X1 = fte_paths_from_dW(
        X0=0.02,
        kappa=2.0,
        theta=0.02,
        sigma=0.2,
        dt=dt,
        dW=dW,
    )

    X2 = fte_paths_from_dW(
        X0=0.02,
        kappa=2.0,
        theta=0.02,
        sigma=0.2,
        dt=dt,
        dW=dW,
    )

    np.testing.assert_allclose(X1, X2)


def test_terminal_matches_last_column_of_paths():
    rng = np.random.default_rng(123)

    dt = 1.0 / 100
    dW = make_brownian_increments(rng, n_paths = 1000, n_steps = 100, dt = dt)

    X = fte_paths_from_dW(
        X0=0.02,
        kappa=2.0,
        theta=0.02,
        sigma=0.2,
        dt=dt,
        dW=dW,
    )

    X_T = fte_terminal_from_dW(
        X0=0.02,
        kappa=2.0,
        theta=0.02,
        sigma=0.2,
        dt=dt,
        dW=dW,
    )

    np.testing.assert_allclose(X_T, X[:, -1])


def test_fte_terminal_has_correct_shape():
    rng = np.random.default_rng(123)

    X_T = fte_terminal(
        X0=0.02,
        kappa=2.0,
        theta=0.02,
        sigma=0.2,
        T=1.0,
        n_steps=100,
        n_paths=1000,
        rng=rng,
    )

    assert X_T.shape == (1000,)