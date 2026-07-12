import numpy as np
import pytest

from src.utils.brownian import aggregate_brownian_increments, brownian_bridge
from src.utils.rng import make_brownian_increments, make_rng


def test_aggregate_brownian_increments_have_correct_shape():
    dW_fine = np.ones((5, 12))
    dW_coarse = aggregate_brownian_increments(dW_fine, factor = 3)

    assert dW_coarse.shape == (5,4)

def test_aggregate_brownian_increments_sum_blocks_correctly():
    dW_fine = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [0.5, 0.5, 1.0, 1.0, 2.0, 2.0],
        ]
    )

    dW_coarse = aggregate_brownian_increments(dW_fine, factor = 2)

    expected = np.array(
        [
            [3.0, 7.0, 11.0],
            [1.0, 2.0, 4.0],
        ]
    )

    np.testing.assert_allclose(dW_coarse, expected)

def test_aggregate_brownian_increments_with_factor_one_return_same():
    dW_fine = np.array(
        [
            [1.0, -2.0, 3.0]
        ]
    )

    dW_coarse = aggregate_brownian_increments(dW_fine, factor = 1)

    np.testing.assert_allclose(dW_coarse, dW_fine)

def test_aggregate_brownian_rejects_negative_factor():
    dW_fine = np.ones(
        (2,4)
    )

    with pytest.raises(ValueError, match = "factor must be positive"):
        aggregate_brownian_increments(dW_fine, factor = 0)

def test_aggregate_brownian_increments_reject_non2darray():
    dW_fine = np.ones(8)

    with pytest.raises(ValueError, match = "dW_fine needs to be a 2D array"):
        aggregate_brownian_increments(dW_fine, factor = 2)

def test_aggregate_brownian_increments_rejects_nondivisible_factor():
    dW_fine = np.ones(
        (2,5)
    )

    with pytest.raises(ValueError, match = "number of fine steps must be divisible by chosen factor"):
        aggregate_brownian_increments(dW_fine, factor = 2)

def test_aggregate_brownian_increments_has_correct_total():
    rng = make_rng(123)
    dt_fine = 1.0 / 16

    dW_fine = make_brownian_increments(
        rng = rng,
        n_paths = 100,
        n_steps = 16,
        dt = dt_fine,
    )

    dW_coarse = aggregate_brownian_increments(dW_fine, factor = 4)

    fine_total = dW_fine.sum(axis = 1)
    coarse_total = dW_coarse.sum(axis = 1)

    np.testing.assert_allclose(coarse_total, fine_total)

def test_aggregate_brownian_increments_have_correct_var_scale():
    rng = make_rng(123)

    n_paths = 50_000
    n_fine_steps = 16
    factor = 4
    dt_fine = 1.0 / n_fine_steps
    dt_coarse = factor * dt_fine

    dW_fine = make_brownian_increments(
        rng = rng,
        n_paths = n_paths,
        n_steps = n_fine_steps,
        dt = dt_fine,
    )

    dW_coarse = aggregate_brownian_increments(dW_fine, factor = factor)

    empirical_variance = np.var(dW_coarse[:, 0], ddof = 1)

    assert abs(empirical_variance - dt_coarse) < 0.01

def test_brownian_bridge_matches_analytic_moments():
    # Many independent draws at one interior time tau, with fixed endpoints,
    # should reproduce the analytic bridge mean and variance.
    rng = make_rng(0)

    t_left, t_right = 0.3, 0.5
    w_left, w_right = 0.1, -0.2
    tau = 0.45

    n = 2_000_000
    samples = brownian_bridge(w_left, w_right, t_left, t_right, np.full(n, tau), rng)

    h = t_right - t_left
    theta = (tau - t_left) / h
    expected_mean = w_left + theta * (w_right - w_left)
    expected_variance = (tau - t_left) * (t_right - tau) / h

    assert abs(samples.mean() - expected_mean) < 2e-3
    assert abs(samples.var() - expected_variance) < 2e-3


def test_brownian_bridge_pins_left_endpoint():
    # At tau = t_left the conditional variance is zero, so the sample must
    # equal w_left exactly, regardless of the random draw.
    rng = make_rng(1)
    sample = brownian_bridge(0.1, -0.2, 0.3, 0.5, np.array([0.3]), rng)
    np.testing.assert_allclose(sample, [0.1])


def test_brownian_bridge_pins_right_endpoint():
    # Symmetrically, at tau = t_right the sample must equal w_right exactly.
    rng = make_rng(2)
    sample = brownian_bridge(0.1, -0.2, 0.3, 0.5, np.array([0.5]), rng)
    np.testing.assert_allclose(sample, [-0.2])

