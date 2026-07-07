# Tests for the BLT splitting scheme and its exact noise machinery.
#
# The assertions pin the paper's exactly-known properties: the running-
# infimum law, the squared-Bessel step mean E[R_{t+h}] = r + h (Lemma 3.2),
# the pathwise comparison / L1 contraction E[x_k - y_k] = e^{-b t_k}(x0 - y0)
# (Proposition 3.3), positivity for a >= 1 (Proposition 3.1), and the
# min(1, a/2)-type strong convergence observed in the paper's Table 1.

import numpy as np
import pytest
import yaml
from scipy.stats import norm

from src.samplers.blt_splitting import (
    besq1_flow,
    blt_ab_from_cir,
    blt_terminal,
    blt_terminal_from_noise,
)
from src.metrics.strong_error import fit_loglog_order
from src.metrics.weak_error import exact_cir_mean
from src.utils.brownian import aggregate_brownian_increments_and_infima
from src.utils.cir_params import cir_delta
from src.utils.io import config_path
from src.utils.rng import (
    make_brownian_increments_with_infima,
    make_rng,
    sample_bridge_infimum,
)


def load_config(filename: str) -> dict:
    with open(config_path(filename), encoding="utf-8") as f:
        return yaml.safe_load(f)


def shared_params():
    regimes = load_config("regimes.yaml")
    experiments = load_config("experiments.yaml")
    return regimes, experiments["shared"]["master_seed"]


# ---------------------------------------------------------------- noise


def test_infimum_is_below_zero_and_endpoint():
    rng = make_rng(7)
    dW, m = make_brownian_increments_with_infima(rng, 200, 50, dt=0.01)

    assert np.all(m <= 0.0)
    assert np.all(m <= dW)


def test_infimum_matches_reflection_law():
    # P( inf_{[0,t]} W <= x ) = 2 * Phi(x / sqrt(t)) for x <= 0.
    rng = make_rng(11)
    t = 0.25
    _, m = make_brownian_increments_with_infima(rng, 200_000, 1, dt=t)
    m = m[:, 0]

    for x in [-1.0 * np.sqrt(t), -0.5 * np.sqrt(t)]:
        expected = 2.0 * norm.cdf(x / np.sqrt(t))
        observed = np.mean(m <= x)
        assert observed == pytest.approx(expected, abs=0.01)


def test_bridge_infimum_conditional_on_increment():
    # Same formula, conditional route: given dW, still m <= min(0, dW), and
    # the unconditional law is recovered when dW is N(0, dt).
    rng = make_rng(13)
    dt = 0.04
    dW = np.sqrt(dt) * rng.standard_normal(100_000)
    m = sample_bridge_infimum(rng, dW, dt)

    assert np.all(m <= np.minimum(0.0, dW))
    x = -0.5 * np.sqrt(dt)
    assert np.mean(m <= x) == pytest.approx(2.0 * norm.cdf(x / np.sqrt(dt)), abs=0.01)


def test_aggregation_identity_two_stage_equals_one_stage():
    rng = make_rng(17)
    dW, m = make_brownian_increments_with_infima(rng, 300, 64, dt=1.0 / 64)

    one_stage = aggregate_brownian_increments_and_infima(dW, m, 16)
    stage_a = aggregate_brownian_increments_and_infima(dW, m, 4)
    two_stage = aggregate_brownian_increments_and_infima(*stage_a, 4)

    np.testing.assert_allclose(two_stage[0], one_stage[0], rtol=1e-12)
    np.testing.assert_allclose(two_stage[1], one_stage[1], rtol=1e-12)

    # Aggregated pairs are still valid (infimum below 0 and endpoint).
    dW_c, m_c = one_stage
    assert np.all(m_c <= 1e-15)
    assert np.all(m_c <= dW_c + 1e-15)


def test_besq1_flow_mean_is_r_plus_h():
    # Lemma 3.2: E[R_{t+h}] = r + h; the exact reflection sampling must
    # reproduce this to Monte Carlo accuracy.
    rng = make_rng(19)
    r, h = 0.3, 0.25
    dW, m = make_brownian_increments_with_infima(rng, 200_000, 1, dt=h)

    R = besq1_flow(np.full(200_000, r), dW[:, 0], m[:, 0])

    assert np.all(R >= 0.0)
    assert np.mean(R) == pytest.approx(r + h, rel=0.02)


# ---------------------------------------------------------------- scheme


def test_blt_ab_dictionary():
    regimes, _ = shared_params()
    shared = regimes["shared"]
    sigma = regimes["regimes"]["B"]["sigma"]

    a, b = blt_ab_from_cir(shared["kappa"], shared["theta"], sigma)
    assert a == pytest.approx(cir_delta(shared["kappa"], shared["theta"], sigma))
    assert b == pytest.approx(shared["kappa"])


def test_blt_terminal_strictly_positive_when_a_geq_1():
    regimes, master_seed = shared_params()
    shared = regimes["shared"]

    for i, regime_name in enumerate(["A", "B", "C"]):
        sigma = regimes["regimes"][regime_name]["sigma"]
        assert cir_delta(shared["kappa"], shared["theta"], sigma) >= 1.0

        rng = make_rng(master_seed + i)
        terminal = blt_terminal(
            X0=shared["x0"],
            kappa=shared["kappa"],
            theta=shared["theta"],
            sigma=sigma,
            T=1.0,
            n_steps=64,
            n_paths=4000,
            rng=rng,
        )

        assert np.all(np.isfinite(terminal))
        assert np.all(terminal > 0.0), f"non-positive BLT terminal in {regime_name}"


def test_blt_modified_scheme_runs_in_boundary_regimes():
    regimes, master_seed = shared_params()
    shared = regimes["shared"]

    for i, regime_name in enumerate(["D", "E"]):
        sigma = regimes["regimes"][regime_name]["sigma"]
        assert cir_delta(shared["kappa"], shared["theta"], sigma) < 1.0

        rng = make_rng(master_seed + i)
        terminal = blt_terminal(
            X0=shared["x0"],
            kappa=shared["kappa"],
            theta=shared["theta"],
            sigma=sigma,
            T=1.0,
            n_steps=64,
            n_paths=4000,
            rng=rng,
        )

        assert np.all(np.isfinite(terminal))
        assert np.all(terminal >= 0.0)
        assert np.mean(terminal > 0.0) > 0.5, "modified BLT collapsed to zero"


def test_blt_pathwise_comparison_and_l1_contraction():
    # Proposition 3.3: with coupled noise, x0 >= y0 implies x_k >= y_k
    # pathwise, and E[x_k - y_k] = e^{-b t_k} (x0 - y0) exactly.
    regimes, master_seed = shared_params()
    shared = regimes["shared"]
    sigma = regimes["regimes"]["B"]["sigma"]

    T, n_steps, n_paths = 1.0, 64, 20000
    dt = T / n_steps
    rng = make_rng(master_seed)
    dW, m = make_brownian_increments_with_infima(rng, n_paths, n_steps, dt)

    X0_high, X0_low = 0.05, 0.02
    common = dict(
        kappa=shared["kappa"], theta=shared["theta"], sigma=sigma,
        dt=dt, dW=dW, m=m,
    )
    high = blt_terminal_from_noise(X0=X0_high, **common)
    low = blt_terminal_from_noise(X0=X0_low, **common)

    assert np.all(high >= low - 1e-14), "pathwise comparison violated"

    expected = np.exp(-shared["kappa"] * T) * (X0_high - X0_low)
    assert np.mean(high - low) == pytest.approx(expected, rel=0.05)


def test_blt_terminal_mean_matches_exact_mean():
    regimes, master_seed = shared_params()
    shared = regimes["shared"]
    sigma = regimes["regimes"]["B"]["sigma"]
    exact = exact_cir_mean(shared["x0"], shared["kappa"], shared["theta"], 1.0)

    rng = make_rng(master_seed)
    terminal = blt_terminal(
        X0=shared["x0"],
        kappa=shared["kappa"],
        theta=shared["theta"],
        sigma=sigma,
        T=1.0,
        n_steps=128,
        n_paths=20000,
        rng=rng,
    )

    rel_error = abs(np.mean(terminal) - exact) / exact
    assert rel_error < 0.05, f"BLT mean off by {rel_error:.2%}"


def test_blt_self_convergence_order_near_one_in_regime_B():
    # a = 4 > 2: proven L1 order 1 up to logs; the coupled self-reference
    # tail order must comfortably exceed the increment-only ceiling 1/2.
    regimes, master_seed = shared_params()
    shared = regimes["shared"]
    sigma = regimes["regimes"]["B"]["sigma"]

    T = 1.0
    reference_n_steps = 2048
    n_paths = 4000
    dt_fine = T / reference_n_steps

    rng = make_rng(master_seed)
    dW_fine, m_fine = make_brownian_increments_with_infima(
        rng, n_paths, reference_n_steps, dt_fine
    )
    common = dict(
        X0=shared["x0"], kappa=shared["kappa"],
        theta=shared["theta"], sigma=sigma,
    )
    reference = blt_terminal_from_noise(dt=dt_fine, dW=dW_fine, m=m_fine, **common)

    dts, errors = [], []
    for n_steps in [16, 64, 256]:
        factor = reference_n_steps // n_steps
        dW, m = aggregate_brownian_increments_and_infima(dW_fine, m_fine, factor)
        terminal = blt_terminal_from_noise(dt=T / n_steps, dW=dW, m=m, **common)
        dts.append(T / n_steps)
        errors.append(np.mean(np.abs(terminal - reference)))

    assert errors[0] > errors[1] > errors[2], f"not decreasing: {errors}"
    order = fit_loglog_order(np.array(dts), np.array(errors))
    assert order > 0.7, f"BLT regime-B order {order:.3f}, expected near 1"


def test_blt_deterministic_given_seed():
    regimes, master_seed = shared_params()
    shared = regimes["shared"]
    sigma = regimes["regimes"]["E"]["sigma"]

    outputs = []
    for _ in range(2):
        rng = make_rng(master_seed)
        outputs.append(
            blt_terminal(
                X0=shared["x0"],
                kappa=shared["kappa"],
                theta=shared["theta"],
                sigma=sigma,
                T=1.0,
                n_steps=32,
                n_paths=500,
                rng=rng,
            )
        )

    np.testing.assert_array_equal(outputs[0], outputs[1])
