# Tests for the BLT splitting step used as a KLM backstop candidate.
#
# The map must satisfy the admissibility conditions: defined for every
# input the adaptive scheme can produce, strictly positive output, and it
# must retake a failed explicit step with the SAME Brownian increment (the
# auxiliary bridge-infimum exponential is the only fresh randomness).

import numpy as np
import pytest
import yaml

from src.metrics.weak_error import exact_cir_mean
from src.samplers.klm_backstop import (
    _backstop_map,
    klm_backstop_terminal,
    klm_backstop_terminal_from_fine_dW,
)
from src.utils.cir_params import kl_alpha, kl_coefficients
from src.utils.io import config_path
from src.utils.rng import make_rng


def load_config(filename: str) -> dict:
    with open(config_path(filename), encoding="utf-8") as f:
        return yaml.safe_load(f)


def shared_params():
    regimes = load_config("regimes.yaml")
    experiments = load_config("experiments.yaml")
    return regimes, experiments["shared"]["master_seed"]


def test_blt_backstop_requires_rng():
    with pytest.raises(ValueError, match="rng"):
        _backstop_map(
            y=np.array([0.1]),
            h=np.array([0.01]),
            dW=np.array([-0.5]),
            alpha=-0.06,
            beta=-1.0,
            gamma=0.4,
            kind="blt",
        )


@pytest.mark.parametrize("regime_name", ["B", "E"])
def test_blt_backstop_strictly_positive_under_extreme_increments(regime_name):
    # Large negative increments are exactly the inputs that trigger the
    # backstop; the map must stay strictly positive for all of them.
    regimes, master_seed = shared_params()
    shared = regimes["shared"]
    sigma = regimes["regimes"][regime_name]["sigma"]
    alpha, beta, gamma = kl_coefficients(shared["kappa"], shared["theta"], sigma)

    rng = make_rng(master_seed)
    y = np.full(64, 0.02)
    h = np.full(64, 1.0 / 128)
    dW = np.linspace(-1.0, -0.01, 64)

    out = _backstop_map(y, h, dW, alpha, beta, gamma, "blt", rng=rng)

    assert np.all(np.isfinite(out))
    assert np.all(out > 0.0)
    # The floor gamma*sqrt(h) is the guaranteed lower bound.
    assert np.all(out >= gamma * np.sqrt(h) - 1e-15)


def test_blt_backstop_single_step_mean_matches_closed_form():
    # One BLT step has a closed-form mean: E[Psi_R] = x + h (Bessel mean,
    # Lemma 3.2) pushed through the linear ODE flow gives
    #   E[x_next] = e^{-b h} (x + h) + (a - 1)(1 - e^{-b h}) / b.
    # Feeding the backstop map unconditioned increments must reproduce this
    # in X = (y)^2 coordinates; this pins the coordinate dictionary, the
    # conditional bridge-infimum sampling, and both exact flows end to end.
    # (Regime B, moderate state: the gamma*sqrt(h) floor is never active.)
    regimes, master_seed = shared_params()
    shared = regimes["shared"]
    sigma = regimes["regimes"]["B"]["sigma"]
    alpha, beta, gamma = kl_coefficients(shared["kappa"], shared["theta"], sigma)

    a = 1.0 + 2.0 * alpha / gamma**2
    b = -2.0 * beta
    scale = sigma**2 / 4.0

    n = 400_000
    h_val = 1.0 / 128
    X_start = shared["x0"]
    y = np.full(n, np.sqrt(X_start))
    h = np.full(n, h_val)

    rng = make_rng(master_seed)
    dW = np.sqrt(h_val) * rng.standard_normal(n)

    y_next = _backstop_map(y, h, dW, alpha, beta, gamma, "blt", rng=rng)
    X_next = y_next**2

    x_norm = X_start / scale
    decay = np.exp(-b * h_val)
    expected_norm = decay * (x_norm + h_val) + (a - 1.0) * (1.0 - decay) / b
    expected = scale * expected_norm

    # No path should have been floored in this configuration.
    assert np.all(y_next > gamma * np.sqrt(h_val) * (1.0 + 1e-12))
    assert np.mean(X_next) == pytest.approx(expected, rel=0.01)


def test_klm_blt_terminal_mean_matches_exact_mean_in_design_regime():
    regimes, master_seed = shared_params()
    shared = regimes["shared"]
    sigma = regimes["regimes"]["B"]["sigma"]
    exact = exact_cir_mean(shared["x0"], shared["kappa"], shared["theta"], 1.0)

    rng = make_rng(master_seed)
    terminal, stats = klm_backstop_terminal(
        X0=shared["x0"],
        kappa=shared["kappa"],
        theta=shared["theta"],
        sigma=sigma,
        T=1.0,
        h_max=1.0 / 128,
        n_paths=20000,
        rng=rng,
        backstop="blt",
    )

    assert stats["backstop_kind"] == "blt"
    assert np.all(terminal > 0.0)
    rel_error = abs(np.mean(terminal) - exact) / exact
    assert rel_error < 0.05, f"KLM-BLT mean off by {rel_error:.2%}"


@pytest.mark.parametrize("regime_name", ["D", "E"])
def test_klm_blt_runs_positive_outside_design_regime(regime_name):
    regimes, master_seed = shared_params()
    shared = regimes["shared"]
    sigma = regimes["regimes"][regime_name]["sigma"]
    assert kl_alpha(shared["kappa"], shared["theta"], sigma) < 0.0

    rng = make_rng(master_seed)
    terminal, stats = klm_backstop_terminal(
        X0=shared["x0"],
        kappa=shared["kappa"],
        theta=shared["theta"],
        sigma=sigma,
        T=1.0,
        h_max=1.0 / 64,
        n_paths=4000,
        rng=rng,
        backstop="blt",
    )

    assert stats["backstop_kind"] == "blt"
    assert np.all(np.isfinite(terminal))
    assert np.all(terminal > 0.0)
    # The backstop is the typical mode in the boundary-active regimes.
    assert stats["backstop_fraction"] > 0.1


def test_klm_blt_coupled_variant_runs_and_is_deterministic():
    regimes, master_seed = shared_params()
    shared = regimes["shared"]
    sigma = regimes["regimes"]["E"]["sigma"]

    T = 1.0
    reference_n_steps = 512
    dt_fine = T / reference_n_steps

    outputs = []
    for _ in range(2):
        rng = make_rng(master_seed)
        dW_fine = np.sqrt(dt_fine) * rng.standard_normal((500, reference_n_steps))
        # rho = 32 keeps h_min = h_max/rho = dt_fine representable on this
        # 512-step fine grid, so the fine-grid guard does not fire.
        terminal, stats = klm_backstop_terminal_from_fine_dW(
            X0=shared["x0"],
            kappa=shared["kappa"],
            theta=shared["theta"],
            sigma=sigma,
            T=T,
            h_max=1.0 / 16,
            dW_fine=dW_fine,
            backstop="blt",
            rng=rng,
            rho=32.0,
        )
        assert stats["backstop_kind"] == "blt"
        assert np.all(terminal > 0.0)
        outputs.append(terminal)

    np.testing.assert_array_equal(outputs[0], outputs[1])
