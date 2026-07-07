import numpy as np
import pytest
import yaml

from src.samplers.cev import (
    cev_exact_mean,
    cev_fte_terminal_from_dW,
    cev_projected_paths_from_dW,
    cev_projected_terminal,
    cev_projected_terminal_from_dW,
)
from src.samplers.projected_euler import projected_euler_terminal_from_dW
from src.utils.io import config_path
from src.utils.rng import make_brownian_increments, make_rng


def load_config(filename: str) -> dict:
    with open(config_path(filename), encoding="utf-8") as f:
        return yaml.safe_load(f)


def shared_params():
    regimes = load_config("regimes.yaml")
    experiments = load_config("experiments.yaml")
    return regimes, experiments["shared"]["master_seed"]


def test_beta_half_reduces_exactly_to_cir_projected_euler():
    # At beta = 1/2 the CEV Lamperti variable is Y = 2*sqrt(F), and the
    # projected CEV update is exactly twice the CIR projected Euler update
    # on the same Brownian increments (same drift function, scaled by two,
    # with the floor scaled accordingly).  The terminal values must agree
    # to floating-point accuracy, not merely statistically.
    regimes, master_seed = shared_params()
    shared = regimes["shared"]
    sigma = regimes["regimes"]["B"]["sigma"]

    n_paths, n_steps = 2000, 64
    dt = 1.0 / n_steps
    rng = make_rng(master_seed)
    dW = make_brownian_increments(rng, n_paths, n_steps, dt)

    y_floor_cir = dt

    cir_terminal = projected_euler_terminal_from_dW(
        X0=shared["x0"],
        kappa=shared["kappa"],
        theta=shared["theta"],
        sigma=sigma,
        dt=dt,
        dW=dW,
        y_floor=y_floor_cir,
    )

    cev_terminal = cev_projected_terminal_from_dW(
        F0=shared["x0"],
        kappa=shared["kappa"],
        theta=shared["theta"],
        sigma=sigma,
        beta=0.5,
        dt=dt,
        dW=dW,
        y_floor=2.0 * y_floor_cir,
    )

    np.testing.assert_allclose(cev_terminal, cir_terminal, rtol=1e-12, atol=1e-15)


def test_cev_projected_terminal_mean_beta_075():
    regimes, master_seed = shared_params()
    shared = regimes["shared"]
    sigma = regimes["regimes"]["B"]["sigma"]

    T = 1.0
    exact = cev_exact_mean(shared["x0"], shared["kappa"], shared["theta"], T)

    rng = make_rng(master_seed)
    terminal = cev_projected_terminal(
        F0=shared["x0"],
        kappa=shared["kappa"],
        theta=shared["theta"],
        sigma=sigma,
        beta=0.75,
        T=T,
        n_steps=256,
        n_paths=10000,
        rng=rng,
    )

    rel_error = abs(np.mean(terminal) - exact) / exact
    assert np.all(np.isfinite(terminal))
    assert rel_error < 0.05, (
        f"CEV mean check failed for beta=0.75: sample mean={np.mean(terminal):.6f}, "
        f"exact={exact:.6f}, relative error={rel_error:.3%}"
    )


def test_cev_projected_paths_strictly_positive():
    regimes, master_seed = shared_params()
    shared = regimes["shared"]
    sigma = regimes["regimes"]["D"]["sigma"]

    n_paths, n_steps = 1000, 128
    dt = 1.0 / n_steps
    rng = make_rng(master_seed)
    dW = make_brownian_increments(rng, n_paths, n_steps, dt)

    F = cev_projected_paths_from_dW(
        F0=shared["x0"],
        kappa=shared["kappa"],
        theta=shared["theta"],
        sigma=sigma,
        beta=0.75,
        dt=dt,
        dW=dW,
    )

    assert F.shape == (n_paths, n_steps + 1)
    assert np.all(np.isfinite(F))
    assert np.all(F > 0.0)


def test_cev_fte_terminal_nonnegative_and_mean():
    regimes, master_seed = shared_params()
    shared = regimes["shared"]
    sigma = regimes["regimes"]["B"]["sigma"]

    T = 1.0
    n_paths, n_steps = 10000, 256
    dt = T / n_steps
    rng = make_rng(master_seed)
    dW = make_brownian_increments(rng, n_paths, n_steps, dt)

    terminal = cev_fte_terminal_from_dW(
        F0=shared["x0"],
        kappa=shared["kappa"],
        theta=shared["theta"],
        sigma=sigma,
        beta=0.75,
        dt=dt,
        dW=dW,
    )

    exact = cev_exact_mean(shared["x0"], shared["kappa"], shared["theta"], T)
    rel_error = abs(np.mean(terminal) - exact) / exact

    assert np.all(terminal >= 0.0)
    assert rel_error < 0.05, f"CEV FTE mean relative error {rel_error:.3%}"


def test_cev_rejects_beta_outside_range():
    dW = np.zeros((10, 4))
    for bad_beta in [0.3, 1.0, 1.2]:
        with pytest.raises(ValueError):
            cev_projected_terminal_from_dW(
                F0=0.02,
                kappa=2.0,
                theta=0.02,
                sigma=0.2,
                beta=bad_beta,
                dt=0.25,
                dW=dW,
            )
