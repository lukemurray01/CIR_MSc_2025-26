import numpy as np
import pytest
import yaml

from src.metrics.weak_error import (
    TERMINAL_PAYOFFS,
    terminal_exact_expectations,
)
from src.samplers.kelly_lord_adaptive import (
    kl_adaptive_terminal_from_fine_dW,
    kl_adaptive_terminal,
    soft_zero_threshold,
)
from src.utils.io import config_path
from src.utils.rng import make_rng


def load_config(filename: str) -> dict:
    with open(config_path(filename), encoding="utf-8") as f:
        return yaml.safe_load(f)


def shared_params(regime_name: str) -> tuple[float, float, float, float]:
    regimes_config = load_config("regimes.yaml")

    kappa = regimes_config["shared"]["kappa"]
    theta = regimes_config["shared"]["theta"]
    x0 = regimes_config["shared"]["x0"]
    sigma = regimes_config["regimes"][regime_name]["sigma"]

    return kappa, theta, x0, sigma


def exact_cir_mean(
    x0: float,
    kappa: float,
    theta: float,
    T: float,
) -> float:
    return theta + (x0 - theta) * np.exp(-kappa * T)


def test_soft_zero_threshold_matches_formula_and_decreases_with_rho():
    kappa = 2.0
    theta = 0.02
    dt_max = 0.01

    threshold = soft_zero_threshold(
        kappa=kappa,
        theta=theta,
        dt_max=dt_max,
        rho=2.0,
    )

    expected = theta * (1.0 - np.exp(-kappa * dt_max)) / 2.0

    assert threshold == pytest.approx(expected)
    assert soft_zero_threshold(kappa, theta, dt_max, rho=4.0) < threshold


def test_kl_adaptive_terminal_shape_finiteness_and_positivity():
    kappa, theta, x0, sigma = shared_params("E")

    terminal = kl_adaptive_terminal(
        X0=x0,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        T=0.005,
        dt_max=0.001,
        n_paths=16,
        rng=make_rng(120339106),
        max_rounds=10_000,
    )

    assert terminal.shape == (16,)
    assert np.all(np.isfinite(terminal))
    assert np.all(terminal >= 0.0)


def test_kl_adaptive_terminal_runs_in_alpha_nonnegative_regimes():
    for i, regime_name in enumerate(["A", "B", "C"]):
        kappa, theta, x0, sigma = shared_params(regime_name)

        terminal = kl_adaptive_terminal(
            X0=x0,
            kappa=kappa,
            theta=theta,
            sigma=sigma,
            T=0.01,
            dt_max=0.005,
            n_paths=16,
            rng=make_rng(120339106 + i),
            max_rounds=10_000,
        )

        assert np.all(np.isfinite(terminal))
        assert np.all(terminal >= 0.0)


def test_kl_adaptive_terminal_handles_alpha_negative_regimes():
    for i, regime_name in enumerate(["D", "E"]):
        kappa, theta, x0, sigma = shared_params(regime_name)

        terminal = kl_adaptive_terminal(
            X0=x0,
            kappa=kappa,
            theta=theta,
            sigma=sigma,
            T=0.005,
            dt_max=0.001,
            n_paths=16,
            rng=make_rng(120339106 + i),
            max_rounds=10_000,
        )

        assert np.all(np.isfinite(terminal))
        assert np.all(terminal >= 0.0)


def test_coupled_kl_adaptive_runs_in_alpha_negative_regimes():
    for i, regime_name in enumerate(["D", "E"]):
        kappa, theta, x0, sigma = shared_params(regime_name)
        n_paths = 8
        n_fine = 512
        T = 0.125
        dt_fine = T / n_fine
        rng = make_rng(120339106 + i)
        dW_fine = np.sqrt(dt_fine) * rng.standard_normal((n_paths, n_fine))

        terminal, stats = kl_adaptive_terminal_from_fine_dW(
            X0=x0,
            kappa=kappa,
            theta=theta,
            sigma=sigma,
            T=T,
            dt_max=0.01,
            dW_fine=dW_fine,
            max_rounds=10_000,
        )

        assert terminal.shape == (n_paths,)
        assert np.all(np.isfinite(terminal))
        assert np.all(terminal >= 0.0)
        assert stats["n_steps_total"] >= n_paths
        assert 0.0 <= stats["soft_zero_fraction"] <= 1.0


def test_kl_adaptive_terminal_is_seed_reproducible_and_seed_sensitive():
    kappa, theta, x0, sigma = shared_params("E")
    kwargs = {
        "X0": x0,
        "kappa": kappa,
        "theta": theta,
        "sigma": sigma,
        "T": 0.005,
        "dt_max": 0.001,
        "n_paths": 16,
        "max_rounds": 10_000,
    }

    first = kl_adaptive_terminal(**kwargs, rng=make_rng(120339106))
    second = kl_adaptive_terminal(**kwargs, rng=make_rng(120339106))
    different_seed = kl_adaptive_terminal(**kwargs, rng=make_rng(120339107))

    np.testing.assert_allclose(first, second)
    assert not np.allclose(first, different_seed)


def test_kl_adaptive_terminal_truncates_final_step_when_dt_max_exceeds_T():
    kappa, theta, x0, sigma = shared_params("B")

    terminal = kl_adaptive_terminal(
        X0=x0,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        T=0.001,
        dt_max=1.0,
        n_paths=16,
        rng=make_rng(120339106),
        max_rounds=10_000,
    )

    assert np.all(np.isfinite(terminal))
    assert np.all(terminal >= 0.0)


def test_kl_adaptive_terminal_soft_zero_branch_matches_deterministic_flow():
    kappa, theta, _x0, sigma = shared_params("E")
    T = 0.001
    dt_max = 0.01
    x0 = 0.5 * soft_zero_threshold(kappa, theta, dt_max)

    terminal = kl_adaptive_terminal(
        X0=x0,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        T=T,
        dt_max=dt_max,
        n_paths=8,
        rng=make_rng(120339106),
    )

    expected = exact_cir_mean(
        x0=x0,
        kappa=kappa,
        theta=theta,
        T=T,
    )

    np.testing.assert_allclose(terminal, expected)


def test_kl_adaptive_terminal_raises_when_max_rounds_too_small():
    kappa, theta, x0, sigma = shared_params("E")

    with pytest.raises(RuntimeError):
        kl_adaptive_terminal(
            X0=x0,
            kappa=kappa,
            theta=theta,
            sigma=sigma,
            T=1.0,
            dt_max=0.001,
            n_paths=4,
            rng=make_rng(120339106),
            max_rounds=1,
        )


@pytest.mark.parametrize("regime_name", ["A", "B", "C"])
def test_kl_adaptive_terminal_short_time_mean_close_to_exact_mean(regime_name):
    kappa, theta, x0, sigma = shared_params(regime_name)
    T = 0.01

    terminal = kl_adaptive_terminal(
        X0=x0,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        T=T,
        dt_max=0.005,
        n_paths=512,
        rng=make_rng(120339106),
        max_rounds=10_000,
    )

    sample_mean = np.mean(terminal)
    exact_mean = exact_cir_mean(
        x0=x0,
        kappa=kappa,
        theta=theta,
        T=T,
    )

    assert abs(sample_mean - exact_mean) / exact_mean < 0.08


@pytest.mark.parametrize("payoff_name", ["g1", "g2", "g3"])
def test_kl_adaptive_terminal_weak_payoffs_close_to_exact_regime_e(payoff_name):
    kappa, theta, x0, sigma = shared_params("E")
    T = 0.005

    terminal = kl_adaptive_terminal(
        X0=x0,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        T=T,
        dt_max=0.001,
        n_paths=128,
        rng=make_rng(120339106),
        max_rounds=10_000,
    )

    sample = np.mean(TERMINAL_PAYOFFS[payoff_name](terminal))
    exact = terminal_exact_expectations(
        x0=x0,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        T=T,
    )[payoff_name]

    absolute_tolerances = {
        "g1": 8.0e-3,
        "g2": 8.0e-4,
        "g3": 8.0e-3,
    }

    assert abs(sample - exact) < absolute_tolerances[payoff_name]


def test_kl_adaptive_terminal_does_not_stall_at_soft_zero_boundary():
    # Regression for the floating-point stall fixed 2026-07-06: the exact
    # soft-zero exit flow could land a few ulp BELOW X_zero, re-entering the
    # region with a vanishing next step and never reaching T.  Coarse dt_max
    # in regime E is the configuration that exposed it; completed exits are
    # now snapped onto the boundary.
    kappa, theta, x0, sigma = shared_params("E")

    rng = make_rng(120339106)
    terminal = kl_adaptive_terminal(
        X0=x0,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        T=1.0,
        dt_max=1.0 / 8,
        n_paths=4000,
        rng=rng,
        max_rounds=20_000,
    )

    assert terminal.shape == (4000,)
    assert np.all(np.isfinite(terminal))
    assert np.all(terminal >= 0.0)
