import numpy as np
import yaml

from src.metrics.distributional import (
    exact_terminal_cdf,
    ks_statistic_vs_exact,
    lower_tail_mass,
    wasserstein1_vs_exact,
)
from src.samplers.exact import cir_ncx2_params
from src.samplers.full_truncation_euler import fte_terminal
from src.utils.io import config_path
from src.utils.rng import make_rng


def load_config(filename: str) -> dict:
    with open(config_path(filename), encoding="utf-8") as f:
        return yaml.safe_load(f)


def shared_params():
    regimes = load_config("regimes.yaml")
    experiments = load_config("experiments.yaml")
    return regimes, experiments["shared"]["master_seed"]


def exact_samples(x0, kappa, theta, sigma, T, n_paths, rng):
    c, df, nc = cir_ncx2_params(x0, kappa, theta, sigma, T)
    return rng.noncentral_chisquare(df, nc, size=n_paths) / c


def test_exact_cdf_is_monotone_from_zero_to_one():
    regimes, _ = shared_params()
    shared = regimes["shared"]
    sigma = regimes["regimes"]["C"]["sigma"]

    grid = np.linspace(0.0, 0.5, 200)
    cdf = exact_terminal_cdf(grid, shared["x0"], shared["kappa"], shared["theta"], sigma, 1.0)

    assert cdf[0] <= 1e-12
    assert cdf[-1] > 0.999
    assert np.all(np.diff(cdf) >= -1e-12)


def test_exact_sampler_ks_and_w1_near_monte_carlo_floor():
    # Exact draws follow the exact law, so KS should sit at the n^{-1/2}
    # Monte Carlo scale and W1 should be a small fraction of the state scale.
    regimes, master_seed = shared_params()
    shared = regimes["shared"]
    sigma = regimes["regimes"]["C"]["sigma"]
    law = (shared["x0"], shared["kappa"], shared["theta"], sigma, 1.0)

    n = 20000
    rng = make_rng(master_seed)
    samples = exact_samples(*law, n_paths=n, rng=rng)

    ks = ks_statistic_vs_exact(samples, *law)
    w1 = wasserstein1_vs_exact(samples, *law)

    assert ks < 2.0 / np.sqrt(n), f"KS={ks:.5f} too large for exact draws"
    assert w1 < 1e-3, f"W1={w1:.2e} too large for exact draws"


def test_fte_terminal_law_error_decreases_with_steps():
    # In the strongly boundary-accessible regime E the discretisation bias
    # dominates Monte Carlo noise, so a coarse grid must look worse than a
    # fine grid under both diagnostics.
    regimes, master_seed = shared_params()
    shared = regimes["shared"]
    sigma = regimes["regimes"]["E"]["sigma"]
    law = (shared["x0"], shared["kappa"], shared["theta"], sigma, 1.0)

    n_paths = 20000
    values = {}
    for n_steps in [8, 256]:
        rng = make_rng(master_seed)
        samples = fte_terminal(
            X0=shared["x0"],
            kappa=shared["kappa"],
            theta=shared["theta"],
            sigma=sigma,
            T=1.0,
            n_steps=n_steps,
            n_paths=n_paths,
            rng=rng,
        )
        values[n_steps] = {
            "ks": ks_statistic_vs_exact(samples, *law),
            "w1": wasserstein1_vs_exact(samples, *law),
        }

    assert values[256]["ks"] < values[8]["ks"], values
    assert values[256]["w1"] < values[8]["w1"], values


def test_lower_tail_mass_bounds():
    samples = np.array([0.0, 0.001, 0.01, 0.1, 1.0])

    assert lower_tail_mass(samples, epsilon=10.0) == 1.0
    assert lower_tail_mass(samples, epsilon=0.005) == 0.4
