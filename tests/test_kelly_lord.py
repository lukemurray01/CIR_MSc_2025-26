import numpy as np
import pytest
import yaml

from src.samplers.kelly_lord import (
    kl_uniform_paths,
    kl_uniform_paths_from_dW,
    kl_uniform_terminal,
    kl_uniform_terminal_from_dW,
)
from src.utils.io import config_path
from src.utils.rng import make_rng, make_brownian_increments

def load_config(filename: str) -> dict:
    with open(config_path(filename), encoding="utf-8") as f:
        return yaml.safe_load(f)
    
def exact_cir_mean(x0: float,
                   kappa: float,
                   theta: float,
                   T: float,) -> float:
    return theta + (x0 - theta) * np.exp(-kappa * T)

def test_kl_uniform_paths_has_correct_shape():
    regimes_config = load_config("regimes.yaml")
    experiments_config = load_config("experiments.yaml")

    kappa = regimes_config["shared"]["kappa"]
    theta = regimes_config["shared"]["theta"]
    x0 = regimes_config["shared"]["x0"]
    sigma = regimes_config["regimes"]["B"]["sigma"]

    master_seed = experiments_config["shared"]["master_seed"]
    rng = make_rng(master_seed)

    n_paths = 1000
    n_steps = 100
    T = 1.0

    X = kl_uniform_paths(
        X0 = x0,
        kappa = kappa,
        theta = theta,
        sigma = sigma,
        T = T,
        n_steps = n_steps,
        n_paths = n_paths,
        rng = rng,
    )
    
    assert X.shape == (n_paths, n_steps + 1)

def test_kl_uniform_paths_start_at_X0():
    regimes_config = load_config("regimes.yaml")
    experiments_config = load_config("experiments.yaml")

    kappa = regimes_config["shared"]["kappa"]
    theta = regimes_config["shared"]["theta"]
    x0 = regimes_config["shared"]["x0"]
    sigma = regimes_config["regimes"]["B"]["sigma"]

    master_seed = experiments_config["shared"]["master_seed"]
    rng = make_rng(master_seed)

    X = kl_uniform_paths(
        X0 = x0,
        kappa = kappa,
        theta = theta,
        sigma = sigma,
        T = 1.0,
        n_steps = 100,
        n_paths = 1000,
        rng = rng,
    )

    assert np.allclose(X[:,0], x0)

def test_kl_uniform_paths_are_nonnegative_for_regimes_A_B_C():
    regimes_config = load_config("regimes.yaml")
    experiments_config = load_config("experiments.yaml")

    kappa = regimes_config["shared"]["kappa"]
    theta = regimes_config["shared"]["theta"]
    x0 = regimes_config["shared"]["x0"]

    master_seed = experiments_config["shared"]["master_seed"]

    T = 1.0
    n_steps = 100
    n_paths = 1000

    for i, regime_name in enumerate(["A", "B", "C"]):
        sigma = regimes_config["regimes"][regime_name]["sigma"]
        rng = make_rng(master_seed + i)

        X = kl_uniform_paths(
            X0=x0,
            kappa=kappa,
            theta=theta,
            sigma=sigma,
            T=T,
            n_steps=n_steps,
            n_paths=n_paths,
            rng=rng,
        )

        assert np.all(X >= 0.0), f"KL produced negative values in regime {regime_name}"


def test_same_dW_gives_same_kl_paths():
    regimes_config = load_config("regimes.yaml")
    experiments_config = load_config("experiments.yaml")

    kappa = regimes_config["shared"]["kappa"]
    theta = regimes_config["shared"]["theta"]
    x0 = regimes_config["shared"]["x0"]
    sigma = regimes_config["regimes"]["B"]["sigma"]

    master_seed = experiments_config["shared"]["master_seed"]
    rng = make_rng(master_seed)

    dt = 0.01
    n_paths = 500
    n_steps = 100

    dW = make_brownian_increments(rng, n_paths, n_steps, dt)

    X1 = kl_uniform_paths_from_dW(
        X0=x0,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        dt=dt,
        dW=dW,
    )

    X2 = kl_uniform_paths_from_dW(
        X0=x0,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        dt=dt,
        dW=dW,
    )

    assert np.allclose(X1, X2)


def test_kl_terminal_matches_last_column_of_paths():
    regimes_config = load_config("regimes.yaml")
    experiments_config = load_config("experiments.yaml")

    kappa = regimes_config["shared"]["kappa"]
    theta = regimes_config["shared"]["theta"]
    x0 = regimes_config["shared"]["x0"]
    sigma = regimes_config["regimes"]["B"]["sigma"]

    master_seed = experiments_config["shared"]["master_seed"]
    rng = make_rng(master_seed)

    dt = 0.01
    n_paths = 500
    n_steps = 100

    dW = make_brownian_increments(rng, n_paths, n_steps, dt)

    X = kl_uniform_paths_from_dW(
        X0=x0,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        dt=dt,
        dW=dW,
    )

    X_T = kl_uniform_terminal_from_dW(
        X0=x0,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        dt=dt,
        dW=dW,
    )

    assert np.allclose(X_T, X[:, -1])


def test_kl_uniform_rejects_alpha_negative_regimes():
    regimes_config = load_config("regimes.yaml")
    experiments_config = load_config("experiments.yaml")

    kappa = regimes_config["shared"]["kappa"]
    theta = regimes_config["shared"]["theta"]
    x0 = regimes_config["shared"]["x0"]

    master_seed = experiments_config["shared"]["master_seed"]

    T = 1.0
    n_steps = 100
    n_paths = 100

    for i, regime_name in enumerate(["D", "E"]):
        sigma = regimes_config["regimes"][regime_name]["sigma"]
        rng = make_rng(master_seed + i)

        with pytest.raises(ValueError):
            kl_uniform_paths(
                X0=x0,
                kappa=kappa,
                theta=theta,
                sigma=sigma,
                T=T,
                n_steps=n_steps,
                n_paths=n_paths,
                rng=rng,
            )

def test_kl_uniform_from_dW_rejects_alpha_negative_regimes():
    regimes_config = load_config("regimes.yaml")

    kappa = regimes_config["shared"]["kappa"]
    theta = regimes_config["shared"]["theta"]
    x0 = regimes_config["shared"]["x0"]

    dt = 0.01
    dW = np.zeros((3,2))

    for regime_name in ["D", "E"]:
        sigma = regimes_config["regimes"][regime_name]["sigma"]

        with pytest.raises(ValueError):
            kl_uniform_paths_from_dW(
                X0 = x0,
                kappa = kappa,
                theta = theta,
                sigma = sigma,
                dt = dt,
                dW = dW,
            )
        with pytest.raises(ValueError):
            kl_uniform_terminal_from_dW(
                X0 = x0,
                kappa = kappa,
                theta = theta,
                sigma = sigma,
                dt = dt,
                dW = dW,
            )
def test_kl_uniform_paths_wrapper_matches_precomputed_increments():
    regimes_config = load_config("regimes.yaml")
    experiments_config = load_config("experiments.yaml")

    kappa = regimes_config["shared"]["kappa"]
    theta = regimes_config["shared"]["theta"]
    x0 = regimes_config["shared"]["x0"]
    sigma = regimes_config["regimes"]["B"]["sigma"]

    master_seed = experiments_config["shared"]["master_seed"]
    T = 1.0
    n_steps = 32
    n_paths = 250
    dt = T / n_steps

    wrapper_rng = make_rng(master_seed)
    dW_rng = make_rng(master_seed)
    dW = make_brownian_increments(dW_rng, n_paths, n_steps, dt)

    X_wrapper = kl_uniform_paths(
        X0=x0,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        T=T,
        n_steps=n_steps,
        n_paths=n_paths,
        rng=wrapper_rng,
    )

    X_from_dW = kl_uniform_paths_from_dW(
        X0=x0,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        dt=dt,
        dW=dW,
    )

    np.testing.assert_allclose(X_wrapper, X_from_dW)


def test_kl_uniform_terminal_wrapper_matches_precomputed_increments():
    regimes_config = load_config("regimes.yaml")
    experiments_config = load_config("experiments.yaml")

    kappa = regimes_config["shared"]["kappa"]
    theta = regimes_config["shared"]["theta"]
    x0 = regimes_config["shared"]["x0"]
    sigma = regimes_config["regimes"]["B"]["sigma"]

    master_seed = experiments_config["shared"]["master_seed"]
    T = 1.0
    n_steps = 32
    n_paths = 250
    dt = T / n_steps

    wrapper_rng = make_rng(master_seed)
    dW_rng = make_rng(master_seed)
    dW = make_brownian_increments(dW_rng, n_paths, n_steps, dt)

    X_wrapper = kl_uniform_terminal(
        X0=x0,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        T=T,
        n_steps=n_steps,
        n_paths=n_paths,
        rng=wrapper_rng,
    )

    X_from_dW = kl_uniform_terminal_from_dW(
        X0=x0,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        dt=dt,
        dW=dW,
    )

    np.testing.assert_allclose(X_wrapper, X_from_dW)

def test_kl_uniform_terminal_mean_close_to_exact_mean_regime_B():
    regimes_config = load_config("regimes.yaml")
    experiments_config = load_config("experiments.yaml")

    kappa = regimes_config["shared"]["kappa"]
    theta = regimes_config["shared"]["theta"]
    x0 = regimes_config["shared"]["x0"]
    sigma = regimes_config["regimes"]["B"]["sigma"]

    master_seed = experiments_config["shared"]["master_seed"]
    rng = make_rng(master_seed)

    T = 1.0
    n_steps = 200
    n_paths = 20_000

    X_T = kl_uniform_terminal(
        X0=x0,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        T=T,
        n_steps=n_steps,
        n_paths=n_paths,
        rng=rng,
    )

    sample_mean = np.mean(X_T)
    expected_mean = exact_cir_mean(x0, kappa, theta, T)

    assert abs(sample_mean - expected_mean) < 0.001