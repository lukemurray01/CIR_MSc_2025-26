import numpy as np
import yaml

from src.samplers.full_truncation_euler import fte_terminal
from src.samplers.kelly_lord import kl_uniform_terminal
from src.utils.io import config_path
from src.utils.rng import make_rng


def load_config(filename: str) -> dict:
    with open(config_path(filename), encoding="utf-8") as f:
        return yaml.safe_load(f)


def exact_cir_mean(
        X0: float, 
        kappa: float, 
        theta: float, 
        T: float
        ) -> float:
    return theta + (X0 - theta) * np.exp(-kappa * T)


def test_fte_terminal_mean_check_abc():
    regimes_config = load_config("regimes.yaml")
    experiments_config = load_config("experiments.yaml")

    kappa = regimes_config["shared"]["kappa"]
    theta = regimes_config["shared"]["theta"]
    x0 = regimes_config["shared"]["x0"]

    master_seed = experiments_config["shared"]["master_seed"]

    experiment = experiments_config["experiments"]["fte_moment_check"]

    T = experiment["T"]
    n_steps = experiment["n_steps"]
    n_paths = experiment["n_paths"]
    relative_tolerance = experiment["relative_tolerance"]

    exact_mean = exact_cir_mean(
        X0=x0,
        kappa=kappa,
        theta=theta,
        T=T,
    )

    for i, regime_name in enumerate(experiment["regimes"]):
        sigma = regimes_config["regimes"][regime_name]["sigma"]
        relative_tolerance = experiment["relative_tolerance"]

        rng = make_rng(master_seed + 2000 + i)

        terminal = fte_terminal(
            X0=x0,
            kappa=kappa,
            theta=theta,
            sigma=sigma,
            T=T,
            n_steps=n_steps,
            n_paths=n_paths,
            rng=rng,
        )

        sample_mean = np.mean(terminal)
        relative_error = abs(sample_mean - exact_mean) / exact_mean

        assert np.all(np.isfinite(terminal)), (
            f"FTE produced non-finite terminal values in regime {regime_name}"
        )

        assert relative_error < relative_tolerance, (
            f"FTE mean check failed in regime {regime_name}. "
            f"Sample mean={sample_mean:.6g}, "
            f"exact mean={exact_mean:.6g}, "
            f"relative error={relative_error:.3%}, "
            f"tolerance={relative_tolerance:.3%}"
        )
    
def test_kl_uniform_terminal_mean_check_valid_regimes():
    regimes_config = load_config("regimes.yaml")
    experiments_config = load_config("experiments.yaml")

    kappa = regimes_config["shared"]["kappa"]
    theta = regimes_config["shared"]["theta"]
    x0 = regimes_config["shared"]["x0"]

    master_seed = experiments_config["shared"]["master_seed"]

    experiment = experiments_config["experiments"]["kl_uniform_moment_check"]

    T = experiment["T"]
    n_steps = experiment["n_steps"]
    n_paths = experiment["n_paths"]
    relative_tolerance = experiment["relative_tolerance"]

    exact_mean = exact_cir_mean(
        X0=x0,
        kappa=kappa,
        theta=theta,
        T=T,
    )

    for i, regime_name in enumerate(experiment["regimes"]):
        sigma = regimes_config["regimes"][regime_name]["sigma"]

        rng = make_rng(master_seed + 3000 + i)

        terminal = kl_uniform_terminal(
            X0=x0,
            kappa=kappa,
            theta=theta,
            sigma=sigma,
            T=T,
            n_steps=n_steps,
            n_paths=n_paths,
            rng=rng,
        )

        sample_mean = np.mean(terminal)
        relative_error = abs(sample_mean - exact_mean) / exact_mean

        assert np.all(np.isfinite(terminal)), (
            f"KL uniform produced non-finite terminal values in regime {regime_name}"
        )

        assert relative_error < relative_tolerance, (
            f"KL uniform mean check failed in regime {regime_name}. "
            f"Sample mean={sample_mean:.6g}, "
            f"exact mean={exact_mean:.6g}, "
            f"relative error={relative_error:.3%}, "
            f"tolerance={relative_tolerance:.3%}"
        )