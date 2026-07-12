import numpy as np
import yaml

from src.samplers.full_truncation_euler import fte_paths
from src.samplers.kelly_lord import kl_uniform_paths
from src.utils.io import config_path
from src.utils.rng import make_rng

def load_config(filename: str) -> dict:
    with open(config_path(filename), encoding="utf-8") as f:
        return yaml.safe_load(f)
    
def test_fte_paths_nonnegative_all_regimes():
    regimes_config = load_config("regimes.yaml")
    experiments_config = load_config("experiments.yaml")

    kappa = regimes_config["shared"]["kappa"]
    theta = regimes_config["shared"]["theta"]
    x0 = regimes_config["shared"]["x0"]

    master_seed = experiments_config["shared"]["master_seed"]

    experiment = experiments_config["experiments"]["fte_all_regime_smoke"]

    T = experiment["T"]
    n_steps = experiment["n_steps"]
    n_paths = experiment["n_paths"]

    for i, regime_name in enumerate(experiment["regimes"]):
        sigma = regimes_config["regimes"][regime_name]["sigma"]

        rng = make_rng(master_seed + i)

        X = fte_paths(
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
        assert np.all(X >= 0.0), f"FTE produced negative values in regime {regime_name}"
        assert np.all(np.isfinite(X)), f"FTE produced non-finite values in regime {regime_name}"

def test_kl_uniform_paths_nonnegative_valid_regimes():
    regimes_config = load_config("regimes.yaml")
    experiments_config = load_config("experiments.yaml")

    kappa = regimes_config["shared"]["kappa"]
    theta = regimes_config["shared"]["theta"]
    x0 = regimes_config["shared"]["x0"]

    master_seed = experiments_config["shared"]["master_seed"]

    experiment = experiments_config["experiments"]["kl_uniform_smoke"]

    T = experiment["T"]
    n_steps = experiment["n_steps"]
    n_paths = experiment["n_paths"]

    for i, regime_name in enumerate(experiment["regimes"]):
        sigma = regimes_config["regimes"][regime_name]["sigma"]

        rng = make_rng(master_seed + 1000 + i)

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
        assert np.all(np.isfinite(X)), f"KL uniform produced non finite values in regime {regime_name}"
        assert np.all(X >= 0.0), f"KL uniform produced negative values in regime {regime_name}"
