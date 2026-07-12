import numpy as np
import yaml

from src.samplers.full_truncation_euler import fte_terminal_from_dW
from src.utils.brownian import aggregate_brownian_increments
from src.utils.io import config_path
from src.utils.rng import make_brownian_increments, make_rng


def load_config(filename: str) -> dict:
    with open(config_path(filename), encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    regimes_config = load_config("regimes.yaml")
    experiments_config = load_config("experiments.yaml")

    kappa = regimes_config["shared"]["kappa"]
    theta = regimes_config["shared"]["theta"]
    x0 = regimes_config["shared"]["x0"]

    master_seed = experiments_config["shared"]["master_seed"]

    regime_name = "B"
    sigma = regimes_config["regimes"][regime_name]["sigma"]

    T = 1.0
    n_paths = 20_000
    reference_n_steps = 4096
    coarse_n_steps_list = [8, 16, 32, 64, 128, 256]

    dt_fine = T / reference_n_steps

    rng = make_rng(master_seed + 9000)

    dW_fine = make_brownian_increments(
        rng=rng,
        n_paths=n_paths,
        n_steps=reference_n_steps,
        dt=dt_fine,
    )

    terminal_fine = fte_terminal_from_dW(
        X0=x0,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        dt=dt_fine,
        dW=dW_fine,
    )

    print(f"\nFTE Brownian-coupled smoke check, regime {regime_name}")
    print(f"Reference grid: {reference_n_steps} steps")
    print()
    print("n_steps    dt          mean_abs_diff")
    print("-" * 42)

    for n_steps in coarse_n_steps_list:
        factor = reference_n_steps // n_steps

        dW_coarse = aggregate_brownian_increments(
            dW_fine=dW_fine,
            factor=factor,
        )

        terminal_coarse = fte_terminal_from_dW(
            X0=x0,
            kappa=kappa,
            theta=theta,
            sigma=sigma,
            dt=T / n_steps,
            dW=dW_coarse,
        )

        mean_abs_diff = np.mean(np.abs(terminal_coarse - terminal_fine))

        print(f"{n_steps:7d}    {T / n_steps:.6f}    {mean_abs_diff:.8e}")


if __name__ == "__main__":
    main()