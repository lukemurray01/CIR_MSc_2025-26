import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.metrics.boundary import (
    bernoulli_standard_error,
    exact_terminal_cdf,
    path_hit_zero_fraction,
    terminal_near_zero_mass,
    terminal_zero_mass,
    zero_time_fraction,
)
from src.samplers.full_truncation_euler import fte_paths_from_dW
from src.utils.io import config_path, ensure_dirs, results_path
from src.utils.rng import make_brownian_increments, make_rng


SEED_OFFSET = 50_000


def load_config(filename: str) -> dict:
    with open(config_path(filename), encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_regime(
    regime_name: str,
    kappa: float,
    theta: float,
    x0: float,
    sigma: float,
    T: float,
    n_paths: int,
    n_steps_list: list[int],
    epsilons: list[float],
    seed: int,
) -> list[dict]:
    rows = []

    for j, n_steps in enumerate(n_steps_list):
        dt = T / n_steps
        rng = make_rng(seed + j)

        dW = make_brownian_increments(
            rng=rng,
            n_paths=n_paths,
            n_steps=n_steps,
            dt=dt,
        )

        paths = fte_paths_from_dW(
            X0=x0,
            kappa=kappa,
            theta=theta,
            sigma=sigma,
            dt=dt,
            dW=dW,
        )

        terminal_values = paths[:, -1]

        terminal_zero = terminal_zero_mass(paths)
        path_hit_zero = path_hit_zero_fraction(paths)
        zero_time = zero_time_fraction(paths)

        for epsilon_raw in epsilons:
            epsilon = float(epsilon_raw)

            empirical_mass = terminal_near_zero_mass(
                terminal_values=terminal_values,
                epsilon=epsilon,
            )

            exact_mass = exact_terminal_cdf(
                epsilon=epsilon,
                x0=x0,
                kappa=kappa,
                theta=theta,
                sigma=sigma,
                T=T,
            )

            signed_error = empirical_mass - exact_mass
            abs_error = abs(signed_error)

            standard_error = bernoulli_standard_error(
                p_hat=empirical_mass,
                n_paths=n_paths,
            )

            error_to_se = (
                abs_error / standard_error
                if standard_error > 0.0
                else float("nan")
            )

            rows.append(
                {
                    "method": "FTE",
                    "regime": regime_name,
                    "sigma": sigma,
                    "n_paths": n_paths,
                    "n_steps": n_steps,
                    "dt": dt,
                    "epsilon": epsilon,
                    "terminal_zero_mass": terminal_zero,
                    "path_hit_zero_fraction": path_hit_zero,
                    "zero_time_fraction": zero_time,
                    "empirical_mass_le_epsilon": empirical_mass,
                    "exact_mass_le_epsilon": exact_mass,
                    "signed_error": signed_error,
                    "abs_error": abs_error,
                    "standard_error": standard_error,
                    "error_to_se": error_to_se,
                }
            )

    return rows


def save_csv(rows: list[dict], output_csv: str) -> None:
    output_path = results_path(output_csv)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved CSV to: {output_path}")


def print_summary(rows: list[dict]) -> None:
    print("\nFTE zero-boundary diagnostic")
    print("=" * 72)

    compact_rows = [
        row
        for row in rows
        if row["epsilon"] == max(r["epsilon"] for r in rows if r["regime"] == row["regime"])
    ]

    for row in compact_rows:
        print(
            f"Regime {row['regime']}, "
            f"N={row['n_steps']:4d}, "
            f"eps={row['epsilon']:.0e}, "
            f"P_num={row['empirical_mass_le_epsilon']:.6e}, "
            f"P_exact={row['exact_mass_le_epsilon']:.6e}, "
            f"hit_zero={row['path_hit_zero_fraction']:.6e}, "
            f"zero_time={row['zero_time_fraction']:.6e}"
        )


def main() -> None:
    ensure_dirs()

    regimes_config = load_config("regimes.yaml")
    experiments_config = load_config("experiments.yaml")

    experiment = experiments_config["experiments"]["fte_zero_boundary_diagnostic"]

    kappa = regimes_config["shared"]["kappa"]
    theta = regimes_config["shared"]["theta"]
    x0 = regimes_config["shared"]["x0"]

    master_seed = experiments_config["shared"]["master_seed"]

    T = experiment["T"]
    n_paths = experiment["n_paths"]
    n_steps_list = experiment["n_steps"]
    epsilons = experiment["epsilons"]

    all_rows = []

    for i, regime_name in enumerate(experiment["regimes"]):
        sigma = regimes_config["regimes"][regime_name]["sigma"]

        print(f"Running FTE zero-boundary diagnostic for regime {regime_name}...")

        rows = run_regime(
            regime_name=regime_name,
            kappa=kappa,
            theta=theta,
            x0=x0,
            sigma=sigma,
            T=T,
            n_paths=n_paths,
            n_steps_list=n_steps_list,
            epsilons=epsilons,
            seed=master_seed + SEED_OFFSET + 1000 * i,
        )

        all_rows.extend(rows)

    print_summary(all_rows)
    save_csv(all_rows, experiment["output_csv"])


if __name__ == "__main__":
    main()