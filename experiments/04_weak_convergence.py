import csv

import numpy as np
import yaml

from src.metrics.weak_error import (
    TERMINAL_PAYOFFS,
    monte_carlo_standard_error,
    terminal_exact_expectations,
)
from src.samplers.full_truncation_euler import fte_terminal_from_dW
from src.utils.io import config_path, ensure_dirs, results_path
from src.utils.rng import make_brownian_increments, make_rng


SEED_OFFSET = 40_000


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
    payoff_names: list[str],
    seed: int,
) -> list[dict]:
    exact_expectations = terminal_exact_expectations(
        x0=x0,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        T=T,
    )

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

        terminal_values = fte_terminal_from_dW(
            X0=x0,
            kappa=kappa,
            theta=theta,
            sigma=sigma,
            dt=dt,
            dW=dW,
        )

        for payoff_name in payoff_names:
            payoff = TERMINAL_PAYOFFS[payoff_name]

            payoff_values = payoff(terminal_values)

            approx_mean = float(np.mean(payoff_values))
            exact_mean = exact_expectations[payoff_name]

            signed_error = approx_mean - exact_mean
            weak_error = abs(signed_error)

            standard_error = monte_carlo_standard_error(payoff_values)

            error_to_se = (
                abs(signed_error) / standard_error
                if standard_error > 0.0
                else np.nan
            )

            rows.append(
                {
                    "method": "FTE",
                    "regime": regime_name,
                    "sigma": sigma,
                    "payoff": payoff_name,
                    "n_paths": n_paths,
                    "n_steps": n_steps,
                    "dt": dt,
                    "approx_mean": approx_mean,
                    "exact_mean": exact_mean,
                    "signed_error": signed_error,
                    "weak_error": weak_error,
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
    print("\nFTE terminal weak-error smoke experiment")
    print("=" * 64)

    regimes = []
    for row in rows:
        if row["regime"] not in regimes:
            regimes.append(row["regime"])

    payoffs = []
    for row in rows:
        if row["payoff"] not in payoffs:
            payoffs.append(row["payoff"])

    for regime_name in regimes:
        print(f"\nRegime {regime_name}")
        print("-" * 64)

        for payoff_name in payoffs:
            payoff_rows = [
                row
                for row in rows
                if row["regime"] == regime_name and row["payoff"] == payoff_name
            ]

            print(f"\nPayoff {payoff_name}")
            print(
                "n_steps    dt          approx_mean       exact_mean        "
                "signed_error      weak_error        error/SE"
                )
            print("-" * 78)

            for row in payoff_rows:
                print(
                    f"{row['n_steps']:7d}    "
                    f"{row['dt']:.6f}    "
                    f"{row['approx_mean']:.8e}    "
                    f"{row['exact_mean']:.8e}    "
                    f"{row['signed_error']:.8e}    "
                    f"{row['weak_error']:.8e}    "
                    f"{row['error_to_se']:.2f}"
                )

def main() -> None:
    ensure_dirs()

    regimes_config = load_config("regimes.yaml")
    experiments_config = load_config("experiments.yaml")

    experiment = experiments_config["experiments"]["fte_weak_convergence_smoke"]

    kappa = regimes_config["shared"]["kappa"]
    theta = regimes_config["shared"]["theta"]
    x0 = regimes_config["shared"]["x0"]

    master_seed = experiments_config["shared"]["master_seed"]

    T = experiment["T"]
    n_paths = experiment["n_paths"]
    n_steps_list = experiment["n_steps"]
    payoff_names = experiment["payoffs"]

    all_rows = []

    for i, regime_name in enumerate(experiment["regimes"]):
        sigma = regimes_config["regimes"][regime_name]["sigma"]

        print(f"Running FTE weak-error smoke for regime {regime_name}...")

        rows = run_regime(
            regime_name=regime_name,
            kappa=kappa,
            theta=theta,
            x0=x0,
            sigma=sigma,
            T=T,
            n_paths=n_paths,
            n_steps_list=n_steps_list,
            payoff_names=payoff_names,
            seed=master_seed + SEED_OFFSET + 1000 * i,
        )

        all_rows.extend(rows)

    print_summary(all_rows)
    save_csv(all_rows, experiment["output_csv"])


if __name__ == "__main__":
    main()
