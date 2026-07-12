import csv

import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.samplers.full_truncation_euler import fte_terminal_from_dW
from src.utils.brownian import aggregate_brownian_increments
from src.utils.io import config_path, ensure_dirs, figure_path, results_path
from src.utils.rng import make_brownian_increments, make_rng
from src.utils.style import REGIME_COLOURS

SEED_OFFSET = 10_000


def load_config(filename: str) -> dict:
    with open(config_path(filename), encoding="utf-8") as f:
        return yaml.safe_load(f)


def add_reference_slope_segment(
    ax,
    slope: float,
    x_start: float,
    x_end: float,
    y_start: float,
    label: str,
) -> None:
    
    x_values = np.array([x_start, x_end], dtype=float)
    y_values = y_start * (x_values / x_start) ** slope

    ax.loglog(
        x_values,
        y_values,
        linestyle="--",
        linewidth=1.6,
        color="black",
        alpha=0.65,
    )

    ax.text(
        x_end * 1.05,
        y_values[-1],
        label,
        fontsize=10,
        va="center",
        ha="left",
    )

def estimate_loglog_slope(dt_values: np.ndarray, errors: np.ndarray) -> float:
    log_dt = np.log(dt_values)
    log_errors = np.log(errors)

    slope, _intercept = np.polyfit(log_dt, log_errors, deg=1)

    return float(slope)


def run_regime(
    regime_name: str,
    kappa: float,
    theta: float,
    x0: float,
    sigma: float,
    T: float,
    n_paths: int,
    reference_n_steps: int,
    coarse_n_steps_list: list[int],
    seed: int,
) -> list[dict]:
    dt_fine = T / reference_n_steps

    rng = make_rng(seed)

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

    rows = []

    for n_steps in coarse_n_steps_list:
        if reference_n_steps % n_steps != 0:
            raise ValueError(
                f"reference_n_steps={reference_n_steps} is not divisible "
                f"by n_steps={n_steps}"
            )

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

        abs_diff = np.abs(terminal_coarse - terminal_fine)

        mean_abs_error = float(np.mean(abs_diff))
        standard_error = float(np.std(abs_diff, ddof=1) / np.sqrt(n_paths))

        rows.append(
            {
                "method": "FTE",
                "regime": regime_name,
                "sigma": sigma,
                "n_paths": n_paths,
                "n_steps": n_steps,
                "reference_n_steps": reference_n_steps,
                "dt": T / n_steps,
                "mean_abs_error": mean_abs_error,
                "standard_error": standard_error,
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


def save_figure(rows: list[dict], output_figure: str) -> None:
    output_path = figure_path(output_figure)

    regimes = []
    for row in rows:
        if row["regime"] not in regimes:
            regimes.append(row["regime"])

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    min_dt = min(row["dt"] for row in rows)
    max_dt = max(row["dt"] for row in rows)

    for regime_name in regimes:
        regime_rows = [row for row in rows if row["regime"] == regime_name]

        dt_values = np.array([row["dt"] for row in regime_rows], dtype=float)
        errors = np.array([row["mean_abs_error"] for row in regime_rows], dtype=float)

        # Sort by dt so the line is drawn left-to-right on the log scale.
        order = np.argsort(dt_values)
        dt_values = dt_values[order]
        errors = errors[order]

        slope = estimate_loglog_slope(dt_values, errors)
        colour = REGIME_COLOURS[regime_name]

        ax.loglog(
            dt_values,
            errors,
            marker="o",
            linewidth=1.8,
            markersize=5,
            color=colour,
        )

        # Label each curve at the largest time step, on the right-hand side.
        ax.text(
            dt_values[-1] * 1.08,
            errors[-1],
            f"{regime_name}: {slope:.2f}",
            color=colour,
            fontsize=9,
            va="center",
            ha="left",
        )

    add_reference_slope_segment(
        ax=ax,
        slope=0.5,
        x_start=0.0045,
        x_end=0.03,
        y_start=1.8e-4,
        label=r"$h^{1/2}$",
    )

    ax.set_xlim(min_dt * 0.85, max_dt * 1.75)

    ax.set_xlabel("Time step size $h$")
    ax.set_ylabel(r"$\mathbb{E}|X_T^{h} - X_T^{\mathrm{ref}}|$")
    ax.set_title("FTE Brownian-coupled self-convergence")

    ax.grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"Saved figure to: {output_path}")

def print_summary(rows: list[dict]) -> None:
    regimes = sorted({row["regime"] for row in rows})

    print("\nFTE Brownian-coupled self-convergence")
    print("=" * 52)

    for regime_name in regimes:
        regime_rows = [row for row in rows if row["regime"] == regime_name]

        dt_values = np.array([row["dt"] for row in regime_rows], dtype=float)
        errors = np.array([row["mean_abs_error"] for row in regime_rows], dtype=float)

        slope = estimate_loglog_slope(dt_values, errors)

        print(f"\nRegime {regime_name}")
        print(f"Estimated log-log slope: {slope:.3f}")
        print("n_steps    dt          mean_abs_error    standard_error")
        print("-" * 62)

        for row in regime_rows:
            print(
                f"{row['n_steps']:7d}    "
                f"{row['dt']:.6f}    "
                f"{row['mean_abs_error']:.8e}    "
                f"{row['standard_error']:.3e}"
            )


def main() -> None:
    ensure_dirs()

    regimes_config = load_config("regimes.yaml")
    experiments_config = load_config("experiments.yaml")

    experiment = experiments_config["experiments"]["fte_self_convergence_diagnostic"]

    kappa = regimes_config["shared"]["kappa"]
    theta = regimes_config["shared"]["theta"]
    x0 = regimes_config["shared"]["x0"]

    master_seed = experiments_config["shared"]["master_seed"]

    T = experiment["T"]
    n_paths = experiment["n_paths"]
    reference_n_steps = experiment["reference_n_steps"]
    coarse_n_steps_list = experiment["coarse_n_steps"]

    all_rows = []

    for i, regime_name in enumerate(experiment["regimes"]):
        sigma = regimes_config["regimes"][regime_name]["sigma"]

        print(f"Running regime {regime_name} with sigma={sigma}...")

        regime_rows = run_regime(
            regime_name=regime_name,
            kappa=kappa,
            theta=theta,
            x0=x0,
            sigma=sigma,
            T=T,
            n_paths=n_paths,
            reference_n_steps=reference_n_steps,
            coarse_n_steps_list=coarse_n_steps_list,
            seed=master_seed + SEED_OFFSET + i,
        )

        all_rows.extend(regime_rows)

    print_summary(all_rows)
    save_csv(all_rows, experiment["output_csv"])
    save_figure(all_rows, experiment["output_figure"])


if __name__ == "__main__":
    main()