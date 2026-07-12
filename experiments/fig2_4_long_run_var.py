import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.utils.io import config_path, ensure_dirs, figure_path
from src.utils.style import REGIME_COLOURS


def load_config(filename: str) -> dict:
    with open(config_path(filename), encoding="utf-8") as f:
        return yaml.safe_load(f)


def cir_variance(
    x0: float,
    kappa: float,
    theta: float,
    sigma: float,
    t: np.ndarray,
) -> np.ndarray:
    return (
        x0 * sigma**2 / kappa * (np.exp(-kappa * t) - np.exp(-2.0 * kappa * t))
        + theta * sigma**2 / (2.0 * kappa) * (1.0 - np.exp(-kappa * t)) ** 2
    )


def cir_stationary_variance(
    kappa: float,
    theta: float,
    sigma: float,
) -> float:
    return theta * sigma**2 / (2.0 * kappa)


def main() -> None:
    ensure_dirs()

    regimes_config = load_config("regimes.yaml")

    kappa = regimes_config["shared"]["kappa"]
    theta = regimes_config["shared"]["theta"]
    x0 = regimes_config["shared"]["x0"]
    regimes = regimes_config["regimes"]

    time_grid = np.linspace(0.0, 4.0, 400)

    fig, ax = plt.subplots(figsize=(3.00, 3.35))

    for regime_name in ["A", "B", "C", "D", "E"]:
        sigma = regimes[regime_name]["sigma"]

        variance = cir_variance(
            x0=x0,
            kappa=kappa,
            theta=theta,
            sigma=sigma,
            t=time_grid,
        )

        stationary_variance = cir_stationary_variance(
            kappa=kappa,
            theta=theta,
            sigma=sigma,
        )

        ax.plot(
            time_grid,
            variance,
            color=REGIME_COLOURS[regime_name],
            linewidth=1.0,
            label=regime_name,
        )

        ax.hlines(
            stationary_variance,
            xmin=time_grid[0],
            xmax=time_grid[-1],
            color=REGIME_COLOURS[regime_name],
            linestyle=":",
            linewidth=0.7,
            alpha=0.65,
        )

    ax.set_title(r"Variance to stationarity", fontsize=8)
    ax.set_xlabel(r"$t$", fontsize=7)
    ax.set_ylabel(r"$\mathrm{Var}(X_t)$", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(alpha=0.18)

    ax.legend(
        title="Regime",
        title_fontsize=5.8,
        fontsize=5.8,
        frameon=True,
        loc="upper right",
        bbox_to_anchor=(1.25, 0.7),
        borderaxespad = 0.0,
    )

    fig.tight_layout(pad=0.0)

    output_path = figure_path("fig2_margin_variance_reversion.pdf")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()