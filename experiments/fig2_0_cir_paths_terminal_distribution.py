
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.stats import ncx2

from src.samplers.exact import cir_ncx2_params, simulate_paths
from src.utils.io import config_path, ensure_dirs, figure_path
from src.utils.rng import make_rng
from src.utils.style import REGIME_COLOURS


def load_config(filename: str) -> dict:
    with open(config_path(filename), encoding="utf-8") as f:
        return yaml.safe_load(f)


def exact_cir_mean(
    x0: float,
    kappa: float,
    theta: float,
    t: np.ndarray,
) -> np.ndarray:
    return theta + (x0 - theta) * np.exp(-kappa * t)


def exact_cir_pdf(
    x_grid: np.ndarray,
    x0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
) -> np.ndarray:
    c, df, nc = cir_ncx2_params(
        x=x0,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        dt=T,
    )

    return c * ncx2.pdf(c * x_grid, df, nc)


def main() -> None:
    ensure_dirs()

    regimes_config = load_config("regimes.yaml")
    experiments_config = load_config("experiments.yaml")

    kappa = regimes_config["shared"]["kappa"]
    theta = regimes_config["shared"]["theta"]
    x0 = regimes_config["shared"]["x0"]

    regime_name = "A"
    sigma = regimes_config["regimes"][regime_name]["sigma"]
    colour = REGIME_COLOURS[regime_name]

    master_seed = experiments_config["shared"]["master_seed"]
    rng = make_rng(master_seed + 250)

    T = 5.0
    n_steps = 250
    n_paths = 250

    dt = T / n_steps
    time_grid = np.linspace(0.0, T, n_steps + 1)

    X = simulate_paths(
        X0=x0,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        dt=dt,
        n_steps=n_steps,
        n_paths=n_paths,
        rng=rng,
    )

    X_T = X[:, -1]
    mean_path = exact_cir_mean(x0, kappa, theta, time_grid)

    x_max = np.quantile(X_T, 0.995)
    x_grid = np.linspace(0.0, x_max, 600)
    pdf = exact_cir_pdf(
        x_grid=x_grid,
        x0=x0,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        T=T,
    )

    terminal_mean = exact_cir_mean(
        x0=x0,
        kappa=kappa,
        theta=theta,
        t=np.array([T]),
    )[0]

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(9.2, 4.6),
        gridspec_kw={"width_ratios": [4.0, 1.15]},
        sharey=True,
    )

    ax_paths = axes[0]
    ax_hist = axes[1]
    
    # Left panel: simulated paths
    
    for i in range(n_paths):
        ax_paths.plot(
            time_grid,
            X[i],
            color=colour,
            alpha=0.12,
            linewidth=0.7,
        )

    ax_paths.plot(
        time_grid,
        mean_path,
        color="black",
        linestyle="--",
        linewidth=1.4,
        label=r"$\mathbb{E}[X_t]$",
    )

    ax_paths.set_xlabel(r"$t$")
    ax_paths.set_ylabel(r"$X_t$")
    ax_paths.set_title(
        rf"CIR sample paths, regime {regime_name} "
        rf"($\kappa={kappa:g}$, $\theta={theta:g}$, $\sigma={sigma:g}$)"
    )
    ax_paths.grid(alpha=0.18)
    ax_paths.legend(frameon=True, fontsize=8)

    # Right panel: terminal distribution
   
    ax_hist.hist(
        X_T,
        bins=35,
        density=True,
        orientation="horizontal",
        color=colour,
        alpha=0.35,
        label=r"terminal samples",
    )

    ax_hist.plot(
        pdf,
        x_grid,
        color="black",
        linewidth=1.3,
        label=r"exact density",
    )

    ax_hist.axhline(
        terminal_mean,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=r"$\mathbb{E}[X_T]$",
    )

    ax_hist.set_xlabel("density")
    ax_hist.set_title(r"$X_T$")
    ax_hist.grid(alpha=0.18)
    ax_hist.legend(frameon=True, fontsize=7, loc="upper right")

    fig.suptitle(
        rf"Exact CIR paths and terminal law "
        rf"($T={T:g}$, $M={n_paths}$)",
        fontsize=11,
    )

    fig.tight_layout()

    output_path = figure_path("fig2_cir_paths_terminal_distribution.pdf")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()