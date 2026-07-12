import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.stats import ncx2

from src.samplers.exact import cir_ncx2_params
from src.samplers.kelly_lord import kl_uniform_terminal
from src.utils.cir_params import cir_delta, kl_alpha
from src.utils.io import config_path, ensure_dirs, figure_path
from src.utils.rng import make_rng
from src.utils.style import REGIME_COLOURS

N_PATHS = 20_000
N_STEPS = 200
T = 1.0
N_BINS = 80
PDF_POINTS = 800
DENSITY_QUANTILE = 0.999
HIST_ALPHA = 0.40


def load_config(filename: str) -> dict:
    with open(config_path(filename), encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def exact_cir_mean(x0: float, kappa: float, theta: float, T: float) -> float:
    return theta + (x0 - theta) * np.exp(-kappa * T)


def exact_cir_quantile(
    probability: float,
    x0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
) -> float:
    c, df, nc = cir_ncx2_params(
        x=x0,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        dt=T,
    )

    return ncx2.ppf(probability, df, nc) / c


def main() -> None:
    ensure_dirs()

    regimes_config = load_config("regimes.yaml")
    experiments_config = load_config("experiments.yaml")

    kappa = regimes_config["shared"]["kappa"]
    theta = regimes_config["shared"]["theta"]
    x0 = regimes_config["shared"]["x0"]

    master_seed = experiments_config["shared"]["master_seed"]

    regimes_to_plot = ["A", "B", "C"]

    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(11.0, 3.4),
        sharey=False,
    )

    print()
    print("Uniform Kelly-Lord versus exact CIR terminal law")
    print("------------------------------------------------")
    print("Regime | sigma      | delta      | alpha      | exact mean | KL mean")
    print("-------|------------|------------|------------|------------|---------")

    for i, regime_name in enumerate(regimes_to_plot):
        ax = axes[i]

        sigma = regimes_config["regimes"][regime_name]["sigma"]
        colour = REGIME_COLOURS[regime_name]
        rng = make_rng(master_seed + i)

        X_kl = kl_uniform_terminal(
            X0=x0,
            kappa=kappa,
            theta=theta,
            sigma=sigma,
            T=T,
            n_steps=N_STEPS,
            n_paths=N_PATHS,
            rng=rng,
        )

        x_max = exact_cir_quantile(
            probability=DENSITY_QUANTILE,
            x0=x0,
            kappa=kappa,
            theta=theta,
            sigma=sigma,
            T=T,
        )

        x_grid = np.linspace(0.0, x_max, PDF_POINTS)
        exact_pdf = exact_cir_pdf(
            x_grid=x_grid,
            x0=x0,
            kappa=kappa,
            theta=theta,
            sigma=sigma,
            T=T,
        )

        ax.hist(
            X_kl,
            bins=N_BINS,
            range=(0.0, x_max),
            density=True,
            alpha=HIST_ALPHA,
            color=colour,
            label="uniform KL histogram",
        )

        ax.plot(
            x_grid,
            exact_pdf,
            color="black",
            linewidth=1.3,
            label="exact density",
        )

        delta = cir_delta(kappa, theta, sigma)
        alpha = kl_alpha(kappa, theta, sigma)

        ax.set_title(
            rf"Regime {regime_name}: $\sigma={sigma:.4g}$, $\delta={delta:.3g}$",
            fontsize=10,
        )
        ax.set_xlabel(r"$X_T$")
        ax.grid(alpha=0.18)

        if i == 0:
            ax.set_ylabel("Density")

        exact_mean = exact_cir_mean(x0, kappa, theta, T)
        kl_mean = float(np.mean(X_kl))

        print(
            f"{regime_name:>6} | "
            f"{sigma:10.6f} | "
            f"{delta:10.6f} | "
            f"{alpha:10.6f} | "
            f"{exact_mean:10.6f} | "
            f"{kl_mean:7.6f}"
        )

    axes[0].legend(frameon=True, fontsize=8)

    fig.suptitle(
        rf"Uniform Kelly-Lord sanity check against exact CIR law "
        rf"($T={T:g}$, $N={N_STEPS}$, $M={N_PATHS}$)",
        fontsize=11,
    )

    fig.tight_layout()

    output_path = figure_path("sanity_kl_vs_exact_uniform.pdf")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    print()
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()