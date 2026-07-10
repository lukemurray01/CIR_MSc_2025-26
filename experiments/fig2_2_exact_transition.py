import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.stats import ncx2

from src.samplers.exact import cir_ncx2_params, simulate_paths
from src.utils.cir_params import cir_delta
from src.utils.io import config_path, figure_path
from src.utils.rng import make_rng
from src.utils.style import REGIME_COLOURS, EXACT_LINEWIDTH, GRID_ALPHA

HIST_ALPHA = 0.40

def load_yaml(filename: str) -> dict:
    with open(config_path(filename), encoding = "utf-8") as f:
        return yaml.safe_load(f)
    
def main() -> None:
    regimes_config = load_yaml("regimes.yaml")
    experiments_config = load_yaml("experiments.yaml")

    experiment = experiments_config["experiments"]["exact_transition_figure"]

    kappa = regimes_config["shared"]["kappa"]
    theta = regimes_config["shared"]["theta"]
    x0 = regimes_config["shared"]["x0"]

    T = experiment["T"]
    n_paths = experiment["n_paths"]
    regimes = experiment["regimes"]
    output = experiment["output"]

    rng = make_rng(experiments_config["shared"]["master_seed"])

    fig, axes = plt.subplots(3, 2, figsize=(10,7.5), sharey = True)
    axes = axes.ravel()

    for ax, regime_name in zip(axes[:5], regimes):
        sigma = regimes_config["regimes"][regime_name]["sigma"]
        colour = REGIME_COLOURS[regime_name]

        c, df, nc = cir_ncx2_params(
            x = x0,
            kappa = kappa,
            theta = theta,
            sigma = sigma,
            dt = T,
        )

        x_max = ncx2.ppf(0.999, df, nc) / c
        x_grid = np.linspace(max(1e-10, x_max * 1e-5), x_max, 600)
        density = c * ncx2.pdf(c * x_grid, df, nc)

        X = simulate_paths(
            X0 = x0,
            kappa = kappa,
            theta = theta,
            sigma = sigma,
            dt = T,
            n_steps = 1,
            n_paths = n_paths,
            rng = rng
        )

        ax.hist(
            X[:,-1],
            bins = 80,
            range=(0.0, x_max),
            density = True,
            color = colour, 
            alpha = HIST_ALPHA,
            edgecolor = colour, 
            linewidth = 0.3,
        )

        ax.plot(x_grid,
                density,
                color = "black",
                linewidth = EXACT_LINEWIDTH)
        delta = cir_delta(kappa, theta, sigma)

        ax.set_title(
            f"{regime_name}\n"
            rf"$\sigma = {sigma:4g},\ \delta={delta:.2f}$"
        )

        ax.set_xlim(0.0, x_max)
        ax.set_ylim(0.0, 90.0)
        ax.grid(alpha = GRID_ALPHA)
    
    # 3x2 grid (row-major):  [A][B] / [C][D] / [E][legend]
    # Left column = axes 0,2,4 -> y-labels; bottom of each column = axes 4,3 -> x-labels.
    axes[0].set_ylabel("Density")
    axes[2].set_ylabel("Density")
    axes[4].set_ylabel("Density")
    axes[3].set_xlabel(r"$X_T$")
    axes[4].set_xlabel(r"$X_T$")

    axes[5].axis("off")

    line = plt.Line2D(
        [0],
        [0],
        color="black",
        linewidth=EXACT_LINEWIDTH,
        label="exact transition density",
    )

    fill = plt.Rectangle(
        (0, 0),
        1,
        1,
        color=REGIME_COLOURS["A"],
        alpha=HIST_ALPHA,
        label="histogram / density fill",
    )

    leg = axes[5].legend(
        handles=[line, fill],
        title="Legend",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=True,
        fontsize=13,
        title_fontsize=14,
    )

    # Regime-colour key, stacked below the legend's actual drawn extent so the
    # two never overlap regardless of the cell height. (The sampling details
    # -- exact law X_T|X_0 ~ noncentral chi-squared, T and M -- live in the
    # LaTeX figure caption to keep this compact cell uncluttered.)
    fig.canvas.draw()
    leg_bb = leg.get_window_extent().transformed(axes[5].transAxes.inverted())
    key_title_y = leg_bb.y0 - 0.10
    dot_y = key_title_y - 0.16
    label_y = dot_y - 0.12

    axes[5].text(
        0.5,
        key_title_y,
        "Regime colours",
        transform=axes[5].transAxes,
        fontsize=13,
        va="top",
        ha="center",
    )

    # Five dots labelled A--E, centred horizontally (span 0.11*4=0.44).
    for i, name in enumerate(["A", "B", "C", "D", "E"]):
        x = 0.28 + 0.11 * i

        axes[5].scatter(
            x,
            dot_y,
            color=REGIME_COLOURS[name],
            s=55,
            transform=axes[5].transAxes,
        )

        axes[5].text(
            x,
            label_y,
            name,
            ha="center",
            va="center",
            transform=axes[5].transAxes,
            fontsize=12,
        )

    fig.tight_layout()

    output_path = figure_path(output)
    fig.savefig(output_path, bbox_inches = "tight")
    print(f"Saved {output_path}")

if __name__ == "__main__":
    main()