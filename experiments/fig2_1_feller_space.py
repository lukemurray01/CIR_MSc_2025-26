import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.utils.cir_params import cir_delta
from src.utils.io import config_path, ensure_dirs, figure_path
from src.utils.style import REGIME_COLOURS


def main(quick: bool = False, outdir: str | None = None) -> None:
    ensure_dirs()

    with open(config_path("regimes.yaml"), encoding="utf-8") as f:
        reg_cfg = yaml.safe_load(f)

    kappa = reg_cfg["shared"]["kappa"]
    theta = reg_cfg["shared"]["theta"]

    sigma_grid = np.linspace(0.05, 0.95, 160 if quick else 400)
    delta_grid = cir_delta(kappa, theta, sigma_grid)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    y_top = 18.0

    ax.fill_between(
        sigma_grid,
        delta_grid,
        y_top,
        where=delta_grid >= 2.0,
        alpha=0.12,
        color="#0D652D",
        label=r"Feller satisfied ($\delta \geq 2$)",
    )

    ax.fill_between(
        sigma_grid,
        0.0,
        np.minimum(delta_grid, 2.0),
        where=delta_grid < 2.0,
        alpha=0.12,
        color="#A50E0E",
        label=r"Feller violated ($\delta < 2$)",
    )

    ax.plot(
        sigma_grid,
        delta_grid,
        color="black",
        linewidth=1.2,
    )

    ax.axhline(
        2.0,
        color="black",
        linestyle="--",
        linewidth=0.9,
        alpha=0.6,
    )

    ax.axhline(
        1.0,
        color="gray",
        linestyle=":",
        linewidth=0.9,
        alpha=0.6,
    )

    ax.text(
        0.88,
        2.15,
        r"Feller boundary $\delta=2$",
        fontsize=7.5,
        ha="right",
        color="0.3",
    )

    ax.text(
        0.88,
        1.15,
        r"$\alpha=0$ boundary $\delta=1$",
        fontsize=7.5,
        ha="right",
        color="0.5",
    )

    for regime_name, regime_data in reg_cfg["regimes"].items():
        sigma = regime_data["sigma"]
        delta = cir_delta(kappa, theta, sigma)
        colour = REGIME_COLOURS[regime_name]

        ax.scatter(
            sigma,
            delta,
            zorder=5,
            s=70,
            c=colour,
            edgecolors="black",
            linewidths=0.5,
        )

        offset = (6, -12) if regime_name == "A" else (6, 5)

        ax.annotate(
            regime_name,
            (sigma, delta),
            textcoords="offset points",
            xytext=offset,
            fontsize=9,
            fontweight="bold",
            color=colour,
        )

    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel(r"$\delta = 4\kappa\theta/\sigma^2$")
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0.0, y_top)

    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(alpha=0.18)

    ax.set_title(
        rf"CIR parameter space ($\kappa={kappa:g}$, $\theta={theta:g}$)"
    )

    fig.tight_layout()

    if outdir is None:
        output_path = figure_path("fig2_1_feller_space.pdf")
    else:
        output_dir = Path(outdir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "fig2_1_feller_space.pdf"

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Figure 2.1 saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--outdir")
    args = parser.parse_args()

    main(quick=args.quick, outdir=args.outdir)