import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from src.samplers.full_truncation_euler import fte_paths_from_dW
from src.utils.brownian import aggregate_brownian_increments
from src.utils.io import ensure_dirs, figure_path
from src.utils.rng import make_brownian_increments, make_rng


# Change this one line to choose the regime.
REGIME = "A"

REGIMES = {
    "A": {"sigma": 0.10, "description": "deep Feller regime"},
    "B": {"sigma": 0.20, "description": "Feller-safe regime"},
    "C": {"sigma": np.sqrt(0.08), "description": "Feller boundary"},
    "D": {"sigma": 0.50, "description": "Feller-violating regime"},
    "E": {"sigma": 0.80, "description": "severe Feller-violating regime"},
}

OUTPUT_FIGURE = "fig_strong_convergence_pathwise_gap_schematic.pdf"


def regime_parameters(regime: str) -> tuple[float, str]:
    if regime not in REGIMES:
        valid = ", ".join(REGIMES)
        raise ValueError(f"Unknown regime '{regime}'. Choose one of: {valid}")

    sigma = REGIMES[regime]["sigma"]
    description = REGIMES[regime]["description"]

    return sigma, description


def main() -> None:
    ensure_dirs()

    x0 = 0.02
    kappa = 2.0
    theta = 0.02
    T = 1.0

    sigma, regime_description = regime_parameters(REGIME)

    delta = 4.0 * kappa * theta / sigma**2
    feller_ratio = delta / 2.0

    # More candidate paths makes it more likely that we find a visually useful path.
    n_candidate_paths = 500

    # Keep the fine reference very fine, and the coarse path coarse enough
    # that the pathwise gap is visible.
    n_fine_steps = 2048
    n_coarse_steps = 16

    if n_fine_steps % n_coarse_steps != 0:
        raise ValueError("n_fine_steps must be divisible by n_coarse_steps")

    factor = n_fine_steps // n_coarse_steps

    dt_fine = T / n_fine_steps
    dt_coarse = T / n_coarse_steps

    rng = make_rng(120339106)

    dW_fine = make_brownian_increments(
        rng=rng,
        n_paths=n_candidate_paths,
        n_steps=n_fine_steps,
        dt=dt_fine,
    )

    dW_coarse = aggregate_brownian_increments(
        dW_fine=dW_fine,
        factor=factor,
    )

    fine_reference_paths = fte_paths_from_dW(
        X0=x0,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        dt=dt_fine,
        dW=dW_fine,
    )

    coarse_paths = fte_paths_from_dW(
        X0=x0,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        dt=dt_coarse,
        dW=dW_coarse,
    )

    fine_reference_at_coarse_times = fine_reference_paths[:, ::factor]

    pathwise_gaps = np.abs(coarse_paths - fine_reference_at_coarse_times)

    # Choose the path with the largest visible coarse/fine discrepancy.
    # Ignore t=0, because both paths start at x0.
    max_gap_by_path = np.max(pathwise_gaps[:, 1:], axis=1)
    selected_path_index = int(np.argmax(max_gap_by_path))

    fine_reference_path = fine_reference_paths[selected_path_index]
    coarse_path = coarse_paths[selected_path_index]
    fine_reference_coarse = fine_reference_at_coarse_times[selected_path_index]
    selected_gaps = pathwise_gaps[selected_path_index]

    t_fine = np.linspace(0.0, T, n_fine_steps + 1)
    t_coarse = np.linspace(0.0, T, n_coarse_steps + 1)

    fig, ax = plt.subplots(figsize=(7.4, 4.6))

    ax.plot(
        t_fine,
        fine_reference_path,
        linewidth=1.8,
        label=rf"Fine-grid FTE reference $Y^{{h_{{\rm ref}}}}$, $N={n_fine_steps}$",
    )

    ax.plot(
        t_coarse,
        coarse_path,
        marker="o",
        linewidth=1.3,
        markersize=4.2,
        label=rf"Coarse FTE approximation $Y^h$, $N={n_coarse_steps}$",
    )

    # Mark the six largest pathwise gaps at coarse time points.
    gap_indices = np.argsort(selected_gaps[1:])[-6:] + 1
    gap_indices = sorted(int(i) for i in gap_indices)

    for idx in gap_indices:
        ax.vlines(
            t_coarse[idx],
            ymin=min(fine_reference_coarse[idx], coarse_path[idx]),
            ymax=max(fine_reference_coarse[idx], coarse_path[idx]),
            linewidth=1.2,
            alpha=0.85,
        )

    largest_gap_index = int(np.argmax(selected_gaps[1:]) + 1)
    largest_gap = selected_gaps[largest_gap_index]

    ax.annotate(
        rf"Example pathwise gap $\approx {largest_gap:.4f}$",
        xy=(
            t_coarse[largest_gap_index],
            max(
                fine_reference_coarse[largest_gap_index],
                coarse_path[largest_gap_index],
            ),
        ),
        xytext=(0.48, 0.88),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "linewidth": 1.0},
        fontsize=9,
    )

    # Zoom the y-axis around the selected pair of paths so the gap is visible.
    displayed_values = np.concatenate([fine_reference_path, coarse_path])
    y_min = float(np.min(displayed_values))
    y_max = float(np.max(displayed_values))
    y_range = y_max - y_min

    if y_range > 0.0:
        padding = 0.08 * y_range
        ax.set_ylim(y_min - padding, y_max + padding)

    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"State value")
    ax.set_title(
        (
            "Brownian-coupled pathwise gap: "
            f"Regime {REGIME} "
            rf"$(\sigma={sigma:.4f},\, \nu={feller_ratio:.3f})$"
        )
    )

    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    ax.text(
        0.02,
        0.04,
        (
            rf"Regime {REGIME}: {regime_description}."
            "\n"
            r"True strong error compares $|X_t-Y^h_t|$ under the same Brownian path."
            "\n"
            r"Plotted proxy compares $|Y^{h_{\rm ref}}_t-Y^h_t|$ using shared Brownian increments."
        ),
        transform=ax.transAxes,
        verticalalignment="bottom",
        fontsize=9,
        bbox={"boxstyle": "round", "alpha": 0.15},
    )

    fig.tight_layout()

    output_path = figure_path(OUTPUT_FIGURE)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Regime: {REGIME}")
    print(f"sigma: {sigma:.6f}")
    print(f"delta: {delta:.6f}")
    print(f"Feller ratio nu: {feller_ratio:.6f}")
    print(f"Selected path index: {selected_path_index}")
    print(f"Largest coarse/fine gap on selected path: {largest_gap:.6f}")
    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()