import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.io import ensure_dirs, figure_path, results_path
from src.utils.style import REGIME_COLOURS


INPUT_CSV = "fte_zero_boundary_diagnostic.csv"

NEAR_ZERO_EPSILON = 1.0e-3

OUTPUT_NEAR_ZERO_FIGURE = "fig_zero_boundary_near_zero_eps1e-3.pdf"
OUTPUT_EXACT_ZERO_FIGURE = "fig_zero_boundary_exact_zero_mass.pdf"
OUTPUT_REGIME_E_DECOMPOSITION_FIGURE = (
    "fig_zero_boundary_regime_E_decomposition_eps1e-3.pdf"
)

def load_results() -> pd.DataFrame:
    path = results_path(INPUT_CSV)
    df = pd.read_csv(path)

    df["epsilon"] = df["epsilon"].astype(float)
    df["dt"] = df["dt"].astype(float)
    df["n_steps"] = df["n_steps"].astype(int)

    return df


def plot_near_zero_mass(df: pd.DataFrame) -> None:
    """
    Plot empirical P(X_T^h <= eps) against dt, with exact probability
    shown as a horizontal reference line.
    """
    eps_df = df[df["epsilon"] == NEAR_ZERO_EPSILON].copy()

    regimes = list(eps_df["regime"].drop_duplicates())

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(regimes),
        figsize=(5.0 * len(regimes), 4.0),
        sharey=True,
    )

    if len(regimes) == 1:
        axes = [axes]

    for ax, regime in zip(axes, regimes):
        regime_df = eps_df[eps_df["regime"] == regime].sort_values("dt")
        colour = REGIME_COLOURS[regime]

        ax.plot(
            regime_df["dt"],
            regime_df["empirical_mass_le_epsilon"],
            marker="o",
            label="FTE empirical",
            color=colour,
        )

        exact_mass = regime_df["exact_mass_le_epsilon"].iloc[0]

        ax.axhline(
            exact_mass,
            linestyle="--",
            linewidth=1.5,
            color="black",
            label="Exact",
        )

        ax.set_xscale("log")
        ax.set_title(f"Regime {regime}")
        ax.set_xlabel(r"$\Delta t$")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(
        rf"$\mathbb{{P}}(X_T \leq {NEAR_ZERO_EPSILON:.0e})$"
    )

    axes[-1].legend()

    fig.suptitle(
        rf"Terminal near-zero mass at $\varepsilon={NEAR_ZERO_EPSILON:.0e}$"
    )
    fig.tight_layout()

    output_path = figure_path(OUTPUT_NEAR_ZERO_FIGURE)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved near-zero mass figure to: {output_path}")


def plot_exact_zero_mass(df: pd.DataFrame) -> None:
    """
    Plot the artificial exact-zero mass produced by FTE.
    Use one row per n_steps by dropping duplicate epsilon rows.
    """
    zero_df = (
        df.drop_duplicates(subset=["regime", "n_steps"])
        .sort_values(["regime", "dt"])
        .copy()
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    for regime in zero_df["regime"].drop_duplicates():
        regime_df = zero_df[zero_df["regime"] == regime].sort_values("dt")
        colour = REGIME_COLOURS[regime]

        ax.plot(
            regime_df["dt"],
            regime_df["terminal_zero_mass"],
            marker="o",
            label=f"Regime {regime}",
            color=colour,
        )

    ax.set_xscale("log")
    ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel(r"$\mathbb{P}(X_N^h = 0)$")
    ax.set_title("Artificial terminal exact-zero mass under FTE")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    output_path = figure_path(OUTPUT_EXACT_ZERO_FIGURE)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved exact-zero mass figure to: {output_path}")

def plot_regime_e_near_zero_decomposition(df: pd.DataFrame) -> None:
    """
    Decompose Regime E terminal near-zero mass into:

        P(X_N^h = 0)

    and

        P(0 < X_N^h <= epsilon).

    The stacked bar total equals empirical P(X_N^h <= epsilon).
    The dashed line shows the exact CIR P(X_T <= epsilon).
    """
    regime = "E"
    epsilon = NEAR_ZERO_EPSILON

    plot_df = (
        df[
            (df["regime"] == regime)
            & (df["epsilon"] == epsilon)
        ]
        .sort_values("n_steps")
        .copy()
    )

    if plot_df.empty:
        raise ValueError("No Regime E rows found for the chosen epsilon")

    plot_df["positive_near_zero_mass"] = (
        plot_df["empirical_mass_le_epsilon"]
        - plot_df["terminal_zero_mass"]
    ).clip(lower=0.0)

    x_positions = list(range(len(plot_df)))
    x_labels = [str(n) for n in plot_df["n_steps"]]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    ax.bar(
        x_positions,
        plot_df["terminal_zero_mass"],
        label=r"$\mathbb{P}(X_N^h = 0)$",
    )

    ax.bar(
        x_positions,
        plot_df["positive_near_zero_mass"],
        bottom=plot_df["terminal_zero_mass"],
        label=rf"$\mathbb{{P}}(0 < X_N^h \leq {epsilon:.0e})$",
    )

    exact_mass = plot_df["exact_mass_le_epsilon"].iloc[0]

    ax.axhline(
        exact_mass,
        linestyle="--",
        linewidth=1.5,
        color="black",
        label=rf"Exact $\mathbb{{P}}(X_T \leq {epsilon:.0e})$",
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)

    ax.set_xlabel(r"Number of time steps $N$")
    ax.set_ylabel("Terminal probability mass")
    ax.set_title(
        rf"Regime E near-zero mass decomposition at $\varepsilon={epsilon:.0e}$"
    )

    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()

    output_path = figure_path(OUTPUT_REGIME_E_DECOMPOSITION_FIGURE)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved Regime E decomposition figure to: {output_path}")


def main() -> None:
    ensure_dirs()

    df = load_results()

    plot_near_zero_mass(df)
    plot_exact_zero_mass(df)
    plot_regime_e_near_zero_decomposition(df)

if __name__ == "__main__":
    main()