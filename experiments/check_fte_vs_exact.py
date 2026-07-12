# Compare the terminal values from the FTE scheme against exact CIR

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ncx2

from src.samplers.full_truncation_euler import fte_terminal

MASTER_SEED = 120339106

X0 = 0.02
KAPPA = 2.0
THETA = 0.02
T = 1
N_PATHS = 10_000 # Cannot use 10,000 this throws an error.
N_STEPS = 200


# Reduced regime check 
REGIMES =[
    ("A", 0.10),
    ("C", 0.2828),
    ("E", 0.80)
]

# Exact CIR

def cir_exact_params(
        x0: float,
        kappa: float,
        theta: float,
        sigma: float,
        T: float,
)-> tuple[float, float, float]:
    
    # Return the exact CIR transition parameters.
    # If Z is noncentral chi-squared then X_T = Z / c
    # so the density of X_T is f_X(x) = c f_Z(c x)

    exp_term = np.exp(-kappa * T)

    c = 4.0 * kappa / (sigma**2 * (1.0 - exp_term))
    df = 4.0 * kappa * theta / sigma**2
    nc = c * x0 * exp_term

    return c, df, nc

def cir_exact_density(
        x_grid: np.ndarray,
        x0: float,
        kappa: float,
        theta: float,
        sigma: float,
        T: float,
) -> np.ndarray:
    
    # Exact CIR terminal density on x_grid

    c, df, nc = cir_exact_params(
        x0 = x0,
        kappa = kappa,
        theta = theta,
        sigma = sigma,
        T=T,
    )

    return c * ncx2.pdf(c * x_grid, df, nc)

def cir_exact_quantile(
        q: float,
        x0: float,
        kappa: float,
        theta: float,
        sigma: float,
        T: float,
) -> float:
    
    c, df, nc = cir_exact_params(
        x0 = x0,
        kappa = kappa,
        theta = theta,
        sigma = sigma,
        T=T,
    )

    return ncx2.ppf(q, df, nc) / c


def cir_exact_mean(
        x0: float,
        kappa: float,
        theta: float,
        T: float,
) -> float:
    
    # Exact CIR conditional mean E[X_T | X_0 = x0]

    exp_term = np.exp(-kappa * T)
    return theta + (x0 - theta) * exp_term

# ----

# Recall that KAPPA, THETA and T are fixed constants, only changing sigma through regimes.

def main() -> None:
    rng = np.random.default_rng(MASTER_SEED)

    fig, axes = plt.subplots(1, 3,
                             figsize = (12, 3.8),
                             sharey = False)
    for ax, (regime_name, sigma) in zip(axes, REGIMES):
        delta = 4.0 * KAPPA * THETA / sigma**2

        X_T_fte = fte_terminal(
            X0 = X0,
            kappa = KAPPA,
            theta = THETA,
            sigma = sigma,
            T = T,
            n_steps = N_STEPS,
            n_paths = N_PATHS,
            rng = rng,
        )

        x_max = cir_exact_quantile(
            q = 0.999,
            x0 = X0,
            kappa = KAPPA,
            theta = THETA,
            sigma = sigma,
            T = T,
        )

        x_grid = np.linspace(1.0e-8, x_max, 700)

        exact_density = cir_exact_density(
            x_grid = x_grid,
            x0 = X0,
            kappa = KAPPA,
            theta = THETA,
            sigma = sigma, # current regimes sigma value from the loop
            T = T,
        )

        ax.hist(
            X_T_fte,
            bins = 70,
            range = (0.0, x_max),
            density = True,
            alpha = 0.35,
            label = "Full truncation euler histogram"
        )

        ax.plot(
            x_grid,
            exact_density,
            linewidth = 1.6,
            label = "Exact density"
        )

        # Since this is a high feller violating regime, the exact CIR density becomes singular near zero
        # Changing to log axis for this regime makes the visual more interpretatble 
        if regime_name == "E":
            ax.set_yscale("log")
            ax.set_ylabel("Density, log scale")
            

        ax.set_title(
            f"Regime {regime_name}\n"
            f"$\\sigma={sigma:g}$, $\\delta={delta:.2f}$"
        )
        ax.set_xlabel("$X_T$")
        ax.grid(True, alpha = 0.25)

    axes[0].set_ylabel("Density")
    axes[-1].legend(fontsize=8)

    fig.suptitle("FTE vs exact CIR terminal density")
    fig.tight_layout()

    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)

    output_path = figures_dir / "sanity_fte_vs_exact_minimal.pdf"
    fig.savefig(output_path, bbox_inches="tight")
    plt.show()

    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()