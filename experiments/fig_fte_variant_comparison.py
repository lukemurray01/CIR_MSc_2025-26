# Where the positive part is taken matters: LKvD full truncation vs
# absorption-at-zero, terminal law in regime E.
#
# Chapter 3 states that the truncation convention has "visible
# distributional consequences near the boundary"; this figure is that
# evidence.  Both variants are driven by the SAME Brownian increments, so
# every visible difference is the variant choice, not Monte Carlo noise.
#
#   LKvD full truncation : carry the possibly negative auxiliary variable,
#                          truncate only the read-off (the scheme covered by
#                          the Cozma--Reisinger analysis).
#   Absorption           : clip the stored state to zero every step (what
#                          the pre-July-2026 code implemented).
#
# Left: terminal density against the exact noncentral chi-squared law.
# Right: lower-tail mass P(X_T <= eps), where absorption visibly parks
# paths at the boundary while the LKvD auxiliary variable can drift back.
#
# Usage:
#   uv run python experiments/fig_fte_variant_comparison.py
#
# Outputs:
#   figures/fig_fte_variant_comparison.pdf (+ .png preview)
#   results/fig_fte_variant_comparison.csv

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.stats import ncx2

from src.metrics.distributional import ks_statistic_vs_exact
from src.samplers.exact import cir_ncx2_params
from src.samplers.full_truncation_euler import fte_terminal_from_dW
from src.utils.io import config_path, figure_path, results_path
from src.utils.rng import make_brownian_increments, make_rng


def absorption_terminal_from_dW(X0, kappa, theta, sigma, dt, dW):
    """Pre-fix variant: clip the STORED state to zero each step."""
    n_paths, n_steps = dW.shape
    x = np.full(n_paths, X0, dtype=float)

    for n in range(n_steps):
        x_pos = np.maximum(x, 0.0)
        x = np.maximum(
            x + kappa * (theta - x_pos) * dt + sigma * np.sqrt(x_pos) * dW[:, n],
            0.0,
        )

    return x


def get_args():
    parser = argparse.ArgumentParser(description="FTE variant comparison figure.")
    parser.add_argument("--regime", default="E")
    parser.add_argument("--n-paths", type=int, default=100000)
    parser.add_argument("--n-steps", type=int, default=64)
    return parser.parse_args()


def load_config(filename):
    with open(config_path(filename), encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    args = get_args()

    regimes_cfg = load_config("regimes.yaml")
    experiments_cfg = load_config("experiments.yaml")

    shared = regimes_cfg["shared"]
    sigma = regimes_cfg["regimes"][args.regime]["sigma"]
    master_seed = experiments_cfg["shared"]["master_seed"]
    T = experiments_cfg["shared"]["T"]
    kappa, theta, x0 = shared["kappa"], shared["theta"], shared["x0"]

    dt = T / args.n_steps
    rng = make_rng(master_seed)
    dW = make_brownian_increments(rng, args.n_paths, args.n_steps, dt)

    lkvd = fte_terminal_from_dW(x0, kappa, theta, sigma, dt, dW)
    absorbed = absorption_terminal_from_dW(x0, kappa, theta, sigma, dt, dW)

    law = (x0, kappa, theta, sigma, T)
    c, df, nc = cir_ncx2_params(*law)

    exact_mean = theta + (x0 - theta) * np.exp(-kappa * T)
    summary = []
    for name, samples in [("LKvD full truncation", lkvd), ("Absorption", absorbed)]:
        summary.append(
            {
                "variant": name,
                "regime": args.regime,
                "n_paths": args.n_paths,
                "n_steps": args.n_steps,
                "terminal_mean": float(np.mean(samples)),
                "exact_mean": float(exact_mean),
                "rel_mean_error": float(abs(np.mean(samples) - exact_mean) / exact_mean),
                "ks_vs_exact": ks_statistic_vs_exact(samples, *law),
                "mass_at_zero": float(np.mean(samples == 0.0)),
                "mass_below_1e4": float(np.mean(samples <= 1e-4)),
            }
        )

    fig, (ax_pdf, ax_tail) = plt.subplots(1, 2, figsize=(10.5, 4.3))

    # -- Terminal density against the exact law -------------------------
    x_hi = float(np.quantile(np.concatenate([lkvd, absorbed]), 0.995))
    bins = np.linspace(0.0, x_hi, 90)
    ax_pdf.hist(
        lkvd, bins=bins, density=True, histtype="step",
        color="#4477AA", lw=1.5, label="LKvD full truncation",
    )
    ax_pdf.hist(
        absorbed, bins=bins, density=True, histtype="step",
        color="#EE6677", lw=1.5, ls="--", label="absorption variant",
    )
    grid = np.linspace(1e-6, x_hi, 600)
    ax_pdf.plot(
        grid, ncx2.pdf(c * grid, df, nc) * c, color="k", lw=1.2,
        label="exact transition density",
    )
    ax_pdf.set_yscale("log")
    ax_pdf.set_xlabel(r"$X_T$")
    ax_pdf.set_ylabel("density (log scale)")
    ax_pdf.set_title("Terminal density")
    ax_pdf.legend(fontsize=8)
    ax_pdf.grid(True, alpha=0.25)

    # -- Lower-tail mass -------------------------------------------------
    eps_grid = np.logspace(-8, np.log10(x_hi), 200)
    ax_tail.loglog(
        eps_grid, np.searchsorted(np.sort(lkvd), eps_grid, side="right") / lkvd.size,
        color="#4477AA", lw=1.5, label="LKvD full truncation",
    )
    ax_tail.loglog(
        eps_grid,
        np.searchsorted(np.sort(absorbed), eps_grid, side="right") / absorbed.size,
        color="#EE6677", lw=1.5, ls="--", label="absorption variant",
    )
    ax_tail.loglog(
        eps_grid, ncx2.cdf(c * eps_grid, df, nc), color="k", lw=1.2,
        label="exact CDF",
    )
    ax_tail.set_xlabel(r"$\varepsilon$")
    ax_tail.set_ylabel(r"$\mathbb{P}(X_T \leq \varepsilon)$")
    ax_tail.set_title("Lower-tail mass")
    ax_tail.legend(fontsize=8, loc="lower right")
    ax_tail.grid(True, which="both", alpha=0.25)

    delta = 4.0 * kappa * theta / sigma**2
    fig.suptitle(
        rf"Full truncation variants, regime {args.regime} "
        rf"($\delta$={delta:g}), {args.n_steps} steps, {args.n_paths} paths, "
        "shared Brownian increments"
    )
    fig.tight_layout()

    pdf_path = figure_path("fig_fte_variant_comparison.pdf")
    fig.savefig(pdf_path)
    fig.savefig(figure_path("fig_fte_variant_comparison.png"), dpi=150)
    print(f"wrote {pdf_path}")

    csv_path = results_path("fig_fte_variant_comparison.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f"wrote {csv_path}")
    for row in summary:
        print(
            f"{row['variant']:<22} mean={row['terminal_mean']:.6f} "
            f"(rel err {row['rel_mean_error']:.2%})  KS={row['ks_vs_exact']:.4f}  "
            f"P(X=0)={row['mass_at_zero']:.4f}"
        )


if __name__ == "__main__":
    main()
