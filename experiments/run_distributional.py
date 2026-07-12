# Terminal-law diagnostics: Kolmogorov--Smirnov and Wasserstein-1 distances
# against the exact noncentral chi-squared CIR transition law.
#
# These compare LAWS, so the exact transition sampler is a valid comparator
# here (unlike in the strong-error experiment).  The sampler itself is
# included as a Monte Carlo noise floor: its KS/W1 values show the sampling
# error at the chosen number of paths, which no scheme can beat.
#
# Usage:
#   uv run python experiments/run_distributional.py
#   uv run python experiments/run_distributional.py --n-paths 5000  # smoke
#
# Outputs:
#   results/distributional_diagnostics.csv
#   figures/distributional_diagnostics.pdf

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.metrics.distributional import ks_statistic_vs_exact, wasserstein1_vs_exact
from src.samplers.exact import cir_ncx2_params
from src.samplers.full_truncation_euler import fte_terminal
from src.samplers.hh_milstein import hh_milstein_terminal
from src.samplers.klm_backstop import klm_backstop_terminal
from src.samplers.projected_euler import projected_euler_terminal
from src.utils.io import config_path, figure_path, results_path
from src.utils.rng import make_rng

N_STEPS_GRID = [8, 16, 32, 64, 128, 256]

SCHEME_STYLES = {
    "FTE": dict(color="tab:blue", marker="o"),
    "HH": dict(color="tab:orange", marker="s"),
    "ProjEuler": dict(color="tab:green", marker="^"),
    "KLM": dict(color="tab:purple", marker="v"),
    "Exact": dict(color="black", marker="x"),
}


def load_config(filename):
    with open(config_path(filename), encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_args():
    parser = argparse.ArgumentParser(description="Terminal-law diagnostics.")
    parser.add_argument("--n-paths", type=int, default=200000)
    parser.add_argument("--regimes", nargs="+", default=["A", "C", "E"])
    return parser.parse_args()


def exact_terminal_samples(x0, kappa, theta, sigma, T, n_paths, rng):
    c, df, nc = cir_ncx2_params(x0, kappa, theta, sigma, T)
    return rng.noncentral_chisquare(df, nc, size=n_paths) / c


def terminal_samples(scheme, params, n_steps, n_paths, rng, T):
    x0, kappa, theta, sigma = (
        params["x0"],
        params["kappa"],
        params["theta"],
        params["sigma"],
    )
    if scheme == "FTE":
        return fte_terminal(x0, kappa, theta, sigma, T, n_steps, n_paths, rng)
    if scheme == "HH":
        return hh_milstein_terminal(x0, kappa, theta, sigma, T, n_steps, n_paths, rng)
    if scheme == "ProjEuler":
        return projected_euler_terminal(
            x0, kappa, theta, sigma, T, n_steps, n_paths, rng
        )
    if scheme == "KLM":
        terminal, _ = klm_backstop_terminal(
            X0=x0,
            kappa=kappa,
            theta=theta,
            sigma=sigma,
            T=T,
            h_max=T / n_steps,
            n_paths=n_paths,
            rng=rng,
        )
        return terminal
    raise ValueError(f"unknown scheme {scheme!r}")


def main():
    args = get_args()

    regimes_cfg = load_config("regimes.yaml")
    experiments_cfg = load_config("experiments.yaml")

    shared = regimes_cfg["shared"]
    master_seed = experiments_cfg["shared"]["master_seed"]
    T = experiments_cfg["shared"]["T"]

    schemes = ["FTE", "HH", "ProjEuler", "KLM"]

    rows = []
    for regime_name in args.regimes:
        sigma = regimes_cfg["regimes"][regime_name]["sigma"]
        params = {
            "kappa": shared["kappa"],
            "theta": shared["theta"],
            "x0": shared["x0"],
            "sigma": sigma,
        }
        law_args = (params["x0"], params["kappa"], params["theta"], sigma, T)

        # Monte Carlo noise floor from the exact sampler itself.
        rng = make_rng(master_seed)
        exact_samples = exact_terminal_samples(
            *law_args, n_paths=args.n_paths, rng=rng
        )
        ks_floor = ks_statistic_vs_exact(exact_samples, *law_args)
        w1_floor = wasserstein1_vs_exact(exact_samples, *law_args)
        rows.append(
            {
                "regime": regime_name,
                "scheme": "Exact",
                "n_steps": 0,
                "ks": ks_floor,
                "w1": w1_floor,
                "n_paths": args.n_paths,
            }
        )
        print(f"[{regime_name}] exact-sampler floor: KS={ks_floor:.4f} W1={w1_floor:.2e}")

        for scheme in schemes:
            for n_steps in N_STEPS_GRID:
                rng = make_rng(master_seed)
                samples = terminal_samples(
                    scheme, params, n_steps, args.n_paths, rng, T
                )
                ks = ks_statistic_vs_exact(samples, *law_args)
                w1 = wasserstein1_vs_exact(samples, *law_args)
                rows.append(
                    {
                        "regime": regime_name,
                        "scheme": scheme,
                        "n_steps": n_steps,
                        "ks": ks,
                        "w1": w1,
                        "n_paths": args.n_paths,
                    }
                )
            print(f"[{regime_name}] {scheme} done")

    csv_path = results_path("distributional_diagnostics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path}")

    fig, axes = plt.subplots(
        2, len(args.regimes), figsize=(4.0 * len(args.regimes), 7.2), squeeze=False
    )
    for j, regime_name in enumerate(args.regimes):
        for metric, ax in zip(["ks", "w1"], [axes[0][j], axes[1][j]]):
            for scheme in schemes:
                scheme_rows = [
                    r
                    for r in rows
                    if r["regime"] == regime_name and r["scheme"] == scheme
                ]
                n = np.array([r["n_steps"] for r in scheme_rows])
                v = np.array([r[metric] for r in scheme_rows])
                ax.loglog(n, v, label=scheme, **SCHEME_STYLES[scheme])

            floor = next(
                r[metric]
                for r in rows
                if r["regime"] == regime_name and r["scheme"] == "Exact"
            )
            ax.axhline(floor, color="k", ls="--", lw=0.8, label="exact-sampler floor")
            ax.set_xlabel("number of steps")
            ax.set_ylabel("KS statistic" if metric == "ks" else "Wasserstein-1")
            ax.set_title(f"Regime {regime_name}")
            ax.grid(True, which="both", alpha=0.3)
            if j == 0:
                ax.legend(fontsize=7)

    fig.suptitle(f"Terminal-law diagnostics vs exact CIR law ({args.n_paths} paths)")
    fig.tight_layout()
    fig_path = figure_path("distributional_diagnostics.pdf")
    fig.savefig(fig_path)
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
