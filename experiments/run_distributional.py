# Terminal-law diagnostics: Kolmogorov--Smirnov and Wasserstein-1 distances
# against the exact noncentral chi-squared CIR transition law.
#
# These compare LAWS, so the exact transition sampler is a valid comparator
# here (unlike in the strong-error experiment).  The sampler itself is
# included as a Monte Carlo noise floor: its KS/W1 values show the sampling
# error at the chosen number of paths, which no scheme can beat.
#
# That floor is a DISTRIBUTION, not a line.  For an exact sample of size n the
# KS statistic has mean 0.8687/sqrt(n) and standard deviation 0.2611/sqrt(n),
# a 30% relative spread, so a scheme whose discretisation error is small
# compared with n^{-1/2} falls below any single exact draw roughly half the
# time.  The floor is therefore estimated by replication and plotted as a
# band; a curve inside it is at the resolution limit of the run and is not
# ranked against the others.
#
# Usage:
#   uv run python experiments/run_distributional.py
#   uv run python experiments/run_distributional.py --n-paths 5000 \
#       --n-floor-replicates 20            # smoke
#
# Outputs:
#   results/distributional_diagnostics.csv
#   results/distributional_floor_band.csv
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
    parser.add_argument(
        "--n-floor-replicates",
        type=int,
        default=200,
        help=(
            "Independent exact samples used to estimate the Monte Carlo floor. "
            "One draw is not a threshold: at 200,000 paths the KS statistic of "
            "an exact sample has a 30%% relative spread, so a scheme sitting at "
            "the floor lands below a single draw about half the time."
        ),
    )
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
    band_rows = []
    for regime_name in args.regimes:
        sigma = regimes_cfg["regimes"][regime_name]["sigma"]
        params = {
            "kappa": shared["kappa"],
            "theta": shared["theta"],
            "x0": shared["x0"],
            "sigma": sigma,
        }
        law_args = (params["x0"], params["kappa"], params["theta"], sigma, T)

        # Monte Carlo noise floor from the exact sampler itself.  This is a
        # DISTRIBUTION, not a threshold: both statistics are random at finite
        # sample size, so a single draw cannot say whether a scheme is at the
        # floor or merely got a lucky seed.  Replicating gives a band.
        ks_reps = np.empty(args.n_floor_replicates)
        w1_reps = np.empty(args.n_floor_replicates)
        for r in range(args.n_floor_replicates):
            rep = exact_terminal_samples(
                *law_args, n_paths=args.n_paths, rng=make_rng(master_seed + 1 + r)
            )
            ks_reps[r] = ks_statistic_vs_exact(rep, *law_args)
            w1_reps[r] = wasserstein1_vs_exact(rep, *law_args)

        band = {"regime": regime_name, "n_paths": args.n_paths,
                "n_replicates": args.n_floor_replicates}
        for name, vals in (("ks", ks_reps), ("w1", w1_reps)):
            band[f"{name}_mean"] = float(vals.mean())
            band[f"{name}_sd"] = float(vals.std(ddof=1))
            for q in (5, 50, 95):
                band[f"{name}_p{q:02d}"] = float(np.percentile(vals, q))
        band_rows.append(band)

        # The single-draw floor is retained so the row schema is unchanged.
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
        print(
            f"[{regime_name}] exact-sampler floor over {args.n_floor_replicates}"
            f" replicates: KS {band['ks_mean']:.5f} +/- {band['ks_sd']:.5f}"
            f" (5-95% {band['ks_p05']:.5f}-{band['ks_p95']:.5f}); "
            f"W1 {band['w1_mean']:.3e} +/- {band['w1_sd']:.3e}"
        )

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

    band_path = results_path("distributional_floor_band.csv")
    with open(band_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(band_rows[0].keys()))
        writer.writeheader()
        writer.writerows(band_rows)
    print(f"wrote {band_path}")

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

            band = next(b for b in band_rows if b["regime"] == regime_name)
            ax.axhspan(
                band[f"{metric}_p05"],
                band[f"{metric}_p95"],
                color="0.80",
                zorder=0,
                label="exact-sampler floor, 5--95%",
            )
            ax.axhline(
                band[f"{metric}_p50"], color="k", ls="--", lw=0.8, zorder=1,
                label="exact-sampler floor, median",
            )
            ax.set_xlabel("number of steps")
            ax.set_ylabel("KS statistic" if metric == "ks" else "Wasserstein-1")
            ax.set_title(f"Regime {regime_name}")
            ax.grid(True, which="both", alpha=0.3)
            if j == 0:
                ax.legend(fontsize=7)

    fig.suptitle(
        f"Terminal-law diagnostics vs exact CIR law ({args.n_paths:,} paths).\n"
        f"Shaded: 5--95% of the exact sampler's own statistic over "
        f"{args.n_floor_replicates} replicates. A curve inside the band is at "
        "the resolution limit, not better than exact.",
        fontsize=9,
    )
    fig.tight_layout()
    fig_path = figure_path("distributional_diagnostics.pdf")
    fig.savefig(fig_path)
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
