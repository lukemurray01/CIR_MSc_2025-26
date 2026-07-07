# KLM backstop-candidate comparison: implicit vs projected vs BLT.
#
# Free-running KLM adaptive scheme with each admissible backstop map, judged
# on terminal-law quality against the exact noncentral chi-squared CIR law:
#
#   - relative terminal-mean bias (the projected backstop's documented
#     positive bias in regimes D/E is the baseline the BLT candidate has to
#     beat),
#   - KS and Wasserstein-1 distances to the exact law,
#   - backstop usage statistics and realised cost.
#
# The implicit backstop is only admissible for alpha > 0, so it appears in
# regime C but not in D/E; that asymmetry is exactly the thesis point.
#
# Usage:
#   uv run python experiments/run_klm_backstop_comparison.py
#   uv run python experiments/run_klm_backstop_comparison.py --n-paths 5000
#
# Outputs:
#   results/klm_backstop_comparison.csv
#   figures/klm_backstop_comparison.pdf

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import argparse
import csv
import time

import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.metrics.distributional import ks_statistic_vs_exact, wasserstein1_vs_exact
from src.metrics.weak_error import exact_cir_mean
from src.samplers.klm_backstop import klm_backstop_terminal
from src.utils.cir_params import kl_alpha
from src.utils.io import config_path, figure_path, results_path
from src.utils.rng import make_rng

BACKSTOP_STYLES = {
    "implicit": dict(color="#4477AA", marker="o"),
    "projected": dict(color="#7B1FA2", marker="v"),
    "blt": dict(color="#CC6677", marker="s"),
}


def load_config(filename):
    with open(config_path(filename), encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_args():
    parser = argparse.ArgumentParser(
        description="Compare KLM backstop candidates on terminal-law quality."
    )
    parser.add_argument("--regimes", nargs="+", default=["C", "D", "E"])
    parser.add_argument("--n-paths", type=int, default=20000)
    parser.add_argument(
        "--h-max-denominators",
        nargs="+",
        type=int,
        default=[16, 32, 64, 128, 256],
        help="h_max = 1/denominator for each grid level.",
    )
    parser.add_argument("--rho", type=float, default=64.0)
    return parser.parse_args()


def backstops_for(alpha):
    if alpha > 0.0:
        return ["implicit", "projected", "blt"]
    return ["projected", "blt"]


def main():
    args = get_args()

    regimes_cfg = load_config("regimes.yaml")
    experiments_cfg = load_config("experiments.yaml")
    shared = regimes_cfg["shared"]
    master_seed = experiments_cfg["shared"]["master_seed"]
    T = experiments_cfg["shared"]["T"]

    rows = []
    for regime_name in args.regimes:
        sigma = regimes_cfg["regimes"][regime_name]["sigma"]
        alpha = kl_alpha(shared["kappa"], shared["theta"], sigma)
        exact_mean = exact_cir_mean(shared["x0"], shared["kappa"], shared["theta"], T)
        law_args = (shared["x0"], shared["kappa"], shared["theta"], sigma, T)

        print(f"Regime {regime_name} (sigma={sigma:g}, alpha={alpha:+.5f}) ...")
        for denom in args.h_max_denominators:
            h_max = 1.0 / denom
            for kind in backstops_for(alpha):
                # Same seed per configuration: the Brownian draws coincide
                # until the first backstop-dependent divergence, so the
                # comparison uses common random numbers as far as possible.
                rng = make_rng(master_seed)

                start = time.perf_counter()
                terminal, stats = klm_backstop_terminal(
                    X0=shared["x0"],
                    kappa=shared["kappa"],
                    theta=shared["theta"],
                    sigma=sigma,
                    T=T,
                    h_max=h_max,
                    n_paths=args.n_paths,
                    rng=rng,
                    rho=args.rho,
                    backstop=kind,
                )
                runtime = time.perf_counter() - start

                if not np.all(np.isfinite(terminal)):
                    raise RuntimeError(
                        f"non-finite terminal ({regime_name}, {kind}, h=1/{denom})"
                    )
                if not np.all(terminal > 0.0):
                    raise RuntimeError(
                        f"non-positive terminal ({regime_name}, {kind}, h=1/{denom})"
                    )

                rows.append(
                    {
                        "regime": regime_name,
                        "alpha": alpha,
                        "backstop": kind,
                        "h_max": h_max,
                        "n_paths": args.n_paths,
                        "rho": args.rho,
                        "rel_mean_bias": float(
                            (np.mean(terminal) - exact_mean) / exact_mean
                        ),
                        "ks": ks_statistic_vs_exact(terminal, *law_args),
                        "w1": wasserstein1_vs_exact(terminal, *law_args),
                        "backstop_fraction": stats["backstop_fraction"],
                        "n_backstop_min": stats["n_backstop_min"],
                        "n_backstop_neg": stats["n_backstop_neg"],
                        "mean_steps_per_path": stats["n_steps_total"] / args.n_paths,
                        "runtime_s": runtime,
                    }
                )
            done = ", ".join(
                f"{r['backstop']}: bias {r['rel_mean_bias']:+.2%}"
                for r in rows
                if r["regime"] == regime_name and r["h_max"] == h_max
            )
            print(f"  h_max=1/{denom}: {done}")

    csv_path = results_path("klm_backstop_comparison.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path}")

    fig_path = figure_path("klm_backstop_comparison.pdf")
    plot_comparison(rows, args, fig_path)
    print(f"wrote {fig_path}")


def plot_comparison(rows, args, out_path):
    regimes = sorted({r["regime"] for r in rows})
    fig, axes = plt.subplots(
        len(regimes), 3, figsize=(12.0, 3.4 * len(regimes)), squeeze=False
    )

    metrics = [
        ("rel_mean_bias", "relative terminal-mean bias", "abs"),
        ("w1", "Wasserstein-1 vs exact law", "raw"),
        ("backstop_fraction", "backstop fraction of steps", "raw"),
    ]

    for i, regime_name in enumerate(regimes):
        regime_rows = [r for r in rows if r["regime"] == regime_name]
        kinds = sorted({r["backstop"] for r in regime_rows})

        for j, (key, ylabel, mode) in enumerate(metrics):
            ax = axes[i][j]
            for kind in kinds:
                pts = sorted(
                    (r for r in regime_rows if r["backstop"] == kind),
                    key=lambda r: r["h_max"],
                )
                h = np.array([r["h_max"] for r in pts])
                v = np.array([r[key] for r in pts])
                if mode == "abs":
                    v = np.abs(v)
                ax.loglog(h, np.maximum(v, 1e-16), label=f"{kind} backstop",
                          **BACKSTOP_STYLES[kind])

            ax.set_xlabel(r"$h_{\max}$")
            ax.set_ylabel(ylabel)
            ax.set_title(f"Regime {regime_name}")
            ax.grid(True, which="both", alpha=0.3)
            if j == 0:
                ax.legend(fontsize=8)

    fig.suptitle(
        f"KLM backstop candidates, free-running, {args.n_paths} paths, "
        f"rho={args.rho:g}; terminal law vs exact ncx2",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
