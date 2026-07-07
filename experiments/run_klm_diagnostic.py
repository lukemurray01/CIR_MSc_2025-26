# KLM backstop-usage diagnostic across regimes.
#
# For each regime and each h_max the free-running backstopped adaptive scheme
# is run once, recording how often each backstop trigger fires and the
# terminal-mean error against the exact CIR mean.  This is the numerical
# companion to KLM Thm. 18 (backstop calls are controlled events, not the
# typical mode of the algorithm) and to the thesis positivity-ledger chapter.
#
# Usage:
#   uv run python experiments/run_klm_diagnostic.py
#   uv run python experiments/run_klm_diagnostic.py --n-paths 2000   # smoke
#
# Outputs:
#   results/klm_backstop_diagnostic.csv
#   figures/klm_backstop_diagnostic.pdf

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.metrics.weak_error import exact_cir_mean
from src.samplers.klm_backstop import klm_backstop_terminal
from src.utils.cir_params import cir_delta, kl_alpha
from src.utils.io import config_path, figure_path, results_path
from src.utils.rng import make_rng

H_MAX_LEVELS = [2.0**-k for k in range(3, 10)]  # 1/8 ... 1/512

REGIME_COLOURS = {
    "A": "tab:blue",
    "B": "tab:orange",
    "C": "tab:green",
    "D": "tab:red",
    "E": "tab:purple",
}


def load_config(filename):
    with open(config_path(filename), encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_args():
    parser = argparse.ArgumentParser(description="KLM backstop-usage diagnostic.")
    parser.add_argument("--n-paths", type=int, default=20000)
    parser.add_argument("--rho", type=float, default=64.0)
    parser.add_argument(
        "--regimes", nargs="+", default=["A", "B", "C", "D", "E"]
    )
    return parser.parse_args()


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
        kappa, theta, x0 = shared["kappa"], shared["theta"], shared["x0"]
        alpha = kl_alpha(kappa, theta, sigma)
        delta = cir_delta(kappa, theta, sigma)
        exact_mean = exact_cir_mean(x0, kappa, theta, T)

        for h_max in H_MAX_LEVELS:
            rng = make_rng(master_seed)
            terminal, stats = klm_backstop_terminal(
                X0=x0,
                kappa=kappa,
                theta=theta,
                sigma=sigma,
                T=T,
                h_max=h_max,
                n_paths=args.n_paths,
                rng=rng,
                rho=args.rho,
            )

            rows.append(
                {
                    "regime": regime_name,
                    "sigma": sigma,
                    "delta": delta,
                    "alpha": alpha,
                    "h_max": h_max,
                    "rho": args.rho,
                    "n_paths": args.n_paths,
                    "backstop_kind": stats["backstop_kind"],
                    "backstop_fraction": stats["backstop_fraction"],
                    "n_backstop_min": stats["n_backstop_min"],
                    "n_backstop_neg": stats["n_backstop_neg"],
                    "mean_steps_per_path": stats["n_steps_total"] / args.n_paths,
                    "terminal_mean": float(np.mean(terminal)),
                    "exact_mean": exact_mean,
                    "rel_mean_error": float(
                        abs(np.mean(terminal) - exact_mean) / exact_mean
                    ),
                    "min_terminal": float(np.min(terminal)),
                }
            )
            r = rows[-1]
            print(
                f"[{regime_name}] h_max={h_max:.5f}  "
                f"backstop={r['backstop_fraction']:.3%}  "
                f"steps/path={r['mean_steps_per_path']:.1f}  "
                f"rel mean err={r['rel_mean_error']:.3%}"
            )

    csv_path = results_path("klm_backstop_diagnostic.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.2))
    for regime_name in args.regimes:
        regime_rows = [r for r in rows if r["regime"] == regime_name]
        h = np.array([r["h_max"] for r in regime_rows])
        frac = np.array([r["backstop_fraction"] for r in regime_rows])
        err = np.array([r["rel_mean_error"] for r in regime_rows])
        colour = REGIME_COLOURS.get(regime_name)
        label = f"{regime_name} ($\\delta$={regime_rows[0]['delta']:.2f})"
        # A zero fraction cannot appear on a log axis; clip to a display floor.
        ax1.loglog(h, np.maximum(frac, 1e-8), "o-", color=colour, label=label)
        ax2.loglog(h, np.maximum(err, 1e-8), "o-", color=colour, label=label)

    ax1.set_xlabel(r"$h_{\max}$")
    ax1.set_ylabel("fraction of steps using the backstop")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=8)

    ax2.set_xlabel(r"$h_{\max}$")
    ax2.set_ylabel("relative terminal-mean error")
    ax2.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        rf"KLM backstopped adaptive scheme: $\rho$={args.rho:g}, "
        f"{args.n_paths} paths, T={T:g}"
    )
    fig.tight_layout()
    fig_path = figure_path("klm_backstop_diagnostic.pdf")
    fig.savefig(fig_path)
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
