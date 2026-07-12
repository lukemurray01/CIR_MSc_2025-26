# Mean-reverting CEV extension: the thesis Pillar-3 minimum experiment.
#
# For each beta the projected Lamperti scheme is compared against a fine
# self-reference on the same Brownian path (no exact CEV law is available),
# and the terminal mean is checked against the closed-form first moment,
# which closes for every beta in the mean-reverting family.
#
# beta = 1/2 doubles as a CIR consistency check: the transformed drift then
# reduces exactly to the CIR Lamperti drift (see tests/test_cev.py for the
# pathwise identity), so the observed convergence behaviour should match the
# CIR projected Euler benchmark.
#
# Usage:
#   uv run python experiments/run_cev_experiment.py
#   uv run python experiments/run_cev_experiment.py --betas 0.5 0.75 0.9
#
# Outputs:
#   results/cev_convergence.csv
#   figures/cev_convergence.pdf

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.metrics.strong_error import fit_loglog_order
from src.samplers.cev import cev_exact_mean, cev_projected_terminal_from_dW
from src.utils.brownian import aggregate_brownian_increments
from src.utils.io import config_path, figure_path, results_path
from src.utils.rng import make_rng

def load_config(filename):
    with open(config_path(filename), encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_args():
    parser = argparse.ArgumentParser(description="CEV convergence experiment.")
    parser.add_argument("--betas", nargs="+", type=float, default=[0.5, 0.75])
    parser.add_argument("--regime", default="B")
    parser.add_argument("--n-paths", type=int, default=20000)
    parser.add_argument(
        "--reference-n-steps",
        type=int,
        default=32768,
        help="Self-convergence reference resolution; matches the CIR strong "
        "benchmark so the CEV extension is not the weakest-referenced "
        "experiment in the suite.",
    )
    parser.add_argument(
        "--coarse-n-steps",
        nargs="+",
        type=int,
        default=[16, 32, 64, 128, 256, 512],
        help="Coarse levels (must divide the reference); matches the CIR "
        "strong-error grid.",
    )
    parser.add_argument(
        "--path-batch-size",
        type=int,
        default=4000,
        help="Paths per batch; keeps the fine-increment buffer ~1 GiB at the "
        "default reference.",
    )
    return parser.parse_args()


def main():
    args = get_args()

    regimes_cfg = load_config("regimes.yaml")
    experiments_cfg = load_config("experiments.yaml")

    shared = regimes_cfg["shared"]
    sigma = regimes_cfg["regimes"][args.regime]["sigma"]
    master_seed = experiments_cfg["shared"]["master_seed"]
    T = experiments_cfg["shared"]["T"]

    kappa, theta, f0 = shared["kappa"], shared["theta"], shared["x0"]
    reference_n_steps = args.reference_n_steps
    coarse_n_steps = args.coarse_n_steps
    for n_steps in coarse_n_steps:
        if reference_n_steps % n_steps != 0:
            raise ValueError(
                f"reference_n_steps={reference_n_steps} is not divisible "
                f"by coarse n_steps={n_steps}"
            )
    dt_fine = T / reference_n_steps
    exact_mean = cev_exact_mean(f0, kappa, theta, T)

    rows = []
    for beta in args.betas:
        rng = make_rng(master_seed)
        acc = {
            n_steps: {"abs_sum": 0.0, "sq_sum": 0.0, "terminal_sum": 0.0}
            for n_steps in coarse_n_steps
        }
        done = 0
        while done < args.n_paths:
            batch_n = min(args.path_batch_size, args.n_paths - done)
            dW_fine = np.sqrt(dt_fine) * rng.standard_normal(
                (batch_n, reference_n_steps)
            )

            reference = cev_projected_terminal_from_dW(
                F0=f0,
                kappa=kappa,
                theta=theta,
                sigma=sigma,
                beta=beta,
                dt=dt_fine,
                dW=dW_fine,
            )

            for n_steps in coarse_n_steps:
                factor = reference_n_steps // n_steps
                dW_coarse = aggregate_brownian_increments(dW_fine, factor)

                terminal = cev_projected_terminal_from_dW(
                    F0=f0,
                    kappa=kappa,
                    theta=theta,
                    sigma=sigma,
                    beta=beta,
                    dt=dt_fine * factor,
                    dW=dW_coarse,
                )

                diff = terminal - reference
                acc[n_steps]["abs_sum"] += float(np.sum(np.abs(diff)))
                acc[n_steps]["sq_sum"] += float(np.sum(diff**2))
                acc[n_steps]["terminal_sum"] += float(np.sum(terminal))
            done += batch_n

        for n_steps in coarse_n_steps:
            factor = reference_n_steps // n_steps
            terminal_mean = acc[n_steps]["terminal_sum"] / args.n_paths
            rows.append(
                {
                    "beta": beta,
                    "regime": args.regime,
                    "sigma": sigma,
                    "dt": dt_fine * factor,
                    "l1": acc[n_steps]["abs_sum"] / args.n_paths,
                    "l2": float(np.sqrt(acc[n_steps]["sq_sum"] / args.n_paths)),
                    "terminal_mean": terminal_mean,
                    "exact_mean": exact_mean,
                    "rel_mean_error": float(
                        abs(terminal_mean - exact_mean) / exact_mean
                    ),
                    "n_paths": args.n_paths,
                    "reference_n_steps": reference_n_steps,
                }
            )

        beta_rows = [r for r in rows if r["beta"] == beta]
        dt = np.array([r["dt"] for r in beta_rows])
        l2 = np.array([r["l2"] for r in beta_rows])
        order = fit_loglog_order(dt, l2)
        finest = beta_rows[-1]
        print(
            f"beta={beta:g}: self-convergence L2 order {order:.3f}, "
            f"finest rel mean error {finest['rel_mean_error']:.3%}"
        )

    csv_path = results_path("cev_convergence.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path}")

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    for beta in args.betas:
        beta_rows = [r for r in rows if r["beta"] == beta]
        dt = np.array([r["dt"] for r in beta_rows])
        l2 = np.array([r["l2"] for r in beta_rows])
        order = fit_loglog_order(dt, l2)
        ax.loglog(dt, l2, "o-", label=rf"$\beta$={beta:g} (order {order:.2f})")

    h = np.array(sorted({r["dt"] for r in rows}))
    anchor = max(r["l2"] for r in rows)
    ax.loglog(h, anchor * (h / h[-1]) ** 0.5, "k--", lw=0.8, label="slope 1/2")
    ax.loglog(h, anchor * (h / h[-1]) ** 1.0, "k:", lw=0.8, label="slope 1")

    ax.set_xlabel(r"step size $h$")
    ax.set_ylabel(r"self-convergence $L^2$ error at $T$")
    ax.set_title(
        f"CEV projected Lamperti scheme, regime {args.regime} "
        rf"($\sigma$={sigma:g}), {args.n_paths} paths"
    )
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig_path = figure_path("cev_convergence.pdf")
    fig.savefig(fig_path)
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
