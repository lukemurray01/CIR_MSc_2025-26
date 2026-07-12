# Coupled strong error of the KLM scheme per BACKSTOP MAP, against the
# Hefter--Herzwurm (HH) fine-grid reference on a shared Brownian path.
#
# Companion to run_klm_backstop_comparison.py, which compares the backstop
# maps in distributional metrics (mean bias, KS, W1) only.  This experiment
# adds the missing strong-error view: every backstop variant consumes
# partial sums of the SAME fine increments as the reference, so an
# overlapping RMSE-vs-dt plot demonstrates that the maps are pathwise
# equivalent (or shows where they are not).  The implicit map is included
# automatically where it is defined (alpha > 0, regime C of the default
# set); D and E compare projected vs blt.
#
# The blt backstop consumes auxiliary exponentials for the bridge infimum;
# these come from a dedicated rng per (backstop, level) so the shared fine
# path stays identical across variants.
#
# Usage:
#   uv run python experiments/run_klm_backstop_strong_error.py
#   uv run python experiments/run_klm_backstop_strong_error.py --regimes D E
#   uv run python experiments/run_klm_backstop_strong_error.py \
#       --n-paths 2000 --reference-n-steps 4096      # smoke run
#
# Outputs:
#   results/klm_backstop_strong_error.csv
#   figures/klm_backstop_strong_error.pdf (+ .png preview)

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

from src.metrics.strong_error import fit_loglog_order
from src.samplers.hh_milstein import hh_milstein_terminal_from_dW
from src.samplers.klm_backstop import klm_backstop_terminal_from_fine_dW
from src.utils.cir_params import kl_alpha
from src.utils.io import config_path, figure_path, results_path
from src.utils.rng import make_rng

BACKSTOP_COLOURS = {
    "implicit": "#4477AA",
    "projected": "#AA3377",
    "blt": "#CC6677",
}
BACKSTOP_INDEX = {"implicit": 1, "projected": 2, "blt": 3}


def load_config(filename):
    with open(config_path(filename), encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_args():
    parser = argparse.ArgumentParser(
        description="KLM coupled strong error per backstop map."
    )
    parser.add_argument("--regimes", nargs="+", default=["C", "D", "E"])
    parser.add_argument("--n-paths", type=int, default=None,
                        help="Override the configured number of paths (smoke runs).")
    parser.add_argument("--reference-n-steps", type=int, default=None,
                        help="Override the configured HH fine-reference step count.")
    parser.add_argument("--rho", type=float, default=64.0,
                        help="KLM backstop floor parameter h_min = h_max / rho.")
    parser.add_argument("--max-brownian-gib", type=float, default=0.5,
                        help="Memory budget for one batch of fine increments.")
    parser.add_argument("--drop-coarsest", type=int, default=1,
                        help="Largest-h points excluded from fitted order labels.")
    return parser.parse_args()


def batch_size_for(n_paths, reference_n_steps, max_brownian_gib):
    bytes_per_path = reference_n_steps * np.dtype(float).itemsize
    budget_paths = int(max_brownian_gib * 2**30 / bytes_per_path)
    return max(1, min(n_paths, budget_paths))


def run_regime(regime_name, params, grid, master_seed, kinds, rho):
    n_paths = grid["n_paths"]
    reference_n_steps = grid["reference_n_steps"]
    coarse_list = grid["coarse_n_steps"]
    T = params["T"]
    dt_fine = T / reference_n_steps
    for n_steps in coarse_list:
        if reference_n_steps % n_steps != 0:
            raise ValueError(
                f"reference_n_steps={reference_n_steps} not divisible by {n_steps}"
            )

    rng = make_rng(master_seed)
    # Dedicated auxiliary streams so the blt bridge-infimum draws cannot
    # perturb the shared fine path or each other across levels.
    aux_rngs = {
        (kind, n_steps): np.random.default_rng(
            np.random.SeedSequence([master_seed, BACKSTOP_INDEX[kind], n_steps])
        )
        for kind in kinds
        for n_steps in coarse_list
    }

    acc = {
        (kind, n_steps): {
            "sum_abs": 0.0, "sum_sq": 0.0, "steps": 0.0,
            "events": 0.0, "runtime_s": 0.0,
        }
        for kind in kinds
        for n_steps in coarse_list
    }

    batch = batch_size_for(n_paths, reference_n_steps, grid["max_brownian_gib"])
    for batch_start in range(0, n_paths, batch):
        batch_n = min(batch, n_paths - batch_start)
        dW_fine = np.sqrt(dt_fine) * rng.standard_normal(
            (batch_n, reference_n_steps)
        )
        reference_terminal = hh_milstein_terminal_from_dW(
            X0=params["x0"], kappa=params["kappa"], theta=params["theta"],
            sigma=params["sigma"], dt=dt_fine, dW=dW_fine,
        )
        if not np.all(np.isfinite(reference_terminal)):
            raise RuntimeError(
                f"non-finite reference terminal values in regime {regime_name}"
            )

        for kind in kinds:
            for n_steps in coarse_list:
                a = acc[(kind, n_steps)]
                dt_coarse = dt_fine * (reference_n_steps // n_steps)

                start = time.perf_counter()
                terminal, stats = klm_backstop_terminal_from_fine_dW(
                    X0=params["x0"], kappa=params["kappa"],
                    theta=params["theta"], sigma=params["sigma"],
                    T=T, h_max=dt_coarse, dW_fine=dW_fine, rho=rho,
                    backstop=kind,
                    rng=aux_rngs[(kind, n_steps)] if kind == "blt" else None,
                )
                a["runtime_s"] += time.perf_counter() - start

                diff = terminal - reference_terminal
                a["sum_abs"] += float(np.sum(np.abs(diff)))
                a["sum_sq"] += float(np.sum(diff**2))
                a["steps"] += stats["n_steps_total"]
                a["events"] += stats["n_backstop_min"] + stats["n_backstop_neg"]

    rows = []
    for kind in kinds:
        for n_steps in coarse_list:
            a = acc[(kind, n_steps)]
            rows.append({
                "regime": regime_name,
                "scheme": "KLM",
                "backstop": kind,
                "reference": "HH",
                "dt": T / n_steps,
                "l1": a["sum_abs"] / n_paths,
                "l2": float(np.sqrt(a["sum_sq"] / n_paths)),
                "mean_steps_per_path": a["steps"] / n_paths,
                "backstop_fraction": a["events"] / max(a["steps"], 1.0),
                "runtime_s": a["runtime_s"],
            })
    return rows


def plot_regimes(all_rows, regimes, grid, drop_coarsest, out_path):
    fig, axes = plt.subplots(
        1, len(regimes), figsize=(4.6 * len(regimes), 4.4), squeeze=False
    )
    for ax, regime in zip(axes[0], regimes):
        rows = [r for r in all_rows if r["regime"] == regime]
        kinds = sorted({r["backstop"] for r in rows}, key=BACKSTOP_INDEX.get)
        for kind in kinds:
            line = sorted(
                (r for r in rows if r["backstop"] == kind),
                key=lambda r: r["dt"],
            )
            dts = np.array([r["dt"] for r in line])
            l2s = np.array([r["l2"] for r in line])
            order = fit_loglog_order(dts[: len(dts) - drop_coarsest or None],
                                     l2s[: len(l2s) - drop_coarsest or None])
            ax.loglog(
                dts, l2s, "o-", color=BACKSTOP_COLOURS[kind], markersize=4,
                label=f"{kind} (slope {order:.2f})",
            )
        ax.set_xlabel(r"$\Delta t_{\max}$")
        ax.set_ylabel(r"RMSE vs HH reference at $T$")
        ax.set_title(f"regime {regime}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(
        "KLM coupled strong error per backstop map -- "
        f"{grid['n_paths']} paths, HH reference at "
        f"{grid['reference_n_steps']} steps, shared Brownian path",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path)
    fig.savefig(Path(out_path).with_suffix(".png"), dpi=150)
    plt.close(fig)


def main():
    args = get_args()

    regimes_cfg = load_config("regimes.yaml")
    experiments_cfg = load_config("experiments.yaml")
    shared = regimes_cfg["shared"]

    grid = dict(experiments_cfg["time_grids"]["strong_error"])
    if args.n_paths is not None:
        grid["n_paths"] = args.n_paths
    if args.reference_n_steps is not None:
        grid["reference_n_steps"] = args.reference_n_steps
    grid["max_brownian_gib"] = args.max_brownian_gib
    master_seed = experiments_cfg["shared"]["master_seed"]
    T = experiments_cfg["shared"]["T"]

    all_rows = []
    for regime_name in args.regimes:
        sigma = regimes_cfg["regimes"][regime_name]["sigma"]
        params = {
            "kappa": shared["kappa"], "theta": shared["theta"],
            "x0": shared["x0"], "sigma": sigma, "T": T,
        }
        alpha = kl_alpha(params["kappa"], params["theta"], params["sigma"])
        kinds = (["implicit", "projected", "blt"] if alpha > 0.0
                 else ["projected", "blt"])
        print(f"Regime {regime_name} (sigma={sigma:g}, alpha={alpha:+.4f}): "
              f"backstops {kinds}")
        all_rows.extend(
            run_regime(regime_name, params, grid, master_seed, kinds, args.rho)
        )

    csv_path = results_path("klm_backstop_strong_error.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"wrote {csv_path}")

    fig_path = figure_path("klm_backstop_strong_error.pdf")
    plot_regimes(all_rows, args.regimes, grid, args.drop_coarsest, fig_path)
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
