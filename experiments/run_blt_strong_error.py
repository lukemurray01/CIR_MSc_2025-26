# BLT splitting strong-error benchmark across regimes A-E.
#
# The BLT scheme consumes exact (increment, running-infimum) pairs; coarse
# grids are obtained from the fine pairs by the exact aggregation identity
# (sum the increments, min-compose the infima), so every level sees the same
# Brownian path.  Errors are reported against TWO references on that path:
#
#   - a BLT self-reference at the fine grid (paper-style, reproduces the
#     min(1, delta/2) rates of the BLT paper's Table 1), and
#   - the HH truncated-Milstein reference on the same increments (the thesis
#     benchmark convention, comparable with run_strong_error.py output).
#
# Divergence between the two error curves at fine levels is itself a
# diagnostic: it bounds the reference contribution to the measured error.
#
# Usage:
#   uv run python experiments/run_blt_strong_error.py
#   uv run python experiments/run_blt_strong_error.py --regimes D E --n-paths 2000
#
# Outputs:
#   results/blt_strong_error.csv
#   figures/blt_strong_error.pdf

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
from src.metrics.weak_error import exact_cir_mean
from src.samplers.blt_splitting import blt_ab_from_cir, blt_terminal_from_noise
from src.samplers.hh_milstein import hh_milstein_terminal_from_dW
from src.utils.brownian import aggregate_brownian_increments_and_infima
from src.utils.cir_params import cir_delta
from src.utils.io import config_path, figure_path, results_path
from src.utils.rng import make_brownian_increments_with_infima, make_rng


def load_config(filename):
    with open(config_path(filename), encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_args():
    parser = argparse.ArgumentParser(
        description="BLT splitting strong-error benchmark (dual reference)."
    )
    parser.add_argument("--regimes", nargs="+", default=["A", "B", "C", "D", "E"])
    parser.add_argument("--n-paths", type=int, default=20000)
    # 32768 matches the CIR strong benchmark; at the old 8192 the finest
    # level (512) sat only a factor 16 below the reference, which biases the
    # fitted slope of an order-1 scheme.
    parser.add_argument("--reference-n-steps", type=int, default=32768)
    parser.add_argument(
        "--coarse-n-steps",
        nargs="+",
        type=int,
        default=[16, 32, 64, 128, 256, 512],
    )
    parser.add_argument("--drop-coarsest", type=int, default=1)
    return parser.parse_args()


def run_regime(regime_name, params, args, master_seed):
    T = params["T"]
    reference_n_steps = args.reference_n_steps
    dt_fine = T / reference_n_steps

    rng = make_rng(master_seed)
    dW_fine, m_fine = make_brownian_increments_with_infima(
        rng, args.n_paths, reference_n_steps, dt_fine
    )

    blt_reference = blt_terminal_from_noise(
        X0=params["x0"],
        kappa=params["kappa"],
        theta=params["theta"],
        sigma=params["sigma"],
        dt=dt_fine,
        dW=dW_fine,
        m=m_fine,
    )
    hh_reference = hh_milstein_terminal_from_dW(
        X0=params["x0"],
        kappa=params["kappa"],
        theta=params["theta"],
        sigma=params["sigma"],
        dt=dt_fine,
        dW=dW_fine,
    )
    for name, ref in [("BLT", blt_reference), ("HH", hh_reference)]:
        if not np.all(np.isfinite(ref)):
            raise RuntimeError(f"non-finite {name} reference in regime {regime_name}")

    exact_mean = exact_cir_mean(params["x0"], params["kappa"], params["theta"], T)
    a, _b = blt_ab_from_cir(params["kappa"], params["theta"], params["sigma"])

    rows = []
    for n_steps in args.coarse_n_steps:
        if reference_n_steps % n_steps != 0:
            raise ValueError(
                f"reference_n_steps={reference_n_steps} not divisible by {n_steps}"
            )
        factor = reference_n_steps // n_steps
        dt = dt_fine * factor
        dW, m = aggregate_brownian_increments_and_infima(dW_fine, m_fine, factor)

        start = time.perf_counter()
        terminal = blt_terminal_from_noise(
            X0=params["x0"],
            kappa=params["kappa"],
            theta=params["theta"],
            sigma=params["sigma"],
            dt=dt,
            dW=dW,
            m=m,
        )
        runtime = time.perf_counter() - start

        diff_blt = terminal - blt_reference
        diff_hh = terminal - hh_reference
        rows.append(
            {
                "regime": regime_name,
                "scheme": "BLT",
                "a": a,
                "delta": cir_delta(params["kappa"], params["theta"], params["sigma"]),
                "reference_n_steps": reference_n_steps,
                "n_paths": args.n_paths,
                "dt": dt,
                "l1_vs_blt_ref": float(np.mean(np.abs(diff_blt))),
                "l2_vs_blt_ref": float(np.sqrt(np.mean(diff_blt**2))),
                "l1_vs_hh_ref": float(np.mean(np.abs(diff_hh))),
                "l2_vs_hh_ref": float(np.sqrt(np.mean(diff_hh**2))),
                "terminal_zero_mass": float(np.mean(terminal == 0.0)),
                "rel_mean_bias": float((np.mean(terminal) - exact_mean) / exact_mean),
                "runtime_s": runtime,
            }
        )

    return rows


def fitted_tail_order(rows, key, drop_coarsest):
    rows = sorted(rows, key=lambda r: r["dt"])
    fit_rows = rows[:-drop_coarsest] if drop_coarsest else rows
    dt = np.array([r["dt"] for r in fit_rows])
    err = np.array([r[key] for r in fit_rows])
    return fit_loglog_order(dt, err)


def plot_summary(all_rows, regimes, args, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.0), sharey=False)
    axes = axes.ravel()

    for i, regime_name in enumerate(regimes):
        ax = axes[i]
        rows = sorted(
            (r for r in all_rows if r["regime"] == regime_name),
            key=lambda r: r["dt"],
        )
        dt = np.array([r["dt"] for r in rows])
        delta = rows[0]["delta"]

        for key, style, label in [
            ("l1_vs_blt_ref", dict(color="#CC6677", marker="o"), "L1 vs BLT ref"),
            ("l1_vs_hh_ref", dict(color="#228833", marker="s"), "L1 vs HH ref"),
        ]:
            err = np.array([r[key] for r in rows])
            order = fitted_tail_order(rows, key, args.drop_coarsest)
            ax.loglog(dt, err, label=f"{label} (tail {order:.2f})", **style)

        # Slope guides: paper rate min(1, delta/2) and order 1.
        anchor = max(r["l1_vs_blt_ref"] for r in rows)
        rate = min(1.0, delta / 2.0)
        ax.loglog(dt, anchor * (dt / dt[-1]) ** rate, "k--", lw=0.8,
                  label=f"slope {rate:g} = min(1, a/2)")
        if rate < 1.0:
            ax.loglog(dt, anchor * (dt / dt[-1]) ** 1.0, "k:", lw=0.8, label="slope 1")

        ax.set_title(f"Regime {regime_name}: a = delta = {delta:g}")
        ax.set_xlabel("step size h")
        ax.set_ylabel(r"strong $L^1$ error at $T$")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)

    axes[-1].axis("off")
    axes[-1].text(
        0.02,
        0.65,
        "BLT splitting scheme, exact (dW, inf) noise.\n"
        f"{args.n_paths} paths, fine grid {args.reference_n_steps} steps.\n"
        "Proven L1 rates: 1 (log) for a>2; a/2-eps for a in (1,2];\n"
        "a<1 uses the modified scheme (positive part), no theory,\n"
        "observed min(1, a/2).  HH-referenced curve shown for\n"
        "comparability with the main benchmark convention.",
        fontsize=8,
        va="top",
    )

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    args = get_args()

    regimes_cfg = load_config("regimes.yaml")
    experiments_cfg = load_config("experiments.yaml")
    shared = regimes_cfg["shared"]
    master_seed = experiments_cfg["shared"]["master_seed"]
    T = experiments_cfg["shared"]["T"]

    all_rows = []
    for regime_name in args.regimes:
        sigma = regimes_cfg["regimes"][regime_name]["sigma"]
        params = {
            "kappa": shared["kappa"],
            "theta": shared["theta"],
            "x0": shared["x0"],
            "sigma": sigma,
            "T": T,
        }
        print(f"Regime {regime_name} (sigma={sigma:g}, a={cir_delta(shared['kappa'], shared['theta'], sigma):g}) ...")
        rows = run_regime(regime_name, params, args, master_seed)
        for key in ["l1_vs_blt_ref", "l1_vs_hh_ref"]:
            order = fitted_tail_order(rows, key, args.drop_coarsest)
            print(f"  tail L1 order ({key}): {order:.3f}")
        all_rows.extend(rows)

    csv_path = results_path("blt_strong_error.csv")
    fieldnames = list(all_rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"wrote {csv_path}")

    fig_path = figure_path("blt_strong_error.pdf")
    plot_summary(all_rows, args.regimes, args, fig_path)
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
