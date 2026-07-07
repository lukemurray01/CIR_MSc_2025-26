# Weak-error benchmark for the terminal test functions g1-g3 and the
# path-dependent bond functional g4, across regimes A-E.
#
# Definitions follow the thesis background chapter (eq:weak):
#   g1(x) = x                      exact: CIR mean
#   g2(x) = (x - K)_+^2, K = 0.02  exact: quadrature vs ncx2 density
#   g3(x) = exp(-x)                exact: CIR Laplace transform
#   g4(path) = exp(-int_0^T x dt)  exact: affine zero-coupon bond price;
#                                  the path integral uses the trapezoidal rule.
#
# Scheme roles follow the thesis method registry (tab:method-registry): the
# weak comparison has no reference-path conflict, so HH and IF are ranked as
# ordinary benchmarked schemes; KL is uniform in A-C and the exploratory
# adaptive soft-zero variant in D-E; IF exists only for alpha > 0; KLM and
# the adaptive variants are terminal-only (no uniform grid), so g4 is not
# defined for them.
#
# Error-floor policy follows the thesis methodology chapter: every point
# records the Monte Carlo standard error and the ratio |error|/s.e.; global
# order fits use ONLY points with |error| >= 2 s.e. and are refused (NaN)
# when fewer than three such points exist.  Consecutive-level local slopes
# are written alongside the global fit.
#
# Noise: schemes on the shared-noise interface consume identical
# pre-generated (increment, infimum) pairs per level; free-running adaptive
# schemes draw their own noise (recorded in the `noise` column).  Weak error
# concerns expectations, so this affects estimator variance, not bias.
#
# Usage:
#   uv run python experiments/run_weak_error.py
#   uv run python experiments/run_weak_error.py --regimes A E --n-paths 20000
#
# Outputs:
#   results/weak_error.csv
#   results/weak_error_orders.csv
#   figures/weak_error_regime_{R}.pdf (+ .png previews)

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
from src.metrics.weak_error import (
    TERMINAL_PAYOFFS,
    affine_cir_bond_price,
    g4_bond_discount_from_paths,
    terminal_exact_expectations,
)
from src.samplers.blt_splitting import blt_paths_from_noise
from src.samplers.full_truncation_euler import fte_paths_from_dW
from src.samplers.hh_milstein import hh_milstein_paths_from_dW
from src.samplers.kelly_lord import kl_uniform_paths_from_dW
from src.samplers.kelly_lord_adaptive import kl_adaptive_terminal
from src.samplers.klm_backstop import klm_backstop_terminal
from src.samplers.lamperti_implicit import if_paths_from_dW
from src.samplers.projected_euler import projected_euler_paths_from_dW
from src.utils.cir_params import kl_alpha
from src.utils.io import config_path, figure_path, results_path
from src.utils.rng import make_brownian_increments_with_infima, make_rng
from src.utils.style import METHOD_COLOURS, METHOD_LABELS

ALL_SCHEMES = ["FTE", "HH", "ProjEuler", "KL", "IF", "KLM", "BLT"]
PAYOFFS = ["g1", "g2", "g3", "g4"]

# IF has no registered style of its own in METHOD_COLOURS; give it one here.
SCHEME_STYLES = {
    "FTE": dict(color=METHOD_COLOURS["FTE"], marker="o"),
    "HH": dict(color=METHOD_COLOURS["HH"], marker="s"),
    "ProjEuler": dict(color=METHOD_COLOURS["ProjEuler"], marker="^"),
    "KL": dict(color=METHOD_COLOURS["KL"], marker="d"),
    "IF": dict(color="#66CCEE", marker="x"),
    "KLM": dict(color=METHOD_COLOURS["KLM"], marker="v"),
    "BLT": dict(color=METHOD_COLOURS["BLT"], marker="P"),
}

SCHEME_LABELS = {**METHOD_LABELS, "IF": "Drift-implicit Lamperti"}


def load_config(filename):
    with open(config_path(filename), encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_args():
    parser = argparse.ArgumentParser(
        description="Weak-error benchmark for g1-g4 across regimes A-E."
    )
    parser.add_argument("--regimes", nargs="+", default=["A", "B", "C", "D", "E"])
    parser.add_argument("--schemes", nargs="+", default=ALL_SCHEMES,
                        choices=ALL_SCHEMES)
    parser.add_argument(
        "--n-paths", type=int, default=None,
        help="Monte Carlo paths (default: time_grids.weak_error in config).",
    )
    parser.add_argument(
        "--n-steps", nargs="+", type=int,
        default=[8, 16, 32, 64, 128, 256],
        help="Step counts; default is the thesis ladder h = 2^-3 .. 2^-8.",
    )
    parser.add_argument("--batch-size", type=int, default=25_000,
                        help="Paths per batch (bounds the stored path arrays).")
    return parser.parse_args()


def path_scheme_runner(name, params, alpha):
    """Batch runner: (dt, dW, m) -> full path array, or None if the scheme
    does not apply in this regime on the shared-noise path interface."""
    common = dict(
        X0=params["x0"], kappa=params["kappa"],
        theta=params["theta"], sigma=params["sigma"],
    )
    if name == "FTE":
        return lambda dt, dW, m: fte_paths_from_dW(dt=dt, dW=dW, **common)
    if name == "HH":
        return lambda dt, dW, m: hh_milstein_paths_from_dW(dt=dt, dW=dW, **common)
    if name == "ProjEuler":
        return lambda dt, dW, m: projected_euler_paths_from_dW(dt=dt, dW=dW, **common)
    if name == "KL" and alpha >= 0.0:
        return lambda dt, dW, m: kl_uniform_paths_from_dW(dt=dt, dW=dW, **common)
    if name == "IF" and alpha > 0.0:
        return lambda dt, dW, m: if_paths_from_dW(dt=dt, dW=dW, **common)
    if name == "BLT":
        return lambda dt, dW, m: blt_paths_from_noise(dt=dt, dW=dW, m=m, **common)
    return None


class _Welford:
    """Streaming mean and standard error over path batches."""

    def __init__(self):
        self.n = 0
        self.total = 0.0
        self.total_sq = 0.0

    def add(self, values):
        self.n += values.size
        self.total += float(np.sum(values))
        self.total_sq += float(np.sum(values.astype(float) ** 2))

    def mean(self):
        return self.total / self.n

    def standard_error(self):
        var = (self.total_sq - self.total**2 / self.n) / (self.n - 1)
        return float(np.sqrt(max(var, 0.0) / self.n))


def run_regime(regime_name, params, args, master_seed, n_paths):
    T = params["T"]
    alpha = kl_alpha(params["kappa"], params["theta"], params["sigma"])

    exact = terminal_exact_expectations(
        x0=params["x0"], kappa=params["kappa"], theta=params["theta"],
        sigma=params["sigma"], T=T,
    )
    exact["g4"] = affine_cir_bond_price(
        x0=params["x0"], kappa=params["kappa"], theta=params["theta"],
        sigma=params["sigma"], T=T,
    )

    rows = []
    for level, n_steps in enumerate(args.n_steps):
        dt = T / n_steps
        rng = make_rng(master_seed + 7919 * level)

        # ---- shared-noise path schemes -------------------------------
        runners = {}
        for name in args.schemes:
            runner = path_scheme_runner(name, params, alpha)
            if runner is not None:
                runners[name] = runner

        stats = {
            (name, payoff): _Welford()
            for name in runners
            for payoff in PAYOFFS
        }
        runtimes = {name: 0.0 for name in runners}

        for batch_start in range(0, n_paths, args.batch_size):
            batch_n = min(args.batch_size, n_paths - batch_start)
            dW, m = make_brownian_increments_with_infima(rng, batch_n, n_steps, dt)

            for name, runner in runners.items():
                start = time.perf_counter()
                paths = runner(dt, dW, m)
                runtimes[name] += time.perf_counter() - start

                terminal = paths[:, -1]
                for payoff in ["g1", "g2", "g3"]:
                    stats[(name, payoff)].add(TERMINAL_PAYOFFS[payoff](terminal))
                stats[(name, "g4")].add(g4_bond_discount_from_paths(paths, dt))

        for name in runners:
            variant = "uniform" if name == "KL" else "fixed"
            for payoff in PAYOFFS:
                rows.append(
                    _make_row(
                        regime_name, name, variant, "shared", n_steps, dt,
                        payoff, stats[(name, payoff)], exact[payoff],
                        runtimes[name], n_paths,
                    )
                )

        # ---- free-running adaptive schemes (terminal payoffs only) ---
        if "KLM" in args.schemes:
            start = time.perf_counter()
            terminal, klm_stats = klm_backstop_terminal(
                X0=params["x0"], kappa=params["kappa"], theta=params["theta"],
                sigma=params["sigma"], T=T, h_max=dt, n_paths=n_paths, rng=rng,
            )
            runtime = time.perf_counter() - start
            for payoff in ["g1", "g2", "g3"]:
                acc = _Welford()
                acc.add(TERMINAL_PAYOFFS[payoff](terminal))
                rows.append(
                    _make_row(
                        regime_name, "KLM", klm_stats["backstop_kind"], "own",
                        n_steps, dt, payoff, acc, exact[payoff], runtime,
                        n_paths,
                    )
                )

        if "KL" in args.schemes and alpha < 0.0:
            start = time.perf_counter()
            terminal = kl_adaptive_terminal(
                X0=params["x0"], kappa=params["kappa"], theta=params["theta"],
                sigma=params["sigma"], T=T, dt_max=dt, n_paths=n_paths, rng=rng,
            )
            runtime = time.perf_counter() - start
            for payoff in ["g1", "g2", "g3"]:
                acc = _Welford()
                acc.add(TERMINAL_PAYOFFS[payoff](terminal))
                rows.append(
                    _make_row(
                        regime_name, "KL", "adaptive-soft-zero", "own",
                        n_steps, dt, payoff, acc, exact[payoff], runtime,
                        n_paths,
                    )
                )

    return rows


def _make_row(regime, scheme, variant, noise, n_steps, dt, payoff, acc,
              exact_value, runtime, n_paths):
    mean = acc.mean()
    se = acc.standard_error()
    signed = mean - exact_value
    return {
        "regime": regime,
        "scheme": scheme,
        "scheme_variant": variant,
        "noise": noise,
        "payoff": payoff,
        "n_steps": n_steps,
        "dt": dt,
        "n_paths": n_paths,
        "approx_mean": mean,
        "exact_value": exact_value,
        "signed_error": signed,
        "weak_error": abs(signed),
        "mc_standard_error": se,
        "error_to_se": abs(signed) / se if se > 0 else np.nan,
        "runtime_s": runtime,
    }


def fit_orders(rows):
    """Global fit over noise-resolved points, plus consecutive local slopes.

    Following the thesis methodology chapter, the global fit is refused
    (NaN) when fewer than three points satisfy |error| >= 2 s.e.
    """
    order_rows = []
    keys = sorted({(r["regime"], r["scheme"], r["payoff"]) for r in rows})
    for regime, scheme, payoff in keys:
        pts = sorted(
            (r for r in rows
             if r["regime"] == regime and r["scheme"] == scheme
             and r["payoff"] == payoff),
            key=lambda r: r["dt"],
        )
        usable = [
            r for r in pts
            if np.isfinite(r["error_to_se"]) and r["error_to_se"] >= 2.0
            and r["weak_error"] > 0.0
        ]

        if len(usable) >= 3:
            dt = np.array([r["dt"] for r in usable])
            err = np.array([r["weak_error"] for r in usable])
            fitted = float(fit_loglog_order(dt, err))
        else:
            fitted = np.nan

        local = []
        for lo, hi in zip(pts[:-1], pts[1:]):
            both_resolved = (
                lo["error_to_se"] >= 2.0 and hi["error_to_se"] >= 2.0
                and lo["weak_error"] > 0 and hi["weak_error"] > 0
            )
            if both_resolved:
                slope = (
                    np.log(hi["weak_error"] / lo["weak_error"])
                    / np.log(hi["dt"] / lo["dt"])
                )
                local.append(f"{slope:.2f}")
            else:
                local.append("noise")

        order_rows.append(
            {
                "regime": regime,
                "scheme": scheme,
                "payoff": payoff,
                "fitted_weak_order": fitted,
                "n_points_used": len(usable),
                "n_points_total": len(pts),
                "local_slopes_fine_to_coarse": ";".join(local),
            }
        )
    return order_rows


def plot_regime(regime_name, rows, args, n_paths, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))
    axes = axes.ravel()

    regime_rows = [r for r in rows if r["regime"] == regime_name]
    schemes = sorted(
        {r["scheme"] for r in regime_rows},
        key=lambda s: ALL_SCHEMES.index(s),
    )

    for ax, payoff in zip(axes, PAYOFFS):
        payoff_rows = [r for r in regime_rows if r["payoff"] == payoff]
        if not payoff_rows:
            ax.set_title(f"{payoff}: not available")
            ax.axis("off")
            continue

        noise_floor = {}
        for name in schemes:
            pts = sorted(
                (r for r in payoff_rows if r["scheme"] == name),
                key=lambda r: r["dt"],
            )
            if not pts:
                continue
            dt = np.array([r["dt"] for r in pts])
            err = np.array([max(r["weak_error"], 1e-16) for r in pts])
            resolved = np.array([r["error_to_se"] >= 2.0 for r in pts])

            style = SCHEME_STYLES[name]
            ax.loglog(dt, err, lw=1.1, ms=5, label=SCHEME_LABELS[name], **style)
            # Noise-dominated points are hollow: their value is an upper
            # bound set by Monte Carlo error, not a measured bias.
            if np.any(~resolved):
                ax.loglog(
                    dt[~resolved], err[~resolved], linestyle="none",
                    marker=style["marker"], ms=9, mfc="none",
                    mec="0.4", mew=1.0,
                )
            for r in pts:
                noise_floor[r["dt"]] = max(
                    noise_floor.get(r["dt"], 0.0), 2.0 * r["mc_standard_error"]
                )

        if noise_floor:
            dts = np.array(sorted(noise_floor))
            ax.loglog(
                dts, [noise_floor[d] for d in dts], "--", color="0.55",
                lw=1.0, label=r"$2\times$ MC s.e.",
            )
            # Slope-one guide anchored at the coarsest resolved level.
            err_max = max(r["weak_error"] for r in payoff_rows)
            ax.loglog(dts, err_max * (dts / dts[-1]), "k:", lw=0.8,
                      label="slope 1")

        ax.set_xlabel("step size h")
        ax.set_ylabel(f"|weak error|, {payoff}")
        ax.set_title(f"{payoff}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=6.5)

    fig.suptitle(
        f"Weak error vs exact functionals, regime {regime_name} "
        f"({n_paths} paths; hollow markers: below the MC noise floor; "
        "g4 unavailable for adaptive schemes)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path)
    fig.savefig(Path(out_path).with_suffix(".png"), dpi=150)
    plt.close(fig)


def save_csv(rows, filename):
    path = results_path(filename)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def main():
    args = get_args()

    regimes_cfg = load_config("regimes.yaml")
    experiments_cfg = load_config("experiments.yaml")
    shared = regimes_cfg["shared"]
    master_seed = experiments_cfg["shared"]["master_seed"]
    T = experiments_cfg["shared"]["T"]
    n_paths = (
        args.n_paths
        if args.n_paths is not None
        else experiments_cfg["time_grids"]["weak_error"]["n_paths"]
    )

    all_rows = []
    for regime_name in args.regimes:
        sigma = regimes_cfg["regimes"][regime_name]["sigma"]
        params = {
            "kappa": shared["kappa"], "theta": shared["theta"],
            "x0": shared["x0"], "sigma": sigma, "T": T,
        }
        print(f"Regime {regime_name} (sigma={sigma:g}) ...")
        rows = run_regime(regime_name, params, args, master_seed, n_paths)
        all_rows.extend(rows)

    csv_path = save_csv(all_rows, "weak_error.csv")
    print(f"wrote {csv_path}")

    order_rows = fit_orders(all_rows)
    orders_path = save_csv(order_rows, "weak_error_orders.csv")
    print(f"wrote {orders_path}")

    for r in order_rows:
        fitted = (
            f"{r['fitted_weak_order']:.3f}"
            if np.isfinite(r["fitted_weak_order"])
            else "refused (noise)"
        )
        print(
            f"  {r['regime']} {r['scheme']:<10} {r['payoff']}: "
            f"order {fitted} "
            f"({r['n_points_used']}/{r['n_points_total']} resolved; "
            f"local {r['local_slopes_fine_to_coarse']})"
        )

    for regime_name in args.regimes:
        fig_path = figure_path(f"weak_error_regime_{regime_name}.pdf")
        plot_regime(regime_name, all_rows, args, n_paths, fig_path)
        print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
