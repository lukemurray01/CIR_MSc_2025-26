# Coupled strong-error benchmark: thesis benchmark schemes against the
# Hefter--Herzwurm (HH) fine-grid reference on a shared Brownian path.
#
# Fixed-step schemes (FTE, ProjEuler, KL) consume aggregated fine increments.
# The KLM backstopped adaptive scheme consumes partial sums of the SAME fine
# increments with its step sizes quantised to the fine grid, so all reported
# errors are Brownian-coupled discretisation diagnostics (thesis background
# chapter, Brownian coupling section).
#
# KL uses the uniform splitting update in alpha >= 0 regimes.  In regimes D/E,
# where the uniform square-root update is not defined for all inputs, the
# Brownian-coupled adaptive Kelly--Lord soft-zero variant is used and recorded
# as scheme_variant=adaptive-soft-zero.
#
# Usage:
#   uv run python experiments/run_strong_error.py                # all regimes
#   uv run python experiments/run_strong_error.py --regimes B D  # subset
#   uv run python experiments/run_strong_error.py --schemes FTE KLM
#
# Outputs (per regime):
#   results/strong_error_regime_{R}.csv
#   figures/strong_error_regime_{R}.pdf
#   figures/strong_error_vs_cost_regime_{R}.pdf

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import argparse
import csv
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.metrics.strong_error import fit_loglog_order
from src.samplers.full_truncation_euler import fte_terminal_from_dW
from src.samplers.hh_milstein import hh_milstein_terminal_from_dW
from src.samplers.kelly_lord import kl_uniform_terminal_from_dW
from src.samplers.kelly_lord_adaptive import kl_adaptive_terminal_from_fine_dW
from src.samplers.klm_backstop import klm_backstop_terminal_from_fine_dW
from src.samplers.projected_euler import (
    projected_euler_terminal_from_dW,
    projected_euler_terminal_with_stats_from_dW,
)
from src.utils.brownian import aggregate_brownian_increments
from src.utils.cir_params import kl_alpha
from src.utils.io import config_path, figure_path, results_path
from src.utils.rng import make_rng

FIXED_STEP_SCHEMES = {
    "FTE": fte_terminal_from_dW,
    "ProjEuler": projected_euler_terminal_from_dW,
    "KL": kl_uniform_terminal_from_dW,
}

ALL_SCHEMES = ["FTE", "ProjEuler", "KL", "KLM"]

SCHEME_STYLES = {
    "FTE": dict(color="tab:blue", marker="o"),
    "ProjEuler": dict(color="tab:green", marker="^"),
    "KL": dict(color="tab:red", marker="d"),
    "KLM": dict(color="tab:purple", marker="v"),
}


def load_config(filename):
    with open(config_path(filename), encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_args():
    parser = argparse.ArgumentParser(
        description="Coupled strong-error benchmark for the thesis schemes."
    )
    parser.add_argument(
        "--regimes",
        nargs="+",
        default=["A", "B", "C", "D", "E"],
        help="CIR regimes to run (default: all five).",
    )
    parser.add_argument(
        "--schemes",
        nargs="+",
        default=ALL_SCHEMES,
        choices=ALL_SCHEMES,
        help="Schemes to include (default: all benchmark schemes).",
    )
    parser.add_argument(
        "--n-paths",
        type=int,
        default=None,
        help="Override the configured number of paths (smoke runs).",
    )
    parser.add_argument(
        "--reference-n-steps",
        type=int,
        default=None,
        help="Override the configured HH fine-reference step count.",
    )
    parser.add_argument(
        "--path-batch-size",
        type=int,
        default=None,
        help=(
            "Number of Monte Carlo paths to process per batch. By default this "
            "is chosen from --max-brownian-gib."
        ),
    )
    parser.add_argument(
        "--max-brownian-gib",
        type=float,
        default=0.5,
        help=(
            "Approximate memory budget for the fine Brownian increment array "
            "inside one path batch (default: 0.5 GiB)."
        ),
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=64.0,
        help="KLM backstop floor parameter h_min = h_max / rho.",
    )
    parser.add_argument(
        "--drop-coarsest",
        type=int,
        default=1,
        help="Number of largest-h points to exclude from fitted order labels.",
    )
    parser.add_argument(
        "--cost-plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write companion L2-error versus mean-steps plots.",
    )
    parser.add_argument(
        "--proj-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record ProjEuler floor/projection diagnostics in the CSV.",
    )
    return parser.parse_args()


def resolve_path_batch_size(
    n_paths,
    reference_n_steps,
    path_batch_size=None,
    max_brownian_gib=0.5,
):
    if n_paths <= 0:
        raise ValueError("n_paths must be positive")
    if reference_n_steps <= 0:
        raise ValueError("reference_n_steps must be positive")

    if path_batch_size is not None:
        if path_batch_size <= 0:
            raise ValueError("path_batch_size must be positive")
        return min(n_paths, path_batch_size)

    if max_brownian_gib <= 0.0 or not math.isfinite(max_brownian_gib):
        raise ValueError("max_brownian_gib must be a positive finite number")

    bytes_per_path = reference_n_steps * np.dtype(float).itemsize
    budget_bytes = max_brownian_gib * (1024**3)
    return max(1, min(n_paths, int(budget_bytes // bytes_per_path)))


def _validate_grid(reference_n_steps, coarse_list):
    for n_steps in coarse_list:
        if n_steps <= 0:
            raise ValueError("coarse_n_steps entries must be positive")
        if reference_n_steps % n_steps != 0:
            raise ValueError(
                f"reference_n_steps={reference_n_steps} is not divisible "
                f"by coarse n_steps={n_steps}"
            )


def _empty_accumulator():
    return {
        "sum_abs": 0.0,
        "sum_sq": 0.0,
        "runtime_s": 0.0,
        "steps_total": 0.0,
        "backstop_events": 0.0,
        "backstop_steps_total": 0.0,
        "proj_pre_count": 0.0,
        "proj_post_count": 0.0,
        "proj_total_updates": 0.0,
        "proj_y_floor": np.nan,
        "scheme_variant": "",
    }


def run_regime(
    regime_name,
    params,
    grid,
    master_seed,
    schemes,
    rho,
    proj_diagnostics=True,
    path_batch_size=None,
    max_brownian_gib=0.5,
):
    n_paths = grid["n_paths"]
    reference_n_steps = grid["reference_n_steps"]
    coarse_list = grid["coarse_n_steps"]
    T = params["T"]
    dt_fine = T / reference_n_steps
    _validate_grid(reference_n_steps, coarse_list)

    alpha = kl_alpha(params["kappa"], params["theta"], params["sigma"])
    resolved_batch_size = resolve_path_batch_size(
        n_paths,
        reference_n_steps,
        path_batch_size=path_batch_size,
        max_brownian_gib=max_brownian_gib,
    )

    # One fine Brownian path per batch, shared by the reference and every scheme.
    rng = make_rng(master_seed)
    reference_name = "HH"
    accumulators = {
        (name, n_steps): _empty_accumulator()
        for name in schemes
        for n_steps in coarse_list
    }

    for batch_start in range(0, n_paths, resolved_batch_size):
        batch_n = min(resolved_batch_size, n_paths - batch_start)
        dW_fine = np.sqrt(dt_fine) * rng.standard_normal(
            (batch_n, reference_n_steps)
        )
        reference_terminal = hh_milstein_terminal_from_dW(
            X0=params["x0"],
            kappa=params["kappa"],
            theta=params["theta"],
            sigma=params["sigma"],
            dt=dt_fine,
            dW=dW_fine,
        )

        if not np.all(np.isfinite(reference_terminal)):
            raise RuntimeError(
                f"non-finite reference terminal values in regime {regime_name}"
            )

        for name in schemes:
            for n_steps in coarse_list:
                acc = accumulators[(name, n_steps)]
                factor = reference_n_steps // n_steps
                dt_coarse = dt_fine * factor

                start = time.perf_counter()
                if name == "KLM":
                    terminal, stats = klm_backstop_terminal_from_fine_dW(
                        X0=params["x0"],
                        kappa=params["kappa"],
                        theta=params["theta"],
                        sigma=params["sigma"],
                        T=T,
                        h_max=dt_coarse,
                        dW_fine=dW_fine,
                        rho=rho,
                    )
                    acc["backstop_events"] += (
                        stats["n_backstop_min"] + stats["n_backstop_neg"]
                    )
                    acc["backstop_steps_total"] += stats["n_steps_total"]
                    acc["steps_total"] += stats["n_steps_total"]
                    scheme_variant = stats["backstop_kind"]
                elif name == "KL" and alpha < 0.0:
                    terminal, stats = kl_adaptive_terminal_from_fine_dW(
                        X0=params["x0"],
                        kappa=params["kappa"],
                        theta=params["theta"],
                        sigma=params["sigma"],
                        T=T,
                        dt_max=dt_coarse,
                        dW_fine=dW_fine,
                    )
                    acc["backstop_events"] += stats["n_soft_zero"]
                    acc["backstop_steps_total"] += stats["n_steps_total"]
                    acc["steps_total"] += stats["n_steps_total"]
                    scheme_variant = "adaptive-soft-zero"
                elif name == "ProjEuler":
                    dW_coarse = aggregate_brownian_increments(dW_fine, factor)
                    if proj_diagnostics:
                        terminal, stats = projected_euler_terminal_with_stats_from_dW(
                            X0=params["x0"],
                            kappa=params["kappa"],
                            theta=params["theta"],
                            sigma=params["sigma"],
                            dt=dt_coarse,
                            dW=dW_coarse,
                        )
                        total_updates = batch_n * n_steps
                        acc["proj_y_floor"] = stats["y_floor"]
                        acc["proj_pre_count"] += (
                            stats["pre_floor_fraction"] * total_updates
                        )
                        acc["proj_post_count"] += (
                            stats["post_projection_fraction"] * total_updates
                        )
                        acc["proj_total_updates"] += total_updates
                    else:
                        terminal = projected_euler_terminal_from_dW(
                            X0=params["x0"],
                            kappa=params["kappa"],
                            theta=params["theta"],
                            sigma=params["sigma"],
                            dt=dt_coarse,
                            dW=dW_coarse,
                        )
                    acc["steps_total"] += float(n_steps * batch_n)
                    scheme_variant = "fixed"
                else:
                    dW_coarse = aggregate_brownian_increments(dW_fine, factor)
                    terminal = FIXED_STEP_SCHEMES[name](
                        X0=params["x0"],
                        kappa=params["kappa"],
                        theta=params["theta"],
                        sigma=params["sigma"],
                        dt=dt_coarse,
                        dW=dW_coarse,
                    )
                    acc["steps_total"] += float(n_steps * batch_n)
                    scheme_variant = "uniform" if name == "KL" else "fixed"
                acc["runtime_s"] += time.perf_counter() - start
                acc["scheme_variant"] = scheme_variant

                diff = terminal - reference_terminal
                acc["sum_abs"] += float(np.sum(np.abs(diff)))
                acc["sum_sq"] += float(np.sum(diff**2))

    rows = []
    for name in schemes:
        for n_steps in coarse_list:
            acc = accumulators[(name, n_steps)]
            factor = reference_n_steps // n_steps
            dt_coarse = dt_fine * factor
            row_extras = {}

            if acc["proj_total_updates"]:
                row_extras = {
                    "proj_y_floor": acc["proj_y_floor"],
                    "proj_pre_floor_fraction": acc["proj_pre_count"]
                    / acc["proj_total_updates"],
                    "proj_post_projection_fraction": acc["proj_post_count"]
                    / acc["proj_total_updates"],
                }

            backstop_fraction = np.nan
            if acc["backstop_steps_total"]:
                backstop_fraction = (
                    acc["backstop_events"] / acc["backstop_steps_total"]
                )

            row = {
                "regime": regime_name,
                "scheme": name,
                "scheme_variant": acc["scheme_variant"],
                "reference": reference_name,
                "dt": dt_coarse,
                "l1": acc["sum_abs"] / n_paths,
                "l2": float(np.sqrt(acc["sum_sq"] / n_paths)),
                "runtime_s": acc["runtime_s"],
                "mean_steps_per_path": acc["steps_total"] / n_paths,
                "backstop_fraction": backstop_fraction,
            }
            row.update(row_extras)
            rows.append(row)

    return rows


def fitted_orders(rows, schemes, drop_coarsest=1):
    if drop_coarsest < 0:
        raise ValueError("drop_coarsest must be nonnegative")

    orders = {}
    for name in schemes:
        scheme_rows = [r for r in rows if r["scheme"] == name]
        if len(scheme_rows) < 2 + drop_coarsest:
            continue

        # Sort from finest to coarsest, then remove the largest h values.
        scheme_rows = sorted(scheme_rows, key=lambda r: r["dt"])
        fit_rows = scheme_rows[:-drop_coarsest] if drop_coarsest else scheme_rows

        dt = np.array([r["dt"] for r in fit_rows])
        l2 = np.array([r["l2"] for r in fit_rows])
        orders[name] = fit_loglog_order(dt, l2)
    return orders


def display_label(name, scheme_rows, orders):
    variant = scheme_rows[0].get("scheme_variant", "") if scheme_rows else ""

    if name == "KL":
        if variant == "adaptive-soft-zero":
            base = "KL adaptive soft-zero"
        elif variant == "uniform":
            base = "KL uniform"
        else:
            base = "KL"
    elif name == "KLM":
        if variant in {"implicit", "projected"}:
            base = f"KLM {variant} backstop"
        elif variant:
            base = f"KLM {variant}"
        else:
            base = "KLM"
    else:
        base = name

    if name in orders:
        return f"{base} (tail order {orders[name]:.2f})"
    return base


def plot_regime(regime_name, rows, params, grid, orders, out_path):
    fig, ax = plt.subplots(figsize=(6.0, 4.5))

    alpha = kl_alpha(params["kappa"], params["theta"], params["sigma"])

    for name in ALL_SCHEMES:
        scheme_rows = sorted(
            [r for r in rows if r["scheme"] == name],
            key=lambda r: r["dt"],
        )
        if not scheme_rows:
            continue
        dt = np.array([r["dt"] for r in scheme_rows])
        l2 = np.array([r["l2"] for r in scheme_rows])
        label = display_label(name, scheme_rows, orders)
        # In alpha < 0 regimes the reference-sensitivity gate shows these
        # slopes are not reference-converged; flag them on the figure itself.
        if alpha < 0.0 and name in ("ProjEuler", "KLM"):
            label += " †"
        ax.loglog(dt, l2, label=label, **SCHEME_STYLES[name])

    # Slope guides anchored at the coarsest step.
    dts = sorted({r["dt"] for r in rows})
    if dts:
        h = np.array(dts)
        anchor = max(r["l2"] for r in rows)
        ax.loglog(h, anchor * (h / h[-1]) ** 0.5, "k--", lw=0.8, label="slope 1/2")
        ax.loglog(h, anchor * (h / h[-1]) ** 1.0, "k:", lw=0.8, label="slope 1")

    ax.set_xlabel(r"nominal step parameter $h$ ($h_{\max}$ for adaptive schemes)")
    ax.set_ylabel(r"strong $L^2$ error at $T$")
    reference_name = rows[0]["reference"] if rows else "HH"
    ax.set_title(
        f"Regime {regime_name}: "
        rf"$\sigma={params['sigma']:g}$, "
        f"{grid['n_paths']} paths\n"
        f"{reference_name} fine reference at {grid['reference_n_steps']} steps"
    )
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    if alpha < 0.0:
        ax.text(
            0.02,
            0.02,
            "† fitted slope not reference-converged\n"
            "(see reference-sensitivity gate)",
            transform=ax.transAxes,
            fontsize=7,
            color="0.35",
            va="bottom",
        )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_regime_cost(regime_name, rows, params, grid, out_path):
    fig, ax = plt.subplots(figsize=(6.0, 4.5))

    for name in ALL_SCHEMES:
        scheme_rows = sorted(
            [r for r in rows if r["scheme"] == name],
            key=lambda r: r["mean_steps_per_path"],
        )
        if not scheme_rows:
            continue
        mean_steps = np.array([r["mean_steps_per_path"] for r in scheme_rows])
        l2 = np.array([r["l2"] for r in scheme_rows])
        label = display_label(name, scheme_rows, {})
        ax.loglog(mean_steps, l2, label=label, **SCHEME_STYLES[name])

    ax.set_xlabel("mean realised steps per path")
    ax.set_ylabel(r"strong $L^2$ error at $T$")
    reference_name = rows[0]["reference"] if rows else "HH"
    ax.set_title(
        f"Regime {regime_name}: cost-normalised view, "
        rf"$\sigma={params['sigma']:g}$, "
        f"{grid['n_paths']} paths\n"
        f"{reference_name} fine reference"
    )
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_csv(rows, filename):
    path = results_path(filename)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


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
    master_seed = experiments_cfg["shared"]["master_seed"]
    T = experiments_cfg["shared"]["T"]
    path_batch_size = resolve_path_batch_size(
        grid["n_paths"],
        grid["reference_n_steps"],
        path_batch_size=args.path_batch_size,
        max_brownian_gib=args.max_brownian_gib,
    )
    print(
        "Strong-error grid: "
        f"{grid['n_paths']} paths, "
        f"{grid['reference_n_steps']} HH reference steps, "
        f"path_batch_size={path_batch_size}"
    )

    for regime_name in args.regimes:
        sigma = regimes_cfg["regimes"][regime_name]["sigma"]
        params = {
            "kappa": shared["kappa"],
            "theta": shared["theta"],
            "x0": shared["x0"],
            "sigma": sigma,
            "T": T,
        }

        print(f"Regime {regime_name} (sigma={sigma:g}) ...")
        rows = run_regime(
            regime_name,
            params,
            grid,
            master_seed,
            args.schemes,
            args.rho,
            proj_diagnostics=args.proj_diagnostics,
            path_batch_size=path_batch_size,
            max_brownian_gib=args.max_brownian_gib,
        )
        orders = fitted_orders(
            rows,
            args.schemes,
            drop_coarsest=args.drop_coarsest,
        )

        for name, order in orders.items():
            print(f"  {name:<9} tail-fitted L2 order: {order:.3f}")

        csv_path = save_csv(rows, f"strong_error_regime_{regime_name}.csv")
        fig_path = figure_path(f"strong_error_regime_{regime_name}.pdf")
        plot_regime(regime_name, rows, params, grid, orders, fig_path)
        print(f"  wrote {csv_path}")
        print(f"  wrote {fig_path}")

        if args.cost_plot:
            cost_fig_path = figure_path(
                f"strong_error_vs_cost_regime_{regime_name}.pdf"
            )
            plot_regime_cost(regime_name, rows, params, grid, cost_fig_path)
            print(f"  wrote {cost_fig_path}")


if __name__ == "__main__":
    main()
