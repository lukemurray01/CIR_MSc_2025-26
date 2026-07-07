"""Reference-resolution sensitivity checks for thesis production figures.

This script is deliberately smaller than the full production runs.  It keeps
the seed, plotted coarse levels, and schemes fixed while varying only the fine
reference resolution.  The output CSVs are intended as a gate before quoting
production fitted slopes in the thesis.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_strong_error import fitted_orders, run_regime
from klm_jax.fig3 import fitted_orders_by_group, run_fig3_experiment
from src.utils.io import config_path


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run strong-error and Fig. 3 reference-sensitivity checks."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=config_path("experiments.yaml"),
        help="Experiment configuration file.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("outputs/reference_sensitivity"),
        help="Directory for sensitivity CSV outputs.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use tiny grids for local smoke testing.",
    )
    parser.add_argument(
        "--strong-only",
        action="store_true",
        help="Run only the strong-error reference sensitivity.",
    )
    parser.add_argument(
        "--fig3-only",
        action="store_true",
        help="Run only the compact Fig. 3 reference sensitivity.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def quick_strong_config(config: dict) -> dict:
    quick = dict(config)
    quick["n_paths"] = 64
    quick["regimes"] = ["D", "E"]
    quick["schemes"] = ["KL", "KLM"]
    quick["coarse_n_steps"] = [8, 16, 32]
    quick["reference_n_steps"] = [512, 1024]
    return quick


def quick_fig3_config(config: dict) -> dict:
    quick = dict(config)
    quick["n_paths"] = 16
    quick["reference_powers"] = [10, 12]
    quick["n_paper_a_values"] = 2
    quick["kappas"] = [2.0]
    quick["levels"] = [4, 5]
    return quick


def run_strong_reference_sensitivity(
    all_config: dict,
    sensitivity_config: dict,
) -> tuple[list[dict], list[dict]]:
    regimes_cfg = load_yaml(config_path("regimes.yaml"))
    shared = regimes_cfg["shared"]
    master_seed = all_config["shared"]["master_seed"]
    T = all_config["shared"]["T"]

    result_rows: list[dict] = []
    order_rows: list[dict] = []

    for reference_n_steps in sensitivity_config["reference_n_steps"]:
        grid = {
            "n_paths": sensitivity_config["n_paths"],
            "coarse_n_steps": sensitivity_config["coarse_n_steps"],
            "reference_n_steps": reference_n_steps,
        }
        for regime_name in sensitivity_config["regimes"]:
            sigma = regimes_cfg["regimes"][regime_name]["sigma"]
            params = {
                "kappa": shared["kappa"],
                "theta": shared["theta"],
                "x0": shared["x0"],
                "sigma": sigma,
                "T": T,
            }
            rows = run_regime(
                regime_name=regime_name,
                params=params,
                grid=grid,
                master_seed=master_seed,
                schemes=sensitivity_config["schemes"],
                rho=sensitivity_config["rho"],
            )
            for row in rows:
                row["sensitivity_kind"] = "strong_error"
                row["reference_n_steps"] = reference_n_steps
            result_rows.extend(rows)

            orders = fitted_orders(
                rows,
                sensitivity_config["schemes"],
                drop_coarsest=sensitivity_config.get("drop_coarsest", 1),
            )
            for scheme, order in orders.items():
                order_rows.append(
                    {
                        "sensitivity_kind": "strong_error",
                        "regime": regime_name,
                        "scheme": scheme,
                        "reference_n_steps": reference_n_steps,
                        "fitted_l2_order": order,
                    }
                )

    return result_rows, order_rows


def fig3_config_for_power(sensitivity_config: dict, reference_power: int) -> dict:
    config = dict(sensitivity_config)
    config["reference_power"] = reference_power
    config.pop("reference_powers", None)
    config.pop("description", None)
    return config


def run_fig3_reference_sensitivity(
    sensitivity_config: dict,
    outdir: Path,
) -> tuple[list[dict], list[dict]]:
    result_rows: list[dict] = []
    order_rows: list[dict] = []

    for reference_power in sensitivity_config["reference_powers"]:
        config = fig3_config_for_power(sensitivity_config, int(reference_power))
        rows = run_fig3_experiment(config, outdir=outdir)
        for row in rows:
            row["sensitivity_kind"] = "klm_fig3_jax"
        result_rows.extend(rows)

        orders = fitted_orders_by_group(rows)
        for row in orders:
            row["sensitivity_kind"] = "klm_fig3_jax"
            row["reference_power"] = reference_power
        order_rows.extend(orders)

    return result_rows, order_rows


def main() -> None:
    args = get_args()
    if args.strong_only and args.fig3_only:
        raise ValueError("Choose at most one of --strong-only and --fig3-only.")

    all_config = load_yaml(args.config)
    sensitivity = all_config["reference_sensitivity"]
    args.outdir.mkdir(parents=True, exist_ok=True)

    if not args.fig3_only:
        strong_config = sensitivity["strong_error"]
        if args.quick:
            strong_config = quick_strong_config(strong_config)
        strong_rows, strong_orders = run_strong_reference_sensitivity(
            all_config, strong_config
        )
        write_csv(strong_rows, args.outdir / "strong_reference_sensitivity.csv")
        write_csv(
            strong_orders,
            args.outdir / "strong_reference_sensitivity_orders.csv",
        )
        print(
            "Strong-error sensitivity rows:",
            len(strong_rows),
            "orders:",
            len(strong_orders),
        )

    if not args.strong_only:
        fig3_config = sensitivity["klm_fig3_jax"]
        if args.quick:
            fig3_config = quick_fig3_config(fig3_config)
        fig3_rows, fig3_orders = run_fig3_reference_sensitivity(
            fig3_config,
            args.outdir,
        )
        write_csv(fig3_rows, args.outdir / "fig3_reference_sensitivity.csv")
        write_csv(
            fig3_orders,
            args.outdir / "fig3_reference_sensitivity_orders.csv",
        )
        print(
            "Fig. 3 sensitivity rows:",
            len(fig3_rows),
            "orders:",
            len(fig3_orders),
        )

    print(f"Reference-sensitivity outputs saved in: {args.outdir}")


if __name__ == "__main__":
    main()
