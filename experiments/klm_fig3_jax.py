import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from klm_jax.fig3 import fitted_orders_by_group, run_fig3_experiment


def get_args():
    parser = argparse.ArgumentParser(description="Run the KLM JAX Fig. 3 experiment.")

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a small fast version for testing.",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments.yaml"),
        help="Path to the experiment configuration file.",
    )

    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("outputs/klm_fig3_jax"),
        help="Folder where results will be saved.",
    )

    return parser.parse_args()


def load_config(path, quick):
    with path.open("r", encoding="utf-8") as file:
        all_configs = yaml.safe_load(file)

    run_name = "quick" if quick else "full"
    return all_configs["klm_fig3_jax"][run_name]


def save_csv(rows, path):
    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def plot_orders(orders, out_path):
    fig, ax = plt.subplots(figsize=(6.0, 4.5))

    kappas = sorted({o["kappa"] for o in orders})
    for kappa in kappas:
        group = [o for o in orders if o["kappa"] == kappa and o["a"] > 0.0]
        a = np.array([o["a"] for o in group])
        order = np.array([o["fitted_order"] for o in group])
        ax.plot(a, order, "o-", label=rf"$\kappa$={kappa:g}")

    ax.axhline(1.0, color="k", ls=":", lw=0.8)
    ax.axhline(0.5, color="k", ls="--", lw=0.8)
    ax.set_xlabel(r"Feller ratio $a = \sigma^2 / (2\kappa\lambda)$")
    ax.set_ylabel(r"fitted strong $L^2$ order")
    ax.set_title("KLM backstopped adaptive scheme: observed order across $a$")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    args = get_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config, args.quick)
    results = run_fig3_experiment(config, outdir=args.outdir)
    orders = fitted_orders_by_group(results)

    with (args.outdir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    save_csv(results, args.outdir / "results.csv")
    save_csv(orders, args.outdir / "fitted_orders.csv")
    plot_orders(orders, args.outdir / "fig3_orders.pdf")

    for o in orders:
        print(
            f"kappa={o['kappa']:g}  a={o['a']:.3f}  "
            f"fitted order={o['fitted_order']:.3f}"
        )

    print("Finished KLM Fig. 3 run.")
    print(f"Results saved in: {args.outdir}")


if __name__ == "__main__":
    main()