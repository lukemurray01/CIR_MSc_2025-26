# Work-precision diagram: strong L1 error vs estimated wall time, per scheme,
# on the measured CPU (NumPy) and GPU (JAX/P100) throughputs.
#
# Raw speed-up (results/cpu_gpu_timing.csv) answers "how much faster is one
# step"; this figure answers the practitioner's question "which scheme is
# cheapest at a given accuracy".  Cost model:
#
#   wall(scheme, level) = n_paths * mean_steps_per_path / throughput
#
# with throughput = paths_steps_per_s from the canonical Kaggle P100 timing
# run at the largest measured path count (saturated regime), and
# mean_steps_per_path from the strong-error benchmark CSVs (for KLM this is
# the ACTUAL adaptive step count per level, so adaptivity is priced in).
#
# Honesty notes, also stated in the caption:
#   - Costs exclude reference generation; both devices use the same model.
#   - The timing run's throughput is normalised per fine-grid step
#     (n_steps=512).  KLM took ~2 fine cells per adaptive step in that run
#     (h = h_max*min(1,|Y|) with |Y| ~ sqrt(0.02)), so its per-adaptive-step
#     throughput is ~0.5x the recorded number (KLM_FINE_PER_STEP below); an
#     estimate until step counters are recorded in the timing harness.
#   - In regime E the KL row is the adaptive soft-zero variant but is priced
#     at the uniform-KL throughput (the variants share the update's cost).
#   - BLT errors are l1_vs_hh_ref from results/blt_strong_error.csv for
#     comparability with the HH-referenced benchmark CSVs.
#
# Usage:
#   uv run python experiments/fig_work_precision.py
#   uv run python experiments/fig_work_precision.py --regimes B D E
#
# Outputs:
#   results/work_precision.csv
#   figures/work_precision.pdf (+ .png preview)

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.io import figure_path, results_path

SCHEME_COLOURS = {
    "FTE": "#4477AA",
    "ProjEuler": "#AA3377",
    "KL": "#EE6677",
    "KLM": "#7B1FA2",
    "BLT": "#CC6677",
}

# Fine-grid cells consumed per KLM adaptive step in the timing configuration
# (see honesty notes above).
KLM_FINE_PER_STEP = 2.0


def get_args():
    parser = argparse.ArgumentParser(description="Work-precision diagram.")
    parser.add_argument("--regimes", nargs="+", default=["B", "E"])
    parser.add_argument("--n-paths", type=int, default=20_000,
                        help="Monte Carlo paths per strong-error level.")
    return parser.parse_args()


def load_throughputs():
    """paths_steps_per_s per (scheme, impl) at the largest measured rung."""
    timing = pd.read_csv(results_path("cpu_gpu_timing.csv"))
    bad_jax = timing[(timing["impl"] == "jax") & (timing["backend"] != "gpu")]
    if not bad_jax.empty:
        raise SystemExit("timing CSV contains non-GPU jax rows; use the "
                         "canonical Kaggle P100 dataset")

    top = timing.loc[timing.groupby(["scheme", "impl"])["n_paths"].idxmax()]
    rate = top.set_index(["scheme", "impl"])["paths_steps_per_s"]
    return rate / pd.Series(
        {(s, i): KLM_FINE_PER_STEP if s == "KLM" else 1.0 for s, i in rate.index}
    )


def load_errors(regimes):
    """Rows of (regime, scheme, dt, l1, steps_per_path) from the benchmark CSVs."""
    frames = []
    for regime in regimes:
        d = pd.read_csv(results_path(f"strong_error_regime_{regime}.csv"))
        frames.append(d.assign(steps_per_path=d["mean_steps_per_path"])[
            ["regime", "scheme", "dt", "l1", "steps_per_path"]
        ])

    blt = pd.read_csv(results_path("blt_strong_error.csv"))
    blt = blt[blt["regime"].isin(regimes)]
    frames.append(pd.DataFrame({
        "regime": blt["regime"], "scheme": blt["scheme"], "dt": blt["dt"],
        "l1": blt["l1_vs_hh_ref"], "steps_per_path": 1.0 / blt["dt"],
    }))

    return pd.concat(frames, ignore_index=True)


def main():
    args = get_args()
    rate = load_throughputs()
    errors = load_errors(args.regimes)

    rows = []
    for _, r in errors.iterrows():
        for impl, backend in (("numpy", "cpu"), ("jax", "gpu")):
            key = (r["scheme"], impl)
            if key not in rate.index:
                continue
            wall = args.n_paths * r["steps_per_path"] / rate[key]
            rows.append({
                "regime": r["regime"], "scheme": r["scheme"], "dt": r["dt"],
                "l1": r["l1"], "steps_per_path": r["steps_per_path"],
                "backend": backend, "est_wall_s": wall,
            })
    table = pd.DataFrame(rows)

    csv_path = results_path("work_precision.csv")
    table.to_csv(csv_path, index=False)
    print(f"wrote {csv_path}")

    fig, axes = plt.subplots(
        1, len(args.regimes), figsize=(5.2 * len(args.regimes), 4.6),
        squeeze=False, sharey=False,
    )
    for ax, regime in zip(axes[0], args.regimes):
        sub = table[table["regime"] == regime]
        for scheme, colour in SCHEME_COLOURS.items():
            for backend, style in (("cpu", "--"), ("gpu", "-")):
                line = sub[(sub["scheme"] == scheme)
                           & (sub["backend"] == backend)].sort_values("est_wall_s")
                if line.empty:
                    continue
                ax.loglog(
                    line["est_wall_s"], line["l1"], style,
                    marker="o" if backend == "gpu" else "s",
                    fillstyle="full" if backend == "gpu" else "none",
                    color=colour, markersize=4,
                    label=f"{scheme} ({backend.upper()})",
                )
        ax.set_xlabel("estimated wall time (s), 20k paths")
        ax.set_ylabel(r"strong $L^1$ error at $T$")
        ax.set_title(f"regime {regime}")
        ax.grid(True, which="both", alpha=0.3)

    axes[0][0].legend(fontsize=7, ncol=2)
    fig.suptitle(
        "Work-precision: error vs estimated wall time -- measured P100/CPU "
        "throughputs, costs exclude reference generation",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    fig_path = figure_path("work_precision.pdf")
    fig.savefig(fig_path)
    fig.savefig(Path(fig_path).with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
