# Cross-regime summary: fitted strong L2 order per scheme against the
# Bessel dimension delta, with the Hefter--Jentzen ceiling overlaid.
#
# This is the figure that ties the benchmark to the theory chapter: in the
# boundary-accessible band 0 < delta < 2 no uniform-mesh Brownian scheme can
# beat order delta/2 (Thm HJ), while the adaptive KLM scheme sits outside
# the theorem's information class.  Reads the CSVs produced by
# experiments/run_strong_error.py, so run that first.
#
# Usage:
#   uv run python experiments/run_strong_error.py     # produces the CSVs
#   uv run python experiments/fig_order_summary.py
#
# Outputs:
#   figures/fig_order_vs_delta_summary.pdf (+ .png preview)
#   results/fig_order_vs_delta_summary.csv

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import csv

import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.metrics.strong_error import fit_loglog_order
from src.utils.cir_params import cir_delta
from src.utils.io import config_path, figure_path, results_path

REGIMES = ["E", "D", "C", "B", "A"]  # ascending delta

SCHEME_STYLES = {
    "FTE": dict(color="#4477AA", marker="o", label="Full truncation Euler"),
    "ProjEuler": dict(color="#AA3377", marker="^", label="Projected Euler"),
    "KL": dict(color="#EE6677", marker="d", label="Kelly–Lord splitting"),
    "KLM": dict(color="#7B1FA2", marker="v", label="KLM backstopped adaptive"),
}


def load_config(filename):
    with open(config_path(filename), encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_regime_csv(regime_name):
    path = results_path(f"strong_error_regime_{regime_name}.csv")
    if not Path(path).exists():
        raise SystemExit(
            f"missing {path}; run experiments/run_strong_error.py first"
        )
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_reference_sensitivity_ranges():
    """Fitted-order min/max per (regime, scheme) across HH reference grids.

    Reads the reference-sensitivity gate output if present; returns {} when the
    gate has not been run.  These ranges are the evidence bound on how much a
    quoted slope depends on the finite reference (essential in regimes D/E,
    where the gate shows the ProjEuler and KLM slopes are not converged).
    """
    path = Path("outputs/reference_sensitivity/strong_reference_sensitivity_orders.csv")
    if not path.exists():
        return {}

    ranges = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("sensitivity_kind") != "strong_error":
                continue
            key = (row["regime"], row["scheme"])
            order = float(row["fitted_l2_order"])
            lo, hi = ranges.get(key, (order, order))
            ranges[key] = (min(lo, order), max(hi, order))
    return ranges


def main():
    regimes_cfg = load_config("regimes.yaml")
    shared = regimes_cfg["shared"]

    sensitivity_ranges = read_reference_sensitivity_ranges()

    summary = []
    for regime_name in REGIMES:
        sigma = regimes_cfg["regimes"][regime_name]["sigma"]
        delta = cir_delta(shared["kappa"], shared["theta"], sigma)

        rows = read_regime_csv(regime_name)
        schemes = sorted({r["scheme"] for r in rows})
        for scheme in schemes:
            scheme_rows = sorted(
                (r for r in rows if r["scheme"] == scheme),
                key=lambda r: float(r["dt"]),
            )
            dt = np.array([float(r["dt"]) for r in scheme_rows])
            l1 = np.array([float(r["l1"]) for r in scheme_rows])
            l2 = np.array([float(r["l2"]) for r in scheme_rows])

            # Full fit uses every level; the tail fit drops the coarsest
            # (largest-h) level, matching run_strong_error --drop-coarsest 1.
            sens_lo, sens_hi = sensitivity_ranges.get(
                (regime_name, scheme), (np.nan, np.nan)
            )
            summary.append(
                {
                    "regime": regime_name,
                    "delta": delta,
                    "scheme": scheme,
                    "order_l1_full": float(fit_loglog_order(dt, l1)),
                    "order_l1_tail": float(fit_loglog_order(dt[:-1], l1[:-1])),
                    "order_l2_full": float(fit_loglog_order(dt, l2)),
                    "order_l2_tail": float(fit_loglog_order(dt[:-1], l2[:-1])),
                    "sens_l2_min": sens_lo,
                    "sens_l2_max": sens_hi,
                    "n_levels": dt.size,
                }
            )

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    deltas_all = sorted({r["delta"] for r in summary})

    # Hefter--Jentzen ceiling for uniform-mesh schemes, active for delta < 2.
    band = np.linspace(min(deltas_all) * 0.8, 2.0, 100)
    ax.fill_between(band, band / 2.0, 1.55, color="0.88", zorder=0)
    ax.plot(band, band / 2.0, color="k", lw=1.4, zorder=1)
    ax.annotate(
        "Hefter–Jentzen ceiling $\\delta/2$\n(uniform-mesh schemes)",
        xy=(0.55, 0.68), fontsize=8, ha="center",
    )
    ax.axvline(2.0, color="k", ls=":", lw=0.9)
    ax.annotate(
        "Feller boundary\n$\\delta = 2$", xy=(2.0, 1.47), fontsize=8,
        ha="center", va="top",
    )
    ax.axhline(0.5, color="0.5", ls="--", lw=0.8, zorder=1)
    ax.axhline(1.0, color="0.5", ls=":", lw=0.8, zorder=1)
    ax.axhline(0.0, color="0.5", lw=0.8, zorder=1)

    for scheme, style in SCHEME_STYLES.items():
        pts = sorted(
            (r for r in summary if r["scheme"] == scheme),
            key=lambda r: r["delta"],
        )
        if not pts:
            continue
        d = np.array([r["delta"] for r in pts])
        order = np.array([r["order_l2_tail"] for r in pts])
        ax.plot(d, order, lw=1.2, ms=6, **style)

        # Reference-sensitivity range (gate output): the span of the fitted
        # order across HH reference grids.  A visible bar means the slope is
        # not reference-converged and must be quoted as a range.
        for r in pts:
            if np.isfinite(r["sens_l2_min"]) and np.isfinite(r["sens_l2_max"]):
                ax.vlines(
                    r["delta"],
                    r["sens_l2_min"],
                    r["sens_l2_max"],
                    color=style["color"],
                    lw=3.0,
                    alpha=0.30,
                    zorder=1,
                )

    ax.set_xscale("log")
    ax.set_xticks(deltas_all)
    ax.set_xticklabels(
        [rf"{d:g}" + f"\n({name})" for d, name in zip(deltas_all, REGIMES)],
        fontsize=8,
    )
    ax.set_xlabel(r"Bessel dimension $\delta = 4\kappa\theta/\sigma^2$  (regime)")
    ax.set_ylabel(r"fitted strong $L^2$ order (tail fit)")
    ax.set_ylim(-0.35, 1.55)
    if sensitivity_ranges:
        fig.text(
            0.01,
            0.005,
            "Shaded bars: fitted-order span across HH references 4096-32768 "
            "(reference-sensitivity gate). Uniform-mesh points above the "
            r"$\delta/2$ band are reference-limited coupled diagnostics, "
            "not true rates.",
            fontsize=6.5,
            color="0.35",
        )
    ax.set_title(
        "Observed strong convergence order across the regime grid"
    )
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, which="major", axis="y", alpha=0.25)

    fig.tight_layout()
    pdf_path = figure_path("fig_order_vs_delta_summary.pdf")
    fig.savefig(pdf_path)
    fig.savefig(figure_path("fig_order_vs_delta_summary.png"), dpi=150)
    print(f"wrote {pdf_path}")

    csv_path = results_path("fig_order_vs_delta_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f"wrote {csv_path}")

    for r in summary:
        sens = ""
        if np.isfinite(r["sens_l2_min"]):
            sens = f"  sens=[{r['sens_l2_min']:.3f}, {r['sens_l2_max']:.3f}]"
        print(
            f"{r['regime']}  delta={r['delta']:<6g} {r['scheme']:<4} "
            f"L2 full={r['order_l2_full']:.3f} tail={r['order_l2_tail']:.3f}  "
            f"L1 full={r['order_l1_full']:.3f} tail={r['order_l1_tail']:.3f}"
            f"{sens}"
        )


if __name__ == "__main__":
    main()
