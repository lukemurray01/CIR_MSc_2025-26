# Anatomy of a single KLM backstopped adaptive path.
#
# Companion figure for the backstop proof-anatomy chapter: it shows, on one
# trajectory, the three mechanisms the chapter analyses in the abstract --
# the adaptive step contraction h_n = h_max * min(1, |Y_n|) as the iterate
# approaches the boundary, the explicit-step failures that trigger the
# backstop retake (same Brownian increment), and the minimum-step floor
# h_min = h_max / rho.
#
# Left column: regime C (Feller boundary, delta = 2, implicit backstop).
# Right column: regime E (delta = 0.25, projected fallback), where the
# backstop becomes the typical mode -- the documented failure regime.
#
# Usage:
#   uv run python experiments/fig_klm_path_anatomy.py
#
# Outputs:
#   figures/fig_klm_path_anatomy.pdf (+ .png preview)
#   results/fig_klm_path_anatomy_meta.csv (seeds and event counts)

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import csv

import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.samplers.klm_backstop import _backstop_map, default_backstop_kind
from src.utils.cir_params import cir_delta, kl_coefficients
from src.utils.io import config_path, figure_path, results_path
from src.utils.style import REGIME_COLOURS

H_MAX = 1.0 / 32
RHO = 64.0

EXPLICIT, BACKSTOP_MIN, BACKSTOP_NEG = 0, 1, 2


def load_config(filename):
    with open(config_path(filename), encoding="utf-8") as f:
        return yaml.safe_load(f)


def record_single_path(x0, kappa, theta, sigma, T, h_max, rho, seed):
    """Run one free-running KLM path, recording every step and trigger."""
    alpha, beta, gamma = kl_coefficients(kappa, theta, sigma)
    kind = default_backstop_kind(alpha)
    h_min = h_max / rho

    rng = np.random.default_rng(seed)
    y = float(np.sqrt(x0))
    t = 0.0

    times, x_values, step_sizes, events = [0.0], [x0], [], []

    while t < T - 1e-14:
        h_prop = h_max * min(1.0, abs(y))
        min_triggered = h_prop < h_min
        h = h_min if min_triggered else h_prop
        h = min(h, T - t)

        dW = np.sqrt(h) * rng.standard_normal()

        if min_triggered:
            y_next = float(
                _backstop_map(
                    np.array([y]), np.array([h]), np.array([dW]),
                    alpha, beta, gamma, kind,
                )[0]
            )
            event = BACKSTOP_MIN
        else:
            y_try = y + (alpha / y + beta * y) * h + gamma * dW
            if y_try <= 0.0:
                y_next = float(
                    _backstop_map(
                        np.array([y]), np.array([h]), np.array([dW]),
                        alpha, beta, gamma, kind,
                    )[0]
                )
                event = BACKSTOP_NEG
            else:
                y_next = y_try
                event = EXPLICIT

        y = y_next
        t += h
        times.append(t)
        x_values.append(y * y)
        step_sizes.append(h)
        events.append(event)

    return {
        "times": np.array(times),
        "x": np.array(x_values),
        "h": np.array(step_sizes),
        "events": np.array(events),
        "backstop_kind": kind,
    }


def select_illustrative_seed(x0, kappa, theta, sigma, T, base_seed, max_tries=200):
    """First seed whose path fires the negative-retake backstop at least twice.

    Deterministic given base_seed, so the figure is exactly reproducible;
    the chosen seed is reported in the metadata CSV and should be quoted in
    the thesis caption.
    """
    for k in range(max_tries):
        record = record_single_path(x0, kappa, theta, sigma, T, H_MAX, RHO, base_seed + k)
        if np.sum(record["events"] == BACKSTOP_NEG) >= 2:
            return base_seed + k, record
    return base_seed, record_single_path(x0, kappa, theta, sigma, T, H_MAX, RHO, base_seed)


def main():
    regimes_cfg = load_config("regimes.yaml")
    experiments_cfg = load_config("experiments.yaml")

    shared = regimes_cfg["shared"]
    master_seed = experiments_cfg["shared"]["master_seed"]
    T = experiments_cfg["shared"]["T"]
    kappa, theta, x0 = shared["kappa"], shared["theta"], shared["x0"]

    picks = ["C", "E"]
    fig, axes = plt.subplots(
        2, len(picks), figsize=(10.5, 6.4), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.2]},
    )

    meta_rows = []
    for j, regime_name in enumerate(picks):
        sigma = regimes_cfg["regimes"][regime_name]["sigma"]
        delta = cir_delta(kappa, theta, sigma)
        seed, rec = select_illustrative_seed(x0, kappa, theta, sigma, T, master_seed)
        colour = REGIME_COLOURS[regime_name]

        t_mid = rec["times"][1:]  # event/step arrays align with step endpoints
        neg = rec["events"] == BACKSTOP_NEG
        floor = rec["events"] == BACKSTOP_MIN

        ax_x = axes[0][j]
        ax_x.plot(rec["times"], rec["x"], color=colour, lw=1.0, zorder=1)
        ax_x.axhline(theta, color="k", ls=":", lw=0.8)
        ax_x.annotate(
            r"$\theta$", xy=(1.001, theta), xycoords=("axes fraction", "data"),
            fontsize=9, va="center",
        )
        if np.any(floor):
            ax_x.scatter(
                t_mid[floor], rec["x"][1:][floor], marker="^", s=28,
                facecolors="none", edgecolors="tab:orange", lw=1.1,
                label=rf"min-step backstop ({int(np.sum(floor))})", zorder=3,
            )
        if np.any(neg):
            ax_x.scatter(
                t_mid[neg], rec["x"][1:][neg], marker="x", s=34,
                color="tab:red", lw=1.3,
                label=rf"negative-retake backstop ({int(np.sum(neg))})", zorder=3,
            )
        ax_x.set_ylabel(r"$X_t = Y_t^2$")
        ax_x.set_title(
            rf"Regime {regime_name}: $\delta$={delta:g}, "
            rf"{rec['backstop_kind']} backstop, {rec['h'].size} steps"
        )
        ax_x.legend(fontsize=8, loc="upper right")
        ax_x.grid(True, alpha=0.25)

        ax_h = axes[1][j]
        ax_h.semilogy(t_mid, rec["h"], color=colour, lw=0.8, drawstyle="steps-post")
        ax_h.axhline(H_MAX, color="k", ls="--", lw=0.8)
        ax_h.axhline(H_MAX / RHO, color="k", ls="-.", lw=0.8)
        ax_h.annotate(
            r"$h_{\max}$", xy=(1.001, H_MAX), xycoords=("axes fraction", "data"),
            fontsize=9, va="center",
        )
        ax_h.annotate(
            r"$h_{\min}$", xy=(1.001, H_MAX / RHO),
            xycoords=("axes fraction", "data"), fontsize=9, va="center",
        )
        if np.any(neg):
            ax_h.scatter(
                t_mid[neg], rec["h"][neg], marker="x", s=30,
                color="tab:red", lw=1.2, zorder=3,
            )
        ax_h.set_xlabel(r"$t$")
        ax_h.set_ylabel(r"step size $h_n$")
        ax_h.grid(True, which="both", alpha=0.25)

        meta_rows.append(
            {
                "regime": regime_name,
                "delta": delta,
                "sigma": sigma,
                "seed": seed,
                "h_max": H_MAX,
                "rho": RHO,
                "n_steps": int(rec["h"].size),
                "n_backstop_min": int(np.sum(floor)),
                "n_backstop_neg": int(np.sum(neg)),
                "backstop_kind": rec["backstop_kind"],
            }
        )

    fig.suptitle(
        rf"KLM backstopped adaptive scheme, single-path anatomy "
        rf"($h_{{\max}}$={H_MAX:g}, $\rho$={RHO:g})"
    )
    fig.tight_layout()

    pdf_path = figure_path("fig_klm_path_anatomy.pdf")
    fig.savefig(pdf_path)
    fig.savefig(figure_path("fig_klm_path_anatomy.png"), dpi=150)
    print(f"wrote {pdf_path}")

    meta_path = results_path("fig_klm_path_anatomy_meta.csv")
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(meta_rows[0].keys()))
        writer.writeheader()
        writer.writerows(meta_rows)
    print(f"wrote {meta_path}")
    for row in meta_rows:
        print(row)


if __name__ == "__main__":
    main()
