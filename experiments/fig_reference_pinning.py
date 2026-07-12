# Reference rate-pinning figure: KLM coupled discrepancy vs dt at three
# reference resolutions, regimes D and E.
#
# The Hefter--Jentzen lower bound pins ANY uniform-mesh reference's L1 error
# at Theta(N^(-delta/2)) for delta < 2, and the HH upper bound matches, so a
# 4x reference refinement can shrink the reference floor by at most
# 4^(-delta/2): a 36% cut in D (delta=0.64) and a 16% cut in E (delta=0.25).
# "You cannot refine your way into a trustworthy reference."
#
# This figure makes the floor visible: KLM's discrepancy is the smallest of
# any scheme, so at fine dt it flattens at the reference floor; refining the
# reference moves the plateau down only by roughly the pinned factor
# (D finest level, 4096 -> 16384: measured x0.63 vs predicted x0.64;
# E: measured x0.80 vs predicted x0.84).  At coarse dt the KLM discrepancy
# RISES with reference resolution -- scheme-reference cancellation being
# removed -- the same signature the sensitivity gate sees in the ProjEuler
# fitted orders.
#
# Honesty notes for the caption: second-refinement ratios are noisier (the
# suite path count is smaller than the main benchmark's, and at the finest
# level the KLM floor h_min crosses the fine grid at the lower reference
# resolutions); the prediction is an asymptotic bound, so "consistent with",
# not "equal to".
#
# Input: results/reference_sensitivity/strong_reference_sensitivity.csv
#        (produced by the Kaggle reference-sensitivity notebook, RUN_MODE=full)
#
# Outputs:
#   figures/reference_pinning.pdf (+ .png preview)

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.io import figure_path, results_path

DELTA = {"D": 0.64, "E": 0.25}
REF_COLOURS = {4096: "#4477AA", 16384: "#EE6677", 32768: "#228833"}


def main():
    d = pd.read_csv(
        results_path("reference_sensitivity") / "strong_reference_sensitivity.csv"
    )
    d = d[(d["scheme"] == "KLM") & (d["regime"].isin(DELTA))]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    for ax, (regime, delta) in zip(axes, DELTA.items()):
        sub = d[d["regime"] == regime]
        for ref, colour in REF_COLOURS.items():
            line = sub[sub["reference_n_steps"] == ref].sort_values("dt")
            ax.loglog(
                line["dt"], line["l1"], "o-", color=colour, markersize=4,
                label=f"HH reference at {ref}",
            )
        pinned = 4.0 ** (-delta / 2.0)
        ax.set_title(
            f"regime {regime} ($\\delta={delta}$): "
            f"pinned shrink $4^{{-\\delta/2}}={pinned:.2f}$",
            fontsize=10,
        )
        ax.set_xlabel(r"$\Delta t_{\max}$")
        ax.set_ylabel(r"KLM coupled $L^1$ discrepancy at $T$")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(
        "Reference rate-pinning: the KLM discrepancy floor moves down only\n"
        "by the pinned factor; coarse-dt discrepancies rise as "
        "scheme-reference cancellation is removed",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))

    out = figure_path("reference_pinning.pdf")
    fig.savefig(out)
    fig.savefig(Path(out).with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
