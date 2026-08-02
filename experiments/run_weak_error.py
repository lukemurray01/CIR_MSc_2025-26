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

from src.metrics.control_variate import (
    G2Table,
    control_from_paths,
    dxu_g1,
    dxu_g3,
)
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
        default=[8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
        help="Step counts; default is the deep ladder h = 2^-3 .. 2^-12.",
    )
    parser.add_argument(
        "--max-adaptive-steps", type=int, default=1024,
        help="Cap on h_max^-1 for the free-running adaptive schemes (KLM, "
             "adaptive KL). Their accepted-step counts grow far faster than "
             "the nominal level, so they stop at a coarser level than the "
             "fixed-step schemes; the cap is recorded in the CSV.",
    )
    parser.add_argument("--batch-size", type=int, default=25_000,
                        help="Paths per batch at the coarsest level; scaled "
                             "down with n_steps to bound path-array memory.")
    parser.add_argument(
        "--no-control-variate", action="store_true",
        help="Disable the analytical martingale control variate (reproduces "
             "the pre-2026-08 direct estimator).",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Write results and figures here instead of the repo tree "
             "(Kaggle: /kaggle/working).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Reuse completed (regime, n_steps) levels from the partial CSV "
             "in --out-dir and continue. Levels are independently seeded, so "
             "a resumed run reproduces an uninterrupted one exactly.",
    )
    parser.add_argument(
        "--time-budget-s", type=float, default=None,
        help="Stop cleanly after this many seconds, leaving a resumable "
             "partial file. Set below the Kaggle session limit.",
    )
    return parser.parse_args()


PARTIAL_CSV = "weak_error_partial.csv"


class TimeBudgetExceeded(Exception):
    """Raised to unwind cleanly when --time-budget-s is hit."""


def _out_path(args, filename):
    if args.out_dir:
        directory = Path(args.out_dir)
        directory.mkdir(parents=True, exist_ok=True)
        return directory / filename
    return results_path(filename)


def load_partial(args, force=False):
    """Rows already computed, as (list_of_rows, {regime: {n_steps}}).

    `force` reads the partial file regardless of --resume; used after a
    time-budget stop, where the partial file is the run's only output.
    """
    path = _out_path(args, PARTIAL_CSV)
    if (not args.resume and not force) or not Path(path).exists():
        return [], {}

    numeric = {
        "dt", "approx_mean", "exact_value", "signed_error", "weak_error",
        "mc_standard_error", "error_to_se", "runtime_s", "signed_error_cv",
        "mc_standard_error_cv", "error_to_se_cv", "variance_reduction",
        "batch_standard_error_cv", "steps_per_path", "backstop_fraction",
    }
    rows, done = [], {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for key in numeric:
                if key in row:
                    row[key] = float(row[key]) if row[key] != "" else np.nan
            row["n_steps"] = int(row["n_steps"])
            row["n_paths"] = int(row["n_paths"])
            rows.append(row)
            done.setdefault(row["regime"], set()).add(row["n_steps"])
    return rows, done


def append_partial(args, rows):
    """Append completed levels so an interrupted session can resume."""
    if not rows:
        return
    path = _out_path(args, PARTIAL_CSV)
    exists = Path(path).exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def adaptive_dxu_fns(params, T, g2_table):
    """d_x u(tau, x) callables for the free-running adaptive schemes.

    These take per-path tau, because an adaptive mesh puts every path at a
    different time within a round.
    """
    kappa, theta, sigma = params["kappa"], params["theta"], params["sigma"]

    def _g2(tau, x):
        # Two 1-D lookups per round rather than one per level: the adaptive
        # taus are heterogeneous, so the table is sampled pointwise.
        rows = g2_table.rows_for_grid(tau)
        return np.array([np.interp(xi, g2_table.x, row)
                         for xi, row in zip(np.atleast_1d(x), rows)])

    return {
        "g1": lambda tau, x: np.full_like(x, 0.0) + dxu_g1(tau, kappa),
        "g2": _g2,
        "g3": lambda tau, x: dxu_g3(tau, x, kappa, theta, sigma),
    }


def batch_size_for(requested, n_steps):
    """Bound the stored (batch, n_steps+1) path array to ~25M float64 entries.

    At 4096 steps a 25,000-path batch would allocate 820 MB for the paths
    alone, and the control-variate temporaries triple that.
    """
    return max(1, min(requested, 25_000_000 // max(n_steps, 1)))


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


class _Accumulator:
    """Streaming bias and standard error over path batches.

    Values are centred on the exact expectation BEFORE accumulation.  At the
    deep levels the control-variate residual has a standard deviation many
    orders of magnitude below the payoff level, and the two-pass form
    (sum_sq - sum^2/n) would then lose most of its significant digits to
    cancellation; centring removes the cancellation entirely.

    Per-batch means are retained so a batch-based interval can be reported
    alongside the pooled one as an independence check.
    """

    def __init__(self):
        self.n = 0
        self.total = 0.0
        self.total_sq = 0.0
        self.batch_means = []

    def add(self, centred):
        centred = np.asarray(centred, dtype=float)
        self.n += centred.size
        self.total += float(np.sum(centred))
        self.total_sq += float(np.sum(centred**2))
        self.batch_means.append(float(np.mean(centred)))

    def bias(self):
        return self.total / self.n

    def standard_error(self):
        var = (self.total_sq - self.total**2 / self.n) / (self.n - 1)
        return float(np.sqrt(max(var, 0.0) / self.n))

    def sd(self):
        var = (self.total_sq - self.total**2 / self.n) / (self.n - 1)
        return float(np.sqrt(max(var, 0.0)))

    def batch_standard_error(self):
        """s.e. from the spread of equal-weight batch means (diagnostic)."""
        if len(self.batch_means) < 2:
            return float("nan")
        b = np.asarray(self.batch_means)
        return float(b.std(ddof=1) / np.sqrt(b.size))


def run_regime(regime_name, params, args, master_seed, n_paths,
               on_level=None, skip_levels=frozenset()):
    """Rows for one regime.

    `on_level(rows_for_level)` is called after each completed level so a long
    run can checkpoint; `skip_levels` holds n_steps values already present in
    a partial result file.  Levels are independent (each reseeds from
    master_seed + 7919*level), so resuming reproduces an uninterrupted run
    bit for bit.
    """
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

    use_cv = not args.no_control_variate

    # One g2 derivative table per regime, resolved down to the finest dt on
    # the ladder and re-sampled onto each level's own tau grid.
    g2_table = None
    if use_cv:
        g2_table = G2Table(
            kappa=params["kappa"], theta=params["theta"], sigma=params["sigma"],
            T=T, tau_min=T / max(args.n_steps),
        )

    rows = []
    for level, n_steps in enumerate(args.n_steps):
        if n_steps in skip_levels:
            continue
        level_rows = []
        dt = T / n_steps
        rng = make_rng(master_seed + 7919 * level)
        batch_size = batch_size_for(args.batch_size, n_steps)

        g2_rows = None
        if use_cv:
            taus = T - dt * np.arange(n_steps)
            g2_rows = {"x": g2_table.x, "rows": g2_table.rows_for_grid(taus)}

        # ---- shared-noise path schemes -------------------------------
        runners = {}
        for name in args.schemes:
            runner = path_scheme_runner(name, params, alpha)
            if runner is not None:
                runners[name] = runner

        stats = {
            (name, payoff, kind): _Accumulator()
            for name in runners
            for payoff in PAYOFFS
            for kind in ("raw", "cv")
        }
        runtimes = {name: 0.0 for name in runners}

        for batch_start in range(0, n_paths, batch_size):
            batch_n = min(batch_size, n_paths - batch_start)
            dW, m = make_brownian_increments_with_infima(rng, batch_n, n_steps, dt)

            for name, runner in runners.items():
                start = time.perf_counter()
                paths = runner(dt, dW, m)
                runtimes[name] += time.perf_counter() - start

                terminal = paths[:, -1]
                for payoff in PAYOFFS:
                    if payoff == "g4":
                        values = g4_bond_discount_from_paths(paths, dt)
                    else:
                        values = TERMINAL_PAYOFFS[payoff](terminal)
                    centred = values - exact[payoff]
                    stats[(name, payoff, "raw")].add(centred)

                    if use_cv:
                        # beta = 1: the martingale-representation value, and
                        # the h -> 0 limit of the optimal coefficient.  Any
                        # fixed beta keeps the estimator unbiased, so this
                        # needs no pilot and leaves the intervals exact.
                        control = control_from_paths(
                            payoff, paths, dW, dt, T, params["kappa"],
                            params["theta"], params["sigma"], g2_rows=g2_rows,
                        )
                        stats[(name, payoff, "cv")].add(centred - control)

        for name in runners:
            variant = "uniform" if name == "KL" else "fixed"
            for payoff in PAYOFFS:
                level_rows.append(
                    _make_row(
                        regime_name, name, variant, "shared", n_steps, dt,
                        payoff, stats[(name, payoff, "raw")],
                        stats[(name, payoff, "cv")] if use_cv else None,
                        runtimes[name], n_paths, exact[payoff],
                    )
                )

        # ---- free-running adaptive schemes (terminal payoffs only) ---
        # These choose their own mesh, so they are NOT put on a shared fine
        # grid: quantising an adaptive step to a 2^-k grid forces m = 1 at
        # every step once h_max reaches the grid spacing, which silently
        # collapses the scheme to a uniform mesh.  They run free, as before.
        if n_steps <= args.max_adaptive_steps:
            # g4 needs a fixed grid for the path integral, so the adaptive
            # schemes carry terminal payoffs only.
            dxu_fns = adaptive_dxu_fns(params, T, g2_table) if use_cv else None

            def _adaptive_rows(scheme, variant, terminal, controls, runtime,
                               steps_per_path=np.nan, backstop_fraction=np.nan):
                for payoff in ["g1", "g2", "g3"]:
                    centred = TERMINAL_PAYOFFS[payoff](terminal) - exact[payoff]
                    raw = _Accumulator()
                    raw.add(centred)
                    cv = None
                    if controls:
                        cv = _Accumulator()
                        cv.add(centred - controls[payoff])
                    level_rows.append(
                        _make_row(regime_name, scheme, variant, "own", n_steps,
                                  dt, payoff, raw, cv, runtime, n_paths,
                                  exact[payoff], steps_per_path,
                                  backstop_fraction)
                    )

            if "KLM" in args.schemes:
                controls = {} if use_cv else None
                start = time.perf_counter()
                terminal, klm_stats = klm_backstop_terminal(
                    X0=params["x0"], kappa=params["kappa"],
                    theta=params["theta"], sigma=params["sigma"], T=T,
                    h_max=dt, n_paths=n_paths, rng=rng,
                    dxu_fns=dxu_fns, controls_out=controls,
                )
                _adaptive_rows("KLM", klm_stats["backstop_kind"], terminal,
                               controls, time.perf_counter() - start,
                               klm_stats["n_steps_total"] / n_paths,
                               klm_stats["backstop_fraction"])

            if "KL" in args.schemes and alpha < 0.0:
                controls = {} if use_cv else None
                start = time.perf_counter()
                terminal = kl_adaptive_terminal(
                    X0=params["x0"], kappa=params["kappa"],
                    theta=params["theta"], sigma=params["sigma"], T=T,
                    dt_max=dt, n_paths=n_paths, rng=rng,
                    dxu_fns=dxu_fns, controls_out=controls,
                )
                _adaptive_rows("KL", "adaptive-soft-zero", terminal, controls,
                               time.perf_counter() - start)

        rows.extend(level_rows)
        if on_level is not None:
            on_level(level_rows)

    return rows


def _make_row(regime, scheme, variant, noise, n_steps, dt, payoff, raw, cv,
              runtime, n_paths, exact_value, steps_per_path=np.nan,
              backstop_fraction=np.nan):
    """One CSV row.

    The direct ("raw") estimator is always recorded so the new run stays
    comparable with the pre-control-variate results.  When a control variate
    is available its columns carry the production estimate: both target the
    same expectation, since E[C_h] = 0 exactly.
    """
    signed = raw.bias()
    se = raw.standard_error()

    row = {
        "regime": regime,
        "scheme": scheme,
        "scheme_variant": variant,
        "noise": noise,
        "payoff": payoff,
        "n_steps": n_steps,
        "dt": dt,
        "n_paths": n_paths,
        "approx_mean": signed + exact_value,
        "exact_value": exact_value,
        "signed_error": signed,
        "weak_error": abs(signed),
        "mc_standard_error": se,
        "error_to_se": abs(signed) / se if se > 0 else np.nan,
        "runtime_s": runtime,
        # Adaptive schemes only: accepted steps per path against the nominal
        # n_steps, and how often the backstop fired.  The ratio is the cost
        # of adaptivity at that nominal level.
        "steps_per_path": steps_per_path,
        "backstop_fraction": backstop_fraction,
    }

    if cv is None:
        row.update({
            "estimator": "direct",
            "signed_error_cv": np.nan, "mc_standard_error_cv": np.nan,
            "error_to_se_cv": np.nan, "variance_reduction": np.nan,
            "batch_standard_error_cv": np.nan,
        })
    else:
        signed_cv = cv.bias()
        se_cv = cv.standard_error()
        row.update({
            "estimator": "control-variate",
            "signed_error_cv": signed_cv,
            "mc_standard_error_cv": se_cv,
            "error_to_se_cv": abs(signed_cv) / se_cv if se_cv > 0 else np.nan,
            "variance_reduction": (raw.sd() / cv.sd()) ** 2 if cv.sd() > 0 else np.nan,
            "batch_standard_error_cv": cv.batch_standard_error(),
        })

    return row


def production_estimate(row):
    """(signed bias, s.e.) from the control variate when present."""
    if np.isfinite(row.get("signed_error_cv", np.nan)):
        return row["signed_error_cv"], row["mc_standard_error_cv"]
    return row["signed_error"], row["mc_standard_error"]


def _weighted_power_fit(h, b, se):
    """Inverse-variance weighted fit of b(h) = C h^alpha, on SIGNED biases.

    Returns (alpha, s.e.(alpha), n_used).  Points are used when they share
    the dominant sign; the weight of a point is (|b| / s.e.)^2, the
    delta-method precision of log|b|, so a level near the noise floor is
    downweighted smoothly instead of being admitted or excluded by a
    threshold.  Thresholding then fitting -- the previous policy -- selects
    on the noise it is trying to avoid and biases the fitted slope.
    """
    h, b, se = np.asarray(h), np.asarray(b), np.asarray(se)
    finite = np.isfinite(b) & np.isfinite(se) & (se > 0) & (b != 0.0)
    if finite.sum() < 3:
        return np.nan, np.nan, int(finite.sum())

    h, b, se = h[finite], b[finite], se[finite]

    # Dominant sign: weight each level by its own significance so that
    # noise-dominated sign flips do not decide it.
    signal = np.abs(b) / se
    sign = np.sign(np.sum(np.sign(b) * signal))
    keep = np.sign(b) == sign
    if keep.sum() < 3:
        return np.nan, np.nan, int(keep.sum())

    h, b, se = h[keep], b[keep], se[keep]
    y, x = np.log(np.abs(b)), np.log(h)
    w = (np.abs(b) / se) ** 2

    sw = w.sum()
    xbar, ybar = (w * x).sum() / sw, (w * y).sum() / sw
    sxx = (w * (x - xbar) ** 2).sum()
    if sxx <= 0:
        return np.nan, np.nan, int(len(h))
    alpha = (w * (x - xbar) * (y - ybar)).sum() / sxx

    resid = y - (ybar + alpha * (x - xbar))
    dof = len(h) - 2
    if dof > 0:
        # Scale the nominal slope error by the observed misfit, so that
        # curvature (a misspecified power law) widens the interval.
        chi2 = (w * resid**2).sum() / dof
        se_alpha = float(np.sqrt(max(chi2, 1.0) / sxx))
    else:
        se_alpha = float(np.sqrt(1.0 / sxx))

    return float(alpha), se_alpha, int(len(h))


def fit_orders(rows):
    """Signed weighted power-law fit, local slopes, and a drift diagnostic.

    Three columns matter downstream:

      fitted_weak_order        signed weighted fit over all levels (production)
      fitted_weak_order_legacy old policy: |error| fit over points >= 2 s.e.
      local_slope_trend        drift of the local slope per halving of h

    A monotonically drifting local slope means no single alpha describes the
    data: the floored schemes below the Feller boundary behave this way, and
    quoting one number for them is an artefact of the window measured.
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
        est = [production_estimate(r) for r in pts]
        h = np.array([r["dt"] for r in pts])
        b = np.array([e[0] for e in est])
        se = np.array([e[1] for e in est])

        alpha, se_alpha, n_used = _weighted_power_fit(h, b, se)

        # Stability under dropping the coarsest / the finest level.
        alpha_drop_coarse, _, _ = _weighted_power_fit(h[:-1], b[:-1], se[:-1])
        alpha_drop_fine, _, _ = _weighted_power_fit(h[1:], b[1:], se[1:])
        spread = np.nanmax(np.abs([alpha_drop_coarse - alpha,
                                   alpha_drop_fine - alpha])) \
            if np.isfinite(alpha) else np.nan

        # Local slopes on signed biases, fine-to-coarse as before.
        local, local_vals = [], []
        for lo, hi in zip(pts[:-1], pts[1:]):
            b_lo, se_lo = production_estimate(lo)
            b_hi, se_hi = production_estimate(hi)
            resolved = (abs(b_lo) >= 2 * se_lo and abs(b_hi) >= 2 * se_hi)
            if not resolved:
                local.append("noise")
            elif np.sign(b_lo) != np.sign(b_hi):
                local.append("sign-change")
            else:
                slope = np.log(abs(b_hi / b_lo)) / np.log(hi["dt"] / lo["dt"])
                local.append(f"{slope:.2f}")
                local_vals.append(slope)

        # Trend in the local slope: fitted per level (per halving of h).
        if len(local_vals) >= 4:
            idx = np.arange(len(local_vals), dtype=float)
            trend = float(np.polyfit(idx, np.asarray(local_vals), 1)[0])
        else:
            trend = np.nan

        # Legacy policy, retained so the rerun stays comparable with the
        # numbers currently quoted in the results chapter.
        usable = [r for r in pts
                  if np.isfinite(r["error_to_se"]) and r["error_to_se"] >= 2.0
                  and r["weak_error"] > 0.0]
        legacy = (
            float(fit_loglog_order(np.array([r["dt"] for r in usable]),
                                   np.array([r["weak_error"] for r in usable])))
            if len(usable) >= 3 else np.nan
        )

        resolved_cv = int(np.sum(np.abs(b) >= 2 * se))
        order_rows.append({
            "regime": regime,
            "scheme": scheme,
            "payoff": payoff,
            "fitted_weak_order": alpha,
            "fitted_weak_order_se": se_alpha,
            "order_stability_spread": float(spread) if np.isfinite(spread) else np.nan,
            "local_slope_trend": trend,
            "order_drifting": bool(np.isfinite(trend) and abs(trend) > 0.03),
            "n_points_used": n_used,
            "n_points_resolved": resolved_cv,
            "n_points_total": len(pts),
            "fitted_weak_order_legacy": legacy,
            "n_points_used_legacy": len(usable),
            "local_slopes_fine_to_coarse": ";".join(local),
        })
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
            est = [production_estimate(r) for r in pts]
            dt = np.array([r["dt"] for r in pts])
            bias = np.array([e[0] for e in est])
            se = np.array([e[1] for e in est])
            resolved = np.abs(bias) >= 2.0 * se

            # Unresolved levels are plotted at their 95% upper bound, not at
            # their point estimate: the point estimate there is a draw from
            # the noise and joining it into the curve manufactures apparent
            # oscillation that is not a measured convergence behaviour.
            err = np.where(resolved, np.abs(bias),
                           np.abs(bias) + 1.96 * se)
            err = np.maximum(err, 1e-18)

            style = SCHEME_STYLES[name]
            ax.loglog(dt[resolved], err[resolved], lw=1.1, ms=5,
                      label=SCHEME_LABELS[name], **style)
            if np.any(~resolved):
                ax.loglog(
                    dt[~resolved], err[~resolved], linestyle="none",
                    marker="v", ms=7, mfc="none", mec=style["color"], mew=1.0,
                )
            for r, s in zip(pts, se):
                noise_floor[r["dt"]] = max(noise_floor.get(r["dt"], 0.0), 2.0 * s)

        if noise_floor:
            dts = np.array(sorted(noise_floor))
            ax.loglog(
                dts, [noise_floor[d] for d in dts], "--", color="0.55",
                lw=1.0, label=r"$2\times$ MC s.e.",
            )
            # Slope-one guide anchored at the coarsest level.
            err_max = max(abs(production_estimate(r)[0]) for r in payoff_rows)
            ax.loglog(dts, err_max * (dts / dts[-1]), "k:", lw=0.8,
                      label="slope 1")

        ax.set_xlabel("step size h")
        ax.set_ylabel(f"|weak error|, {payoff}")
        ax.set_title(f"{payoff}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=6.5)

    estimator = ("direct" if args.no_control_variate
                 else "martingale control variate")
    fig.suptitle(
        f"Weak error vs exact functionals, regime {regime_name} "
        f"({n_paths} paths, {estimator}; open triangles: 95% upper bound "
        "where the bias is below the noise floor; "
        "g4 unavailable for adaptive schemes)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path)
    fig.savefig(Path(out_path).with_suffix(".png"), dpi=150)
    plt.close(fig)


def save_csv(rows, filename, args=None):
    path = _out_path(args, filename) if args is not None else results_path(filename)
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

    started = time.perf_counter()
    all_rows, done = load_partial(args)
    if all_rows:
        have = ", ".join(f"{r}:{len(v)}" for r, v in sorted(done.items()))
        print(f"resuming; {len(all_rows)} rows already complete ({have})",
              flush=True)

    out_of_time = False
    for regime_name in args.regimes:
        if out_of_time:
            break
        sigma = regimes_cfg["regimes"][regime_name]["sigma"]
        params = {
            "kappa": shared["kappa"], "theta": shared["theta"],
            "x0": shared["x0"], "sigma": sigma, "T": T,
        }
        print(f"Regime {regime_name} (sigma={sigma:g}) ...", flush=True)

        def checkpoint(level_rows, _regime=regime_name):
            nonlocal out_of_time
            append_partial(args, level_rows)
            elapsed = time.perf_counter() - started
            n_steps = level_rows[0]["n_steps"] if level_rows else "?"
            print(f"  {_regime} n_steps={n_steps} done "
                  f"({elapsed / 60:.1f} min elapsed)", flush=True)
            if args.time_budget_s and elapsed > args.time_budget_s:
                out_of_time = True
                raise TimeBudgetExceeded

        try:
            rows = run_regime(regime_name, params, args, master_seed, n_paths,
                              on_level=checkpoint,
                              skip_levels=done.get(regime_name, frozenset()))
            all_rows.extend(rows)
        except TimeBudgetExceeded:
            print(f"time budget reached during regime {regime_name}; "
                  f"partial results saved. Rerun with --resume to continue.",
                  flush=True)
            break

    if out_of_time:
        # Reload so the summary reflects exactly what is on disk.
        all_rows, _ = load_partial(args, force=True)

    if not all_rows:
        print("no results produced")
        return

    csv_path = save_csv(all_rows, "weak_error.csv", args)
    print(f"wrote {csv_path}")

    order_rows = fit_orders(all_rows)
    orders_path = save_csv(order_rows, "weak_error_orders.csv", args)
    print(f"wrote {orders_path}")

    for r in order_rows:
        if np.isfinite(r["fitted_weak_order"]):
            fitted = f"{r['fitted_weak_order']:.3f} +/- {r['fitted_weak_order_se']:.3f}"
            if r["order_drifting"]:
                fitted += f"  [DRIFTING {r['local_slope_trend']:+.3f}/level]"
        else:
            fitted = "refused (no resolved levels)"
        print(
            f"  {r['regime']} {r['scheme']:<10} {r['payoff']}: order {fitted} "
            f"({r['n_points_resolved']}/{r['n_points_total']} resolved; "
            f"legacy {r['fitted_weak_order_legacy']:.3f}; "
            f"local {r['local_slopes_fine_to_coarse']})"
        )

    present = {r["regime"] for r in all_rows}
    for regime_name in args.regimes:
        if regime_name not in present:
            continue
        fig_path = (_out_path(args, f"weak_error_regime_{regime_name}.pdf")
                    if args.out_dir
                    else figure_path(f"weak_error_regime_{regime_name}.pdf"))
        plot_regime(regime_name, all_rows, args, n_paths, fig_path)
        print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
