"""Reproduce the Kelly--Lord adaptive-splitting paper figures.

This is a paper-specific reproduction lane for Fig. 2 and Fig. 3 of

    C. Kelly and G. J. Lord, "An adaptive splitting method for the
    Cox-Ingersoll-Ross process", Applied Numerical Mathematics 186 (2023).

The paper setup is:

    kappa = 2, theta = 0.02, X0 = 0, T = 1,
    M = 1000 paths, 20 batches of 50 paths,
    HH/Milstein reference with dt_ref = 1e-5,
    dtmax in {0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001}.

The adaptive splitting paths are Brownian-bridge coupled to the same fine
HH path used by the reference. This keeps the experiment a genuine strong
error comparison.

Examples
--------
Smoke run, intended for local checks:

    python experiments/kl_adaptive_splitting_paper.py --mode smoke

Paper-scale run, intended for a larger machine:

    python experiments/kl_adaptive_splitting_paper.py --mode paper
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.io import figure_path, results_path


KAPPA = 2.0
THETA = 0.02
X0 = 0.0
FINAL_TIME = 1.0
SOFT_ZERO_RHO = 2.0

PAPER_DT_REF = 1.0e-5
PAPER_DTMAX = (0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001)
PAPER_N_PATHS = 1000
PAPER_N_BATCHES = 20
PAPER_SIGMAS = tuple(np.round(np.linspace(0.0, 1.0, 41), 6))

SMOKE_DT_REF = 1.0e-3
SMOKE_DTMAX = (0.1, 0.02, 0.01)
SMOKE_N_PATHS = 60
SMOKE_N_BATCHES = 5
SMOKE_SIGMAS = (0.2, 0.3, 0.5, 0.8)

FIG3_SIGMA = 0.3
EPS = 1.0e-14


@dataclass(frozen=True)
class ExperimentConfig:
    n_paths: int
    n_batches: int
    dt_ref: float
    dtmax_values: tuple[float, ...]
    sigmas: tuple[float, ...]
    fig3_sigma: float
    seed: int
    output_stem: str


def kl_alpha(kappa: float, theta: float, sigma: float) -> float:
    return (4.0 * kappa * theta - sigma**2) / 8.0


def soft_zero_threshold(kappa: float, theta: float, dt_max: float, rho: float) -> float:
    return theta * (1.0 - math.exp(-kappa * dt_max)) / rho


def projected_floor(dt: float) -> float:
    """Paper projected-Euler floor: N^(-1/4), with T=1 so N=1/dt."""
    return dt**0.25


def key_sigma_values(kappa: float = KAPPA, theta: float = THETA) -> dict[str, float]:
    return {
        "projected 1/4": math.sqrt(2.0 * kappa * theta / 3.0),
        "sqrt(kappa theta)": math.sqrt(kappa * theta),
        "Feller": math.sqrt(2.0 * kappa * theta),
        "alpha=0": 2.0 * math.sqrt(kappa * theta),
    }


def hh_milstein_step(
    x: np.ndarray,
    kappa: float,
    theta: float,
    sigma: float,
    dt: float,
    dW: np.ndarray,
) -> np.ndarray:
    floor = 0.25 * sigma**2 * dt
    r1 = np.maximum(
        0.5 * sigma * math.sqrt(dt),
        np.sqrt(np.maximum(floor, x)) + 0.5 * sigma * dW,
    )
    x_hat = r1**2 + dt * (kappa * (theta - x) - 0.25 * sigma**2)
    return np.maximum(x_hat, 0.0)


def fte_step(
    x_aux: np.ndarray,
    kappa: float,
    theta: float,
    sigma: float,
    dt: float,
    dW: np.ndarray,
) -> np.ndarray:
    x_pos = np.maximum(x_aux, 0.0)
    return x_aux + kappa * (theta - x_pos) * dt + sigma * np.sqrt(x_pos) * dW


def projected_euler_step(
    y: np.ndarray,
    alpha: float,
    kappa: float,
    sigma: float,
    dt: float,
    dW: np.ndarray,
) -> np.ndarray:
    y_floor = projected_floor(dt)
    y_safe = np.maximum(y, y_floor)
    y_hat = y_safe + (alpha / y_safe - 0.5 * kappa * y_safe) * dt
    y_hat = y_hat + 0.5 * sigma * dW
    return np.maximum(y_hat, y_floor)


def implicit_lamperti_step(
    y: np.ndarray,
    alpha: float,
    kappa: float,
    sigma: float,
    dt: float,
    dW: np.ndarray,
) -> np.ndarray:
    beta = -0.5 * kappa
    gamma = 0.5 * sigma
    a = 1.0 - beta * dt
    b = -(y + gamma * dW)
    c = -alpha * dt
    return (-b + np.sqrt(b**2 - 4.0 * a * c)) / (2.0 * a)


def splitting_step(
    x: np.ndarray,
    alpha: float,
    kappa: float,
    sigma: float,
    dt: np.ndarray | float,
    dW: np.ndarray,
) -> np.ndarray:
    inside = np.maximum(x + 2.0 * alpha * dt, 0.0)
    return np.exp(-kappa * dt) * (np.sqrt(inside) + 0.5 * sigma * dW) ** 2


def terminal_fixed_scheme(
    method: str,
    dW: np.ndarray,
    dt: float,
    *,
    kappa: float,
    theta: float,
    sigma: float,
    x0: float,
) -> np.ndarray:
    n_paths, n_steps = dW.shape
    alpha = kl_alpha(kappa, theta, sigma)

    if method == "Milstein":
        x = np.full(n_paths, x0, dtype=float)
        for n in range(n_steps):
            x = hh_milstein_step(x, kappa, theta, sigma, dt, dW[:, n])
        return x

    if method == "Fully Truncated":
        x_aux = np.full(n_paths, x0, dtype=float)
        for n in range(n_steps):
            x_aux = fte_step(x_aux, kappa, theta, sigma, dt, dW[:, n])
        return np.maximum(x_aux, 0.0)

    if method == "Projected":
        y = np.full(n_paths, max(math.sqrt(x0), projected_floor(dt)), dtype=float)
        for n in range(n_steps):
            y = projected_euler_step(y, alpha, kappa, sigma, dt, dW[:, n])
        return y**2

    if method == "Implicit":
        if alpha <= 0.0:
            return np.full(n_paths, np.nan)
        y = np.full(n_paths, math.sqrt(x0), dtype=float)
        for n in range(n_steps):
            y = implicit_lamperti_step(y, alpha, kappa, sigma, dt, dW[:, n])
        return y**2

    if method == "Splitting":
        if alpha < 0.0:
            return np.full(n_paths, np.nan)
        x = np.full(n_paths, x0, dtype=float)
        for n in range(n_steps):
            x = splitting_step(x, alpha, kappa, sigma, dt, dW[:, n])
        return x

    raise ValueError(f"unknown fixed method {method!r}")


def aggregate_fine_increments(dW_ref: np.ndarray, factor: int) -> np.ndarray:
    n_paths, n_ref_steps = dW_ref.shape
    if n_ref_steps % factor != 0:
        raise ValueError("coarse step must divide the reference grid")
    n_steps = n_ref_steps // factor
    return dW_ref.reshape(n_paths, n_steps, factor).sum(axis=2)


def brownian_bridge_values(
    W_grid: np.ndarray,
    path_index: np.ndarray,
    base_t: np.ndarray,
    base_w: np.ndarray,
    target_t: np.ndarray,
    dt_ref: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample W(target_t) on the fine reference path by Brownian bridge."""
    n_ref = W_grid.shape[1] - 1
    target_t = np.minimum(target_t, FINAL_TIME)
    raw_left = np.floor(np.minimum(target_t, FINAL_TIME - EPS) / dt_ref).astype(int)
    left_index = np.clip(raw_left, 0, n_ref - 1)
    t_left = left_index * dt_ref
    t_right = (left_index + 1) * dt_ref
    w_left = W_grid[path_index, left_index]
    w_right = W_grid[path_index, left_index + 1]

    use_base = base_t >= t_left - 10.0 * EPS
    bridge_left_t = np.where(use_base, base_t, t_left)
    bridge_left_w = np.where(use_base, base_w, w_left)

    denom = np.maximum(t_right - bridge_left_t, EPS)
    theta = np.clip((target_t - bridge_left_t) / denom, 0.0, 1.0)
    variance = np.maximum((target_t - bridge_left_t) * (t_right - target_t) / denom, 0.0)
    sampled = bridge_left_w + theta * (w_right - bridge_left_w)
    sampled = sampled + np.sqrt(variance) * rng.standard_normal(path_index.size)

    sampled = np.where(np.abs(target_t - bridge_left_t) < 10.0 * EPS, bridge_left_w, sampled)
    sampled = np.where(np.abs(target_t - t_right) < 10.0 * EPS, w_right, sampled)
    sampled = np.where(target_t >= FINAL_TIME - 10.0 * EPS, W_grid[path_index, -1], sampled)
    return sampled


def adaptive_splitting_terminal(
    W_grid: np.ndarray,
    dt_ref: float,
    *,
    kappa: float,
    theta: float,
    sigma: float,
    x0: float,
    dt_max: float,
    rng: np.random.Generator,
    mode: str,
    rho: float = SOFT_ZERO_RHO,
    max_rounds: int = 2_000_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Adaptive splitting terminal values coupled to a fine Brownian path.

    mode="soft_zero" implements the alpha<0 paper extension.
    mode="heuristic" implements the Fig. 3 optional adaptive rule
    dt_n = dtmax / (1 + 3 exp(-150 X_n)).
    """
    if mode not in {"soft_zero", "heuristic"}:
        raise ValueError("mode must be 'soft_zero' or 'heuristic'")

    n_paths = W_grid.shape[0]
    alpha = kl_alpha(kappa, theta, sigma)
    x = np.full(n_paths, x0, dtype=float)
    t = np.zeros(n_paths, dtype=float)
    w = np.zeros(n_paths, dtype=float)
    step_count = np.zeros(n_paths, dtype=np.int64)
    deterministic_count = np.zeros(n_paths, dtype=np.int64)
    x_zero = soft_zero_threshold(kappa, theta, dt_max, rho)

    for _round in range(max_rounds):
        active = t < FINAL_TIME - 10.0 * EPS
        if not np.any(active):
            mean_dt = FINAL_TIME / np.maximum(step_count, 1)
            return x, mean_dt, deterministic_count

        idx = np.flatnonzero(active)
        x_a = x[idx]
        remaining = FINAL_TIME - t[idx]

        if mode == "heuristic":
            dt = dt_max / (1.0 + 3.0 * np.exp(-150.0 * x_a))
            deterministic = np.zeros_like(dt, dtype=bool)
        else:
            deterministic = x_a < x_zero
            dt = np.empty_like(x_a)

            if np.any(deterministic):
                x_det = x_a[deterministic]
                ratio = np.clip((x_zero - theta) / (x_det - theta), EPS, 1.0)
                dt[deterministic] = -np.log(ratio) / kappa

            if np.any(~deterministic):
                if alpha < 0.0:
                    dt_adaptive = 0.95 * x_a[~deterministic] / (2.0 * abs(alpha))
                    dt[~deterministic] = np.minimum(dt_adaptive, dt_max)
                else:
                    dt[~deterministic] = dt_max

        dt = np.minimum(dt, remaining)
        dt = np.maximum(dt, EPS)
        target_t = t[idx] + dt
        target_w = brownian_bridge_values(W_grid, idx, t[idx], w[idx], target_t, dt_ref, rng)
        dW = target_w - w[idx]

        x_next = np.empty_like(x_a)
        if np.any(deterministic):
            h = dt[deterministic]
            decay = np.exp(-kappa * h)
            x_next[deterministic] = decay * x_a[deterministic] + theta * (1.0 - decay)

        stochastic = ~deterministic
        if np.any(stochastic):
            x_next[stochastic] = splitting_step(
                x_a[stochastic],
                alpha,
                kappa,
                sigma,
                dt[stochastic],
                dW[stochastic],
            )

        x[idx] = np.maximum(x_next, 0.0)
        t[idx] = target_t
        w[idx] = target_w
        step_count[idx] += 1
        deterministic_count[idx] += deterministic.astype(np.int64)

    raise RuntimeError("adaptive splitting did not reach T; increase max_rounds")


def fit_loglog_rate(x_values: np.ndarray, y_values: np.ndarray) -> float:
    mask = np.isfinite(x_values) & np.isfinite(y_values) & (x_values > 0.0) & (y_values > 0.0)
    if np.count_nonzero(mask) < 2:
        return float("nan")
    slope, _intercept = np.polyfit(np.log(x_values[mask]), np.log(y_values[mask]), 1)
    return float(slope)


def batch_slices(n_paths: int, n_batches: int) -> list[slice]:
    if n_paths % n_batches != 0:
        raise ValueError("n_paths must be divisible by n_batches")
    batch_size = n_paths // n_batches
    return [slice(i * batch_size, (i + 1) * batch_size) for i in range(n_batches)]


def simulate_sigma(
    sigma: float,
    sigma_index: int,
    config: ExperimentConfig,
) -> list[dict[str, float | int | str]]:
    n_ref_steps = int(round(FINAL_TIME / config.dt_ref))
    if not math.isclose(n_ref_steps * config.dt_ref, FINAL_TIME, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("dt_ref must divide T=1")

    rng = np.random.default_rng(config.seed + 1009 * sigma_index)
    dW_ref = math.sqrt(config.dt_ref) * rng.standard_normal((config.n_paths, n_ref_steps))
    W_grid = np.concatenate(
        [np.zeros((config.n_paths, 1), dtype=float), np.cumsum(dW_ref, axis=1)],
        axis=1,
    )
    reference = terminal_fixed_scheme(
        "Milstein",
        dW_ref,
        config.dt_ref,
        kappa=KAPPA,
        theta=THETA,
        sigma=sigma,
        x0=X0,
    )

    records: list[dict[str, float | int | str]] = []
    batches = batch_slices(config.n_paths, config.n_batches)
    alpha = kl_alpha(KAPPA, THETA, sigma)

    for level, dt_max in enumerate(config.dtmax_values):
        factor = int(round(dt_max / config.dt_ref))
        if factor < 1 or not math.isclose(factor * config.dt_ref, dt_max, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(f"dtmax={dt_max:g} must be a multiple of dt_ref={config.dt_ref:g}")
        dW_coarse = aggregate_fine_increments(dW_ref, factor)

        method_outputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for method in ("Fully Truncated", "Projected", "Milstein", "Implicit"):
            terminal = terminal_fixed_scheme(
                method,
                dW_coarse,
                dt_max,
                kappa=KAPPA,
                theta=THETA,
                sigma=sigma,
                x0=X0,
            )
            method_outputs[method] = (terminal, np.full(config.n_paths, dt_max))

        if alpha >= 0.0:
            terminal = terminal_fixed_scheme(
                "Splitting",
                dW_coarse,
                dt_max,
                kappa=KAPPA,
                theta=THETA,
                sigma=sigma,
                x0=X0,
            )
            method_outputs["Splitting/Adaptive"] = (
                terminal,
                np.full(config.n_paths, dt_max),
            )
        else:
            terminal, mean_dt, _det_count = adaptive_splitting_terminal(
                W_grid,
                config.dt_ref,
                kappa=KAPPA,
                theta=THETA,
                sigma=sigma,
                x0=X0,
                dt_max=dt_max,
                rng=rng,
                mode="soft_zero",
            )
            method_outputs["Splitting/Adaptive"] = (terminal, mean_dt)

        if math.isclose(sigma, config.fig3_sigma, rel_tol=0.0, abs_tol=1.0e-12) and alpha >= 0.0:
            terminal = terminal_fixed_scheme(
                "Splitting",
                dW_coarse,
                dt_max,
                kappa=KAPPA,
                theta=THETA,
                sigma=sigma,
                x0=X0,
            )
            method_outputs["Splitting"] = (terminal, np.full(config.n_paths, dt_max))
            terminal, mean_dt, _det_count = adaptive_splitting_terminal(
                W_grid,
                config.dt_ref,
                kappa=KAPPA,
                theta=THETA,
                sigma=sigma,
                x0=X0,
                dt_max=dt_max,
                rng=rng,
                mode="heuristic",
            )
            method_outputs["Adaptive Splitting"] = (terminal, mean_dt)

        for method, (terminal, mean_dt_path) in method_outputs.items():
            error = terminal - reference
            for batch_id, batch in enumerate(batches):
                e = error[batch]
                dt_batch = mean_dt_path[batch]
                l1 = float(np.mean(np.abs(e))) if np.all(np.isfinite(e)) else float("nan")
                l2 = float(np.sqrt(np.mean(e**2))) if np.all(np.isfinite(e)) else float("nan")
                for metric, value in (("L1", l1), ("L2", l2)):
                    records.append(
                        {
                            "sigma": float(sigma),
                            "alpha": float(alpha),
                            "level": int(level),
                            "dtmax": float(dt_max),
                            "dtmean": float(np.mean(dt_batch)),
                            "method": method,
                            "batch": int(batch_id),
                            "metric": metric,
                            "error": value,
                        }
                    )

    return records


def rate_records(error_records: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    rates: list[dict[str, float | int | str]] = []
    keys = sorted(
        {
            (row["sigma"], row["alpha"], row["method"], row["metric"], row["batch"])
            for row in error_records
            if row["method"] != "Adaptive Splitting"
        }
    )
    for sigma, alpha, method, metric, batch in keys:
        rows = [
            row
            for row in error_records
            if row["sigma"] == sigma
            and row["method"] == method
            and row["metric"] == metric
            and row["batch"] == batch
        ]
        x = np.array([float(row["dtmean"]) for row in rows], dtype=float)
        y = np.array([float(row["error"]) for row in rows], dtype=float)
        rates.append(
            {
                "sigma": float(sigma),
                "alpha": float(alpha),
                "method": str(method),
                "metric": str(metric),
                "batch": int(batch),
                "rate": fit_loglog_rate(x, y),
            }
        )
    return rates


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def grouped_mean_std(
    rows: list[dict[str, float | int | str]],
    *,
    x_key: str,
    y_key: str,
    method: str,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = [r for r in rows if r["method"] == method and r["metric"] == metric]
    x_values = sorted({float(r[x_key]) for r in selected})
    means = []
    stds = []
    for x_value in x_values:
        vals = np.array([float(r[y_key]) for r in selected if float(r[x_key]) == x_value])
        vals = vals[np.isfinite(vals)]
        means.append(float(np.mean(vals)) if vals.size else np.nan)
        stds.append(float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0)
    return np.array(x_values), np.array(means), np.array(stds)


def error_by_level(
    rows: list[dict[str, float | int | str]],
    *,
    method: str,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = [r for r in rows if r["method"] == method and r["metric"] == metric]
    levels = sorted({int(r["level"]) for r in selected})
    x_values = []
    means = []
    stds = []
    for level in levels:
        level_rows = [r for r in selected if int(r["level"]) == level]
        errors = np.array([float(r["error"]) for r in level_rows], dtype=float)
        dtmeans = np.array([float(r["dtmean"]) for r in level_rows], dtype=float)
        mask = np.isfinite(errors) & np.isfinite(dtmeans)
        if not np.any(mask):
            continue
        x_values.append(float(np.mean(dtmeans[mask])))
        means.append(float(np.mean(errors[mask])))
        stds.append(float(np.std(errors[mask], ddof=1)) if np.count_nonzero(mask) > 1 else 0.0)
    return np.array(x_values), np.array(means), np.array(stds)


def plot_combined(
    error_rows: list[dict[str, float | int | str]],
    rates: list[dict[str, float | int | str]],
    config: ExperimentConfig,
) -> Path:
    styles = {
        "Splitting/Adaptive": dict(color="#1f77b4", marker="o", label="Splitting/Adaptive"),
        "Splitting": dict(color="#1f77b4", marker="o", label="Splitting"),
        "Fully Truncated": dict(color="#ff7f0e", marker="v", label="Fully Truncated"),
        "Projected": dict(color="#d62728", marker="D", label="Projected"),
        "Milstein": dict(color="#9467bd", marker="s", label="Milstein"),
        "Implicit": dict(color="#2ca02c", marker="x", label="Implicit"),
        "Adaptive Splitting": dict(color="#17becf", marker="*", label="Adaptive Splitting"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), constrained_layout=True)

    for ax, metric, title in (
        (axes[0, 0], "L1", r"(a) $L_1$ error"),
        (axes[0, 1], "L2", r"(b) $L_2$ error"),
    ):
        for method in ("Splitting/Adaptive", "Fully Truncated", "Projected", "Milstein", "Implicit"):
            x, y, yerr = grouped_mean_std(rates, x_key="sigma", y_key="rate", method=method, metric=metric)
            style = styles[method]
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                lw=1.1,
                ms=3.0,
                capsize=2,
                color=style["color"],
                marker=style["marker"],
                label=style["label"],
            )
        for value in key_sigma_values().values():
            ax.axvline(value, color="0.45", lw=0.8, ls="--", alpha=0.75)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.2)
        ax.set_xlabel("sigma")
        ax.set_ylabel("Rates")
        ax.grid(True, alpha=0.25)
        ax.set_title(title)
        ax.legend(fontsize=7)

    fig3_rows = [r for r in error_rows if math.isclose(float(r["sigma"]), config.fig3_sigma, abs_tol=1.0e-12)]
    for ax, metric, ylabel, title in (
        (axes[1, 0], "L1", "L1 Error", r"(a)"),
        (axes[1, 1], "L2", "RMSE", r"(b)"),
    ):
        for method in (
            "Splitting",
            "Fully Truncated",
            "Projected",
            "Milstein",
            "Adaptive Splitting",
            "Implicit",
        ):
            x, y, yerr = error_by_level(fig3_rows, method=method, metric=metric)
            if x.size == 0:
                continue
            order = np.argsort(x)
            x = x[order]
            y = y[order]
            yerr = yerr[order]
            style = styles[method]
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                lw=1.1,
                ms=3.2,
                capsize=2,
                color=style["color"],
                marker=style["marker"],
                label=style["label"],
            )
        x_ref = np.array([min(config.dtmax_values), max(config.dtmax_values)])
        ax.plot(x_ref, 0.7 * x_ref**0.5, color="0.35", lw=0.8, label="Ref slope 0.5")
        ax.plot(x_ref, 0.4 * x_ref**0.25, color="0.6", lw=0.8, label="Ref slope 0.25")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Dtmean")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.20)
        ax.set_title(title)
        ax.legend(fontsize=7)

    fig.suptitle(
        "Kelly-Lord adaptive splitting paper reproduction "
        f"(M={config.n_paths}, batches={config.n_batches}, dt_ref={config.dt_ref:g})",
        fontsize=11,
    )
    path = figure_path(f"{config.output_stem}_combined.pdf")
    fig.savefig(path)
    plt.close(fig)
    return path


def parse_float_list(raw: str | None) -> tuple[float, ...] | None:
    if raw is None:
        return None
    return tuple(float(part.strip()) for part in raw.split(",") if part.strip())


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    if args.mode == "paper":
        n_paths = PAPER_N_PATHS
        n_batches = PAPER_N_BATCHES
        dt_ref = PAPER_DT_REF
        dtmax_values = PAPER_DTMAX
        sigmas = PAPER_SIGMAS
        output_stem = "kl_adaptive_splitting_paper"
    else:
        n_paths = SMOKE_N_PATHS
        n_batches = SMOKE_N_BATCHES
        dt_ref = SMOKE_DT_REF
        dtmax_values = SMOKE_DTMAX
        sigmas = SMOKE_SIGMAS
        output_stem = "kl_adaptive_splitting_smoke"

    if args.n_paths is not None:
        n_paths = args.n_paths
    if args.n_batches is not None:
        n_batches = args.n_batches
    if args.dt_ref is not None:
        dt_ref = args.dt_ref
    if args.dtmax_values is not None:
        dtmax_values = args.dtmax_values
    if args.sigmas is not None:
        sigmas = args.sigmas
    if args.output_stem is not None:
        output_stem = args.output_stem

    return ExperimentConfig(
        n_paths=n_paths,
        n_batches=n_batches,
        dt_ref=dt_ref,
        dtmax_values=tuple(dtmax_values),
        sigmas=tuple(sigmas),
        fig3_sigma=args.fig3_sigma,
        seed=args.seed,
        output_stem=output_stem,
    )


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "paper"), default="smoke")
    parser.add_argument("--n-paths", type=int)
    parser.add_argument("--n-batches", type=int)
    parser.add_argument("--dt-ref", type=float)
    parser.add_argument("--dtmax", dest="dtmax_values", type=parse_float_list)
    parser.add_argument("--sigmas", type=parse_float_list)
    parser.add_argument("--fig3-sigma", type=float, default=FIG3_SIGMA)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--output-stem")
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = build_config(args)
    print("Kelly-Lord adaptive splitting reproduction")
    print(config)

    start = time.perf_counter()
    error_rows: list[dict[str, float | int | str]] = []
    for sigma_index, sigma in enumerate(config.sigmas):
        print(f"  sigma={sigma:g} ({sigma_index + 1}/{len(config.sigmas)})")
        error_rows.extend(simulate_sigma(sigma, sigma_index, config))

    rates = rate_records(error_rows)
    errors_csv = results_path(f"{config.output_stem}_errors.csv")
    rates_csv = results_path(f"{config.output_stem}_rates.csv")
    write_csv(errors_csv, error_rows)
    write_csv(rates_csv, rates)
    figure = plot_combined(error_rows, rates, config)

    elapsed = time.perf_counter() - start
    print(f"Wrote {errors_csv}")
    print(f"Wrote {rates_csv}")
    print(f"Wrote {figure}")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
