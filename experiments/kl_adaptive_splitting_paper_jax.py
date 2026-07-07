"""JAX/GPU reproduction of the Kelly--Lord adaptive-splitting paper figures.

This mirrors ``experiments/kl_adaptive_splitting_paper.py`` but ports the CIR
scheme kernels to JAX:

* HH/Milstein fine reference via ``jax.lax.scan``.
* Fixed FTE, projected Euler, implicit Lamperti, and splitting via scans.
* Adaptive splitting via a lockstep ``jax.lax.while_loop`` with per-path masks.
* Brownian-bridge off-grid samples gathered from the shared HH fine path.

The plotting and CSV aggregation remain ordinary Python/NumPy. The expensive
path evolution is JAX and will run on Kaggle's P100 when a GPU is enabled.

Examples
--------
Local smoke run:

    python experiments/kl_adaptive_splitting_paper_jax.py --mode smoke

Paper-scale Kaggle run:

    python experiments/kl_adaptive_splitting_paper_jax.py --mode paper
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import sys
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

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


def hh_milstein_step_jax(x, kappa, theta, sigma, dt, dW):
    floor = 0.25 * sigma**2 * dt
    r1 = jnp.maximum(
        0.5 * sigma * jnp.sqrt(dt),
        jnp.sqrt(jnp.maximum(floor, x)) + 0.5 * sigma * dW,
    )
    x_hat = r1**2 + dt * (kappa * (theta - x) - 0.25 * sigma**2)
    return jnp.maximum(x_hat, 0.0)


def fte_step_jax(x_aux, kappa, theta, sigma, dt, dW):
    x_pos = jnp.maximum(x_aux, 0.0)
    return x_aux + kappa * (theta - x_pos) * dt + sigma * jnp.sqrt(x_pos) * dW


def projected_euler_step_jax(y, alpha, kappa, sigma, dt, dW):
    y_floor = dt**0.25
    y_safe = jnp.maximum(y, y_floor)
    y_hat = y_safe + (alpha / y_safe - 0.5 * kappa * y_safe) * dt
    y_hat = y_hat + 0.5 * sigma * dW
    return jnp.maximum(y_hat, y_floor)


def implicit_lamperti_step_jax(y, alpha, kappa, sigma, dt, dW):
    beta = -0.5 * kappa
    gamma = 0.5 * sigma
    a = 1.0 - beta * dt
    b = -(y + gamma * dW)
    c = -alpha * dt
    return (-b + jnp.sqrt(b**2 - 4.0 * a * c)) / (2.0 * a)


def splitting_step_jax(x, alpha, kappa, sigma, dt, dW):
    inside = jnp.maximum(x + 2.0 * alpha * dt, 0.0)
    return jnp.exp(-kappa * dt) * (jnp.sqrt(inside) + 0.5 * sigma * dW) ** 2


@jax.jit
def hh_terminal_jax(dW, x0, kappa, theta, sigma, dt):
    n_paths = dW.shape[0]
    x_init = jnp.full((n_paths,), x0, dtype=jnp.float64)

    def body(x, dW_n):
        return hh_milstein_step_jax(x, kappa, theta, sigma, dt, dW_n), None

    x_final, _ = jax.lax.scan(body, x_init, jnp.swapaxes(dW, 0, 1))
    return x_final


@jax.jit
def fte_terminal_jax(dW, x0, kappa, theta, sigma, dt):
    n_paths = dW.shape[0]
    x_init = jnp.full((n_paths,), x0, dtype=jnp.float64)

    def body(x_aux, dW_n):
        return fte_step_jax(x_aux, kappa, theta, sigma, dt, dW_n), None

    x_aux_final, _ = jax.lax.scan(body, x_init, jnp.swapaxes(dW, 0, 1))
    return jnp.maximum(x_aux_final, 0.0)


@jax.jit
def projected_terminal_jax(dW, x0, kappa, theta, sigma, dt):
    n_paths = dW.shape[0]
    alpha = (4.0 * kappa * theta - sigma**2) / 8.0
    y0 = jnp.maximum(jnp.sqrt(x0), dt**0.25)
    y_init = jnp.full((n_paths,), y0, dtype=jnp.float64)

    def body(y, dW_n):
        return projected_euler_step_jax(y, alpha, kappa, sigma, dt, dW_n), None

    y_final, _ = jax.lax.scan(body, y_init, jnp.swapaxes(dW, 0, 1))
    return y_final**2


@jax.jit
def implicit_terminal_jax(dW, x0, kappa, theta, sigma, dt):
    n_paths = dW.shape[0]
    alpha = (4.0 * kappa * theta - sigma**2) / 8.0
    y_init = jnp.full((n_paths,), jnp.sqrt(x0), dtype=jnp.float64)

    def body(y, dW_n):
        return implicit_lamperti_step_jax(y, alpha, kappa, sigma, dt, dW_n), None

    y_final, _ = jax.lax.scan(body, y_init, jnp.swapaxes(dW, 0, 1))
    return y_final**2


@jax.jit
def splitting_terminal_jax(dW, x0, kappa, theta, sigma, dt):
    n_paths = dW.shape[0]
    alpha = (4.0 * kappa * theta - sigma**2) / 8.0
    x_init = jnp.full((n_paths,), x0, dtype=jnp.float64)

    def body(x, dW_n):
        return splitting_step_jax(x, alpha, kappa, sigma, dt, dW_n), None

    x_final, _ = jax.lax.scan(body, x_init, jnp.swapaxes(dW, 0, 1))
    return x_final


def aggregate_fine_increments_jax(dW_ref, factor: int):
    n_paths, n_ref_steps = dW_ref.shape
    if n_ref_steps % factor != 0:
        raise ValueError("coarse step must divide the reference grid")
    n_steps = n_ref_steps // factor
    return dW_ref.reshape((n_paths, n_steps, factor)).sum(axis=2)


def brownian_bridge_values_jax(W_grid, base_t, base_w, target_t, dt_ref, key):
    n_paths = W_grid.shape[0]
    n_ref = W_grid.shape[1] - 1
    path_index = jnp.arange(n_paths)
    target_t = jnp.minimum(target_t, FINAL_TIME)
    raw_left = jnp.floor(jnp.minimum(target_t, FINAL_TIME - EPS) / dt_ref).astype(jnp.int32)
    left_index = jnp.clip(raw_left, 0, n_ref - 1)
    t_left = left_index.astype(jnp.float64) * dt_ref
    t_right = (left_index.astype(jnp.float64) + 1.0) * dt_ref
    w_left = W_grid[path_index, left_index]
    w_right = W_grid[path_index, left_index + 1]

    use_base = base_t >= t_left - 10.0 * EPS
    bridge_left_t = jnp.where(use_base, base_t, t_left)
    bridge_left_w = jnp.where(use_base, base_w, w_left)

    denom = jnp.maximum(t_right - bridge_left_t, EPS)
    theta = jnp.clip((target_t - bridge_left_t) / denom, 0.0, 1.0)
    variance = jnp.maximum((target_t - bridge_left_t) * (t_right - target_t) / denom, 0.0)
    sampled = bridge_left_w + theta * (w_right - bridge_left_w)
    sampled = sampled + jnp.sqrt(variance) * jax.random.normal(
        key, (n_paths,), dtype=jnp.float64
    )

    sampled = jnp.where(jnp.abs(target_t - bridge_left_t) < 10.0 * EPS, bridge_left_w, sampled)
    sampled = jnp.where(jnp.abs(target_t - t_right) < 10.0 * EPS, w_right, sampled)
    sampled = jnp.where(target_t >= FINAL_TIME - 10.0 * EPS, W_grid[path_index, -1], sampled)
    return sampled


@partial(jax.jit, static_argnames=("mode", "max_rounds"))
def adaptive_splitting_terminal_jax(
    W_grid,
    key,
    dt_ref,
    kappa,
    theta,
    sigma,
    x0,
    dt_max,
    mode: str,
    rho=SOFT_ZERO_RHO,
    max_rounds=2_000_000,
):
    if mode not in {"soft_zero", "heuristic"}:
        raise ValueError("mode must be 'soft_zero' or 'heuristic'")

    n_paths = W_grid.shape[0]
    alpha = (4.0 * kappa * theta - sigma**2) / 8.0
    x_zero = theta * (1.0 - jnp.exp(-kappa * dt_max)) / rho

    x0_vec = jnp.full((n_paths,), x0, dtype=jnp.float64)
    zeros = jnp.zeros((n_paths,), dtype=jnp.float64)
    counts0 = jnp.zeros((n_paths,), dtype=jnp.int64)
    state0 = (x0_vec, zeros, zeros, counts0, counts0, key, jnp.array(0, dtype=jnp.int64))

    def cond_fun(state):
        _x, t, _w, _step_count, _det_count, _key, rounds = state
        return jnp.any(t < FINAL_TIME - 10.0 * EPS) & (rounds < max_rounds)

    def body_fun(state):
        x, t, w, step_count, deterministic_count, key_in, rounds = state
        active = t < FINAL_TIME - 10.0 * EPS
        remaining = FINAL_TIME - t

        if mode == "heuristic":
            dt_raw = dt_max / (1.0 + 3.0 * jnp.exp(-150.0 * x))
            deterministic = jnp.zeros((n_paths,), dtype=bool)
        else:
            deterministic = x < x_zero
            denom = jnp.where(jnp.abs(x - theta) > EPS, x - theta, -EPS)
            ratio = jnp.clip((x_zero - theta) / denom, EPS, 1.0)
            dt_det = -jnp.log(ratio) / kappa
            alpha_abs = jnp.maximum(jnp.abs(alpha), EPS)
            dt_adaptive = 0.95 * x / (2.0 * alpha_abs)
            dt_stochastic = jnp.where(alpha < 0.0, jnp.minimum(dt_adaptive, dt_max), dt_max)
            dt_raw = jnp.where(deterministic, dt_det, dt_stochastic)

        dt = jnp.where(active, jnp.minimum(jnp.maximum(dt_raw, EPS), remaining), 0.0)
        target_t = t + dt
        key_next, bridge_key = jax.random.split(key_in)
        target_w = brownian_bridge_values_jax(W_grid, t, w, target_t, dt_ref, bridge_key)
        dW = target_w - w

        decay = jnp.exp(-kappa * dt)
        x_det = decay * x + theta * (1.0 - decay)
        x_stochastic = splitting_step_jax(x, alpha, kappa, sigma, dt, dW)
        x_candidate = jnp.where(deterministic, x_det, x_stochastic)

        x_next = jnp.where(active, jnp.maximum(x_candidate, 0.0), x)
        t_next = jnp.where(active, target_t, t)
        w_next = jnp.where(active, target_w, w)
        step_count_next = step_count + active.astype(jnp.int64)
        det_count_next = deterministic_count + (active & deterministic).astype(jnp.int64)

        return (
            x_next,
            t_next,
            w_next,
            step_count_next,
            det_count_next,
            key_next,
            rounds + 1,
        )

    x, t, _w, step_count, deterministic_count, _key, rounds = jax.lax.while_loop(
        cond_fun, body_fun, state0
    )
    reached = ~jnp.any(t < FINAL_TIME - 10.0 * EPS)
    mean_dt = FINAL_TIME / jnp.maximum(step_count, 1)
    return x, mean_dt, deterministic_count, reached, rounds


def terminal_fixed_scheme_jax(method: str, dW, dt: float, *, sigma: float):
    alpha = kl_alpha(KAPPA, THETA, sigma)
    if method == "Milstein":
        terminal = hh_terminal_jax(dW, X0, KAPPA, THETA, sigma, dt)
    elif method == "Fully Truncated":
        terminal = fte_terminal_jax(dW, X0, KAPPA, THETA, sigma, dt)
    elif method == "Projected":
        terminal = projected_terminal_jax(dW, X0, KAPPA, THETA, sigma, dt)
    elif method == "Implicit":
        if alpha <= 0.0:
            return np.full(dW.shape[0], np.nan)
        terminal = implicit_terminal_jax(dW, X0, KAPPA, THETA, sigma, dt)
    elif method == "Splitting":
        if alpha < 0.0:
            return np.full(dW.shape[0], np.nan)
        terminal = splitting_terminal_jax(dW, X0, KAPPA, THETA, sigma, dt)
    else:
        raise ValueError(f"unknown fixed method {method!r}")
    return np.asarray(terminal.block_until_ready())


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


def simulate_sigma_jax(
    sigma: float,
    sigma_index: int,
    config: ExperimentConfig,
) -> list[dict[str, float | int | str]]:
    n_ref_steps = int(round(FINAL_TIME / config.dt_ref))
    if not math.isclose(n_ref_steps * config.dt_ref, FINAL_TIME, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("dt_ref must divide T=1")

    key = jax.random.PRNGKey(config.seed + 1009 * sigma_index)
    key, ref_key = jax.random.split(key)
    dW_ref = math.sqrt(config.dt_ref) * jax.random.normal(
        ref_key,
        (config.n_paths, n_ref_steps),
        dtype=jnp.float64,
    )
    W_grid = jnp.concatenate(
        [
            jnp.zeros((config.n_paths, 1), dtype=jnp.float64),
            jnp.cumsum(dW_ref, axis=1),
        ],
        axis=1,
    )
    reference = np.asarray(
        hh_terminal_jax(dW_ref, X0, KAPPA, THETA, sigma, config.dt_ref).block_until_ready()
    )

    records: list[dict[str, float | int | str]] = []
    batches = batch_slices(config.n_paths, config.n_batches)
    alpha = kl_alpha(KAPPA, THETA, sigma)

    for level, dt_max in enumerate(config.dtmax_values):
        factor = int(round(dt_max / config.dt_ref))
        if factor < 1 or not math.isclose(factor * config.dt_ref, dt_max, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(f"dtmax={dt_max:g} must be a multiple of dt_ref={config.dt_ref:g}")
        dW_coarse = aggregate_fine_increments_jax(dW_ref, factor)

        method_outputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for method in ("Fully Truncated", "Projected", "Milstein", "Implicit"):
            terminal = terminal_fixed_scheme_jax(method, dW_coarse, dt_max, sigma=sigma)
            method_outputs[method] = (terminal, np.full(config.n_paths, dt_max))

        if alpha >= 0.0:
            terminal = terminal_fixed_scheme_jax("Splitting", dW_coarse, dt_max, sigma=sigma)
            method_outputs["Splitting/Adaptive"] = (
                terminal,
                np.full(config.n_paths, dt_max),
            )
        else:
            key, adaptive_key = jax.random.split(key)
            terminal, mean_dt, _det_count, reached, rounds = adaptive_splitting_terminal_jax(
                W_grid,
                adaptive_key,
                config.dt_ref,
                KAPPA,
                THETA,
                sigma,
                X0,
                dt_max,
                mode="soft_zero",
            )
            if not bool(np.asarray(reached)):
                raise RuntimeError(
                    f"soft-zero adaptive splitting did not reach T at sigma={sigma:g}, "
                    f"dtmax={dt_max:g}; rounds={int(np.asarray(rounds))}"
                )
            method_outputs["Splitting/Adaptive"] = (
                np.asarray(terminal.block_until_ready()),
                np.asarray(mean_dt),
            )

        if math.isclose(sigma, config.fig3_sigma, rel_tol=0.0, abs_tol=1.0e-12) and alpha >= 0.0:
            terminal = terminal_fixed_scheme_jax("Splitting", dW_coarse, dt_max, sigma=sigma)
            method_outputs["Splitting"] = (terminal, np.full(config.n_paths, dt_max))
            key, adaptive_key = jax.random.split(key)
            terminal, mean_dt, _det_count, reached, rounds = adaptive_splitting_terminal_jax(
                W_grid,
                adaptive_key,
                config.dt_ref,
                KAPPA,
                THETA,
                sigma,
                X0,
                dt_max,
                mode="heuristic",
            )
            if not bool(np.asarray(reached)):
                raise RuntimeError(
                    f"heuristic adaptive splitting did not reach T at sigma={sigma:g}, "
                    f"dtmax={dt_max:g}; rounds={int(np.asarray(rounds))}"
                )
            method_outputs["Adaptive Splitting"] = (
                np.asarray(terminal.block_until_ready()),
                np.asarray(mean_dt),
            )

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
                            "backend": jax.default_backend(),
                        }
                    )

    del dW_ref, W_grid
    gc.collect()
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
                "backend": rows[0].get("backend", jax.default_backend()) if rows else jax.default_backend(),
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
        "Kelly-Lord adaptive splitting paper reproduction, JAX "
        f"(M={config.n_paths}, batches={config.n_batches}, dt_ref={config.dt_ref:g}, "
        f"backend={jax.default_backend()})",
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
        output_stem = "kl_adaptive_splitting_paper_jax"
    else:
        n_paths = SMOKE_N_PATHS
        n_batches = SMOKE_N_BATCHES
        dt_ref = SMOKE_DT_REF
        dtmax_values = SMOKE_DTMAX
        sigmas = SMOKE_SIGMAS
        output_stem = "kl_adaptive_splitting_smoke_jax"

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
    print("Kelly-Lord adaptive splitting reproduction, JAX")
    print(config)
    print("jax", jax.__version__, "backend:", jax.default_backend())
    print("devices:", jax.devices())
    if jax.default_backend() == "cpu":
        print("WARNING: no GPU visible. Enable Kaggle's P100 accelerator for the full run.")

    start = time.perf_counter()
    error_rows: list[dict[str, float | int | str]] = []
    for sigma_index, sigma in enumerate(config.sigmas):
        print(f"  sigma={sigma:g} ({sigma_index + 1}/{len(config.sigmas)})")
        error_rows.extend(simulate_sigma_jax(sigma, sigma_index, config))

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
