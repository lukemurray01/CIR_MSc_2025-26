# Regression tests for the projected Euler floor scaling (fixed 2026-07-05).
#
# The default floor must scale like the one-step noise, y_floor ~ gamma*sqrt(dt)
# with gamma = sigma/2.  The earlier dt-scaled floor made the drift kick at the
# floor O(1) against O(sqrt(dt)) noise for alpha < 0, so escaping the floor
# required a Gaussian tail event N > 2|alpha|/(sigma*sqrt(dt)); paths absorbed
# at X = dt^2 as dt -> 0 and the strong error grew instead of converging
# (docs/thesis_alignment_changes.md, item 13).

import numpy as np
import pytest
import yaml

from src.samplers.hh_milstein import hh_milstein_terminal_from_dW
from src.samplers.projected_euler import (
    default_y_floor,
    projected_euler_terminal_from_dW,
    projected_euler_terminal_with_stats_from_dW,
)
from src.utils.brownian import aggregate_brownian_increments
from src.utils.cir_params import kl_alpha
from src.utils.io import config_path
from src.utils.rng import make_rng


def load_config(filename: str) -> dict:
    with open(config_path(filename), encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_default_y_floor_scales_like_sqrt_dt():
    sigma, dt = 0.8, 0.25
    floor = default_y_floor(sigma, dt)

    assert floor == pytest.approx(0.5 * sigma * np.sqrt(dt))
    # In X-coordinates the default floor is the HH truncation level.
    assert floor**2 == pytest.approx(sigma**2 * dt / 4.0)

    # The stats variant reports the same default when y_floor is not given.
    _, stats = projected_euler_terminal_with_stats_from_dW(
        X0=0.02,
        kappa=2.0,
        theta=0.02,
        sigma=sigma,
        dt=dt,
        dW=np.zeros((3, 4)),
    )
    assert stats["y_floor"] == pytest.approx(floor)


def test_projected_euler_floor_trap_regression():
    # Regime E: sigma = 0.8, alpha = -0.06 < 0, delta = 0.25.  With the old
    # dt-scaled floor the scheme degenerates towards absorption at the floor
    # as dt shrinks (error INCREASES); with the default sqrt(dt)-scaled floor
    # the Brownian-coupled error decreases monotonically.
    regimes = load_config("regimes.yaml")
    experiments = load_config("experiments.yaml")
    shared = regimes["shared"]
    sigma = regimes["regimes"]["E"]["sigma"]
    assert kl_alpha(shared["kappa"], shared["theta"], sigma) < 0.0

    T = 1.0
    reference_n_steps = 4096
    n_paths = 4000
    coarse_levels = [16, 64, 256]

    rng = make_rng(experiments["shared"]["master_seed"])
    dt_fine = T / reference_n_steps
    dW_fine = np.sqrt(dt_fine) * rng.standard_normal((n_paths, reference_n_steps))
    reference = hh_milstein_terminal_from_dW(
        X0=shared["x0"],
        kappa=shared["kappa"],
        theta=shared["theta"],
        sigma=sigma,
        dt=dt_fine,
        dW=dW_fine,
    )

    l1 = {"dt_floor": [], "default_floor": []}
    stuck_at_finest = {}

    for n_steps in coarse_levels:
        factor = reference_n_steps // n_steps
        dt = dt_fine * factor
        dW = aggregate_brownian_increments(dW_fine, factor)

        for label, y_floor in [("dt_floor", dt), ("default_floor", None)]:
            terminal = projected_euler_terminal_from_dW(
                X0=shared["x0"],
                kappa=shared["kappa"],
                theta=shared["theta"],
                sigma=sigma,
                dt=dt,
                dW=dW,
                y_floor=y_floor,
            )
            l1[label].append(float(np.mean(np.abs(terminal - reference))))

            if n_steps == coarse_levels[-1]:
                floor = dt if y_floor is not None else default_y_floor(sigma, dt)
                stuck_at_finest[label] = float(
                    np.mean(terminal <= (floor * (1.0 + 1e-9)) ** 2)
                )

    # Old floor: absorbing trap.  Error grows from the coarsest to the finest
    # level and most paths finish exactly at the floor.
    assert l1["dt_floor"][-1] > l1["dt_floor"][0], (
        f"dt-scaled floor no longer shows the trap: {l1['dt_floor']}"
    )
    assert stuck_at_finest["dt_floor"] > 0.5

    # Default floor: monotone convergence and far fewer floored paths.
    assert l1["default_floor"][0] > l1["default_floor"][1] > l1["default_floor"][2], (
        f"default floor errors not monotone decreasing: {l1['default_floor']}"
    )
    assert stuck_at_finest["default_floor"] < 0.35
    assert stuck_at_finest["default_floor"] < stuck_at_finest["dt_floor"]
