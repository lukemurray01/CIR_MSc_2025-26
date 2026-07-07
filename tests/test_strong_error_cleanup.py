import csv

import numpy as np
import pytest

import experiments.run_strong_error as strong
from src.samplers.projected_euler import (
    projected_euler_terminal_from_dW,
    projected_euler_terminal_with_stats_from_dW,
)


def test_fitted_orders_can_drop_pre_asymptotic_coarsest_point():
    rows = [
        {"scheme": "ProjEuler", "dt": 0.125, "l2": 100.0},
        {"scheme": "ProjEuler", "dt": 0.0625, "l2": 0.0625},
        {"scheme": "ProjEuler", "dt": 0.03125, "l2": 0.03125},
        {"scheme": "ProjEuler", "dt": 0.015625, "l2": 0.015625},
    ]

    tail_order = strong.fitted_orders(
        rows,
        ["ProjEuler"],
        drop_coarsest=1,
    )["ProjEuler"]

    assert tail_order == pytest.approx(1.0)


def test_display_label_uses_scheme_variant_and_tail_order():
    assert (
        strong.display_label(
            "KL",
            [{"scheme_variant": "adaptive-soft-zero"}],
            {"KL": 0.347},
        )
        == "KL adaptive soft-zero (tail order 0.35)"
    )
    assert (
        strong.display_label(
            "KLM",
            [{"scheme_variant": "projected"}],
            {"KLM": 0.154},
        )
        == "KLM projected backstop (tail order 0.15)"
    )


def test_projected_euler_stats_leave_terminal_values_unchanged():
    dW = np.zeros((3, 4), dtype=float)
    kwargs = dict(
        X0=0.02,
        kappa=2.0,
        theta=0.02,
        sigma=0.8,
        dt=0.25,
        dW=dW,
    )

    terminal = projected_euler_terminal_from_dW(**kwargs)
    terminal_with_stats, stats = projected_euler_terminal_with_stats_from_dW(
        **kwargs
    )

    np.testing.assert_allclose(terminal_with_stats, terminal)
    # default floor is 0.5 * sigma * sqrt(dt) = 0.5 * 0.8 * 0.5
    assert stats["y_floor"] == pytest.approx(0.2)
    assert 0.0 <= stats["pre_floor_fraction"] <= 1.0
    assert 0.0 <= stats["post_projection_fraction"] <= 1.0
    assert stats["post_projection_fraction"] > 0.0


def test_batched_strong_error_matches_single_batch():
    params = {
        "kappa": 2.0,
        "theta": 0.02,
        "x0": 0.02,
        "sigma": 0.2,
        "T": 1.0,
    }
    grid = {
        "n_paths": 6,
        "reference_n_steps": 64,
        "coarse_n_steps": [8, 16],
    }

    # rho = 4 keeps h_min = h_max/rho >= dt_fine on this small 64-step grid,
    # so the KLM fine-grid guard does not fire (the test targets batching
    # equality, not the backstop floor).
    single = strong.run_regime(
        "B",
        params,
        grid,
        master_seed=1234,
        schemes=["FTE", "ProjEuler", "KL", "KLM"],
        rho=4.0,
        path_batch_size=6,
    )
    batched = strong.run_regime(
        "B",
        params,
        grid,
        master_seed=1234,
        schemes=["FTE", "ProjEuler", "KL", "KLM"],
        rho=4.0,
        path_batch_size=2,
    )

    assert len(single) == len(batched)
    for left, right in zip(single, batched):
        assert left["regime"] == right["regime"]
        assert left["scheme"] == right["scheme"]
        assert left["scheme_variant"] == right["scheme_variant"]
        for key in [
            "dt",
            "l1",
            "l2",
            "mean_steps_per_path",
            "backstop_fraction",
            "proj_pre_floor_fraction",
            "proj_post_projection_fraction",
        ]:
            left_value = left.get(key, np.nan)
            right_value = right.get(key, np.nan)
            if np.isnan(left_value) and np.isnan(right_value):
                continue
            assert left_value == pytest.approx(right_value)


def test_save_csv_uses_union_of_row_keys(tmp_path, monkeypatch):
    monkeypatch.setattr(strong, "results_path", lambda filename: tmp_path / filename)

    path = strong.save_csv(
        [
            {"scheme": "FTE", "l2": 1.0},
            {
                "scheme": "ProjEuler",
                "l2": 2.0,
                "proj_post_projection_fraction": 0.5,
            },
        ],
        "strong.csv",
    )

    with path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert "proj_post_projection_fraction" in rows[0]
    assert rows[0]["proj_post_projection_fraction"] == ""
    assert rows[1]["proj_post_projection_fraction"] == "0.5"
