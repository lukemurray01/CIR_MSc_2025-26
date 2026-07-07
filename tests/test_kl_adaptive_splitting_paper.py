import math

import numpy as np
import pytest

from experiments.kl_adaptive_splitting_paper import (
    ExperimentConfig,
    key_sigma_values,
    projected_floor,
    rate_records,
    simulate_sigma,
    soft_zero_threshold,
)


def tiny_config(sigmas=(0.3,)) -> ExperimentConfig:
    return ExperimentConfig(
        n_paths=8,
        n_batches=4,
        dt_ref=0.01,
        dtmax_values=(0.1, 0.05),
        sigmas=tuple(sigmas),
        fig3_sigma=0.3,
        seed=2023,
        output_stem="pytest_kl_paper_smoke",
    )


def test_projected_floor_matches_paper_n_minus_quarter():
    assert projected_floor(1.0e-4) == pytest.approx(0.1)
    assert projected_floor(0.01) == pytest.approx(0.01**0.25)


def test_key_sigma_values_match_kl_paper_table():
    values = key_sigma_values(kappa=2.0, theta=0.02)

    assert values["projected 1/4"] == pytest.approx(math.sqrt(0.08 / 3.0))
    assert values["sqrt(kappa theta)"] == pytest.approx(0.2)
    assert values["Feller"] == pytest.approx(math.sqrt(0.08))
    assert values["alpha=0"] == pytest.approx(0.4)


def test_soft_zero_threshold_matches_formula():
    threshold = soft_zero_threshold(kappa=2.0, theta=0.02, dt_max=0.01, rho=2.0)
    expected = 0.02 * (1.0 - math.exp(-2.0 * 0.01)) / 2.0

    assert threshold == pytest.approx(expected)


@pytest.mark.parametrize("sigma", [0.3, 0.5])
def test_simulate_sigma_smoke_produces_finite_errors_and_rates(sigma):
    config = tiny_config(sigmas=(sigma,))
    rows = simulate_sigma(sigma, 0, config)

    assert rows
    finite_errors = [row for row in rows if np.isfinite(float(row["error"]))]
    assert finite_errors
    assert {"L1", "L2"}.issubset({row["metric"] for row in rows})
    assert "Splitting/Adaptive" in {row["method"] for row in finite_errors}

    rates = rate_records(rows)
    finite_rates = [row for row in rates if np.isfinite(float(row["rate"]))]
    assert finite_rates
