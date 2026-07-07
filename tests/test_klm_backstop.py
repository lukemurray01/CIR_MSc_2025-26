import warnings

import numpy as np
import pytest
import yaml

from src.metrics.weak_error import exact_cir_mean
from src.samplers.klm_backstop import (
    _backstop_map,
    default_backstop_kind,
    klm_backstop_terminal,
    klm_backstop_terminal_from_fine_dW,
)
from src.samplers.lamperti_implicit import if_terminal_from_dW
from src.utils.cir_params import kl_alpha
from src.utils.io import config_path
from src.utils.rng import make_rng


def load_config(filename: str) -> dict:
    with open(config_path(filename), encoding="utf-8") as f:
        return yaml.safe_load(f)


def shared_params():
    regimes = load_config("regimes.yaml")
    experiments = load_config("experiments.yaml")
    return regimes, experiments["shared"]["master_seed"]


def test_default_backstop_kind_switches_on_alpha_sign():
    assert default_backstop_kind(0.01) == "implicit"
    assert default_backstop_kind(0.0) == "projected"
    assert default_backstop_kind(-0.01) == "projected"


def test_implicit_backstop_rejects_negative_alpha():
    with pytest.raises(ValueError):
        _backstop_map(
            y=np.array([0.1]),
            h=np.array([0.01]),
            dW=np.array([0.0]),
            alpha=-0.06,
            beta=-1.0,
            gamma=0.4,
            kind="implicit",
        )


def test_backstop_maps_return_strictly_positive_values():
    # Large negative Brownian increments would drive the explicit map negative;
    # both admissible backstops must stay strictly positive (condition (ii)).
    y = np.full(5, 0.05)
    h = np.full(5, 0.01)
    dW = np.linspace(-0.5, -0.1, 5)

    implicit = _backstop_map(y, h, dW, alpha=0.01875, beta=-1.0, gamma=0.05, kind="implicit")
    projected = _backstop_map(y, h, dW, alpha=-0.06, beta=-1.0, gamma=0.4, kind="projected")

    assert np.all(implicit > 0.0)
    assert np.all(projected > 0.0)


def test_klm_terminal_strictly_positive_all_regimes():
    regimes, master_seed = shared_params()
    shared = regimes["shared"]

    for i, (name, regime) in enumerate(regimes["regimes"].items()):
        rng = make_rng(master_seed + i)
        terminal, stats = klm_backstop_terminal(
            X0=shared["x0"],
            kappa=shared["kappa"],
            theta=shared["theta"],
            sigma=regime["sigma"],
            T=1.0,
            h_max=1.0 / 32,
            n_paths=2000,
            rng=rng,
        )

        assert terminal.shape == (2000,)
        assert np.all(np.isfinite(terminal)), f"non-finite terminal in regime {name}"
        assert np.all(terminal > 0.0), f"non-positive terminal in regime {name}"
        assert stats["n_steps_total"] > 0


def test_klm_terminal_mean_matches_exact_mean_in_design_regimes():
    # The KLM construction assumes alpha > 0 (delta > 1); regimes A-C satisfy
    # this and the implicit backstop applies.  Measured relative errors at
    # h_max = 1/128 with 20000 paths are 0.24% (A), 0.21% (B), 0.70% (C).
    regimes, master_seed = shared_params()
    shared = regimes["shared"]
    exact = exact_cir_mean(shared["x0"], shared["kappa"], shared["theta"], 1.0)

    for i, regime_name in enumerate(["A", "B", "C"]):
        sigma = regimes["regimes"][regime_name]["sigma"]
        rng = make_rng(master_seed + i)
        terminal, stats = klm_backstop_terminal(
            X0=shared["x0"],
            kappa=shared["kappa"],
            theta=shared["theta"],
            sigma=sigma,
            T=1.0,
            h_max=1.0 / 128,
            n_paths=20000,
            rng=rng,
        )

        assert stats["backstop_kind"] == "implicit"
        rel_error = abs(np.mean(terminal) - exact) / exact
        assert rel_error < 0.05, (
            f"KLM mean check failed in regime {regime_name}: "
            f"sample mean={np.mean(terminal):.6f}, exact={exact:.6f}, "
            f"relative error={rel_error:.3%}"
        )


def test_klm_outside_design_regime_runs_but_bias_is_documented():
    # For alpha < 0 (regimes D, E) the KLM design premise fails: the backstop
    # becomes the typical mode (>50% of steps in E) and the projected
    # fallback introduces a positive terminal-mean bias that decays only
    # slowly with h_max (measured: +33% at h_max=1/128, +17% at 1/1024,
    # 10000 paths).  See notes/diagnostics.md (01.07.2026).  This test pins
    # the documented behaviour so any silent change is caught; it is NOT an
    # accuracy guarantee.
    regimes, master_seed = shared_params()
    shared = regimes["shared"]
    sigma = regimes["regimes"]["E"]["sigma"]
    exact = exact_cir_mean(shared["x0"], shared["kappa"], shared["theta"], 1.0)

    rng = make_rng(master_seed)
    terminal, stats = klm_backstop_terminal(
        X0=shared["x0"],
        kappa=shared["kappa"],
        theta=shared["theta"],
        sigma=sigma,
        T=1.0,
        h_max=1.0 / 128,
        n_paths=10000,
        rng=rng,
    )

    assert stats["backstop_kind"] == "projected"
    assert np.all(terminal > 0.0)
    # Backstop is the typical mode outside the design regime.
    assert stats["backstop_fraction"] > 0.25
    # Positive bias present but bounded; tracked as a documented limitation.
    rel_bias = (np.mean(terminal) - exact) / exact
    assert 0.0 < rel_bias < 0.60, f"documented regime-E bias changed: {rel_bias:.3%}"


def test_klm_backstop_usage_rare_in_regime_A_heavy_in_regime_E():
    regimes, master_seed = shared_params()
    shared = regimes["shared"]

    fractions = {}
    for regime_name in ["A", "E"]:
        sigma = regimes["regimes"][regime_name]["sigma"]
        rng = make_rng(master_seed)
        _, stats = klm_backstop_terminal(
            X0=shared["x0"],
            kappa=shared["kappa"],
            theta=shared["theta"],
            sigma=sigma,
            T=1.0,
            h_max=1.0 / 64,
            n_paths=2000,
            rng=rng,
        )
        fractions[regime_name] = stats["backstop_fraction"]

    assert fractions["A"] < 0.05, f"regime A backstop fraction {fractions['A']:.3%}"
    assert fractions["E"] > fractions["A"], (
        "backstop should fire more often in the boundary-accessible regime: "
        f"A={fractions['A']:.3%}, E={fractions['E']:.3%}"
    )


def test_klm_terminal_deterministic_given_seed():
    regimes, master_seed = shared_params()
    shared = regimes["shared"]
    sigma = regimes["regimes"]["B"]["sigma"]

    outputs = []
    for _ in range(2):
        rng = make_rng(master_seed)
        terminal, _ = klm_backstop_terminal(
            X0=shared["x0"],
            kappa=shared["kappa"],
            theta=shared["theta"],
            sigma=sigma,
            T=1.0,
            h_max=1.0 / 32,
            n_paths=500,
            rng=rng,
        )
        outputs.append(terminal)

    np.testing.assert_array_equal(outputs[0], outputs[1])


def test_klm_coupled_error_decreases_with_h_max():
    regimes, master_seed = shared_params()
    shared = regimes["shared"]
    sigma = regimes["regimes"]["B"]["sigma"]

    T = 1.0
    reference_n_steps = 2048
    dt_fine = T / reference_n_steps
    n_paths = 4000

    rng = make_rng(master_seed)
    dW_fine = np.sqrt(dt_fine) * rng.standard_normal((n_paths, reference_n_steps))

    reference = if_terminal_from_dW(
        X0=shared["x0"],
        kappa=shared["kappa"],
        theta=shared["theta"],
        sigma=sigma,
        dt=dt_fine,
        dW=dW_fine,
    )

    errors = {}
    for h_max in [1.0 / 8, 1.0 / 64]:
        # rho = 32 keeps h_min = h_max/rho >= dt_fine at both levels, so the
        # backstop floor is representable on the fine grid (no guard warning).
        terminal, stats = klm_backstop_terminal_from_fine_dW(
            X0=shared["x0"],
            kappa=shared["kappa"],
            theta=shared["theta"],
            sigma=sigma,
            T=T,
            h_max=h_max,
            dW_fine=dW_fine,
            rho=32.0,
        )
        assert np.all(terminal > 0.0)
        errors[h_max] = np.sqrt(np.mean((terminal - reference) ** 2))

    assert errors[1.0 / 64] < errors[1.0 / 8], (
        f"coupled L2 error did not decrease: {errors}"
    )


def test_klm_coupled_alpha_negative_regime_runs_with_projected_backstop():
    regimes, master_seed = shared_params()
    shared = regimes["shared"]
    sigma = regimes["regimes"]["E"]["sigma"]
    assert kl_alpha(shared["kappa"], shared["theta"], sigma) < 0.0

    T = 1.0
    reference_n_steps = 1024
    dt_fine = T / reference_n_steps
    n_paths = 1000

    rng = make_rng(master_seed)
    dW_fine = np.sqrt(dt_fine) * rng.standard_normal((n_paths, reference_n_steps))

    terminal, stats = klm_backstop_terminal_from_fine_dW(
        X0=shared["x0"],
        kappa=shared["kappa"],
        theta=shared["theta"],
        sigma=sigma,
        T=T,
        h_max=1.0 / 16,
        dW_fine=dW_fine,
    )

    assert stats["backstop_kind"] == "projected"
    assert np.all(np.isfinite(terminal))
    assert np.all(terminal > 0.0)


def test_klm_coupled_warns_when_fine_grid_coarser_than_h_min():
    regimes, master_seed = shared_params()
    shared = regimes["shared"]
    sigma = regimes["regimes"]["B"]["sigma"]

    T = 1.0
    reference_n_steps = 64  # dt_fine = 1/64
    rng = make_rng(master_seed)
    dt_fine = T / reference_n_steps
    dW_fine = np.sqrt(dt_fine) * rng.standard_normal((50, reference_n_steps))

    common = dict(
        X0=shared["x0"],
        kappa=shared["kappa"],
        theta=shared["theta"],
        sigma=sigma,
        T=T,
        dW_fine=dW_fine,
    )

    # h_min = (1/8)/64 = 1/512 < dt_fine = 1/64: floor not representable.
    with pytest.warns(RuntimeWarning, match="fine grid coarser"):
        klm_backstop_terminal_from_fine_dW(h_max=1.0 / 8, rho=64.0, **common)

    # h_min = (1/8)/8 = 1/64 = dt_fine: representable, no warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        klm_backstop_terminal_from_fine_dW(h_max=1.0 / 8, rho=8.0, **common)
