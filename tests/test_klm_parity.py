# CPU/JAX parity for the KLM backstopped adaptive scheme.
#
# The NumPy implementation in src/samplers/klm_backstop.py is the reference
# logic; the JAX implementation in klm_jax/backstop.py is the accelerated
# logic.  Both consume the same fine Brownian increments with the same
# grid-quantised step policy, so their terminal values must agree to
# floating-point tolerance path by path.  This is the CPU/JAX validation
# required by the thesis reproducibility pillar.

import numpy as np
import pytest
import yaml

jax = pytest.importorskip("jax")

from klm_jax.backstop import (  # noqa: E402
    if_terminal_from_fine_dW_jax,
    klm_backstop_terminal_from_fine_dW_jax,
)
from src.samplers.klm_backstop import (  # noqa: E402
    klm_backstop_terminal_from_fine_dW,
)
from src.samplers.lamperti_implicit import if_terminal_from_dW  # noqa: E402
from src.utils.io import config_path  # noqa: E402
from src.utils.rng import make_rng  # noqa: E402

PARITY_TOLERANCE = dict(rtol=1e-9, atol=1e-12)


def load_config(filename: str) -> dict:
    with open(config_path(filename), encoding="utf-8") as f:
        return yaml.safe_load(f)


def shared_setup(regime_name, n_paths=200, reference_n_steps=1024, T=1.0):
    regimes = load_config("regimes.yaml")
    experiments = load_config("experiments.yaml")

    shared = regimes["shared"]
    sigma = regimes["regimes"][regime_name]["sigma"]
    master_seed = experiments["shared"]["master_seed"]

    dt_fine = T / reference_n_steps
    rng = make_rng(master_seed)
    dW_fine = np.sqrt(dt_fine) * rng.standard_normal((n_paths, reference_n_steps))

    params = dict(
        X0=shared["x0"],
        kappa=shared["kappa"],
        theta=shared["theta"],
        sigma=sigma,
        T=T,
    )
    return params, dW_fine


@pytest.mark.parametrize("regime_name", ["A", "B", "C"])
def test_klm_coupled_parity_implicit_backstop(regime_name):
    params, dW_fine = shared_setup(regime_name)

    numpy_terminal, numpy_stats = klm_backstop_terminal_from_fine_dW(
        **params, h_max=1.0 / 16, dW_fine=dW_fine
    )
    jax_terminal, jax_stats = klm_backstop_terminal_from_fine_dW_jax(
        **params, h_max=1.0 / 16, dW_fine=dW_fine
    )

    np.testing.assert_allclose(
        np.asarray(jax_terminal), numpy_terminal, **PARITY_TOLERANCE
    )
    assert numpy_stats["backstop_kind"] == jax_stats["backstop_kind"] == "implicit"
    assert numpy_stats["n_steps_total"] == jax_stats["n_steps_total"]
    assert numpy_stats["n_backstop_min"] == jax_stats["n_backstop_min"]
    assert numpy_stats["n_backstop_neg"] == jax_stats["n_backstop_neg"]


def test_klm_coupled_parity_projected_backstop_regime_E():
    params, dW_fine = shared_setup("E")

    numpy_terminal, numpy_stats = klm_backstop_terminal_from_fine_dW(
        **params, h_max=1.0 / 16, dW_fine=dW_fine
    )
    jax_terminal, jax_stats = klm_backstop_terminal_from_fine_dW_jax(
        **params, h_max=1.0 / 16, dW_fine=dW_fine
    )

    np.testing.assert_allclose(
        np.asarray(jax_terminal), numpy_terminal, **PARITY_TOLERANCE
    )
    assert numpy_stats["backstop_kind"] == jax_stats["backstop_kind"] == "projected"
    assert numpy_stats["n_steps_total"] == jax_stats["n_steps_total"]


def test_if_reference_parity():
    params, dW_fine = shared_setup("B")
    dt_fine = params["T"] / dW_fine.shape[1]

    numpy_reference = if_terminal_from_dW(
        X0=params["X0"],
        kappa=params["kappa"],
        theta=params["theta"],
        sigma=params["sigma"],
        dt=dt_fine,
        dW=dW_fine,
    )
    jax_reference = if_terminal_from_fine_dW_jax(**params, dW_fine=dW_fine)

    np.testing.assert_allclose(
        np.asarray(jax_reference), numpy_reference, **PARITY_TOLERANCE
    )
