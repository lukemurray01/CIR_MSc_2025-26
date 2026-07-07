# CPU/JAX parity for the fixed-step kernels used in the timing comparison.
#
# Same convention as tests/test_klm_parity.py: the NumPy samplers are the
# reference logic; given identical noise the JAX kernels must agree path by
# path to floating-point tolerance.  This is the correctness leg of the
# CPU-vs-GPU comparison (the timing leg is experiments/run_cpu_gpu_timing.py).

import numpy as np
import pytest
import yaml

jax = pytest.importorskip("jax")

from klm_jax.fixed_step import (  # noqa: E402
    blt_terminal_from_noise_jax,
    brownian_increments_with_infima_jax,
    fte_terminal_from_dW_jax,
    kl_uniform_terminal_from_dW_jax,
    projected_euler_terminal_from_dW_jax,
)
from src.samplers.blt_splitting import blt_terminal_from_noise  # noqa: E402
from src.samplers.full_truncation_euler import fte_terminal_from_dW  # noqa: E402
from src.samplers.kelly_lord import kl_uniform_terminal_from_dW  # noqa: E402
from src.samplers.projected_euler import (  # noqa: E402
    projected_euler_terminal_from_dW,
)
from src.utils.io import config_path  # noqa: E402
from src.utils.rng import (  # noqa: E402
    make_brownian_increments_with_infima,
    make_rng,
)

PARITY_TOLERANCE = dict(rtol=1e-9, atol=1e-12)


def load_config(filename: str) -> dict:
    with open(config_path(filename), encoding="utf-8") as f:
        return yaml.safe_load(f)


def shared_setup(regime_name, n_paths=300, n_steps=256, T=1.0):
    regimes = load_config("regimes.yaml")
    experiments = load_config("experiments.yaml")

    shared = regimes["shared"]
    sigma = regimes["regimes"][regime_name]["sigma"]
    master_seed = experiments["shared"]["master_seed"]

    dt = T / n_steps
    rng = make_rng(master_seed)
    dW, m = make_brownian_increments_with_infima(rng, n_paths, n_steps, dt)

    params = dict(
        X0=shared["x0"],
        kappa=shared["kappa"],
        theta=shared["theta"],
        sigma=sigma,
        dt=dt,
    )
    return params, dW, m


# FTE evaluates sqrt(max(x, 0)) whose derivative is unbounded at the kink;
# in the boundary-active regime E a 1-ulp XLA reassociation difference on a
# path near zero is amplified to ~1e-6 relative.  Parity is therefore
# asserted at 1e-9 away from the boundary and 1e-4 in regime E.
@pytest.mark.parametrize(
    ("regime_name", "rtol"), [("B", 1e-9), ("E", 1e-4)]
)
def test_fte_parity(regime_name, rtol):
    params, dW, _ = shared_setup(regime_name)

    numpy_out = fte_terminal_from_dW(dW=dW, **params)
    jax_out = fte_terminal_from_dW_jax(dW=dW, **params)

    np.testing.assert_allclose(np.asarray(jax_out), numpy_out, rtol=rtol, atol=1e-12)


@pytest.mark.parametrize("regime_name", ["B", "E"])
def test_projected_euler_parity(regime_name):
    params, dW, _ = shared_setup(regime_name)

    numpy_out = projected_euler_terminal_from_dW(dW=dW, **params)
    jax_out = projected_euler_terminal_from_dW_jax(dW=dW, **params)

    np.testing.assert_allclose(np.asarray(jax_out), numpy_out, **PARITY_TOLERANCE)


def test_kl_uniform_parity_alpha_positive():
    params, dW, _ = shared_setup("B")

    numpy_out = kl_uniform_terminal_from_dW(dW=dW, **params)
    jax_out = kl_uniform_terminal_from_dW_jax(dW=dW, **params)

    np.testing.assert_allclose(np.asarray(jax_out), numpy_out, **PARITY_TOLERANCE)


@pytest.mark.parametrize("regime_name", ["B", "E"])
def test_blt_parity(regime_name):
    params, dW, m = shared_setup(regime_name)

    numpy_out = blt_terminal_from_noise(dW=dW, m=m, **params)
    jax_out = blt_terminal_from_noise_jax(dW=dW, m=m, **params)

    np.testing.assert_allclose(np.asarray(jax_out), numpy_out, **PARITY_TOLERANCE)


def test_device_joint_noise_is_a_valid_infimum():
    key = jax.random.PRNGKey(0)
    dW, m = brownian_increments_with_infima_jax(key, 200, 64, dt=1.0 / 64)

    dW = np.asarray(dW)
    m = np.asarray(m)
    assert np.all(m <= 0.0)
    assert np.all(m <= dW)
    # FP64 is active (values would collapse at FP32 resolution otherwise).
    assert dW.dtype == np.float64
