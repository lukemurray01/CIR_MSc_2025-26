import json
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from experiments.kl_adaptive_splitting_paper_jax import (
    FINAL_TIME,
    KAPPA,
    THETA,
    X0,
    adaptive_splitting_terminal_jax,
    fte_terminal_jax,
    hh_terminal_jax,
    implicit_terminal_jax,
    projected_terminal_jax,
    splitting_terminal_jax,
)


def tiny_dW() -> jnp.ndarray:
    return jnp.array(
        [
            [0.02, -0.01, 0.03, -0.02],
            [-0.03, 0.04, -0.02, 0.01],
            [0.01, 0.02, -0.01, -0.02],
        ],
        dtype=jnp.float64,
    )


def test_fixed_jax_scheme_kernels_return_finite_terminals():
    dW = tiny_dW()
    sigma = 0.3
    dt = 0.25

    terminals = [
        hh_terminal_jax(dW, X0, KAPPA, THETA, sigma, dt),
        fte_terminal_jax(dW, X0, KAPPA, THETA, sigma, dt),
        projected_terminal_jax(dW, X0, KAPPA, THETA, sigma, dt),
        implicit_terminal_jax(dW, X0, KAPPA, THETA, sigma, dt),
        splitting_terminal_jax(dW, X0, KAPPA, THETA, sigma, dt),
    ]

    for terminal in terminals:
        arr = np.asarray(terminal)
        assert arr.shape == (3,)
        assert np.all(np.isfinite(arr))
        assert np.all(arr >= 0.0)


@pytest.mark.parametrize("mode,sigma", [("heuristic", 0.3), ("soft_zero", 0.5)])
def test_adaptive_jax_kernel_reaches_terminal_time(mode, sigma):
    n_paths = 4
    n_ref = 20
    dt_ref = FINAL_TIME / n_ref
    W_grid = jnp.zeros((n_paths, n_ref + 1), dtype=jnp.float64)

    terminal, mean_dt, det_count, reached, rounds = adaptive_splitting_terminal_jax(
        W_grid,
        jax.random.PRNGKey(123),
        dt_ref,
        KAPPA,
        THETA,
        sigma,
        X0,
        0.1,
        mode=mode,
        max_rounds=1000,
    )

    assert bool(np.asarray(reached))
    assert int(np.asarray(rounds)) > 0
    assert np.asarray(terminal).shape == (n_paths,)
    assert np.all(np.asarray(terminal) >= 0.0)
    assert np.all(np.asarray(mean_dt) > 0.0)
    assert np.asarray(det_count).shape == (n_paths,)


def test_jax_kaggle_notebook_is_single_cell_and_standalone():
    notebook = Path("notebooks/kaggle/kaggle_kl_adaptive_splitting_paper_JAX.ipynb")
    nb = json.loads(notebook.read_text(encoding="utf-8"))
    source = "".join(nb["cells"][0]["source"])

    assert len(nb["cells"]) == 1
    assert nb["cells"][0]["cell_type"] == "code"
    assert "from src.utils.io" not in source
    assert "REPO_ROOT" not in source
    assert "jax.config.update(\"jax_enable_x64\", True)" in source
    assert "main(RUN_ARGS)" in source
