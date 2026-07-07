# Guard tests for the integrated Kaggle streaming scripts.
#
# The scripts experiments/klm_fig2a_streaming.py and
# experiments/klm_fig3_full_diagnostic.py are verbatim integrations of the
# July 2026 Kaggle notebooks (see their header comments).  Full-scale runs
# happen on Kaggle; these tests pin the deterministic building blocks that
# the bit-reproducibility claim rests on, at a tiny configuration.
#
# Reduced-scale end-to-end equivalence against the original notebook code
# was verified at integration time (power 14 fig2a table and power 13 fig3
# CSVs bit-identical); these tests protect the pieces that make that hold:
# deterministic chunk regeneration, the bridge endpoints, backstop
# positivity, and the adaptive step rule.

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

# Tiny configuration; must be set before the module executes its globals.
os.environ.setdefault("FIG2A_M", "8")
os.environ.setdefault("FIG2A_REFERENCE_POWER", "8")
os.environ.setdefault("FIG2A_CHUNK_STEPS", "64")


def load_fig2a_module():
    spec = importlib.util.spec_from_file_location(
        "klm_fig2a_streaming",
        REPO_ROOT / "experiments" / "klm_fig2a_streaming.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def fig2a():
    return load_fig2a_module()


def test_env_overrides_applied(fig2a):
    assert fig2a.NUMBER_OF_PATHS == 8
    assert fig2a.REFERENCE_POWER == 8
    assert fig2a.FINE_STEPS_PER_CHUNK == 64
    assert fig2a.NUMBER_OF_CHUNKS == 4


def test_brownian_chunks_are_deterministic_and_distinct(fig2a):
    # Deterministic chunk regeneration is what lets pass 2 replay the exact
    # Brownian path of pass 1 without storing it.
    first = np.asarray(fig2a.make_brownian_increment_chunk(3))
    again = np.asarray(fig2a.make_brownian_increment_chunk(3))
    other = np.asarray(fig2a.make_brownian_increment_chunk(4))

    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, other)
    assert first.shape == (fig2a.FINE_STEPS_PER_CHUNK, fig2a.NUMBER_OF_PATHS)

    # Increment variance matches the fine step at the 3-sigma Monte Carlo level.
    assert np.isclose(
        first.var(), fig2a.REFERENCE_STEP, rtol=6.0 / np.sqrt(first.size)
    )


def test_implicit_backstop_stays_positive_for_extreme_increments(fig2a):
    y_old = jnp.full((5,), 0.02)
    step = 2.0**-9
    extreme_negative_dW = jnp.linspace(-1.0, -0.1, 5)

    y_new = fig2a.implicit_lamperti_step(y_old, extreme_negative_dW, step)

    assert np.all(np.asarray(y_new) > 0.0)


def test_bridge_is_exact_at_the_fine_grid_endpoints(fig2a):
    # At tau = t_left / t_right the bridge variance vanishes and the sample
    # must equal the known endpoint value exactly.
    n_paths = fig2a.NUMBER_OF_PATHS
    w_left = jnp.linspace(-1.0, 1.0, n_paths)
    w_right = w_left + 0.3
    t_left, t_right = 0.5, 0.5 + fig2a.REFERENCE_STEP
    key = jax.random.PRNGKey(0)

    levels = fig2a.NUMBER_OF_LEVELS
    at_left = fig2a.sample_brownian_bridge_value(
        w_left, w_right, t_left, t_right,
        jnp.full((levels, n_paths), t_left), key,
    )
    at_right = fig2a.sample_brownian_bridge_value(
        w_left, w_right, t_left, t_right,
        jnp.full((levels, n_paths), t_right), key,
    )

    np.testing.assert_allclose(np.asarray(at_left), np.broadcast_to(w_left, (levels, n_paths)))
    np.testing.assert_allclose(np.asarray(at_right), np.broadcast_to(w_right, (levels, n_paths)))


def test_adaptive_step_rule_bounds(fig2a):
    n_paths = fig2a.NUMBER_OF_PATHS
    levels = fig2a.NUMBER_OF_LEVELS

    # Interior state: step equals h_max * min(1, |Y|), no minimum trigger.
    y_interior = jnp.full((levels, n_paths), 0.5)
    h, used_min = fig2a.choose_adaptive_step_size(y_interior, 0.0)
    expected = np.broadcast_to(
        np.asarray(fig2a.HMAX_BY_LEVEL)[:, None] * 0.5, (levels, n_paths)
    )
    np.testing.assert_allclose(np.asarray(h), expected)
    assert not np.any(np.asarray(used_min))

    # Near-boundary state: the floor h_max / rho engages.
    y_boundary = jnp.full((levels, n_paths), 1e-6)
    h, used_min = fig2a.choose_adaptive_step_size(y_boundary, 0.0)
    np.testing.assert_allclose(
        np.asarray(h), np.asarray(fig2a.HMAX_BY_LEVEL)[:, None] / fig2a.RHO * np.ones((levels, n_paths))
    )
    assert np.all(np.asarray(used_min))

    # Terminal clipping: remaining time caps the step.
    h, _ = fig2a.choose_adaptive_step_size(y_interior, fig2a.FINAL_TIME - 1e-4)
    assert np.all(np.asarray(h) <= 1e-4 + 1e-15)
