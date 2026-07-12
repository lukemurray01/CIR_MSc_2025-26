# Tests for random-number generators

import numpy as np
import pytest
from src.utils.rng import make_rng, make_brownian_increments

def test_make_rng_reproducible():
    rng1 = make_rng(123)
    rng2 = make_rng(123)
    np.testing.assert_allclose(rng1.standard_normal(5),
                               rng2.standard_normal(5))

def test_make_rng_rejects_neg_seed():
    with pytest.raises(ValueError):
        make_rng(-1)

def test_make_brownian_increments_shape():
    rng = make_rng(123)
    dW = make_brownian_increments(rng, n_paths=100, n_steps=50, dt=0.01)
    assert dW.shape == (100, 50)

def test_make_brownian_increments_variance():
    rng = make_rng(123)
    dW = make_brownian_increments(rng, n_paths=5000, n_steps=200, dt=0.01)
    assert abs(dW.var() - 0.01) < 0.001