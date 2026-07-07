import numpy as np

from klm_jax.coefficients import sigma_from_a


def test_sigma_from_a_matches_formula():
    sigma = sigma_from_a(1.0, 2.0, 0.05)
    np.testing.assert_allclose(sigma, np.sqrt(2.0 * 2.0 * 0.05))