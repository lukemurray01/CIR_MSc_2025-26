import jax.numpy as jnp

def sigma_from_a(a, kappa, lam):
    # Feller ratio a = sigma^{2} / (2 * kappa * lambda)
    # so sigma = \sqrt{2 * kappa * lambda * a}
    return jnp.sqrt(2.0 * kappa * lam * a)