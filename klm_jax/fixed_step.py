# JAX kernels for the fixed-step thesis schemes (FTE, projected Euler,
# uniform Kelly--Lord, BLT splitting) plus device-side noise generation.
#
# Each kernel mirrors its NumPy reference in src/samplers/ step for step, so
# given the same increments the two implementations must agree to
# floating-point tolerance (tests/test_jax_fixed_step_parity.py).  The KLM
# adaptive kernel lives in klm_jax/backstop.py.
#
# All arrays are float64 (jax_enable_x64); FP32 is insufficient at the
# strong-error scales used in the thesis benchmarks.

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def fte_terminal_from_dW_jax(X0, kappa, theta, sigma, dt, dW):
    """Full truncation Euler; mirrors fte_terminal_from_dW."""
    dW = jnp.asarray(dW, dtype=jnp.float64)

    def step(x_aux, dW_col):
        x_pos = jnp.maximum(x_aux, 0.0)
        x_aux = x_aux + kappa * (theta - x_pos) * dt + sigma * jnp.sqrt(x_pos) * dW_col
        return x_aux, None

    x0 = jnp.full((dW.shape[0],), X0, dtype=jnp.float64)
    x_aux, _ = jax.lax.scan(step, x0, dW.T)

    return jnp.maximum(x_aux, 0.0)


def projected_euler_terminal_from_dW_jax(X0, kappa, theta, sigma, dt, dW, y_floor=None):
    """Projected Euler on the Lamperti scale; mirrors the NumPy default
    floor y_floor = 0.5 * sigma * sqrt(dt)."""
    dW = jnp.asarray(dW, dtype=jnp.float64)
    alpha = (4.0 * kappa * theta - sigma * sigma) / 8.0
    if y_floor is None:
        y_floor = 0.5 * sigma * dt ** 0.5

    def step(y, dW_col):
        y_safe = jnp.maximum(y, y_floor)
        y_hat = (
            y_safe
            + (alpha / y_safe - 0.5 * kappa * y_safe) * dt
            + 0.5 * sigma * dW_col
        )
        return jnp.maximum(y_hat, y_floor), None

    y0 = jnp.full((dW.shape[0],), max(float(X0) ** 0.5, float(y_floor)), dtype=jnp.float64)
    y, _ = jax.lax.scan(step, y0, dW.T)

    return y * y


def kl_uniform_terminal_from_dW_jax(X0, kappa, theta, sigma, dt, dW):
    """Uniform Kelly--Lord splitting; valid for alpha >= 0 (regimes A-C)."""
    dW = jnp.asarray(dW, dtype=jnp.float64)
    alpha = (4.0 * kappa * theta - sigma * sigma) / 8.0
    decay = jnp.exp(-kappa * dt)

    def step(x, dW_col):
        inside = x + 2.0 * alpha * dt
        x_next = decay * (jnp.sqrt(inside) + 0.5 * sigma * dW_col) ** 2
        return x_next, None

    x0 = jnp.full((dW.shape[0],), X0, dtype=jnp.float64)
    x, _ = jax.lax.scan(step, x0, dW.T)

    return x


def blt_terminal_from_noise_jax(X0, kappa, theta, sigma, dt, dW, m):
    """BLT splitting; mirrors blt_terminal_from_noise (modified for a < 1)."""
    dW = jnp.asarray(dW, dtype=jnp.float64)
    m = jnp.asarray(m, dtype=jnp.float64)

    a = 4.0 * kappa * theta / (sigma * sigma)
    b = kappa
    scale = sigma * sigma / 4.0

    decay = jnp.exp(-b * dt)
    ode_shift = (
        (a - 1.0) * dt if b == 0.0 else (a - 1.0) * (1.0 - decay) / b
    )

    def step(x, noise_col):
        dW_col, m_col = noise_col
        root = jnp.sqrt(jnp.maximum(x, 0.0))
        reflection = jnp.maximum(0.0, -(root + m_col))
        bessel = (root + dW_col + reflection) ** 2
        return bessel * decay + ode_shift, None

    x0 = jnp.full((dW.shape[0],), X0 / scale, dtype=jnp.float64)
    x, _ = jax.lax.scan(step, x0, (dW.T, m.T))

    return scale * jnp.maximum(x, 0.0)


def brownian_increments_jax(key, n_paths, n_steps, dt):
    """Device-side Brownian increments, shape (n_paths, n_steps)."""
    return (dt ** 0.5) * jax.random.normal(key, (n_paths, n_steps), dtype=jnp.float64)


def brownian_increments_with_infima_jax(key, n_paths, n_steps, dt):
    """Device-side joint (increment, running infimum) pairs.

    Same Asmussen--Glynn--Pitman construction as
    src.utils.rng.make_brownian_increments_with_infima:
    m = (dW - sqrt(V + dW^2)) / 2 with V ~ Exp(1/(2*dt)) independent.
    """
    key_w, key_v = jax.random.split(key)
    dW = brownian_increments_jax(key_w, n_paths, n_steps, dt)
    V = 2.0 * dt * jax.random.exponential(key_v, (n_paths, n_steps), dtype=jnp.float64)
    m = 0.5 * (dW - jnp.sqrt(V + dW * dW))
    return dW, m
