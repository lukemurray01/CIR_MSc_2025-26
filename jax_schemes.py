# JAX implementations of the thesis schemes: the accelerated analogues of the
# NumPy reference samplers in src/samplers/.
#
# Fixed-step kernels (FTE, projected Euler, uniform Kelly--Lord, BLT
# splitting) mirror their NumPy twins step for step, so given the same
# increments the two implementations must agree to floating-point tolerance
# (tests/test_jax_fixed_step_parity.py).
#
# The KLM backstopped adaptive kernel mirrors
# src/samplers/klm_backstop.klm_backstop_terminal_from_fine_dW: identical
# step policy (h = h_max * min(1, |Y|), floor h_min = h_max / rho), identical
# backstop triggers, and step sizes quantised to the same fine Brownian grid.
# Given the same fine increments the two implementations must agree to
# floating-point tolerance; tests/test_klm_parity.py asserts this CPU/JAX
# parity, which is the validation demanded by the thesis reproducibility
# pillar.
#
# All arrays are float64 (jax_enable_x64); FP32 is insufficient for
# strong-error diagnostics at the 1e-5 error scale (thesis background
# chapter, computational-cost section).

from functools import partial

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Fixed-step scheme kernels (NumPy twins in src/samplers/)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Device-side noise generation
# ---------------------------------------------------------------------------


def brownian_increments_jax(key, n_paths, n_steps, dt):
    """Device-side Brownian increments, shape (n_paths, n_steps)."""
    return (dt ** 0.5) * jax.random.normal(key, (n_paths, n_steps), dtype=jnp.float64)


@partial(jax.jit, static_argnums=(1, 2))
def brownian_increments_with_infima_jax(key, n_paths, n_steps, dt):
    """Device-side joint (increment, running infimum) pairs.

    Same Asmussen--Glynn--Pitman construction as
    src.utils.rng.make_brownian_increments_with_infima:
    m = (dW - sqrt(V + dW^2)) / 2 with V ~ Exp(1/(2*dt)) independent.
    Jitted so XLA fuses the elementwise chain: run eagerly, V and two
    temporaries are materialised as full-size FP64 buffers, which exceeds
    the 16 GB P100 pool at 1e6 paths x 512 steps.
    """
    key_w, key_v = jax.random.split(key)
    dW = brownian_increments_jax(key_w, n_paths, n_steps, dt)
    V = 2.0 * dt * jax.random.exponential(key_v, (n_paths, n_steps), dtype=jnp.float64)
    m = 0.5 * (dW - jnp.sqrt(V + dW * dW))
    return dW, m


# ---------------------------------------------------------------------------
# KLM backstopped adaptive scheme and drift-implicit reference
# ---------------------------------------------------------------------------


def implicit_backstop_step(y, h, dW, alpha, beta, gamma):
    """Drift-implicit Lamperti step; strictly positive for alpha > 0."""
    a = 1.0 - beta * h
    u = y + gamma * dW

    return (u + jnp.sqrt(u * u + 4.0 * a * alpha * h)) / (2.0 * a)


def projected_backstop_step(y, h, dW, alpha, beta, gamma):
    """Explicit step clipped to the strictly positive floor gamma * sqrt(h)."""
    y_floor = gamma * jnp.sqrt(h)
    y_hat = y + (alpha / y + beta * y) * h + gamma * dW

    return jnp.maximum(y_hat, y_floor)


def klm_backstop_terminal_from_fine_dW_jax(
    X0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
    h_max: float,
    dW_fine,
    rho: float = 64.0,
    backstop: str | None = None,
):
    """Brownian-coupled KLM scheme on the fine grid; returns (X_T, stats).

    dW_fine has shape (n_paths, n_fine).  Finished paths take zero-length
    steps (m = 0), which leave the state unchanged, so no explicit masking
    of the state update is required; counters are masked by activity.
    """
    alpha = (4.0 * kappa * theta - sigma * sigma) / 8.0
    beta = -kappa / 2.0
    gamma = sigma / 2.0

    if backstop is None:
        backstop = "implicit" if alpha > 0.0 else "projected"
    if backstop == "implicit" and alpha <= 0.0:
        raise ValueError(
            "implicit backstop requires alpha > 0 (delta > 1); "
            "use backstop='projected' in this regime"
        )
    backstop_fn = (
        implicit_backstop_step if backstop == "implicit" else projected_backstop_step
    )

    dW_fine = jnp.asarray(dW_fine, dtype=jnp.float64)
    n_paths, n_fine = dW_fine.shape
    dt_fine = T / n_fine

    h_min = h_max / rho
    m_min = max(int(round(h_min / dt_fine)), 1)
    m_max = max(int(jnp.floor(h_max / dt_fine)), 1)

    # Bare cumulative sums; W_q is recovered as cs[q-1] with W_0 = 0.  Building
    # the padded (n_paths, n_fine + 1) path array via concatenate keeps three
    # full-size FP64 buffers live at once (dW_fine, the cumsum, the result),
    # which exceeds the 16 GB P100 pool at 1e6 paths x 512 steps.
    cs = jnp.cumsum(dW_fine, axis=1)

    y0 = jnp.full((n_paths,), jnp.sqrt(X0), dtype=jnp.float64)
    pos0 = jnp.zeros((n_paths,), dtype=jnp.int64)
    counters0 = jnp.zeros(3, dtype=jnp.int64)  # steps, min-trigger, neg-retake

    rows = jnp.arange(n_paths)

    def w_at(pos):
        return jnp.where(pos > 0, cs[rows, jnp.maximum(pos - 1, 0)], 0.0)

    def cond(state):
        _, pos, _ = state
        return jnp.any(pos < n_fine)

    def body(state):
        y, pos, counters = state
        active = pos < n_fine

        h_prop = h_max * jnp.minimum(1.0, jnp.abs(y))
        m = jnp.floor(h_prop / dt_fine).astype(jnp.int64)

        min_triggered = m < m_min
        m = jnp.where(min_triggered, m_min, jnp.minimum(m, m_max))
        m = jnp.minimum(m, n_fine - pos)

        h = m * dt_fine
        dW = w_at(pos + m) - w_at(pos)

        # Zero-length steps (finished paths) keep y fixed through both maps.
        y_explicit = jnp.where(
            h > 0.0,
            y + (alpha / y + beta * y) * h + gamma * dW,
            y,
        )
        y_backstop = jnp.where(h > 0.0, backstop_fn(y, h, dW, alpha, beta, gamma), y)

        neg_retake = (~min_triggered) & (y_explicit <= 0.0)
        use_backstop = min_triggered | neg_retake

        y_next = jnp.where(use_backstop, y_backstop, y_explicit)

        counters = counters + jnp.array(
            [
                jnp.sum(active),
                jnp.sum(active & min_triggered),
                jnp.sum(active & neg_retake),
            ],
            dtype=jnp.int64,
        )

        return y_next, pos + m, counters

    y, pos, counters = jax.lax.while_loop(cond, body, (y0, pos0, counters0))

    n_steps_total = int(counters[0])
    stats = {
        "n_steps_total": n_steps_total,
        "n_backstop_min": int(counters[1]),
        "n_backstop_neg": int(counters[2]),
        "backstop_fraction": (int(counters[1]) + int(counters[2]))
        / max(n_steps_total, 1),
        "backstop_kind": backstop,
        "m_min_fine_steps": m_min,
    }
    return y * y, stats


def if_terminal_from_fine_dW_jax(
    X0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
    dW_fine,
):
    """Drift-implicit reference on the uniform fine grid (JAX)."""
    alpha = (4.0 * kappa * theta - sigma * sigma) / 8.0
    beta = -kappa / 2.0
    gamma = sigma / 2.0

    dW_fine = jnp.asarray(dW_fine, dtype=jnp.float64)
    n_paths, n_fine = dW_fine.shape
    dt = T / n_fine

    def step(y, dW_col):
        return implicit_backstop_step(y, dt, dW_col, alpha, beta, gamma), None

    y0 = jnp.full((n_paths,), jnp.sqrt(X0), dtype=jnp.float64)
    y, _ = jax.lax.scan(step, y0, dW_fine.T)

    return y * y
