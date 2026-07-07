# JAX implementation of the KLM backstopped adaptive scheme.
#
# Mirrors src/samplers/klm_backstop.klm_backstop_terminal_from_fine_dW:
# identical step policy (h = h_max * min(1, |Y|), floor h_min = h_max / rho),
# identical backstop triggers, and step sizes quantised to the same fine
# Brownian grid.  Given the same fine increments the two implementations
# must agree to floating-point tolerance; tests/test_klm_parity.py asserts
# this CPU/JAX parity, which is the validation demanded by the thesis
# reproducibility pillar.
#
# All arrays are float64 (jax_enable_x64); FP32 is insufficient for
# strong-error diagnostics at the 1e-5 error scale (thesis background
# chapter, computational-cost section).

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


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

    W = jnp.concatenate(
        [jnp.zeros((n_paths, 1), dtype=jnp.float64), jnp.cumsum(dW_fine, axis=1)],
        axis=1,
    )

    y0 = jnp.full((n_paths,), jnp.sqrt(X0), dtype=jnp.float64)
    pos0 = jnp.zeros((n_paths,), dtype=jnp.int64)
    counters0 = jnp.zeros(3, dtype=jnp.int64)  # steps, min-trigger, neg-retake

    rows = jnp.arange(n_paths)

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
        dW = W[rows, pos + m] - W[rows, pos]

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
