# Backstopped adaptive scheme of Kelly--Lord--Maulana (2022) for CIR.
#
# The scheme works in the Lamperti coordinate Y = sqrt(X), where
#
#   dY = (alpha / Y + beta * Y) dt + gamma dW,
#   alpha = (4*kappa*theta - sigma^2) / 8,  beta = -kappa/2,  gamma = sigma/2.
#
# At each step the adaptive rule proposes  h = h_max * min(1, |Y_n|)  with a
# floor h_min = h_max / rho  (KLM Def. 7).  The backstop map is invoked in two
# cases (KLM Def. 11):
#
#   (a) the rule proposes h < h_min          -> backstop step at h_min;
#   (b) the explicit Lamperti Euler step
#       returns Y <= 0                       -> retake the SAME step (same h,
#                                               same Brownian increment) with
#                                               the backstop map.
#
# Backstop maps (see thesis chapter "Positivity in the backstop convergence
# proof", Admissible backstops):
#
#   "implicit"  : drift-implicit Lamperti step.  Strictly positive for every
#                 Brownian increment when alpha > 0 (delta > 1).  Not defined
#                 for alpha <= 0, where the quadratic can lose its real root.
#   "projected" : explicit step clipped to a strictly positive floor
#                 y_floor = gamma * sqrt(h).  Always defined; used as the
#                 default in the boundary-accessible regimes D and E where
#                 alpha < 0 and the KLM theory does not apply anyway.
#   "blt"       : one step of the BLT splitting scheme (Brehier--Cohen--
#                 Herzwurm--Kelly--Neuenkirch, in preparation 2026) taken in
#                 X-coordinates and mapped back to the Lamperti scale.  The
#                 retake uses the SAME Brownian increment that made the
#                 explicit step fail plus one auxiliary exponential that
#                 supplies the bridge running infimum (Asmussen--Glynn--
#                 Pitman), so the coupling requirement is preserved.  For
#                 delta >= 1 the output is a.s. strictly positive; for
#                 delta < 1 the modified BLT flow can reach zero, so the
#                 result is clipped to the same gamma*sqrt(h) floor as the
#                 projected map.  Requires an rng for the exponentials.
#
# All maps return strictly positive values, as condition (ii) of the
# admissibility definition requires.

import warnings

import numpy as np

from src.utils.cir_params import kl_coefficients
from src.samplers.blt_splitting import blt_step
from src.samplers.lamperti_implicit import if_step
from src.utils.rng import sample_bridge_infimum


def default_backstop_kind(alpha: float) -> str:
    return "implicit" if alpha > 0.0 else "projected"


def _backstop_map(
    y: np.ndarray,
    h: np.ndarray,
    dW: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
    kind: str,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if kind == "implicit":
        if alpha <= 0.0:
            raise ValueError(
                "implicit backstop requires alpha > 0 (delta > 1); "
                "use backstop='projected' in this regime"
            )
        return if_step(y=y, alpha=alpha, beta=beta, gamma=gamma, dt=h, dW=dW)

    if kind == "projected":
        y_floor = gamma * np.sqrt(h)
        y_hat = y + (alpha / y + beta * y) * h + gamma * dW
        return np.maximum(y_hat, y_floor)

    if kind == "blt":
        if rng is None:
            raise ValueError(
                "backstop='blt' needs an rng for the auxiliary bridge-"
                "infimum exponentials"
            )
        # Lamperti coefficients -> normal-form (a, b): sigma = 2*gamma,
        # a = 4*kappa*theta/sigma^2 = 1 + 2*alpha/gamma^2, b = kappa = -2*beta.
        a = 1.0 + 2.0 * alpha / gamma**2
        b = -2.0 * beta

        # Same increment as the failed explicit step; only the running
        # infimum is drawn afresh, conditionally on dW.
        m = sample_bridge_infimum(rng, dW, h)

        # Y -> normal-form x = (y/gamma)^2, one BLT step, back to Y.
        # blt_step is elementwise, so per-path step sizes h are fine.
        x_next = blt_step((y / gamma) ** 2, a, b, h, dW, m)

        y_floor = gamma * np.sqrt(h)
        return np.maximum(gamma * np.sqrt(np.maximum(x_next, 0.0)), y_floor)

    raise ValueError(f"unknown backstop kind: {kind!r}")


def klm_backstop_terminal(
    X0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
    h_max: float,
    n_paths: int,
    rng: np.random.Generator,
    rho: float = 64.0,
    backstop: str | None = None,
    max_rounds: int = 10_000_000,
) -> tuple[np.ndarray, dict]:
    """Free-running KLM backstopped adaptive scheme (own Brownian increments).

    Returns (X_T, stats) where X_T = Y_T**2 and stats records total steps and
    how often each backstop trigger fired.
    """
    alpha, beta, gamma = kl_coefficients(kappa, theta, sigma)
    kind = backstop if backstop is not None else default_backstop_kind(alpha)
    h_min = h_max / rho

    y = np.full(n_paths, np.sqrt(X0), dtype=float)
    t = np.zeros(n_paths, dtype=float)

    n_steps_total = 0
    n_backstop_min = 0
    n_backstop_neg = 0

    rounds = 0
    while np.any(t < T - 1e-14):
        rounds += 1
        if rounds > max_rounds:
            raise RuntimeError("klm_backstop did not reach T; check parameters")

        active = t < T - 1e-14
        idx = np.flatnonzero(active)

        y_a = y[idx]
        h_prop = h_max * np.minimum(1.0, np.abs(y_a))

        min_triggered = h_prop < h_min
        h = np.where(min_triggered, h_min, h_prop)
        h = np.minimum(h, T - t[idx])

        dW = np.sqrt(h) * rng.standard_normal(idx.size)

        y_next = np.empty_like(y_a)

        # (a) sub-floor proposals go straight to the backstop map.
        if np.any(min_triggered):
            y_next[min_triggered] = _backstop_map(
                y_a[min_triggered],
                h[min_triggered],
                dW[min_triggered],
                alpha,
                beta,
                gamma,
                kind,
                rng=rng,
            )
            n_backstop_min += int(np.count_nonzero(min_triggered))

        # Explicit Lamperti Euler step everywhere else.
        explicit = ~min_triggered
        if np.any(explicit):
            y_e = y_a[explicit]
            y_try = y_e + (alpha / y_e + beta * y_e) * h[explicit] + gamma * dW[explicit]

            # (b) non-positive outputs are retaken with the backstop map,
            # using the same step size and the same Brownian increment.
            bad = y_try <= 0.0
            if np.any(bad):
                y_try[bad] = _backstop_map(
                    y_e[bad],
                    h[explicit][bad],
                    dW[explicit][bad],
                    alpha,
                    beta,
                    gamma,
                    kind,
                    rng=rng,
                )
                n_backstop_neg += int(np.count_nonzero(bad))

            y_next[explicit] = y_try

        y[idx] = y_next
        t[idx] = t[idx] + h
        n_steps_total += idx.size

    stats = {
        "n_steps_total": n_steps_total,
        "n_backstop_min": n_backstop_min,
        "n_backstop_neg": n_backstop_neg,
        "backstop_fraction": (n_backstop_min + n_backstop_neg) / max(n_steps_total, 1),
        "backstop_kind": kind,
    }
    return y**2, stats


def klm_backstop_terminal_from_fine_dW(
    X0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
    h_max: float,
    dW_fine: np.ndarray,
    rho: float = 64.0,
    backstop: str | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict]:
    """Brownian-coupled KLM scheme for strong-error experiments.

    ``rng`` is only consumed by backstop='blt': it supplies the auxiliary
    bridge-infimum exponentials, which are independent of the coupled
    Brownian path, so the strong-error coupling on W is unaffected.

    Adaptive step sizes are quantised DOWN to whole multiples of the fine
    reference grid (minimum one fine step), so every Brownian increment the
    scheme consumes is an exact partial sum of the fine increments.  The
    scheme and the fine-grid reference therefore see the same Brownian path,
    which is the coupling required by the strong-error definition.

    Quantisation replaces the continuous adaptive rule by its floor on the
    fine grid; choose reference_n_steps so that dt_fine <= h_max / rho, in
    which case the floor h_min is representable and the quantisation error is
    at most one fine step per adaptive step.
    """
    alpha, beta, gamma = kl_coefficients(kappa, theta, sigma)
    kind = backstop if backstop is not None else default_backstop_kind(alpha)

    n_paths, n_fine = dW_fine.shape
    dt_fine = T / n_fine

    h_min = h_max / rho
    if dt_fine > h_min * (1.0 + 1e-12):
        warnings.warn(
            f"fine grid coarser than the KLM backstop floor: "
            f"dt_fine={dt_fine:.3e} > h_min=h_max/rho={h_min:.3e}; the "
            f"quantised floor becomes one fine step and the adaptive rule is "
            f"distorted below h_min. Increase reference_n_steps so that "
            f"dt_fine <= h_max/rho.",
            RuntimeWarning,
            stacklevel=2,
        )
    m_min = max(int(round(h_min / dt_fine)), 1)
    m_max = max(int(np.floor(h_max / dt_fine)), 1)

    # Cumulative Brownian path at fine grid times, W(0) = 0.
    W = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(dW_fine, axis=1)], axis=1
    )

    y = np.full(n_paths, np.sqrt(X0), dtype=float)
    pos = np.zeros(n_paths, dtype=np.int64)  # index into the fine grid

    n_steps_total = 0
    n_backstop_min = 0
    n_backstop_neg = 0

    while np.any(pos < n_fine):
        active = pos < n_fine
        idx = np.flatnonzero(active)

        y_a = y[idx]
        h_prop = h_max * np.minimum(1.0, np.abs(y_a))
        m = np.floor(h_prop / dt_fine).astype(np.int64)

        min_triggered = m < m_min
        m = np.where(min_triggered, m_min, np.minimum(m, m_max))
        m = np.minimum(m, n_fine - pos[idx])

        h = m * dt_fine
        dW = W[idx, pos[idx] + m] - W[idx, pos[idx]]

        y_next = np.empty_like(y_a)

        if np.any(min_triggered):
            y_next[min_triggered] = _backstop_map(
                y_a[min_triggered],
                h[min_triggered],
                dW[min_triggered],
                alpha,
                beta,
                gamma,
                kind,
                rng=rng,
            )
            n_backstop_min += int(np.count_nonzero(min_triggered))

        explicit = ~min_triggered
        if np.any(explicit):
            y_e = y_a[explicit]
            y_try = y_e + (alpha / y_e + beta * y_e) * h[explicit] + gamma * dW[explicit]

            bad = y_try <= 0.0
            if np.any(bad):
                y_try[bad] = _backstop_map(
                    y_e[bad],
                    h[explicit][bad],
                    dW[explicit][bad],
                    alpha,
                    beta,
                    gamma,
                    kind,
                    rng=rng,
                )
                n_backstop_neg += int(np.count_nonzero(bad))

            y_next[explicit] = y_try

        y[idx] = y_next
        pos[idx] = pos[idx] + m
        n_steps_total += idx.size

    stats = {
        "n_steps_total": n_steps_total,
        "n_backstop_min": n_backstop_min,
        "n_backstop_neg": n_backstop_neg,
        "backstop_fraction": (n_backstop_min + n_backstop_neg) / max(n_steps_total, 1),
        "backstop_kind": kind,
        "m_min_fine_steps": m_min,
    }
    return y**2, stats
