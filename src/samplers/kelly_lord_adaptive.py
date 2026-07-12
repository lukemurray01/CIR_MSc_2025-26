# for regimes D-E
import numpy as np

from src.utils.cir_params import kl_alpha


def soft_zero_threshold(kappa, theta, dt_max, rho=2.0):
    # Soft-zero boundary X_zero 
    return theta * (1.0 - np.exp(-kappa * dt_max)) / rho


def kl_adaptive_terminal(
        X0: float,
        kappa: float,
        theta: float,
        sigma: float,
        T: float,
        dt_max: float,
        n_paths: int,
        rng: np.random.Generator,
        rho: float = 2.0,
        max_rounds: int = 1_000_000,
) -> np.ndarray:
    alpha = kl_alpha(kappa, theta, sigma)
    X_zero = soft_zero_threshold(kappa, theta, dt_max, rho)

    x = np.full(n_paths, X0, dtype=float)
    t = np.zeros(n_paths, dtype=float)

    rounds = 0
    # Each path advances until it reaches T. Finished paths are masked out.
    while np.any(t < T - 1e-12):
        rounds += 1
        if rounds > max_rounds:
            raise RuntimeError("kl_adaptive did not reach T; check parameters")

        active = t < T - 1e-12
        dt_remaining = T - t

        in_soft_zero = active & (x < X_zero)
        in_splitting = active & ~in_soft_zero

        dt = np.zeros(n_paths, dtype=float)

        # --- soft-zero region deterministic ODE
        soft_exit_completed = None
        if np.any(in_soft_zero):
            xs = x[in_soft_zero]
            dt_sz = -np.log((X_zero - theta) / (xs - theta)) / kappa
            remaining_sz = dt_remaining[in_soft_zero]
            soft_exit_completed = dt_sz <= remaining_sz
            dt[in_soft_zero] = np.minimum(dt_sz, remaining_sz)

        # --- splitting region do the adaptive step
        if np.any(in_splitting):
            if alpha < 0.0:
                # Shrink so the inner sqrt argument stays positive
                dt_adaptive = 0.95 * x[in_splitting] / (2.0 * abs(alpha))
                dt[in_splitting] = np.minimum(
                    np.minimum(dt_adaptive, dt_max), dt_remaining[in_splitting]
                )
            else:
                # alpha >= 0: positivity is automatic; no shrinking needed.
                dt[in_splitting] = np.minimum(dt_max, dt_remaining[in_splitting])

        x_next = x.copy()

        # apply soft-zero deterministic flow
        if np.any(in_soft_zero):
            h = dt[in_soft_zero]
            decay = np.exp(-kappa * h)
            x_new = decay * x[in_soft_zero] + theta * (1.0 - decay)
            # dt_sz is by definition the exact hitting time of X_zero, but
            # the floating-point flow can land a few ulp BELOW the boundary,
            # re-entering the soft-zero region with a vanishing next step and
            # stalling the path until max_rounds (observed in regime E at
            # coarse dt_max).  Snap completed exits onto the boundary; steps
            # clipped by the remaining horizon are left where the flow ends.
            x_next[in_soft_zero] = np.where(
                soft_exit_completed, X_zero, x_new
            )

        # apply splitting update
        if np.any(in_splitting):
            h = dt[in_splitting]
            dW = np.sqrt(h) * rng.standard_normal(np.count_nonzero(in_splitting))
            inside_sqrt = x[in_splitting] + 2.0 * alpha * h
            # stop floating point round off such that it guarantees inside_sqrt > 0.
            if np.any(inside_sqrt < -1e-14):
                raise RuntimeError("negative square-root argument")
            inside_sqrt = np.maximum(inside_sqrt, 0.0)

            x_next[in_splitting] = np.exp(-kappa * h) * (
                np.sqrt(inside_sqrt) + 0.5 * sigma * dW
            ) ** 2

        x = x_next
        t = t + dt

    return x


def kl_adaptive_terminal_from_fine_dW(
        X0: float,
        kappa: float,
        theta: float,
        sigma: float,
        T: float,
        dt_max: float,
        dW_fine: np.ndarray,
        rho: float = 2.0,
        safety: float = 0.95,
        max_rounds: int = 10_000_000,
) -> tuple[np.ndarray, dict]:
    """Brownian-coupled adaptive Kelly--Lord terminal values.

    This is the strong-error counterpart of ``kl_adaptive_terminal``.  Adaptive
    times are quantised to the fine Brownian grid so the KL path and the HH
    reference path consume the same underlying Brownian trajectory.  The
    routine is mainly used when alpha < 0, where the uniform KL square-root
    update is not defined for all inputs.
    """
    alpha = kl_alpha(kappa, theta, sigma)
    X_zero = soft_zero_threshold(kappa, theta, dt_max, rho)

    n_paths, n_fine = dW_fine.shape
    dt_fine = T / n_fine

    W = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(dW_fine, axis=1)], axis=1
    )
    rows = np.arange(n_paths)

    x = np.full(n_paths, X0, dtype=float)
    pos = np.zeros(n_paths, dtype=np.int64)

    n_steps_total = 0
    n_soft_zero = 0
    n_splitting = 0

    rounds = 0
    while np.any(pos < n_fine):
        rounds += 1
        if rounds > max_rounds:
            raise RuntimeError("coupled kl_adaptive did not reach T")

        idx = np.flatnonzero(pos < n_fine)
        x_a = x[idx]
        remaining_steps = n_fine - pos[idx]
        remaining_time = remaining_steps * dt_fine

        in_soft_zero = x_a < X_zero
        in_splitting = ~in_soft_zero

        h_continuous = np.zeros_like(x_a)

        if np.any(in_soft_zero):
            xs = x_a[in_soft_zero]
            dt_sz = -np.log((X_zero - theta) / (xs - theta)) / kappa
            h_continuous[in_soft_zero] = np.minimum(
                dt_sz, remaining_time[in_soft_zero]
            )

        if np.any(in_splitting):
            if alpha < 0.0:
                dt_adaptive = safety * x_a[in_splitting] / (2.0 * abs(alpha))
                h_continuous[in_splitting] = np.minimum(
                    np.minimum(dt_adaptive, dt_max),
                    remaining_time[in_splitting],
                )
            else:
                h_continuous[in_splitting] = np.minimum(
                    dt_max, remaining_time[in_splitting]
                )

        m = np.floor(h_continuous / dt_fine).astype(np.int64)
        m = np.maximum(m, 1)
        m = np.minimum(m, remaining_steps)
        h = m * dt_fine

        x_next = x_a.copy()

        if np.any(in_soft_zero):
            h_soft = h[in_soft_zero]
            decay = np.exp(-kappa * h_soft)
            x_next[in_soft_zero] = (
                decay * x_a[in_soft_zero] + theta * (1.0 - decay)
            )
            n_soft_zero += int(np.count_nonzero(in_soft_zero))

        if np.any(in_splitting):
            h_split = h[in_splitting]
            split_idx = idx[in_splitting]
            dW = W[split_idx, pos[split_idx] + m[in_splitting]] - W[
                split_idx, pos[split_idx]
            ]
            inside_sqrt = x_a[in_splitting] + 2.0 * alpha * h_split
            if np.any(inside_sqrt < -1e-14):
                raise RuntimeError(
                    "negative square-root argument after fine-grid quantisation"
                )
            inside_sqrt = np.maximum(inside_sqrt, 0.0)
            x_next[in_splitting] = np.exp(-kappa * h_split) * (
                np.sqrt(inside_sqrt) + 0.5 * sigma * dW
            ) ** 2
            n_splitting += int(np.count_nonzero(in_splitting))

        x[idx] = x_next
        pos[idx] = pos[idx] + m
        n_steps_total += idx.size

    stats = {
        "n_steps_total": n_steps_total,
        "n_soft_zero": n_soft_zero,
        "n_splitting": n_splitting,
        "mean_steps_per_path": n_steps_total / n_paths,
        "soft_zero_fraction": n_soft_zero / max(n_steps_total, 1),
    }
    return x, stats
