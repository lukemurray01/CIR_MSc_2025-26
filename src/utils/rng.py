# random number generator script

import numpy as np

def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)

def make_brownian_increments(
        rng: np.random.Generator,
        n_paths: int,
        n_steps: int,
        dt: float,
) -> np.ndarray:
    return np.sqrt(dt) * rng.standard_normal((n_paths, n_steps))


def sample_bridge_infimum(
        rng: np.random.Generator,
        dW: np.ndarray,
        dt: float | np.ndarray,
) -> np.ndarray:
    """Running infimum of a Brownian step, conditional on its endpoint.

    Given an increment dW = W_{t+dt} - W_t, samples

        m  =  inf_{u in [t, t+dt]} (W_u - W_t)
           =  (dW - sqrt(V + dW^2)) / 2,    V ~ Exp(1/(2*dt)),

    which is the Asmussen--Glynn--Pitman representation of the Brownian
    bridge minimum (BLT paper, Lemma 2.2 with gamma_1 = 0, gamma_2 = 1).
    Always m <= min(0, dW), strictly when V > 0.  Because the auxiliary
    exponential is independent of dW, a backstop retake can reuse the SAME
    increment that made an explicit step fail and only draw V afresh.
    """
    V = rng.exponential(scale=2.0 * dt, size=np.shape(dW))
    return 0.5 * (dW - np.sqrt(V + dW**2))


def make_brownian_increments_with_infima(
        rng: np.random.Generator,
        n_paths: int,
        n_steps: int,
        dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Joint exact sample of per-step increments and per-step running infima.

    Returns (dW, m) with dW[i, j] = W_{t_{j+1}} - W_{t_j} and
    m[i, j] = inf over [t_j, t_{j+1}] of (W_u - W_{t_j}).  The marginal law
    of dW is exactly N(0, dt), so schemes that only need increments can
    consume dW unchanged and remain coupled to schemes that also use m.
    """
    dW = make_brownian_increments(rng, n_paths, n_steps, dt)
    m = sample_bridge_infimum(rng, dW, dt)
    return dW, m
