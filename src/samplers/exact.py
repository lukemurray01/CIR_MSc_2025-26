import numpy as np


def cir_ncx2_params(x, kappa, theta, sigma, dt):
    df = 4.0 * kappa * theta / sigma**2
    c = 4.0 * kappa / (sigma**2 * (1.0 - np.exp(-kappa * dt)))
    nc = c * x * np.exp(-kappa * dt)
    return c, df, nc


def simulate_paths(X0, kappa, theta, sigma, dt, n_steps, n_paths, rng):
    X = np.empty((n_paths, n_steps + 1))
    X[:, 0] = X0

    for i in range(n_steps):
        c, df, nc = cir_ncx2_params(X[:, i], kappa, theta, sigma, dt)
        Z = rng.noncentral_chisquare(df, nc)
        X[:, i + 1] = Z / c

    return X
