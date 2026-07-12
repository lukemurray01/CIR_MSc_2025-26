import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ncx2

from src.samplers.exact import simulate_paths


# Parameters
X0 = 0.05
kappa = 0.05
theta = 0.05
sigma = 0.02

T = 1.0
n_steps = 100
n_paths = 50_000
dt = T / n_steps

seed = 123
rng = np.random.default_rng(seed)


# Simulate exact CIR paths
X = simulate_paths(
    X0=X0,
    kappa=kappa,
    theta=theta,
    sigma=sigma,
    dt=dt,
    n_steps=n_steps,
    n_paths=n_paths,
    rng=rng,
)
# We only need the terminal value for this diagnostic
X_T = X[:, -1]


# Theoretical mean and variance of CIR terminal value
exp_term = np.exp(-kappa * T)

theoretical_mean = theta + (X0 - theta) * exp_term

theoretical_variance = (
    X0 * sigma**2 * exp_term * (1.0 - exp_term) / kappa
    + theta * sigma**2 * (1.0 - exp_term) ** 2 / (2.0 * kappa)
)

sample_mean = np.mean(X_T)
sample_variance = np.var(X_T, ddof=1)


print("Exact CIR sampler sanity check")
print("--------------------------------")
print(f"n_paths:              {n_paths}")
print(f"X0:                   {X0}")
print(f"kappa:                {kappa}")
print(f"theta:                {theta}")
print(f"sigma:                {sigma}")
print(f"T:                    {T}")
print()
print(f"Theoretical mean:     {theoretical_mean:.8f}")
print(f"Sample mean:          {sample_mean:.8f}")
print(f"Absolute mean error:  {abs(sample_mean - theoretical_mean):.8e}")
print()
print(f"Theoretical variance: {theoretical_variance:.8e}")
print(f"Sample variance:      {sample_variance:.8e}")
print(f"Absolute var error:   {abs(sample_variance - theoretical_variance):.8e}")


# Exact terminal density using the noncentral chi-square law
c = 4.0 * kappa / (sigma**2 * (1.0 - exp_term))
df = 4.0 * kappa * theta / sigma**2
nc = c * X0 * exp_term

x_grid = np.linspace(np.min(X_T), np.max(X_T), 500)

# If Z has density f_Z and X_T = Z / c, then f_X(x) = c f_Z(cx)
density = c * ncx2.pdf(c * x_grid, df, nc)


# Plot histogram against exact density
plt.hist(
    X_T,
    bins=80,
    density=True,
    alpha=0.5,
    label="Simulated terminal values",
)

plt.plot(
    x_grid,
    density,
    label="Exact terminal density",
)

plt.xlabel("X_T")
plt.ylabel("Density")
plt.title("Exact CIR sampler: terminal distribution check")
plt.legend()
plt.grid(True)
plt.show()